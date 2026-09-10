"""Freeze the forward value-growth predictions so they can be scored later.

Notebook 02's predictions only become a true out-of-sample test once the upstream
dataset is re-scraped with valuations dated after they were made. This module writes
them to a tracked, dated file, and scores every frozen file against whatever
valuations the current dataset holds:

    uv run python -m src.forecast            # freeze today's predictions
    uv run python -m src.forecast --score    # score every frozen file against the data

Each frozen file trains the project's default model on every snapshot whose outcome
had resolved, predicts each player's latest snapshot, and records the prediction, a
10-90% quantile band, and the date by which the outcome will be known.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from src.data_loader import PROJECT_ROOT, download_dataset
from src.modeling import (
    _backtest_metrics,
    fit_model,
    fit_quantile_models,
    latest_snapshot_per_player,
    modelable_rows,
    predict_value_growth,
    predict_value_growth_interval,
    target_column,
    target_resolution_date,
)
from src.panel import HORIZONS_MONTHS, add_horizon_targets, build_snapshot_panel, load_all_valuations

FORECAST_DIR = PROJECT_ROOT / "forecasts"
DEFAULT_MODEL = "xgboost"
BAND_QUANTILES = (0.1, 0.9)

FORECAST_COLUMNS = [
    "player_id",
    "name",
    "sub_position",
    "age",
    "snapshot_date",
    "horizon_months",
    "outcome_due_date",
    "model",
    "current_value_eur",
    "predicted_pct_change",
    "predicted_value_eur",
    "pct_q10",
    "pct_q90",
]


def make_forward_predictions(
    panel: pd.DataFrame,
    horizons: Sequence[int] = HORIZONS_MONTHS,
    model_name: str = DEFAULT_MODEL,
    quantiles: Sequence[float] = BAND_QUANTILES,
    max_age_days: int = 365,
) -> pd.DataFrame:
    """One row per (player, horizon): the prediction for each player's latest snapshot.

    For each horizon the model is fit on every row of the panel that has that horizon's
    outcome, then applied to each player's most recent snapshot (within `max_age_days`
    of the panel's end). `outcome_due_date` is the latest date at which the outcome can
    still be matched, i.e. when the row becomes scoreable. Only rows with no outcome yet
    are kept: if the panel already holds a valuation inside the window, that row was in
    the training data; if the window has closed with nothing in it, nothing will come."""
    as_of = panel["snapshot_date"].max()
    latest = latest_snapshot_per_player(panel, max_age_days)
    frames = []
    for months in horizons:
        train = modelable_rows(panel, months)
        unresolved = latest[
            latest[target_column(months)].isna()
            & (target_resolution_date(latest["snapshot_date"], months) > as_of)
        ]
        if train.empty or unresolved.empty:
            continue
        pipeline = fit_model(train, model_name, months)
        preds = predict_value_growth(pipeline, unresolved)
        band = predict_value_growth_interval(fit_quantile_models(train, months, quantiles), unresolved)
        band_cols = [f"pct_q{int(round(q * 100)):02d}" for q in quantiles]
        preds = preds.merge(band[["player_id", *band_cols]], on="player_id", how="left")
        preds = preds.merge(latest[["player_id", "snapshot_date"]], on="player_id", how="left")
        preds["horizon_months"] = months
        preds["outcome_due_date"] = target_resolution_date(preds["snapshot_date"], months)
        preds["model"] = model_name
        frames.append(preds)
    out = pd.concat(frames, ignore_index=True)
    columns = [c for c in FORECAST_COLUMNS if c in out]
    return out[columns].sort_values(["horizon_months", "player_id"]).reset_index(drop=True)


def forecast_path(as_of: pd.Timestamp, dataset_version: str) -> Path:
    return FORECAST_DIR / f"forward_{pd.Timestamp(as_of).date()}_data{dataset_version}.csv"


def save_forward_predictions(predictions: pd.DataFrame, as_of: pd.Timestamp, dataset_version: str) -> Path:
    path = forecast_path(as_of, dataset_version)
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(path, index=False, date_format="%Y-%m-%d")
    return path


def load_forecasts(directory: Path = FORECAST_DIR) -> pd.DataFrame:
    """Every frozen forecast file, stacked, with a `forecast_file` column."""
    frames = []
    for path in sorted(directory.glob("forward_*.csv")):
        df = pd.read_csv(path, parse_dates=["snapshot_date", "outcome_due_date"])
        df.insert(0, "forecast_file", path.name)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["forecast_file", *FORECAST_COLUMNS])
    return pd.concat(frames, ignore_index=True)


def resolve_forecast_outcomes(forecast: pd.DataFrame, valuations: pd.DataFrame) -> pd.DataFrame:
    """Attach each forecast row's realised outcome, where the data now holds one.

    `valuations` is the all-league valuation table (see `load_all_valuations`). Matching
    is exactly the panel's: the nearest valuation to snapshot + horizon within that
    horizon's tolerance. Adds `actual_pct_change` (NaN where nothing matched), `resolved`,
    and `due` (whether the matching window had closed by the data's latest valuation)."""
    keys = forecast[["player_id", "snapshot_date", "current_value_eur"]].drop_duplicates()
    matched = add_horizon_targets(keys, valuations.sort_values("date"))
    out = forecast.copy()
    out["actual_pct_change"] = float("nan")
    for months, part in out.groupby("horizon_months"):
        col = target_column(int(months))
        lookup = matched.set_index(["player_id", "snapshot_date"])[col]
        idx = pd.MultiIndex.from_frame(part[["player_id", "snapshot_date"]])
        out.loc[part.index, "actual_pct_change"] = lookup.reindex(idx).to_numpy()
    out["resolved"] = out["actual_pct_change"].notna()
    out["due"] = out["outcome_due_date"] <= valuations["date"].max()
    return out


def score_forecasts(resolved: pd.DataFrame) -> pd.DataFrame:
    """Accuracy per (forecast file, horizon) on the rows whose outcome is known, with the
    share of rows that are due and resolved so a partial score is read as partial."""
    rows = []
    for (file, months), part in resolved.groupby(["forecast_file", "horizon_months"]):
        scored = part[part["resolved"]]
        row = {
            "forecast_file": file,
            "horizon_months": months,
            "n": len(part),
            "share_due": float(part["due"].mean()),
            "share_resolved": float(part["resolved"].mean()),
        }
        if len(scored) >= 2:
            row.update(_backtest_metrics(scored["predicted_pct_change"], scored["actual_pct_change"]))
            inside = (scored["actual_pct_change"] >= scored["pct_q10"]) & (
                scored["actual_pct_change"] <= scored["pct_q90"]
            )
            row["band_coverage"] = float(inside.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--score", action="store_true", help="score every frozen forecast instead")
    args = parser.parse_args()

    data_dir = download_dataset()
    if args.score:
        forecasts = load_forecasts()
        if forecasts.empty:
            print(f"no forecasts found in {FORECAST_DIR}")
            return
        valuations = load_all_valuations(data_dir)
        print(f"dataset version {data_dir.name}, valuations to {valuations['date'].max().date()}")
        resolved = resolve_forecast_outcomes(forecasts, valuations)
        pd.set_option("display.width", 200)
        scores = score_forecasts(resolved)
        show = [
            c
            for c in [
                "forecast_file",
                "horizon_months",
                "n",
                "share_due",
                "share_resolved",
                "spearman",
                "mae",
                "calibration_slope",
                "direction_accuracy",
                "band_coverage",
            ]
            if c in scores
        ]
        print(scores[show].round(3).to_string(index=False))
        if not resolved["due"].any():
            print("nothing is due yet: re-run after the dataset has been re-scraped past the due dates")
        return

    panel = build_snapshot_panel()
    as_of = panel["snapshot_date"].max()
    predictions = make_forward_predictions(panel)
    path = save_forward_predictions(predictions, as_of, data_dir.name)
    per_horizon = predictions.groupby("horizon_months").size().to_dict()
    print(
        f"wrote {len(predictions):,} predictions to {path.relative_to(PROJECT_ROOT)} (as of {as_of.date()})"
    )
    print("per horizon:", per_horizon)
    print(
        "first due:",
        predictions["outcome_due_date"].min().date(),
        "| last due:",
        predictions["outcome_due_date"].max().date(),
    )


if __name__ == "__main__":
    main()
