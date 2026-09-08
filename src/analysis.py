"""Diagnostics that ask *why* the value-growth model works and when it doesn't:
transfer-driven re-valuations, market-wide drift, survivorship, and so on.
Used by notebook 06; the heavy lifting (training, walk-forward) stays in modeling.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from src.data_loader import PREMIER_LEAGUE_COMPETITION_ID
from src.modeling import TARGET_CLIP, XGB_DEFAULTS, _backtest_metrics
from src.panel import HORIZON_TOLERANCE_DAYS


def flag_transfers_within_horizon(
    predictions: pd.DataFrame, transfers: pd.DataFrame, months: int
) -> pd.DataFrame:
    """Mark each backtest prediction with whether the player changed club inside the
    outcome window (snapshot date to snapshot + months + matching tolerance).

    Transfermarkt re-values a player straight after a transfer, usually toward the fee,
    so a "prediction" that lands in this window may really be a prediction that a move
    happens. `paid_transfer` narrows it to moves with a recorded fee above zero.

    `transfers` needs player_id, transfer_date, transfer_fee, and from/to club ids;
    rows where the club doesn't change (loan returns recorded as moves) are ignored."""
    window_days = HORIZON_TOLERANCE_DAYS[months]
    moves = transfers.dropna(subset=["player_id", "transfer_date"]).copy()
    if {"from_club_id", "to_club_id"} <= set(moves.columns):
        moves = moves[moves["from_club_id"] != moves["to_club_id"]]
    moves = moves[["player_id", "transfer_date", "transfer_fee"]]

    keyed = predictions[["player_id", "snapshot_date"]].reset_index()
    joined = keyed.merge(moves, on="player_id", how="left")
    window_end = joined["snapshot_date"] + pd.DateOffset(months=months) + pd.Timedelta(days=window_days)
    in_window = (joined["transfer_date"] > joined["snapshot_date"]) & (joined["transfer_date"] <= window_end)
    joined = joined[in_window]

    flags = pd.DataFrame(
        {
            "transferred": joined.groupby("index").size() > 0,
            "paid_transfer": joined.groupby("index")["transfer_fee"].max() > 0,
        }
    )
    out = predictions.copy()
    out["transferred"] = flags["transferred"].reindex(out.index).fillna(False).astype(bool)
    out["paid_transfer"] = flags["paid_transfer"].reindex(out.index).fillna(False).astype(bool)
    return out


def grouped_backtest_metrics(predictions: pd.DataFrame, group_col: str, min_rows: int = 30) -> pd.DataFrame:
    """Backtest scores per (cutoff, group), for splitting the eval set by some flag."""
    rows = []
    for (cutoff, group), part in predictions.groupby(["cutoff", group_col]):
        if len(part) < min_rows:
            continue
        rows.append(
            {
                "cutoff": cutoff,
                group_col: group,
                "n": len(part),
                "share": len(part) / (predictions["cutoff"] == cutoff).sum(),
                **_backtest_metrics(part["predicted_pct_change"], part["actual_pct_change"]),
            }
        )
    return pd.DataFrame(rows)


def market_drift_decomposition(predictions: pd.DataFrame) -> pd.DataFrame:
    """Per cutoff: how much the whole market moved, what the model implicitly assumed it
    would, and how much error disappears once that market-wide bias is removed.

    Works on log(value ratio) so up and down moves are symmetric. `mae_demeaned` is the
    error after subtracting each cutoff's mean error from every prediction, i.e. the
    error a model with perfect knowledge of the market-wide move would have had; the gap
    to `mae` is the price of not knowing where the market is heading."""
    rows = []
    for cutoff, part in predictions.groupby("cutoff"):
        actual = np.log1p(part["actual_pct_change"].clip(*TARGET_CLIP))
        predicted = np.log1p(part["predicted_pct_change"].clip(*TARGET_CLIP))
        error = predicted - actual
        rows.append(
            {
                "cutoff": cutoff,
                "n": len(part),
                "market_actual_log_change": float(actual.mean()),
                "market_predicted_log_change": float(predicted.mean()),
                "market_bias": float(error.mean()),
                "mae": float(error.abs().mean()),
                "mae_demeaned": float((error - error.mean()).abs().mean()),
            }
        )
    out = pd.DataFrame(rows)
    out["share_of_mae_from_market"] = 1 - out["mae_demeaned"] / out["mae"]
    return out


def augment_targets_with_other_leagues(
    panel: pd.DataFrame, valuations_all_leagues: pd.DataFrame, months: int
) -> pd.DataFrame:
    """Fill in horizon targets for players who left the Premier League, using their
    valuation in whichever league they went to.

    The panel only matches targets against PL valuations, so a player who grows and moves
    abroad has no outcome and silently drops out of every accuracy number. This adds a
    `target_source_{months}m` column ("pl", "other_league", or "none") and fills the
    future-value / change columns for the "other_league" rows so they can be evaluated.

    `valuations_all_leagues` is the raw player_valuations table (dates parsed)."""
    target_col = f"value_change_{months}m_pct"
    future_col = f"future_value_{months}m_eur"
    source_col = f"target_source_{months}m"
    out = panel.copy()
    out[source_col] = np.where(out[target_col].notna(), "pl", "none")

    abroad = valuations_all_leagues[
        valuations_all_leagues["player_club_domestic_competition_id"] != PREMIER_LEAGUE_COMPETITION_ID
    ][["player_id", "date", "market_value_in_eur", "player_club_domestic_competition_id"]].dropna()
    abroad = abroad.sort_values("date")

    unmatched = out[out[source_col] == "none"][["player_id", "snapshot_date"]].copy()
    unmatched["target_date"] = unmatched["snapshot_date"] + pd.DateOffset(months=months)
    ordered = unmatched.sort_values("target_date").reset_index()
    found = pd.merge_asof(
        ordered,
        abroad,
        left_on="target_date",
        right_on="date",
        by="player_id",
        direction="nearest",
        tolerance=pd.Timedelta(days=HORIZON_TOLERANCE_DAYS[months]),
    ).set_index("index")
    found = found[found["market_value_in_eur"].notna()]

    out.loc[found.index, future_col] = found["market_value_in_eur"]
    out.loc[found.index, f"value_change_{months}m_eur"] = (
        found["market_value_in_eur"] - out.loc[found.index, "current_value_eur"]
    )
    out.loc[found.index, target_col] = (
        out.loc[found.index, f"value_change_{months}m_eur"] / out.loc[found.index, "current_value_eur"]
    )
    out.loc[found.index, source_col] = "other_league"
    out[f"future_league_{months}m"] = np.where(out[source_col] == "pl", PREMIER_LEAGUE_COMPETITION_ID, None)
    out.loc[found.index, f"future_league_{months}m"] = found["player_club_domestic_competition_id"]
    return out


STATS_ONLY_NUMERIC = [
    "age",
    "age_sq",
    "trailing_goals_per90",
    "trailing_assists_per90",
    "trailing_minutes",
    "trailing_appearances",
    "trailing_avg_club_position",
]
STATS_ONLY_CATEGORICAL = ["sub_position", "foot"]


def _stats_only_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), STATS_ONLY_NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), STATS_ONLY_CATEGORICAL),
        ]
    )
    return Pipeline([("preprocess", preprocessor), ("model", XGBRegressor(**XGB_DEFAULTS))])


def add_stats_only_residual(panel: pd.DataFrame, min_train_rows: int = 2000) -> pd.DataFrame:
    """The "eye-test" residual from notebook 03, computed without looking forward.

    Fits a stats-only model of log10(value) (age, output, minutes, team context, role -
    no market signals) on all snapshots *before* each calendar year and predicts that
    year's snapshots. `stats_residual` = actual log10 value minus that prediction, so a
    positive value means "valued above what the stats alone suggest". Years without
    `min_train_rows` of history get NaN. Because it only uses the past, it can be a
    feature in the walk-forward growth model without leakage."""
    df = panel.copy()
    df["age_sq"] = df["age"] ** 2
    df["sub_position"] = df["sub_position"].fillna("unknown")
    df["foot"] = df["foot"].fillna("unknown")
    df["log_value"] = np.log10(df["current_value_eur"])
    features = STATS_ONLY_NUMERIC + STATS_ONLY_CATEGORICAL
    usable = df.dropna(subset=features + ["log_value", "snapshot_date"])

    prediction = pd.Series(np.nan, index=panel.index)
    for year in sorted(usable["snapshot_date"].dt.year.unique()):
        year_start = pd.Timestamp(year=year, month=1, day=1)
        train = usable[usable["snapshot_date"] < year_start]
        target = usable[usable["snapshot_date"].dt.year == year]
        if len(train) < min_train_rows or target.empty:
            continue
        pipeline = _stats_only_pipeline()
        pipeline.fit(train[features], train["log_value"])
        prediction.loc[target.index] = pipeline.predict(target[features])

    out = panel.copy()
    out["stats_only_log_value_pred"] = prediction
    out["stats_residual"] = df["log_value"] - prediction
    return out
