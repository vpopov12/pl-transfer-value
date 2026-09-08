from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.modeling import (
    FEATURE_COLUMNS,
    latest_snapshot_per_player,
    predict_value_growth,
    prepare_features,
    time_based_split,
)


def test_prepare_features_adds_log_and_squared_terms_and_fills_categoricals() -> None:
    panel = pd.DataFrame(
        {
            "current_value_eur": [1_000_000, 10_000_000],
            "age": [20.0, 30.0],
            "last_transfer_fee_eur": [0.0, 999.0],
            "foot": ["right", None],
            "sub_position": [None, "Centre-Forward"],
        }
    )
    result = prepare_features(panel)
    assert result["log_current_value_eur"].tolist() == pytest.approx([6.0, 7.0])
    assert result["age_sq"].tolist() == [400.0, 900.0]
    assert result["foot"].tolist() == ["right", "unknown"]
    assert result["sub_position"].tolist() == ["unknown", "Centre-Forward"]


def test_time_based_split_keeps_later_dates_in_test() -> None:
    df = pd.DataFrame({"snapshot_date": pd.date_range("2020-01-01", periods=10, freq="30D")})
    train, test = time_based_split(df, test_frac=0.2)
    assert len(train) + len(test) == len(df)
    assert train["snapshot_date"].max() <= test["snapshot_date"].min()


def test_latest_snapshot_per_player_picks_most_recent_and_drops_stale_players() -> None:
    panel = pd.DataFrame(
        {
            "player_id": [1, 1, 2],
            "snapshot_date": [
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-06-01"),
                pd.Timestamp("2022-01-01"),
            ],
        }
    )
    # Relative to the panel's max date (2024-06-01), player 2's only snapshot is more
    # than max_age_days old and should be excluded.
    result = latest_snapshot_per_player(panel, max_age_days=365)
    assert set(result["player_id"]) == {1}
    assert result.iloc[0]["snapshot_date"] == pd.Timestamp("2024-06-01")


class _StubPipeline:
    """Stands in for a fitted sklearn pipeline: always predicts a fixed % change,
    so the test isolates predict_value_growth's own arithmetic wiring."""

    def __init__(self, constant_pred: float) -> None:
        self.constant_pred = constant_pred

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.constant_pred)


def test_predict_value_growth_computes_dollar_change_consistently() -> None:
    current_players = pd.DataFrame(
        {
            "player_id": [1],
            "name": ["Test Player"],
            "sub_position": ["Centre-Forward"],
            "foot": ["right"],
            "age": [25.0],
            "current_value_eur": [10_000_000.0],
            "trailing_goals_per90": [0.3],
            "trailing_assists_per90": [0.1],
            "trailing_minutes": [2000.0],
            "trailing_appearances": [22.0],
            "trailing_avg_club_position": [8.0],
            "has_transfer_history": [1],
            "months_since_last_transfer": [12.0],
            "num_prior_transfers": [2],
            "last_transfer_fee_eur": [5_000_000.0],
        }
    )
    result = predict_value_growth(_StubPipeline(0.1), current_players)
    assert result.iloc[0]["predicted_pct_change"] == pytest.approx(0.1)
    assert result.iloc[0]["predicted_eur_change"] == pytest.approx(1_000_000.0)
    assert result.iloc[0]["predicted_value_eur"] == pytest.approx(11_000_000.0)


def test_predict_value_growth_drops_rows_missing_required_features() -> None:
    current_players = pd.DataFrame(
        {
            "player_id": [1, 2],
            "name": ["Has Everything", "Missing Minutes"],
            "sub_position": ["Centre-Forward", "Centre-Forward"],
            "foot": ["right", "right"],
            "age": [25.0, 25.0],
            "current_value_eur": [10_000_000.0, 10_000_000.0],
            "trailing_goals_per90": [0.3, 0.3],
            "trailing_assists_per90": [0.1, 0.1],
            "trailing_minutes": [2000.0, np.nan],
            "trailing_appearances": [22.0, 22.0],
            "trailing_avg_club_position": [8.0, 8.0],
            "has_transfer_history": [1, 1],
            "months_since_last_transfer": [12.0, 12.0],
            "num_prior_transfers": [2, 2],
            "last_transfer_fee_eur": [5_000_000.0, 5_000_000.0],
        }
    )
    result = predict_value_growth(_StubPipeline(0.1), current_players)
    assert result["player_id"].tolist() == [1]


def _synthetic_panel(n_players: int = 40, n_dates: int = 30, seed: int = 0) -> pd.DataFrame:
    """A small multi-year panel where the 12m target is a noisy function of the features."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2016-01-01", periods=n_dates, freq="90D")
    rows = []
    for player_id in range(n_players):
        age0 = rng.uniform(18, 30)
        for i, date in enumerate(dates):
            age = age0 + i * 0.25
            goals_per90 = rng.uniform(0, 1)
            rows.append(
                {
                    "player_id": player_id,
                    "name": f"player {player_id}",
                    "snapshot_date": date,
                    "current_value_eur": 10 ** rng.uniform(6, 7.5),
                    "age": age,
                    "trailing_goals_per90": goals_per90,
                    "trailing_assists_per90": rng.uniform(0, 0.5),
                    "trailing_minutes": rng.uniform(0, 3000),
                    "trailing_appearances": rng.uniform(0, 38),
                    "trailing_avg_club_position": rng.uniform(1, 20),
                    "has_transfer_history": 1,
                    "months_since_last_transfer": rng.uniform(0, 48),
                    "num_prior_transfers": rng.integers(0, 5),
                    "last_transfer_fee_eur": rng.uniform(0, 5e7),
                    "sub_position": rng.choice(["Centre-Forward", "Centre-Back"]),
                    "foot": "right",
                    "value_change_12m_pct": 0.5 * goals_per90 - 0.02 * (age - 24) + rng.normal(0, 0.05),
                }
            )
    return pd.DataFrame(rows)


def test_strict_training_panel_excludes_rows_whose_outcome_window_was_still_open() -> None:
    from src.modeling import strict_training_panel
    from src.panel import HORIZON_TOLERANCE_DAYS

    cutoff = pd.Timestamp("2024-06-01")
    tolerance_days = HORIZON_TOLERANCE_DAYS[12]  # 150
    twelve_months_before = cutoff - pd.DateOffset(months=12)
    fully_resolved = twelve_months_before - pd.Timedelta(days=tolerance_days + 10)
    window_still_open = twelve_months_before - pd.Timedelta(days=tolerance_days - 50)
    panel = pd.DataFrame({"snapshot_date": [fully_resolved, window_still_open, twelve_months_before, cutoff]})

    kept = strict_training_panel(panel, cutoff, months=12)
    assert kept["snapshot_date"].tolist() == [fully_resolved]


def test_default_backtest_cutoffs_are_spaced_and_end_before_unresolved_outcomes() -> None:
    from src.modeling import default_backtest_cutoffs
    from src.panel import HORIZON_TOLERANCE_DAYS

    panel = _synthetic_panel()
    cutoffs = default_backtest_cutoffs(panel, months=12, step_months=6, min_train_rows_with_form=100)

    assert len(cutoffs) >= 3
    assert cutoffs == sorted(cutoffs)
    assert all(c.day == 1 for c in cutoffs)
    # Every evaluation outcome must be fully known: the tolerance window closes before the data ends.
    tolerance = pd.Timedelta(days=HORIZON_TOLERANCE_DAYS[12])
    assert cutoffs[-1] + pd.DateOffset(months=12) + tolerance <= panel["snapshot_date"].max()
    gaps = {(b - a).days for a, b in zip(cutoffs, cutoffs[1:])}
    assert gaps <= {181, 182, 183, 184}


def test_walk_forward_backtest_scores_each_cutoff_and_model_without_leakage() -> None:
    from src.modeling import walk_forward_backtest

    panel = _synthetic_panel()
    cutoffs = [pd.Timestamp("2020-06-01"), pd.Timestamp("2021-06-01")]
    metrics, predictions = walk_forward_backtest(
        panel, months=12, cutoffs=cutoffs, model_names=("linear", "ridge")
    )

    assert len(metrics) == 4
    assert set(metrics["model"]) == {"linear", "ridge"}
    assert set(metrics["cutoff"]) == set(cutoffs)
    # Each evaluation set is the latest snapshot per player as of the cutoff.
    for cutoff in cutoffs:
        assert (predictions.loc[predictions["cutoff"] == cutoff, "snapshot_date"] <= cutoff).all()
    # The target is linear in the features, so even the plain linear model should rank players well.
    assert (metrics["spearman"] > 0.8).all()
    assert (metrics["n_train"] > 0).all()
    # Later cutoffs see strictly more training data.
    by_cutoff = metrics[metrics["model"] == "linear"].sort_values("cutoff")
    assert by_cutoff["n_train"].is_monotonic_increasing


def test_summarize_backtest_aggregates_across_cutoffs() -> None:
    from src.modeling import summarize_backtest

    metrics = pd.DataFrame(
        {
            "cutoff": pd.to_datetime(["2020-06-01", "2021-06-01"] * 2),
            "model": ["linear", "linear", "ridge", "ridge"],
            "n_train": [10, 20, 10, 20],
            "n_eval": [5, 5, 5, 5],
            "mae": [0.2, 0.4, 0.1, 0.3],
            "rmse": [0.3, 0.5, 0.2, 0.4],
            "pearson": [0.5, 0.7, 0.6, 0.8],
            "spearman": [0.4, 0.6, 0.5, 0.7],
            "direction_accuracy": [0.6, 0.7, 0.65, 0.75],
            "calibration_slope": [0.9, 1.1, 0.8, 1.2],
        }
    )
    summary = summarize_backtest(metrics)
    assert summary.index.tolist() == ["ridge", "linear"]  # sorted by mean spearman, best first
    assert summary.loc["linear", ("mae", "mean")] == pytest.approx(0.3)
    assert summary.loc["ridge", "n_cutoffs"] == 2


def test_fit_model_with_log_ratio_target_predicts_on_pct_scale() -> None:
    from src.modeling import fit_model, modelable_rows

    panel = _synthetic_panel(n_players=30, n_dates=10)
    train = modelable_rows(panel, months=12)
    pct_model = fit_model(train, "linear", months=12, target_transform="pct")
    log_model = fit_model(train, "linear", months=12, target_transform="log_ratio")

    pct_pred = pct_model.predict(train[FEATURE_COLUMNS])
    log_pred = log_model.predict(train[FEATURE_COLUMNS])
    # Both are on the same (% change) scale: the target here is a small linear signal with
    # noise, so the two fits should agree closely and neither should return log values.
    assert np.corrcoef(pct_pred, log_pred)[0, 1] > 0.99
    assert abs(pct_pred.mean() - log_pred.mean()) < 0.05
    assert log_pred.min() > -1.0  # expm1 of anything is > -1, i.e. a value can't drop below zero


def test_fit_quantile_models_orders_quantiles_and_reports_intervals() -> None:
    from src.modeling import fit_quantile_models, modelable_rows, predict_value_growth_interval

    panel = _synthetic_panel(n_players=40, n_dates=12)
    train = modelable_rows(panel, months=12)
    models = fit_quantile_models(train, months=12, quantiles=(0.1, 0.5, 0.9))
    intervals = predict_value_growth_interval(models, train.head(50))

    assert {"pct_q10", "pct_q50", "pct_q90", "value_eur_q10", "value_eur_q90"} <= set(intervals.columns)
    # Quantile fits aren't guaranteed monotone row by row, but should be on average.
    assert intervals["pct_q10"].mean() < intervals["pct_q50"].mean() < intervals["pct_q90"].mean()
    assert (intervals["value_eur_q10"] == intervals["current_value_eur"] * (1 + intervals["pct_q10"])).all()


def test_walk_forward_interval_backtest_reports_coverage_per_cutoff() -> None:
    from src.modeling import walk_forward_interval_backtest

    panel = _synthetic_panel()
    cutoffs = [pd.Timestamp("2020-06-01"), pd.Timestamp("2021-06-01")]
    coverage = walk_forward_interval_backtest(panel, months=12, cutoffs=cutoffs)

    assert coverage["cutoff"].tolist() == cutoffs
    assert ((coverage["coverage"] + coverage["below_low"] + coverage["above_high"]).round(6) == 1.0).all()
    assert (coverage["median_width"] > 0).all()
