from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis import (
    add_stats_only_residual,
    augment_targets_with_other_leagues,
    flag_transfers_within_horizon,
    grouped_backtest_metrics,
    market_drift_decomposition,
)


def _dt(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def test_flag_transfers_uses_the_outcome_window_and_ignores_same_club_rows() -> None:
    predictions = pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4],
            "snapshot_date": [_dt("2020-01-01")] * 4,
            "predicted_pct_change": [0.1, 0.2, 0.3, 0.4],
        },
        index=[10, 11, 12, 13],
    )
    transfers = pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4, 4],
            # 1: paid move inside the 12m window (+150d tolerance ends 2021-05-31)
            # 2: free move inside the window; 3: move after the window; 4: same-club row, then before snapshot
            "transfer_date": [
                _dt("2020-08-01"),
                _dt("2021-05-01"),
                _dt("2021-07-01"),
                _dt("2020-06-01"),
                _dt("2019-12-01"),
            ],
            "transfer_fee": [5e6, 0.0, 1e6, 2e6, 3e6],
            "from_club_id": [1, 1, 1, 9, 1],
            "to_club_id": [2, 2, 2, 9, 2],
        }
    )
    flagged = flag_transfers_within_horizon(predictions, transfers, months=12)

    assert flagged.index.tolist() == [10, 11, 12, 13]
    assert flagged["transferred"].tolist() == [True, True, False, False]
    assert flagged["paid_transfer"].tolist() == [True, False, False, False]


def test_grouped_backtest_metrics_scores_each_group_per_cutoff() -> None:
    n = 40
    predictions = pd.DataFrame(
        {
            "cutoff": [_dt("2020-01-01")] * n + [_dt("2021-01-01")] * n,
            "flag": ([True] * 30 + [False] * 10) * 2,
            "predicted_pct_change": list(range(n)) * 2,
            "actual_pct_change": [x * 0.05 + 0.1 for x in range(n)] * 2,
        }
    )
    metrics = grouped_backtest_metrics(predictions, "flag", min_rows=5)

    assert len(metrics) == 4
    assert metrics.loc[metrics["flag"], "share"].tolist() == pytest.approx([0.75, 0.75])
    assert metrics["spearman"].tolist() == pytest.approx([1.0] * 4)


def test_market_drift_decomposition_removes_a_uniform_bias_entirely() -> None:
    # The model gets every player's relative move right but misses a market-wide -20%.
    actual = pd.Series([0.1, 0.3, -0.2, 0.0]) * 0.8 - 0.2
    predictions = pd.DataFrame(
        {
            "cutoff": [_dt("2020-01-01")] * 4,
            "predicted_pct_change": [0.1, 0.3, -0.2, 0.0],
            "actual_pct_change": actual,
        }
    )
    out = market_drift_decomposition(predictions)

    assert len(out) == 1
    assert out["market_bias"].iloc[0] == pytest.approx(-np.log(0.8), abs=1e-9)
    assert out["mae_demeaned"].iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert out["share_of_mae_from_market"].iloc[0] == pytest.approx(1.0)


def test_augment_targets_fills_leavers_from_other_leagues_only() -> None:
    panel = pd.DataFrame(
        {
            "player_id": [1, 2, 3],
            "snapshot_date": [_dt("2020-01-01")] * 3,
            "current_value_eur": [1e6, 2e6, 3e6],
            "future_value_12m_eur": [1.5e6, np.nan, np.nan],
            "value_change_12m_eur": [0.5e6, np.nan, np.nan],
            "value_change_12m_pct": [0.5, np.nan, np.nan],
        }
    )
    valuations = pd.DataFrame(
        {
            "player_id": [2, 2, 3],
            # player 2: Serie A valuation near the 12m mark; player 3: only a PL row (which the
            # panel would already have matched if it were inside tolerance) -> stays "none".
            "date": [_dt("2021-02-01"), _dt("2023-01-01"), _dt("2021-01-15")],
            "market_value_in_eur": [4e6, 9e6, 5e6],
            "player_club_domestic_competition_id": ["IT1", "IT1", "GB1"],
        }
    )
    out = augment_targets_with_other_leagues(panel, valuations, months=12)

    assert out["target_source_12m"].tolist() == ["pl", "other_league", "none"]
    assert out["value_change_12m_pct"].tolist()[:2] == pytest.approx([0.5, 1.0])
    assert np.isnan(out["value_change_12m_pct"].iloc[2])
    assert out["future_league_12m"].tolist()[:2] == ["GB1", "IT1"]


def test_stats_only_residual_only_uses_earlier_years() -> None:
    rng = np.random.default_rng(1)
    n = 300
    dates = pd.to_datetime(rng.choice(pd.date_range("2015-01-01", "2018-12-31"), size=n))
    age = rng.uniform(18, 35, n)
    panel = pd.DataFrame(
        {
            "snapshot_date": dates,
            "age": age,
            "trailing_goals_per90": rng.uniform(0, 1, n),
            "trailing_assists_per90": rng.uniform(0, 0.5, n),
            "trailing_minutes": rng.uniform(0, 3000, n),
            "trailing_appearances": rng.uniform(0, 38, n),
            "trailing_avg_club_position": rng.uniform(1, 20, n),
            "sub_position": rng.choice(["Centre-Forward", "Centre-Back"], n),
            "foot": "right",
            "current_value_eur": 10 ** (6 + 0.05 * (28 - abs(age - 26)) + rng.normal(0, 0.1, n)),
        }
    )
    out = add_stats_only_residual(panel, min_train_rows=50)
    years = out["snapshot_date"].dt.year
    # 2015 has no history before it -> NaN; later years are predicted.
    assert out.loc[years == 2015, "stats_residual"].isna().all()
    assert out.loc[years >= 2016, "stats_residual"].notna().all()
    assert out.loc[years >= 2016, "stats_residual"].abs().mean() < 0.3
    assert (out["stats_residual"] == np.log10(out["current_value_eur"]) - out["stats_only_log_value_pred"])[
        years >= 2016
    ].all()
