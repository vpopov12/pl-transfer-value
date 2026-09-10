from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis import (
    add_stats_only_residual,
    flag_relegation_in_window,
    flag_transfers_within_horizon,
    grouped_backtest_metrics,
    market_drift_decomposition,
    partial_dependence_by_group,
    pl_only_targets,
    season_final_positions,
    target_source,
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


def _panel_with_target_leagues() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4],
            "snapshot_date": [_dt("2020-01-01")] * 4,
            "current_value_eur": [1e6, 2e6, 3e6, 4e6],
            "future_value_12m_eur": [1.5e6, 4e6, np.nan, 2e6],
            "value_change_12m_eur": [0.5e6, 2e6, np.nan, -2e6],
            "value_change_12m_pct": [0.5, 1.0, np.nan, -0.5],
            "target_league_12m": ["GB1", "IT1", None, "untracked"],
        }
    )


def test_target_source_labels_pl_other_league_and_none() -> None:
    expected = ["pl", "other_league", "none", "untracked"]
    assert target_source(_panel_with_target_leagues(), months=12).tolist() == expected


def test_pl_only_targets_blanks_outcomes_found_abroad_and_nothing_else() -> None:
    panel = _panel_with_target_leagues()
    out = pl_only_targets(panel, months=12)
    assert out.loc[0, "value_change_12m_pct"] == 0.5
    assert np.isnan(out.loc[1, "value_change_12m_pct"])
    assert np.isnan(out.loc[1, "future_value_12m_eur"])
    assert np.isnan(out.loc[3, "value_change_12m_pct"])
    assert target_source(out, months=12).tolist() == ["pl", "none", "none", "none"]
    # the input is untouched
    assert panel.loc[1, "value_change_12m_pct"] == 1.0


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


class _SumPipeline:
    """Stub with the shape pipeline_feature_columns falls back on (uses FEATURE_COLUMNS)."""

    def predict(self, X):
        return X["age"].to_numpy() + X["trailing_minutes"].to_numpy()


def test_partial_dependence_sweeps_one_feature_and_averages_per_group() -> None:
    from src.modeling import FEATURE_COLUMNS

    rows = pd.DataFrame({c: 0.0 for c in FEATURE_COLUMNS}, index=range(4))
    rows["sub_position"], rows["foot"] = "x", "y"
    rows["trailing_minutes"] = [10.0, 20.0, 100.0, 200.0]
    group = pd.Series(["cheap", "cheap", "dear", "dear"], index=rows.index)

    pdp = partial_dependence_by_group(_SumPipeline(), rows, "age", grid=[0, 5], group=group)

    assert pdp.loc["cheap"].tolist() == [15.0, 20.0]
    assert pdp.loc["dear"].tolist() == [150.0, 155.0]


def test_season_final_positions_uses_last_matchday_and_marks_bottom_three() -> None:
    games = pd.DataFrame(
        {
            "competition_id": ["GB1", "GB1", "GB1", "ES1"],
            "season": [2022, 2022, 2022, 2022],
            "date": ["2023-05-20", "2023-05-28", "2023-05-28", "2023-05-28"],
            "home_club_id": [1, 1, 3, 9],
            "away_club_id": [2, 3, 2, 8],
            "home_club_position": [17, 18, 3, 1],  # club 1 slipped from 17th to 18th on the last day
            "away_club_position": [2, 4, 1, 2],
        }
    )
    final = season_final_positions(games).set_index("club_id")
    assert final.loc[1, "position"] == 18 and bool(final.loc[1, "relegated"])
    assert final.loc[2, "position"] == 1 and not bool(final.loc[2, "relegated"])
    assert 9 not in final.index
    assert final.loc[1, "season_end"] == _dt("2023-06-01")


def test_flag_relegation_only_counts_seasons_ending_inside_the_window() -> None:
    predictions = pd.DataFrame(
        {
            "current_club_id": [1, 1, 2],
            "snapshot_date": [_dt("2023-01-01"), _dt("2023-07-01"), _dt("2023-01-01")],
        },
        index=[5, 6, 7],
    )
    final = pd.DataFrame(
        {"club_id": [1, 2], "season_end": [_dt("2023-06-01"), _dt("2023-06-01")], "relegated": [True, False]}
    )
    out = flag_relegation_in_window(predictions, final, months=12)
    # Club 1 relegated June 2023: inside the window for the Jan snapshot, before it for the July one.
    assert out["club_relegated_in_window"].tolist() == [True, False, False]
