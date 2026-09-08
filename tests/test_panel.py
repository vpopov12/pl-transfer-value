from __future__ import annotations

import pandas as pd
import pytest

from src.panel import (
    _asof_cumulative,
    _career_cumulative_stats,
    add_horizon_targets,
    add_trailing_form,
    add_transfer_history,
)


def _dt(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


@pytest.fixture
def appearances() -> pd.DataFrame:
    """Player 1: three PL appearances spanning two seasons. Player 2: one, on the same day."""
    return pd.DataFrame(
        {
            "player_id": [1, 1, 1, 2],
            "date": [_dt("2023-08-01"), _dt("2023-12-01"), _dt("2024-08-01"), _dt("2023-08-01")],
            "goals": [1, 2, 0, 5],
            "assists": [0, 1, 1, 0],
            "minutes_played": [90, 90, 90, 90],
            "club_position": [5, 3, 10, 1],
        }
    )


def test_career_cumulative_stats_accumulates_per_player(appearances: pd.DataFrame) -> None:
    daily = _career_cumulative_stats(appearances)
    p1 = daily[daily["player_id"] == 1].sort_values("date")
    assert p1["cum_goals"].tolist() == [1, 3, 3]
    assert p1["cum_apps"].tolist() == [1, 2, 3]
    # player 2's stats must not leak into player 1's running totals
    p2 = daily[daily["player_id"] == 2]
    assert p2["cum_goals"].tolist() == [5]


def test_asof_cumulative_uses_last_value_on_or_before(appearances: pd.DataFrame) -> None:
    daily = _career_cumulative_stats(appearances)
    snapshots = pd.DataFrame(
        {
            "player_id": [1, 1, 1],
            "snapshot_date": [_dt("2023-07-01"), _dt("2023-09-01"), _dt("2024-09-01")],
        }
    )
    result = _asof_cumulative(snapshots, "snapshot_date", daily)
    assert result.iloc[0]["cum_goals"] == 0  # before any appearance
    assert result.iloc[1]["cum_goals"] == 1  # after the first appearance only
    assert result.iloc[2]["cum_goals"] == 3  # after all three


def test_trailing_form_excludes_events_older_than_365_days(appearances: pd.DataFrame) -> None:
    snapshots = pd.DataFrame({"player_id": [1], "snapshot_date": [_dt("2024-08-15")]})
    result = add_trailing_form(snapshots, appearances)
    # 2023-08-01 is >365 days before 2024-08-15 and must drop out of the window;
    # only the 2023-12-01 and 2024-08-01 appearances count.
    assert result.iloc[0]["trailing_goals"] == 2
    assert result.iloc[0]["trailing_appearances"] == 2


def test_trailing_per90_requires_minimum_minutes() -> None:
    thin_appearances = pd.DataFrame(
        {
            "player_id": [1],
            "date": [_dt("2024-01-01")],
            "goals": [1],
            "assists": [0],
            "minutes_played": [10],  # well below the 180-minute floor
            "club_position": [5],
        }
    )
    snapshots = pd.DataFrame({"player_id": [1], "snapshot_date": [_dt("2024-06-01")]})
    result = add_trailing_form(snapshots, thin_appearances)
    assert result.iloc[0]["trailing_goals_per90"] == 0.0


def test_trailing_avg_club_position_defaults_when_player_has_no_appearances(
    appearances: pd.DataFrame,
) -> None:
    # player_id 3 never appears in the appearances table at all.
    snapshots = pd.DataFrame({"player_id": [3], "snapshot_date": [_dt("2024-01-01")]})
    result = add_trailing_form(snapshots, appearances)
    assert result.iloc[0]["trailing_appearances"] == 0
    assert result.iloc[0]["trailing_avg_club_position"] == 10.5


@pytest.fixture
def transfers() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": [1, 1, 2],
            "transfer_date": [_dt("2020-01-01"), _dt("2022-06-15"), _dt("2021-01-01")],
            "transfer_fee": [500_000.0, 3_000_000.0, 0.0],
        }
    )


def test_transfer_history_counts_and_uses_most_recent_move(transfers: pd.DataFrame) -> None:
    snapshots = pd.DataFrame({"player_id": [1], "snapshot_date": [_dt("2023-01-01")]})
    result = add_transfer_history(snapshots, transfers)
    assert result.iloc[0]["num_prior_transfers"] == 2
    assert result.iloc[0]["has_transfer_history"] == 1
    assert result.iloc[0]["last_transfer_fee_eur"] == 3_000_000.0
    # ~200 days between 2022-06-15 and 2023-01-01
    assert result.iloc[0]["months_since_last_transfer"] == pytest.approx(6.6, abs=0.3)


def test_transfer_history_defaults_when_player_has_no_transfers(transfers: pd.DataFrame) -> None:
    snapshots = pd.DataFrame({"player_id": [99], "snapshot_date": [_dt("2023-01-01")]})
    result = add_transfer_history(snapshots, transfers)
    assert result.iloc[0]["has_transfer_history"] == 0
    assert result.iloc[0]["num_prior_transfers"] == 0
    assert result.iloc[0]["months_since_last_transfer"] == 0.0
    assert result.iloc[0]["last_transfer_fee_eur"] == 0.0


def test_transfer_history_ignores_future_transfers() -> None:
    # A transfer dated after the snapshot (pre-agreed future move) must not count as history yet.
    transfers = pd.DataFrame({"player_id": [1], "transfer_date": [_dt("2025-01-01")], "transfer_fee": [1.0]})
    snapshots = pd.DataFrame({"player_id": [1], "snapshot_date": [_dt("2023-01-01")]})
    result = add_transfer_history(snapshots, transfers)
    assert result.iloc[0]["has_transfer_history"] == 0


@pytest.fixture
def valuations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": [1, 1, 1],
            "date": [_dt("2023-01-01"), _dt("2023-07-15"), _dt("2024-06-01")],
            "market_value_in_eur": [1_000_000, 2_000_000, 5_000_000],
        }
    )


def test_horizon_target_matches_nearest_valuation_within_tolerance(valuations: pd.DataFrame) -> None:
    snapshots = pd.DataFrame(
        {"player_id": [1], "snapshot_date": [_dt("2023-01-01")], "current_value_eur": [1_000_000]}
    )
    result = add_horizon_targets(snapshots, valuations)
    # 6-month target date is 2023-07-01; the nearest valuation (2023-07-15) is 14 days
    # away, well inside the 70-day tolerance for this horizon.
    assert result.iloc[0]["future_value_6m_eur"] == 2_000_000
    assert result.iloc[0]["value_change_6m_eur"] == 1_000_000
    assert result.iloc[0]["value_change_6m_pct"] == pytest.approx(1.0)


def test_horizon_target_is_nan_when_nothing_is_within_tolerance() -> None:
    # The only valuation on record is the snapshot's own -- ~90 days from the 3-month
    # target date, which is outside the 40-day tolerance, so no match should be found.
    valuations = pd.DataFrame(
        {"player_id": [1], "date": [_dt("2023-01-01")], "market_value_in_eur": [1_000_000]}
    )
    snapshots = pd.DataFrame(
        {"player_id": [1], "snapshot_date": [_dt("2023-01-01")], "current_value_eur": [1_000_000]}
    )
    result = add_horizon_targets(snapshots, valuations)
    assert pd.isna(result.iloc[0]["future_value_3m_eur"])
    assert pd.isna(result.iloc[0]["value_change_3m_pct"])


def test_horizon_target_never_matches_a_valuation_before_the_snapshot() -> None:
    # A stale valuation sitting well before the snapshot must not be mistaken for a
    # "future" value just because it's the closest thing on record for the player.
    valuations = pd.DataFrame(
        {"player_id": [1], "date": [_dt("2020-01-01")], "market_value_in_eur": [500_000]}
    )
    snapshots = pd.DataFrame(
        {"player_id": [1], "snapshot_date": [_dt("2023-01-01")], "current_value_eur": [1_000_000]}
    )
    result = add_horizon_targets(snapshots, valuations)
    assert pd.isna(result.iloc[0]["future_value_12m_eur"])
