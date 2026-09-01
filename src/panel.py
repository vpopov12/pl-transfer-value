"""Build a player-valuation snapshot panel with trailing form features and
forward-looking value-change targets, for predicting Premier League transfer
value growth over multiple horizons.

Each row is one historical valuation event ("snapshot") for a player. Features
describe what was knowable as of that date (trailing 365-day performance,
age, current value, position); targets describe the value at each horizon
afterward, found by matching the nearest actual valuation to the target date.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import PREMIER_LEAGUE_COMPETITION_ID, download_dataset, load_players

HORIZONS_MONTHS = (3, 6, 9, 12)
TRAILING_WINDOW_DAYS = 365
# Looser tolerance for longer horizons: this dataset re-values players roughly
# every 150-190 days on average, so a tight window around a 3-month target
# would match almost nothing.
HORIZON_TOLERANCE_DAYS = {3: 40, 6: 70, 9: 110, 12: 150}

CUM_COLS = ["cum_goals", "cum_assists", "cum_minutes", "cum_apps", "cum_club_position"]


def load_valuations(data_dir: Path) -> pd.DataFrame:
    valuations = pd.read_csv(data_dir / "player_valuations.csv")
    valuations["date"] = pd.to_datetime(valuations["date"], errors="coerce")
    valuations = valuations[
        valuations["player_club_domestic_competition_id"] == PREMIER_LEAGUE_COMPETITION_ID
    ]
    return (
        valuations[["player_id", "date", "market_value_in_eur", "current_club_name"]]
        .dropna(subset=["player_id", "date", "market_value_in_eur"])
        .sort_values("date")
        .reset_index(drop=True)
    )


def load_pl_games(data_dir: Path) -> pd.DataFrame:
    games = pd.read_csv(data_dir / "games.csv")
    games = games[games["competition_id"] == PREMIER_LEAGUE_COMPETITION_ID]
    return games[
        ["game_id", "home_club_id", "away_club_id", "home_club_position", "away_club_position"]
    ]


def load_pl_appearances(data_dir: Path) -> pd.DataFrame:
    """PL appearances, each tagged with the player's club's league position in that match
    (a proxy for team strength/context around the player's performance)."""
    appearances = pd.read_csv(data_dir / "appearances.csv")
    appearances["date"] = pd.to_datetime(appearances["date"], errors="coerce")
    appearances = appearances[appearances["competition_id"] == PREMIER_LEAGUE_COMPETITION_ID]

    games = load_pl_games(data_dir)
    appearances = appearances.merge(games, on="game_id", how="left")
    appearances["club_position"] = np.where(
        appearances["player_club_id"] == appearances["home_club_id"],
        appearances["home_club_position"],
        appearances["away_club_position"],
    )
    return appearances.sort_values("date").reset_index(drop=True)


def load_transfers(data_dir: Path) -> pd.DataFrame:
    """Full transfer history (any league), for time-since-last-move / fee features."""
    transfers = pd.read_csv(data_dir / "transfers.csv", low_memory=False)
    transfers["transfer_date"] = pd.to_datetime(transfers["transfer_date"], errors="coerce")
    return (
        transfers[["player_id", "transfer_date", "transfer_fee"]]
        .dropna(subset=["player_id", "transfer_date"])
        .sort_values("transfer_date")
        .reset_index(drop=True)
    )


def _career_cumulative_stats(appearances: pd.DataFrame) -> pd.DataFrame:
    """Per player_id + date, career-to-date cumulative goals/assists/minutes/apps/club position."""
    cum = appearances.sort_values(["player_id", "date"]).copy()
    cum["cum_goals"] = cum.groupby("player_id")["goals"].cumsum()
    cum["cum_assists"] = cum.groupby("player_id")["assists"].cumsum()
    cum["cum_minutes"] = cum.groupby("player_id")["minutes_played"].cumsum()
    cum["cum_apps"] = cum.groupby("player_id").cumcount() + 1
    cum["cum_club_position"] = cum.groupby("player_id")["club_position"].cumsum()
    daily = cum.groupby(["player_id", "date"], as_index=False)[CUM_COLS].last()
    return daily.sort_values("date").reset_index(drop=True)


def _asof_cumulative(snapshots: pd.DataFrame, on_col: str, daily_cum: pd.DataFrame) -> pd.DataFrame:
    """Career-cumulative stats as of (on or before) each snapshot's `on_col` date."""
    ordered = snapshots[["player_id", on_col]].sort_values(on_col).reset_index()
    merged = pd.merge_asof(
        ordered, daily_cum, left_on=on_col, right_on="date", by="player_id", direction="backward"
    )
    merged = merged.set_index("index")
    return merged[CUM_COLS].reindex(snapshots.index).fillna(0.0)


def add_trailing_form(snapshots: pd.DataFrame, appearances: pd.DataFrame) -> pd.DataFrame:
    """Add trailing-365-day goals/assists/minutes/appearances (+ per-90 rates)."""
    daily_cum = _career_cumulative_stats(appearances)

    snapshots = snapshots.copy()
    now = _asof_cumulative(snapshots, "snapshot_date", daily_cum)

    window_start = snapshots[["player_id"]].copy()
    window_start["snapshot_date"] = snapshots["snapshot_date"] - pd.Timedelta(days=TRAILING_WINDOW_DAYS)
    before = _asof_cumulative(window_start, "snapshot_date", daily_cum)

    trailing = (now - before).clip(lower=0)
    snapshots["trailing_goals"] = trailing["cum_goals"]
    snapshots["trailing_assists"] = trailing["cum_assists"]
    snapshots["trailing_minutes"] = trailing["cum_minutes"]
    snapshots["trailing_appearances"] = trailing["cum_apps"]

    # Require at least two matches' worth of minutes before trusting a per-90 rate,
    # otherwise a single goal in a handful of minutes produces an extreme ratio.
    MIN_MINUTES_FOR_RATE = 180
    nineties = snapshots["trailing_minutes"] / 90
    enough_minutes = snapshots["trailing_minutes"] >= MIN_MINUTES_FOR_RATE
    snapshots["trailing_goals_per90"] = np.where(enough_minutes, snapshots["trailing_goals"] / nineties, 0.0)
    snapshots["trailing_assists_per90"] = np.where(enough_minutes, snapshots["trailing_assists"] / nineties, 0.0)

    # Average league position of the player's club over the trailing window (1 = top of table),
    # a proxy for the strength of the team context the player's performance happened in.
    # With no trailing PL appearances, default to a neutral mid-table position (20-team league).
    NEUTRAL_LEAGUE_POSITION = 10.5
    has_trailing_apps = snapshots["trailing_appearances"] > 0
    snapshots["trailing_avg_club_position"] = np.where(
        has_trailing_apps,
        trailing["cum_club_position"] / snapshots["trailing_appearances"],
        NEUTRAL_LEAGUE_POSITION,
    )
    return snapshots


def add_transfer_history(snapshots: pd.DataFrame, transfers: pd.DataFrame) -> pd.DataFrame:
    """Add time-since-last-transfer, prior-transfer count, and last transfer fee, all as of
    the snapshot date (genuinely dated, unlike players.csv's current-only contract fields)."""
    snapshots = snapshots.copy()
    history = transfers.sort_values(["player_id", "transfer_date"]).copy()
    history["transfer_seq"] = history.groupby("player_id").cumcount() + 1
    history = history.sort_values("transfer_date")

    ordered = snapshots[["player_id", "snapshot_date"]].sort_values("snapshot_date").reset_index()
    merged = pd.merge_asof(
        ordered,
        history,
        left_on="snapshot_date",
        right_on="transfer_date",
        by="player_id",
        direction="backward",
    )
    merged = merged.set_index("index").reindex(snapshots.index)

    months_since = (snapshots["snapshot_date"] - merged["transfer_date"]).dt.days / 30.44
    snapshots["has_transfer_history"] = merged["transfer_date"].notna().astype(int)
    snapshots["months_since_last_transfer"] = months_since.fillna(0.0)
    snapshots["num_prior_transfers"] = merged["transfer_seq"].fillna(0)
    snapshots["last_transfer_fee_eur"] = merged["transfer_fee"].fillna(0.0)
    return snapshots


def add_horizon_targets(snapshots: pd.DataFrame, valuations: pd.DataFrame) -> pd.DataFrame:
    """Add future_value / value_change (EUR and %) for each horizon in HORIZONS_MONTHS."""
    snapshots = snapshots.copy()
    for months in HORIZONS_MONTHS:
        target_date_col = f"_target_date_{months}m"
        dated = snapshots[["player_id"]].copy()
        dated[target_date_col] = snapshots["snapshot_date"] + pd.DateOffset(months=months)

        ordered = dated.sort_values(target_date_col).reset_index()
        merged = pd.merge_asof(
            ordered,
            valuations,
            left_on=target_date_col,
            right_on="date",
            by="player_id",
            direction="nearest",
            tolerance=pd.Timedelta(days=HORIZON_TOLERANCE_DAYS[months]),
        )
        merged = merged.set_index("index")
        future_value = merged["market_value_in_eur"].reindex(snapshots.index)

        snapshots[f"future_value_{months}m_eur"] = future_value
        snapshots[f"value_change_{months}m_eur"] = future_value - snapshots["current_value_eur"]
        snapshots[f"value_change_{months}m_pct"] = (
            snapshots[f"value_change_{months}m_eur"] / snapshots["current_value_eur"]
        )
    return snapshots


def build_snapshot_panel() -> pd.DataFrame:
    """Build the full valuation-snapshot panel: one row per historical PL valuation event."""
    data_dir = download_dataset()
    players = load_players(data_dir)
    valuations = load_valuations(data_dir)
    appearances = load_pl_appearances(data_dir)
    transfers = load_transfers(data_dir)

    snapshots = valuations.rename(
        columns={"date": "snapshot_date", "market_value_in_eur": "current_value_eur"}
    )
    snapshots = add_trailing_form(snapshots, appearances)
    snapshots = add_transfer_history(snapshots, transfers)
    snapshots = add_horizon_targets(snapshots, valuations)

    snapshots = snapshots.merge(
        players[
            [
                "player_id",
                "name",
                "date_of_birth",
                "position",
                "sub_position",
                "foot",
                "height_in_cm",
                "country_of_citizenship",
            ]
        ],
        on="player_id",
        how="left",
    )
    snapshots["age"] = (snapshots["snapshot_date"] - snapshots["date_of_birth"]).dt.days / 365.25
    return snapshots


if __name__ == "__main__":
    panel = build_snapshot_panel()
    print(panel.shape)
    for months in HORIZONS_MONTHS:
        n = panel[f"value_change_{months}m_pct"].notna().sum()
        print(f"{months}m horizon: {n} rows with a matched target")
