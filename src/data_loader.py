"""Load and merge the Transfermarkt player-scores dataset, filtered to the Premier League."""

from __future__ import annotations

import os
from pathlib import Path

import kagglehub
import pandas as pd

DATASET_SLUG = "davidcariboo/player-scores"
PREMIER_LEAGUE_COMPETITION_ID = "GB1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def download_dataset() -> Path:
    """Download (or reuse the cached copy of) the dataset into data/raw and return its path."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("KAGGLEHUB_CACHE", str(RAW_DATA_DIR))
    return Path(kagglehub.dataset_download(DATASET_SLUG))


def _season_from_date(date: pd.Series) -> pd.Series:
    """Map a date to its football season, labeled by the year the season starts (Jul-Jun)."""
    return date.dt.year.where(date.dt.month >= 7, date.dt.year - 1)


def load_players(data_dir: Path) -> pd.DataFrame:
    players = pd.read_csv(data_dir / "players.csv")
    players["date_of_birth"] = pd.to_datetime(players["date_of_birth"], errors="coerce")
    return players


def load_player_valuations(data_dir: Path) -> pd.DataFrame:
    valuations = pd.read_csv(data_dir / "player_valuations.csv")
    valuations["date"] = pd.to_datetime(valuations["date"], errors="coerce")
    valuations["season"] = _season_from_date(valuations["date"])
    valuations = valuations[
        valuations["player_club_domestic_competition_id"] == PREMIER_LEAGUE_COMPETITION_ID
    ]
    # Keep the latest valuation on record for each player within a season.
    return (
        valuations.sort_values("date")
        .groupby(["player_id", "season"], as_index=False)
        .last()[["player_id", "season", "market_value_in_eur"]]
    )


def load_appearances(data_dir: Path) -> pd.DataFrame:
    appearances = pd.read_csv(data_dir / "appearances.csv")
    appearances["date"] = pd.to_datetime(appearances["date"], errors="coerce")
    appearances["season"] = _season_from_date(appearances["date"])
    appearances = appearances[appearances["competition_id"] == PREMIER_LEAGUE_COMPETITION_ID]
    return appearances.groupby(["player_id", "season"], as_index=False).agg(
        goals=("goals", "sum"),
        assists=("assists", "sum"),
        minutes_played=("minutes_played", "sum"),
        num_appearances=("appearance_id", "count"),
        yellow_cards=("yellow_cards", "sum"),
        red_cards=("red_cards", "sum"),
    )


def build_dataset() -> pd.DataFrame:
    """Join players, player_valuations, and appearances on player_id + season, PL only."""
    data_dir = download_dataset()

    players = load_players(data_dir)
    valuations = load_player_valuations(data_dir)
    appearances = load_appearances(data_dir)

    df = valuations.merge(appearances, on=["player_id", "season"], how="inner")
    df = df.merge(
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
    df["age"] = df["season"] - df["date_of_birth"].dt.year
    df = df.rename(columns={"market_value_in_eur": "market_value_eur"})
    return df


if __name__ == "__main__":
    dataset = build_dataset()
    print(dataset.shape)
    print(dataset.head())
