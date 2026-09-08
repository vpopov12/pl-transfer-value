"""Report how current the cached Kaggle dataset is, and optionally force a re-download.

The value-growth predictions in notebook 02 only become a true out-of-sample test
once the upstream dataset is re-scraped with valuations dated after the predictions
were made. This module makes that check a one-liner:

    uv run python -m src.data_refresh            # report status
    uv run python -m src.data_refresh --refresh  # force re-download, then report
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from src.data_loader import DATASET_SLUG, PREMIER_LEAGUE_COMPETITION_ID, RAW_DATA_DIR, download_dataset


def dataset_status(data_dir: Path) -> dict[str, object]:
    """Summarise how far the dataset at `data_dir` extends in time."""
    valuations = pd.read_csv(
        data_dir / "player_valuations.csv",
        usecols=["player_id", "date", "player_club_domestic_competition_id"],
    )
    valuations["date"] = pd.to_datetime(valuations["date"], errors="coerce")
    pl_valuations = valuations[
        valuations["player_club_domestic_competition_id"] == PREMIER_LEAGUE_COMPETITION_ID
    ]

    appearances = pd.read_csv(data_dir / "appearances.csv", usecols=["date"])
    appearances["date"] = pd.to_datetime(appearances["date"], errors="coerce")

    return {
        "dataset_version": data_dir.name,
        "data_dir": str(data_dir),
        "latest_valuation_date": valuations["date"].max(),
        "latest_pl_valuation_date": pl_valuations["date"].max(),
        "num_pl_valuations": int(len(pl_valuations)),
        "latest_appearance_date": appearances["date"].max(),
    }


def months_of_new_outcomes(status: dict[str, object], predictions_as_of: pd.Timestamp) -> float:
    """How many months of PL valuations exist beyond the date predictions were made."""
    latest = pd.Timestamp(status["latest_pl_valuation_date"])
    return max(0.0, (latest - predictions_as_of).days / 30.44)


def refresh_dataset() -> Path:
    """Force kagglehub to re-download the dataset (picks up a newer version if one exists)."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("KAGGLEHUB_CACHE", str(RAW_DATA_DIR))
    import kagglehub

    return Path(kagglehub.dataset_download(DATASET_SLUG, force_download=True))


def _cached_panel_max_snapshot() -> pd.Timestamp | None:
    """Latest snapshot date in the most recent cached panel, if any."""
    from src.panel import PROCESSED_DATA_DIR

    caches = sorted(PROCESSED_DATA_DIR.glob("panel_data*_schema*.parquet"))
    if not caches:
        return None
    dates = pd.read_parquet(caches[-1], columns=["snapshot_date"])["snapshot_date"]
    return pd.Timestamp(dates.max())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--refresh", action="store_true", help="force a re-download before reporting")
    args = parser.parse_args()

    data_dir = refresh_dataset() if args.refresh else download_dataset()
    status = dataset_status(data_dir)

    print(f"dataset version:           {status['dataset_version']}")
    print(f"latest valuation (any):    {pd.Timestamp(status['latest_valuation_date']).date()}")
    print(f"latest valuation (PL):     {pd.Timestamp(status['latest_pl_valuation_date']).date()}")
    print(f"latest appearance:         {pd.Timestamp(status['latest_appearance_date']).date()}")
    print(f"PL valuation rows:         {status['num_pl_valuations']:,}")

    panel_max = _cached_panel_max_snapshot()
    if panel_max is None:
        print("cached panel:              none (run build_snapshot_panel() to create one)")
        return
    new_months = months_of_new_outcomes(status, panel_max)
    print(f"cached panel extends to:   {panel_max.date()}")
    if new_months == 0:
        print("new outcomes since panel:  none - notebook 02's predictions can't be checked yet")
    else:
        print(f"new outcomes since panel:  {new_months:.1f} months - rebuild the panel with cache=False")


if __name__ == "__main__":
    main()
