from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_refresh import dataset_status, months_of_new_outcomes


def _write_fake_dataset(data_dir: Path) -> None:
    data_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "player_id": [1, 1, 2, 3],
            "date": ["2025-01-01", "2026-06-03", "2026-09-01", "2024-03-01"],
            "market_value_in_eur": [1e6, 2e6, 3e6, 4e6],
            "player_club_domestic_competition_id": ["GB1", "GB1", "ES1", "GB1"],
        }
    ).to_csv(data_dir / "player_valuations.csv", index=False)
    pd.DataFrame({"appearance_id": [1, 2], "date": ["2026-05-20", "2026-06-28"]}).to_csv(
        data_dir / "appearances.csv", index=False
    )


def test_dataset_status_reports_pl_and_overall_extents(tmp_path: Path) -> None:
    data_dir = tmp_path / "versions" / "679"
    _write_fake_dataset(data_dir)

    status = dataset_status(data_dir)

    assert status["dataset_version"] == "679"
    # The Spanish-league row extends the overall max but not the PL one.
    assert status["latest_valuation_date"] == pd.Timestamp("2026-09-01")
    assert status["latest_pl_valuation_date"] == pd.Timestamp("2026-06-03")
    assert status["num_pl_valuations"] == 3
    assert status["latest_appearance_date"] == pd.Timestamp("2026-06-28")


def test_months_of_new_outcomes_is_zero_when_nothing_newer_exists() -> None:
    status = {"latest_pl_valuation_date": pd.Timestamp("2026-06-03")}
    assert months_of_new_outcomes(status, pd.Timestamp("2026-06-03")) == 0.0
    assert months_of_new_outcomes(status, pd.Timestamp("2026-08-01")) == 0.0


def test_months_of_new_outcomes_counts_forward_from_prediction_date() -> None:
    status = {"latest_pl_valuation_date": pd.Timestamp("2027-06-03")}
    assert round(months_of_new_outcomes(status, pd.Timestamp("2026-06-03"))) == 12
