from __future__ import annotations

import pandas as pd

from src.data_loader import _season_from_date


def test_season_from_date_uses_july_as_the_season_boundary() -> None:
    dates = pd.Series(
        pd.to_datetime(["2023-06-30", "2023-07-01", "2023-12-15", "2024-05-01"])
    )
    seasons = _season_from_date(dates)
    # A season runs Jul-Jun and is labeled by the year it starts in.
    assert seasons.tolist() == [2022, 2023, 2023, 2023]
