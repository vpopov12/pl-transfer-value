"""End-to-end checks against the real panel, skipped when the dataset isn't cached.

The rest of the suite runs on small synthetic DataFrames so it stays fast and offline.
That leaves one thing untested: the contract between `src/panel.py`, which builds the
columns, and `src/modeling.py`, which names the ones it wants. Nothing in the unit tests
fails if a feature is added to NUMERIC_FEATURES but never built, or if a panel column is
renamed out from under the model; the first sign is a KeyError deep in a notebook run.

These tests close that gap when a cached copy of the Kaggle dataset is present locally.
They are skipped otherwise, so CI and a fresh clone are unaffected.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_loader import RAW_DATA_DIR
from src.modeling import (
    CATEGORICAL_FEATURES,
    CLUB_STANDING_FEATURES,
    CONTEXT_COLUMNS,
    MARKET_FEATURES,
    NUMERIC_FEATURES,
    ROLE_FEATURES,
    TREND_STEP_FEATURES,
    modelable_rows,
    prepare_features,
    target_column,
)
from src.panel import HORIZONS_MONTHS, build_snapshot_panel

pytestmark = pytest.mark.integration

_HAS_DATASET = any(RAW_DATA_DIR.rglob("player_valuations.csv"))
_NEEDS_DATA = pytest.mark.skipif(
    not _HAS_DATASET, reason="no cached Kaggle dataset; run `uv run python -m src.data_refresh`"
)


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    """The real cached panel (parquet-cached, so this is fast after the first build)."""
    return build_snapshot_panel()


@_NEEDS_DATA
def test_panel_supplies_every_column_the_models_ask_for(panel: pd.DataFrame) -> None:
    """Every feature any model can be built with must survive prepare_features."""
    prepared = prepare_features(panel)
    wanted = (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
        + CONTEXT_COLUMNS
        + ROLE_FEATURES
        + MARKET_FEATURES
        + CLUB_STANDING_FEATURES
        + TREND_STEP_FEATURES
    )
    missing = sorted(set(wanted) - set(prepared.columns))
    assert not missing, f"model asks for columns the panel doesn't build: {missing}"


@_NEEDS_DATA
def test_panel_has_a_usable_target_and_rows_for_every_horizon(panel: pd.DataFrame) -> None:
    for months in HORIZONS_MONTHS:
        assert target_column(months) in panel
        assert f"target_league_{months}m" in panel
        usable = modelable_rows(panel, months)
        assert len(usable) > 1000, f"{months}m horizon has only {len(usable)} modelable rows"


@_NEEDS_DATA
def test_panel_features_are_populated_not_silently_all_default(panel: pd.DataFrame) -> None:
    """A feature that is constant across the whole panel is a build bug, not a signal.

    This is the failure mode a column-presence check misses: the column exists, the model
    trains, and the feature is uniformly zero because a merge quietly matched nothing."""
    prepared = prepare_features(panel)
    constant = [c for c in NUMERIC_FEATURES if prepared[c].nunique(dropna=True) <= 1]
    assert not constant, f"features with no variation across the panel: {constant}"


@_NEEDS_DATA
def test_club_standing_is_known_for_most_recent_snapshots(panel: pd.DataFrame) -> None:
    """Club standing falls back to a neutral default when nothing is known, so a broken
    join would look like a valid panel. Match data starts in 2012-13; from 2014 on, most
    snapshots should carry a real standing."""
    recent = panel[panel["snapshot_date"] >= "2014-01-01"]
    known = (recent["club_season_progress"] > 0).mean()
    assert known > 0.6, f"only {known:.0%} of post-2014 snapshots have a real club standing"
