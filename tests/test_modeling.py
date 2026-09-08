from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.modeling import (
    latest_snapshot_per_player,
    prepare_features,
    predict_value_growth,
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
