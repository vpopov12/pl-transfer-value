from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.forecast import (
    forecast_path,
    load_forecasts,
    make_forward_predictions,
    resolve_forecast_outcomes,
    save_forward_predictions,
    score_forecasts,
)
from src.panel import HORIZON_TOLERANCE_DAYS
from tests.test_modeling import _synthetic_panel


def _dt(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def _panel_with_open_outcomes(**kwargs) -> pd.DataFrame:
    """Synthetic panel whose latest snapshot per player has no outcome yet, as in real data."""
    panel = _synthetic_panel(**kwargs)
    latest_idx = panel.groupby("player_id")["snapshot_date"].idxmax()
    panel.loc[latest_idx, "value_change_12m_pct"] = np.nan
    return panel


def test_make_forward_predictions_gives_one_row_per_player_and_horizon_with_a_band() -> None:
    panel = _panel_with_open_outcomes(n_players=30, n_dates=24)
    preds = make_forward_predictions(panel, horizons=(12,), quantiles=(0.1, 0.9))

    as_of = panel["snapshot_date"].max()
    latest_players = panel[panel["snapshot_date"] >= as_of - pd.Timedelta(days=365)]
    assert len(preds) == latest_players["player_id"].nunique()
    assert preds["horizon_months"].eq(12).all()
    assert (preds["outcome_due_date"] > as_of).all()
    assert preds["model"].eq("xgboost").all()
    assert (preds["pct_q10"] <= preds["pct_q90"]).all()
    expected_due = (
        preds["snapshot_date"] + pd.DateOffset(months=12) + pd.Timedelta(days=HORIZON_TOLERANCE_DAYS[12])
    )
    assert preds["outcome_due_date"].equals(expected_due)
    assert np.allclose(
        preds["predicted_value_eur"], preds["current_value_eur"] * (1 + preds["predicted_pct_change"])
    )


def test_make_forward_predictions_keeps_only_rows_without_an_outcome_yet() -> None:
    panel = _panel_with_open_outcomes(n_players=30, n_dates=24)
    as_of = panel["snapshot_date"].max()
    # Player 0: latest snapshot already has a matched outcome (it was training data).
    latest_0 = panel[panel["player_id"] == 0]["snapshot_date"].idxmax()
    panel.loc[latest_0, "value_change_12m_pct"] = 0.3
    # Player 1: no outcome, but its 12-month window (plus tolerance) closed before the
    # panel ends, so nothing can ever be matched.
    stale = panel["player_id"] == 1
    panel.loc[stale, "snapshot_date"] = panel.loc[stale, "snapshot_date"] - pd.Timedelta(days=600)
    assert panel.loc[stale, "snapshot_date"].max() >= as_of - pd.Timedelta(days=3650)

    preds = make_forward_predictions(panel, horizons=(12,), quantiles=(0.1, 0.9), max_age_days=3650)
    assert 0 not in preds["player_id"].tolist()
    assert 1 not in preds["player_id"].tolist()
    assert len(preds) == 28


def _forecast() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "forecast_file": ["f.csv"] * 3,
            "player_id": [1, 2, 3],
            "snapshot_date": [_dt("2026-06-01")] * 3,
            "horizon_months": [12, 12, 3],
            "outcome_due_date": [_dt("2027-10-29"), _dt("2027-10-29"), _dt("2026-10-11")],
            "current_value_eur": [1e6, 2e6, 3e6],
            "predicted_pct_change": [0.5, -0.2, 0.1],
            "pct_q10": [0.0, -0.5, -0.2],
            "pct_q90": [1.0, 0.2, 0.4],
        }
    )


def test_resolve_forecast_outcomes_matches_like_the_panel_and_flags_due_rows() -> None:
    valuations = pd.DataFrame(
        {
            "player_id": [1, 2, 3],
            # 1: inside the 12m window; 2: nothing near 12m (only a stale row); 3: inside the 3m window
            "date": [_dt("2027-06-20"), _dt("2026-06-01"), _dt("2026-09-10")],
            "market_value_in_eur": [1.8e6, 2e6, 2.4e6],
            "player_club_domestic_competition_id": ["GB1", "GB1", "IT1"],
        }
    )
    out = resolve_forecast_outcomes(_forecast(), valuations)

    assert out["resolved"].tolist() == [True, False, True]
    assert out["actual_pct_change"].iloc[0] == 0.8
    assert np.isnan(out["actual_pct_change"].iloc[1])
    assert out["actual_pct_change"].iloc[2] == -0.2
    # data ends 2027-06-20: the 3m row is due, the 12m rows (due 2027-10-29) are not
    assert out["due"].tolist() == [False, False, True]


def test_score_forecasts_reports_shares_and_scores_only_with_enough_resolved_rows() -> None:
    resolved = _forecast()
    resolved["actual_pct_change"] = [0.6, 0.0, np.nan]
    resolved["resolved"] = [True, True, False]
    resolved["due"] = [True, True, False]
    scores = score_forecasts(resolved).set_index("horizon_months")

    assert scores.loc[12, "n"] == 2
    assert scores.loc[12, "share_resolved"] == 1.0
    assert scores.loc[12, "band_coverage"] == 1.0
    assert "mae" in scores and not np.isnan(scores.loc[12, "mae"])
    assert scores.loc[3, "share_due"] == 0.0
    assert np.isnan(scores.loc[3, "mae"])


def test_save_and_load_forecasts_round_trip(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("src.forecast.FORECAST_DIR", tmp_path)
    preds = _forecast().drop(columns="forecast_file")
    path = save_forward_predictions(preds, _dt("2026-06-01"), "679")

    assert path == forecast_path(_dt("2026-06-01"), "679")
    assert path.name == "forward_2026-06-01_data679.csv"
    loaded = load_forecasts(tmp_path)
    assert loaded["forecast_file"].eq(path.name).all()
    assert loaded["snapshot_date"].dtype.kind == "M"
    assert len(loaded) == 3
