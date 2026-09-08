from __future__ import annotations

import pandas as pd
import pytest

from src.analysis import flag_transfers_within_horizon, grouped_backtest_metrics


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
