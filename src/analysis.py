"""Diagnostics that ask *why* the value-growth model works and when it doesn't:
transfer-driven re-valuations, market-wide drift, survivorship, and so on.
Used by notebook 06; the heavy lifting (training, walk-forward) stays in modeling.py.
"""

from __future__ import annotations

import pandas as pd

from src.modeling import _backtest_metrics
from src.panel import HORIZON_TOLERANCE_DAYS


def flag_transfers_within_horizon(
    predictions: pd.DataFrame, transfers: pd.DataFrame, months: int
) -> pd.DataFrame:
    """Mark each backtest prediction with whether the player changed club inside the
    outcome window (snapshot date to snapshot + months + matching tolerance).

    Transfermarkt re-values a player straight after a transfer, usually toward the fee,
    so a "prediction" that lands in this window may really be a prediction that a move
    happens. `paid_transfer` narrows it to moves with a recorded fee above zero.

    `transfers` needs player_id, transfer_date, transfer_fee, and from/to club ids;
    rows where the club doesn't change (loan returns recorded as moves) are ignored."""
    window_days = HORIZON_TOLERANCE_DAYS[months]
    moves = transfers.dropna(subset=["player_id", "transfer_date"]).copy()
    if {"from_club_id", "to_club_id"} <= set(moves.columns):
        moves = moves[moves["from_club_id"] != moves["to_club_id"]]
    moves = moves[["player_id", "transfer_date", "transfer_fee"]]

    keyed = predictions[["player_id", "snapshot_date"]].reset_index()
    joined = keyed.merge(moves, on="player_id", how="left")
    window_end = joined["snapshot_date"] + pd.DateOffset(months=months) + pd.Timedelta(days=window_days)
    in_window = (joined["transfer_date"] > joined["snapshot_date"]) & (joined["transfer_date"] <= window_end)
    joined = joined[in_window]

    flags = pd.DataFrame(
        {
            "transferred": joined.groupby("index").size() > 0,
            "paid_transfer": joined.groupby("index")["transfer_fee"].max() > 0,
        }
    )
    out = predictions.copy()
    out["transferred"] = flags["transferred"].reindex(out.index).fillna(False).astype(bool)
    out["paid_transfer"] = flags["paid_transfer"].reindex(out.index).fillna(False).astype(bool)
    return out


def grouped_backtest_metrics(predictions: pd.DataFrame, group_col: str, min_rows: int = 30) -> pd.DataFrame:
    """Backtest scores per (cutoff, group), for splitting the eval set by some flag."""
    rows = []
    for (cutoff, group), part in predictions.groupby(["cutoff", group_col]):
        if len(part) < min_rows:
            continue
        rows.append(
            {
                "cutoff": cutoff,
                group_col: group,
                "n": len(part),
                "share": len(part) / (predictions["cutoff"] == cutoff).sum(),
                **_backtest_metrics(part["predicted_pct_change"], part["actual_pct_change"]),
            }
        )
    return pd.DataFrame(rows)
