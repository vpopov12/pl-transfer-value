"""Why does the model predict what it predicts for a particular player?

The notebooks explain the model in aggregate: ablations say which feature groups
matter, partial-dependence curves say what the model does with one feature on
average. Neither says why *this* player was flagged. XGBoost can split any single
prediction into a contribution per input feature (TreeSHAP, built into the library),
and those contributions add up exactly to the prediction, so a watch-list entry can
carry its reasons with it.

Contributions are in the model's own units: log(future value / current value). A
contribution of +0.10 pushes the predicted log change up by 0.10, roughly ten percent.
They are additive in that space and only there, so they are reported as-is rather
than converted to percentages that would no longer add up.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost
from sklearn.compose import TransformedTargetRegressor

from src.modeling import pipeline_feature_columns, prepare_features

BASELINE = "baseline"

# Plain-language names for the features, for anything shown to a reader.
FEATURE_LABELS = {
    "log_current_value_eur": "current value",
    "age": "age",
    "age_sq": "age",
    "trailing_goals_per90": "goals per 90",
    "trailing_assists_per90": "assists per 90",
    "trailing_minutes": "minutes played",
    "trailing_appearances": "appearances",
    "trailing_avg_club_position": "club's average league position",
    "has_transfer_history": "transfer history",
    "months_since_last_transfer": "time since last transfer",
    "num_prior_transfers": "number of past transfers",
    "log_last_transfer_fee_eur": "last transfer fee",
    "has_prev_valuation": "valuation history",
    "log_prev_value_ratio": "last re-valuation",
    "months_since_prev_valuation": "time since last re-valuation",
    "value_vs_peak": "value vs. career peak",
    "club_league_position": "club's current league position",
    "club_season_progress": "point in the season",
    "club_in_drop_zone": "club in the relegation zone",
    "sub_position": "position",
    "foot": "preferred foot",
}


def contribution_columns(explanations: pd.DataFrame) -> list[str]:
    """The per-feature contribution columns of an `explain_predictions` result."""
    return [c for c in explanations.columns if c not in ("player_id", BASELINE, "predicted_log_change")]


def _unwrap(pipeline):
    """The fitted preprocessing step and XGBoost model inside a project pipeline."""
    inner = pipeline.regressor_ if isinstance(pipeline, TransformedTargetRegressor) else pipeline
    steps = getattr(inner, "named_steps", {})
    model = steps.get("model")
    if not isinstance(model, xgboost.XGBModel):
        raise TypeError("explanations need an XGBoost pipeline (per-feature contributions are tree-based)")
    return steps["preprocess"], model


def _source_feature_per_column(preprocess) -> list[str]:
    """Map every column the preprocessor outputs back to the input feature it came from,
    so the contributions of a one-hot encoded feature can be summed into one."""
    sources: list[str] = []
    for _, transformer, selected in preprocess.transformers_:
        if transformer == "drop":
            continue
        if hasattr(transformer, "categories_"):
            for column, categories in zip(selected, transformer.categories_):
                sources.extend([column] * len(categories))
        else:
            sources.extend(selected)
    return sources


def explain_predictions(pipeline, rows: pd.DataFrame) -> pd.DataFrame:
    """Per-player feature contributions to the predicted log change.

    Returns one row per player and a column per input feature, plus `baseline` (the
    model's starting point before any feature is considered) and `predicted_log_change`.
    For every row, baseline + all feature columns == predicted_log_change exactly.
    One-hot encoded inputs are summed back into their source feature."""
    preprocess, model = _unwrap(pipeline)
    columns = pipeline_feature_columns(pipeline)
    prepared = prepare_features(rows).dropna(subset=columns)
    matrix = preprocess.transform(prepared[columns])
    contributions = model.get_booster().predict(xgboost.DMatrix(matrix), pred_contribs=True)

    sources = _source_feature_per_column(preprocess)
    per_column = pd.DataFrame(contributions[:, :-1], columns=range(len(sources)))
    grouped = per_column.T.groupby(np.array(sources)).sum().T
    grouped = grouped[[c for c in dict.fromkeys(sources)]]
    grouped[BASELINE] = contributions[:, -1]
    grouped["predicted_log_change"] = contributions.sum(axis=1)
    grouped.index = prepared.index
    grouped.insert(0, "player_id", prepared["player_id"].to_numpy())
    return grouped


def top_reasons(explanations: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """The `n` features that moved each player's prediction the most, in either direction.

    Long format: player_id, rank, feature, label, contribution. Features the model sees
    twice (age and age squared) are combined under one label first."""
    feature_cols = contribution_columns(explanations)
    long = explanations.melt(
        id_vars="player_id", value_vars=feature_cols, var_name="feature", value_name="contribution"
    )
    long["label"] = long["feature"].map(FEATURE_LABELS).fillna(long["feature"])
    by_label = long.groupby(["player_id", "label"], as_index=False)["contribution"].sum()
    by_label["magnitude"] = by_label["contribution"].abs()
    ranked = by_label.sort_values(["player_id", "magnitude"], ascending=[True, False])
    ranked["rank"] = ranked.groupby("player_id").cumcount() + 1
    return ranked[ranked["rank"] <= n].drop(columns="magnitude").reset_index(drop=True)


def reasons_text(reasons: pd.DataFrame) -> pd.Series:
    """One readable line per player, e.g. "last re-valuation +0.31, age +0.12, current value -0.08"."""
    return reasons.groupby("player_id").apply(
        lambda part: ", ".join(f"{r.label} {r.contribution:+.2f}" for r in part.itertuples()),
        include_groups=False,
    )
