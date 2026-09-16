from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.explain import BASELINE, contribution_columns, explain_predictions, reasons_text, top_reasons
from src.modeling import CATEGORICAL_FEATURES, NUMERIC_FEATURES, fit_model, modelable_rows
from tests.test_modeling import _synthetic_panel


@pytest.fixture(scope="module")
def fitted():
    panel = modelable_rows(_synthetic_panel(n_players=40, n_dates=24), months=12)
    pipeline = fit_model(panel, "xgboost", months=12)
    return pipeline, panel.tail(40)


def test_contributions_add_up_exactly_to_the_prediction(fitted) -> None:
    pipeline, rows = fitted
    explanations = explain_predictions(pipeline, rows)
    feature_cols = contribution_columns(explanations)
    reconstructed = explanations[feature_cols].sum(axis=1) + explanations[BASELINE]
    np.testing.assert_allclose(reconstructed, explanations["predicted_log_change"], atol=1e-4)
    # and the log change they add up to is the pipeline's own prediction, on its % scale
    predicted_pct = pipeline.predict(rows)
    np.testing.assert_allclose(np.expm1(explanations["predicted_log_change"]), predicted_pct, atol=1e-4)


def test_one_hot_columns_are_summed_back_into_their_source_feature(fitted) -> None:
    pipeline, rows = fitted
    explanations = explain_predictions(pipeline, rows)
    feature_cols = contribution_columns(explanations)
    assert set(feature_cols) == set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES)
    assert not any(c.startswith(("num__", "cat__")) for c in feature_cols)


def test_top_reasons_ranks_by_size_in_either_direction_and_merges_age_terms() -> None:
    explanations = pd.DataFrame(
        {
            "player_id": [7],
            "age": [0.05],
            "age_sq": [0.04],  # same label as age: combined to +0.09
            "log_prev_value_ratio": [-0.30],
            "trailing_minutes": [0.01],
            BASELINE: [0.02],
            "predicted_log_change": [-0.18],
        }
    )
    reasons = top_reasons(explanations, n=2)
    assert reasons["label"].tolist() == ["last re-valuation", "age"]
    assert reasons["contribution"].tolist() == pytest.approx([-0.30, 0.09])
    assert reasons_text(reasons).loc[7] == "last re-valuation -0.30, age +0.09"


def test_explanations_need_an_xgboost_pipeline() -> None:
    panel = modelable_rows(_synthetic_panel(n_players=20, n_dates=20), months=12)
    ridge = fit_model(panel, "ridge", months=12)
    with pytest.raises(TypeError, match="XGBoost"):
        explain_predictions(ridge, panel.tail(5))
