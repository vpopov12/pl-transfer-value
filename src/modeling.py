"""Train and evaluate value-growth models (linear baseline vs. regularized vs.
tree-based) for each prediction horizon, and run inference on current players.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

NUMERIC_FEATURES = [
    "log_current_value_eur",
    "age",
    "age_sq",
    "trailing_goals_per90",
    "trailing_assists_per90",
    "trailing_minutes",
    "trailing_appearances",
]
CATEGORICAL_FEATURES = ["sub_position", "foot"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

MODEL_FACTORIES = {
    "linear": lambda: LinearRegression(),
    "ridge": lambda: Ridge(alpha=1.0),
    "lasso": lambda: Lasso(alpha=0.01),
    "xgboost": lambda: XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    ),
}

TARGET_CLIP = (-0.95, 5.0)  # cap training/eval target so a few breakout outliers don't dominate squared-error fits


def prepare_features(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    df["log_current_value_eur"] = np.log10(df["current_value_eur"])
    df["age_sq"] = df["age"] ** 2
    df["foot"] = df["foot"].fillna("unknown")
    df["sub_position"] = df["sub_position"].fillna("unknown")
    return df


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def time_based_split(df: pd.DataFrame, test_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = df["snapshot_date"].quantile(1 - test_frac)
    return df[df["snapshot_date"] <= cutoff], df[df["snapshot_date"] > cutoff]


def train_horizon_models(panel: pd.DataFrame, months: int, test_frac: float = 0.2) -> dict:
    """Fit every model in MODEL_FACTORIES for one horizon and report held-out metrics."""
    target_col = f"value_change_{months}m_pct"
    df = prepare_features(panel).dropna(subset=FEATURE_COLUMNS + [target_col, "snapshot_date"])
    train, test = time_based_split(df, test_frac)

    X_train, y_train = train[FEATURE_COLUMNS], train[target_col].clip(*TARGET_CLIP)
    X_test, y_test = test[FEATURE_COLUMNS], test[target_col].clip(*TARGET_CLIP)

    results = {}
    for name, factory in MODEL_FACTORIES.items():
        pipeline = Pipeline([("preprocess", build_preprocessor()), ("model", factory())])
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        results[name] = {
            "pipeline": pipeline,
            "mae": mean_absolute_error(y_test, pred),
            "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
            "r2": r2_score(y_test, pred),
            "n_train": len(train),
            "n_test": len(test),
        }
    return results


def latest_snapshot_per_player(panel: pd.DataFrame, max_age_days: int = 365) -> pd.DataFrame:
    """Most recent valuation snapshot per player, restricted to reasonably current ones."""
    as_of = panel["snapshot_date"].max()
    recent = panel[panel["snapshot_date"] >= as_of - pd.Timedelta(days=max_age_days)]
    idx = recent.groupby("player_id")["snapshot_date"].idxmax()
    return recent.loc[idx]


def predict_value_growth(pipeline: Pipeline, current_players: pd.DataFrame) -> pd.DataFrame:
    """Predict % and $ value change for the given current-player snapshot rows."""
    df = prepare_features(current_players).dropna(subset=FEATURE_COLUMNS)
    predicted_pct = pipeline.predict(df[FEATURE_COLUMNS])
    out = df[["player_id", "name", "sub_position", "age", "current_value_eur"]].copy()
    out["predicted_pct_change"] = predicted_pct
    out["predicted_eur_change"] = predicted_pct * out["current_value_eur"]
    out["predicted_value_eur"] = out["current_value_eur"] + out["predicted_eur_change"]
    return out
