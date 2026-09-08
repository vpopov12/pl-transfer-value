"""Train and evaluate value-growth models (linear baseline vs. regularized vs.
tree-based) for each prediction horizon, run inference on current players, and
backtest the whole pipeline walk-forward across many historical cutoff dates.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from src.panel import HORIZON_TOLERANCE_DAYS

NUMERIC_FEATURES = [
    "log_current_value_eur",
    "age",
    "age_sq",
    "trailing_goals_per90",
    "trailing_assists_per90",
    "trailing_minutes",
    "trailing_appearances",
    "trailing_avg_club_position",
    "has_transfer_history",
    "months_since_last_transfer",
    "num_prior_transfers",
    "log_last_transfer_fee_eur",
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

# Cap training/eval target so a few breakout outliers don't dominate squared-error fits.
TARGET_CLIP = (-0.95, 5.0)


def target_column(months: int) -> str:
    return f"value_change_{months}m_pct"


def prepare_features(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    df["log_current_value_eur"] = np.log10(df["current_value_eur"])
    df["age_sq"] = df["age"] ** 2
    df["log_last_transfer_fee_eur"] = np.log1p(df["last_transfer_fee_eur"])
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


def fit_model(train: pd.DataFrame, model_name: str, months: int) -> Pipeline:
    """Fit one named model on already-prepared training rows for the given horizon."""
    pipeline = Pipeline([("preprocess", build_preprocessor()), ("model", MODEL_FACTORIES[model_name]())])
    y_train = train[target_column(months)].clip(*TARGET_CLIP)
    pipeline.fit(train[FEATURE_COLUMNS], y_train)
    return pipeline


def modelable_rows(panel: pd.DataFrame, months: int) -> pd.DataFrame:
    """Prepared rows with every feature and this horizon's target present."""
    return prepare_features(panel).dropna(subset=FEATURE_COLUMNS + [target_column(months), "snapshot_date"])


def time_based_split(df: pd.DataFrame, test_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = df["snapshot_date"].quantile(1 - test_frac)
    return df[df["snapshot_date"] <= cutoff], df[df["snapshot_date"] > cutoff]


def train_horizon_models(panel: pd.DataFrame, months: int, test_frac: float = 0.2) -> dict:
    """Fit every model in MODEL_FACTORIES for one horizon and report held-out metrics."""
    df = modelable_rows(panel, months)
    train, test = time_based_split(df, test_frac)
    y_test = test[target_column(months)].clip(*TARGET_CLIP)

    results = {}
    for name in MODEL_FACTORIES:
        pipeline = fit_model(train, name, months)
        pred = pipeline.predict(test[FEATURE_COLUMNS])
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


# --------------------------------------------------------------------------------------
# Walk-forward backtesting
# --------------------------------------------------------------------------------------


def target_resolution_date(snapshot_date: pd.Series, months: int) -> pd.Series:
    """Latest date at which a snapshot's `months`-horizon target could still be matched.

    The panel matches each target to the nearest valuation within a tolerance window
    around snapshot + months, so the outcome isn't fully known until the window closes."""
    return snapshot_date + pd.DateOffset(months=months) + pd.Timedelta(days=HORIZON_TOLERANCE_DAYS[months])


def strict_training_panel(panel: pd.DataFrame, cutoff: pd.Timestamp, months: int) -> pd.DataFrame:
    """Rows whose `months`-horizon outcome had *fully* resolved by `cutoff`.

    Stricter than `snapshot_date <= cutoff - months`: it also excludes rows whose
    matching window would still have been open at the cutoff, so a model trained here
    genuinely had no access to any valuation dated after the cutoff."""
    return panel[target_resolution_date(panel["snapshot_date"], months) <= cutoff]


def default_backtest_cutoffs(
    panel: pd.DataFrame,
    months: int,
    step_months: int = 6,
    min_train_rows_with_form: int = 2000,
) -> list[pd.Timestamp]:
    """Evenly spaced month-start cutoffs, every `step_months`, for which a backtest is meaningful.

    Walks back from the latest cutoff whose evaluation outcomes have all resolved, until
    the strict training panel would have fewer than `min_train_rows_with_form` rows that
    carry any trailing-form data. That last condition matters: match-level appearance
    data starts years after the valuation history does, so early cutoffs would train on
    rows with all-zero form features and then extrapolate wildly on fully-populated ones."""
    usable = modelable_rows(panel, months)
    data_end = panel["snapshot_date"].max()
    tolerance = pd.Timedelta(days=HORIZON_TOLERANCE_DAYS[months])
    last_cutoff = (data_end - pd.DateOffset(months=months) - tolerance).normalize().replace(day=1)

    cutoffs: list[pd.Timestamp] = []
    cutoff = last_cutoff
    while True:
        train = strict_training_panel(usable, cutoff, months)
        if (train["trailing_appearances"] > 0).sum() < min_train_rows_with_form:
            break
        cutoffs.append(cutoff)
        cutoff = cutoff - pd.DateOffset(months=step_months)
    return sorted(cutoffs)


def _backtest_metrics(predicted: pd.Series, actual: pd.Series) -> dict[str, float]:
    """Accuracy of point predictions of % change against realised % change.

    `actual` is clipped to TARGET_CLIP for the error metrics so a couple of extreme
    breakouts don't swamp the MAE; the rank metrics are unaffected either way."""
    actual_clipped = actual.clip(*TARGET_CLIP)
    errors = predicted - actual_clipped
    nonzero = actual != 0
    top_decile = predicted >= predicted.quantile(0.9)
    return {
        "mae": float(errors.abs().mean()),
        "rmse": float(np.sqrt((errors**2).mean())),
        "pearson": float(predicted.corr(actual_clipped)),
        "spearman": float(spearmanr(predicted, actual).statistic),
        "direction_accuracy": float((np.sign(predicted[nonzero]) == np.sign(actual[nonzero])).mean()),
        "mean_actual_top_decile": float(actual[top_decile].mean()),
        "mean_actual_all": float(actual.mean()),
    }


def walk_forward_backtest(
    panel: pd.DataFrame,
    months: int,
    cutoffs: Iterable[pd.Timestamp] | None = None,
    model_names: Sequence[str] = ("linear", "ridge", "lasso", "xgboost"),
    max_age_days: int = 365,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repeat "train up to T, predict T + months, score against what actually happened"
    for several cutoff dates T and every named model.

    At each cutoff the model sees only rows whose outcome had resolved by T (see
    `strict_training_panel`), then predicts for each player's latest snapshot as of T.
    Those predictions are scored against the panel's realised target for that snapshot.

    Returns:
        metrics: one row per (cutoff, model) with n_train, n_eval, and the scores
            from `_backtest_metrics`.
        predictions: one row per (cutoff, model, player) with predicted and actual
            % change, for plotting and case studies.
    """
    usable = modelable_rows(panel, months)
    if cutoffs is None:
        cutoffs = default_backtest_cutoffs(panel, months)
    target_col = target_column(months)

    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for cutoff in cutoffs:
        cutoff = pd.Timestamp(cutoff)
        train = strict_training_panel(usable, cutoff, months)
        eval_rows = latest_snapshot_per_player(usable[usable["snapshot_date"] <= cutoff], max_age_days)
        if train.empty or eval_rows.empty:
            continue
        for name in model_names:
            pipeline = fit_model(train, name, months)
            preds = predict_value_growth(pipeline, eval_rows)
            preds = preds.merge(
                eval_rows[["player_id", "snapshot_date", target_col]].rename(
                    columns={target_col: "actual_pct_change"}
                ),
                on="player_id",
                how="left",
            )
            preds.insert(0, "model", name)
            preds.insert(0, "cutoff", cutoff)
            prediction_frames.append(preds)
            metric_rows.append(
                {
                    "cutoff": cutoff,
                    "model": name,
                    "n_train": len(train),
                    "n_eval": len(preds),
                    **_backtest_metrics(preds["predicted_pct_change"], preds["actual_pct_change"]),
                }
            )

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    return metrics, predictions


def summarize_backtest(metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean and spread of each score across cutoffs, per model."""
    score_cols = ["mae", "rmse", "pearson", "spearman", "direction_accuracy"]
    summary = metrics.groupby("model")[score_cols].agg(["mean", "std", "min", "max"])
    summary["n_cutoffs"] = metrics.groupby("model").size()
    return summary.sort_values(("spearman", "mean"), ascending=False)
