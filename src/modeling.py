"""Train and evaluate value-growth models (linear baseline vs. regularized vs.
tree-based) for each prediction horizon, run inference on current players, and
backtest the whole pipeline walk-forward across many historical cutoff dates.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from src.panel import HORIZON_TOLERANCE_DAYS

# The original feature set: current value, age, trailing on-pitch form, transfer history.
BASE_NUMERIC_FEATURES = [
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
# Added later, and kept: the player's own valuation trajectory. Walk-forward (notebook 05)
# these lift 12-month Spearman from ~0.56 to ~0.59 and cut MAE by ~4%.
VALUE_TREND_FEATURES = [
    "has_prev_valuation",
    "log_prev_value_ratio",
    "months_since_prev_valuation",
    "value_vs_peak",
]
# Tested and *not* kept: discipline and starter-vs-substitute role added nothing walk-forward.
# Still built into the panel so the ablation in notebook 05 can be re-run.
ROLE_FEATURES = ["trailing_cards_per90", "trailing_start_share"]
NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + VALUE_TREND_FEATURES
CATEGORICAL_FEATURES = ["sub_position", "foot"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Carried through predictions for context, never used as a model input: players.csv only
# knows the *current* contract, so for past snapshots it leaks later extensions.
CONTEXT_COLUMNS = ["months_to_contract_expiry"]

XGB_DEFAULTS = dict(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)

MODEL_FACTORIES = {
    "linear": lambda: LinearRegression(),
    "ridge": lambda: Ridge(alpha=1.0),
    "lasso": lambda: Lasso(alpha=0.01),
    "xgboost": lambda: XGBRegressor(**XGB_DEFAULTS),
    # Pseudo-Huber loss: squared error near zero, absolute error in the tails, so a
    # handful of extreme breakouts pull the fit less than under plain squared error.
    "xgboost_huber": lambda: XGBRegressor(objective="reg:pseudohubererror", huber_slope=1.0, **XGB_DEFAULTS),
}

# Small grids searched by `tune_model` with time-ordered cross-validation. Kept small on
# purpose: each combination is a full fit, and the walk-forward backtest refits at every
# cutoff. Keys are the estimator's own parameter names.
PARAM_GRIDS: dict[str, dict[str, list]] = {
    "linear": {},
    "ridge": {"alpha": [1.0, 10.0, 100.0, 1000.0]},
    "lasso": {"alpha": [0.0003, 0.001, 0.003, 0.01]},
    "xgboost": {"max_depth": [2, 3, 4, 6], "n_estimators": [100, 200, 400], "learning_rate": [0.03, 0.1]},
    "xgboost_huber": {
        "max_depth": [2, 3, 4, 6],
        "n_estimators": [100, 200, 400],
        "learning_rate": [0.03, 0.1],
    },
}

# Cap training/eval target so a few breakout outliers don't dominate squared-error fits.
TARGET_CLIP = (-0.95, 5.0)

# How the % change target is represented inside the model. Predictions are always
# returned on the % change scale regardless.
#   pct:       fit the raw fraction (0.5 == +50%).
#   log_ratio: fit log(future / current) == log1p(pct). Symmetric in up/down moves
#              and compresses breakouts, so the loss isn't dominated by them and
#              no clip is needed to keep squared error stable.
TARGET_TRANSFORMS: dict[str, tuple] = {
    "pct": (None, None),
    "log_ratio": (np.log1p, np.expm1),
}
# log_ratio wins on every walk-forward metric (lower MAE, higher Spearman, calibration
# slope ~1 instead of ~0.7), see notebook 05, so it is the default everywhere.
DEFAULT_TARGET_TRANSFORM = "log_ratio"


def target_column(months: int) -> str:
    return f"value_change_{months}m_pct"


def prepare_features(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    df["log_current_value_eur"] = np.log10(df["current_value_eur"])
    df["age_sq"] = df["age"] ** 2
    df["log_last_transfer_fee_eur"] = np.log1p(df["last_transfer_fee_eur"])
    # Previous change is heavy-tailed (a few +50,000% academy re-valuations); log the ratio.
    if "prev_value_change_pct" in df:
        df["log_prev_value_ratio"] = np.log1p(df["prev_value_change_pct"].clip(lower=-0.99))
    df["foot"] = df["foot"].fillna("unknown")
    df["sub_position"] = df["sub_position"].fillna("unknown")
    return df


def build_preprocessor(numeric_features: Sequence[str] = NUMERIC_FEATURES) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), list(numeric_features)),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def feature_columns(numeric_features: Sequence[str] = NUMERIC_FEATURES) -> list[str]:
    return list(numeric_features) + CATEGORICAL_FEATURES


def pipeline_feature_columns(pipeline) -> list[str]:
    """The input columns a fitted pipeline selects, read back from its ColumnTransformer.

    Falls back to FEATURE_COLUMNS for anything that isn't one of our pipelines (tests
    pass in stubs)."""
    inner = pipeline.regressor_ if isinstance(pipeline, TransformedTargetRegressor) else pipeline
    steps = getattr(inner, "named_steps", None)
    if not steps or "preprocess" not in steps:
        return FEATURE_COLUMNS
    columns: list[str] = []
    for _, transformer, selected in steps["preprocess"].transformers_:
        if transformer != "drop":
            columns.extend(selected)
    return columns


def _clip_target(y: pd.Series, clip: tuple[float, float] | None) -> pd.Series:
    return y if clip is None else y.clip(*clip)


def build_pipeline(
    model,
    target_transform: str = DEFAULT_TARGET_TRANSFORM,
    numeric_features: Sequence[str] = NUMERIC_FEATURES,
):
    """Preprocessing + model, optionally fitting on a transformed target.

    With a transform, `.predict()` still returns values on the % change scale.
    The fitted pipeline selects its own columns by name, so it can be given the full
    feature frame at predict time whichever `numeric_features` it was built with."""
    pipeline = Pipeline([("preprocess", build_preprocessor(numeric_features)), ("model", model)])
    func, inverse = TARGET_TRANSFORMS[target_transform]
    if func is None:
        return pipeline
    return TransformedTargetRegressor(regressor=pipeline, func=func, inverse_func=inverse)


def _model_param_prefix(pipeline) -> str:
    """Where the estimator's own params live inside the (possibly target-wrapped) pipeline."""
    return "regressor__model__" if isinstance(pipeline, TransformedTargetRegressor) else "model__"


def tune_model(
    train: pd.DataFrame,
    model_name: str,
    months: int,
    target_transform: str = DEFAULT_TARGET_TRANSFORM,
    clip: tuple[float, float] | None = TARGET_CLIP,
    n_splits: int = 3,
    numeric_features: Sequence[str] = NUMERIC_FEATURES,
) -> tuple[object, dict[str, object]]:
    """Grid-search PARAM_GRIDS[model_name] with time-ordered CV on the training rows.

    Rows are sorted by snapshot date first so each TimeSeriesSplit fold validates on
    snapshots later than everything it trained on. Scored on MAE of % change.
    Returns the refitted best pipeline and its chosen parameters (estimator names)."""
    grid = PARAM_GRIDS.get(model_name, {})
    train = train.sort_values("snapshot_date")
    pipeline = build_pipeline(MODEL_FACTORIES[model_name](), target_transform, numeric_features)
    columns = feature_columns(numeric_features)
    y_train = _clip_target(train[target_column(months)], clip)
    if not grid:
        pipeline.fit(train[columns], y_train)
        return pipeline, {}

    prefix = _model_param_prefix(pipeline)
    search = GridSearchCV(
        pipeline,
        {prefix + key: values for key, values in grid.items()},
        cv=TimeSeriesSplit(n_splits=n_splits),
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    search.fit(train[columns], y_train)
    best_params = {key.removeprefix(prefix): value for key, value in search.best_params_.items()}
    return search.best_estimator_, best_params


def fit_model(
    train: pd.DataFrame,
    model_name: str,
    months: int,
    target_transform: str = DEFAULT_TARGET_TRANSFORM,
    clip: tuple[float, float] | None = TARGET_CLIP,
    tune: bool = False,
    numeric_features: Sequence[str] = NUMERIC_FEATURES,
):
    """Fit one named model on already-prepared training rows for the given horizon.

    With `tune=True` the hyper-parameters come from `tune_model` instead of the
    hard-coded defaults in MODEL_FACTORIES. `numeric_features` narrows the inputs
    (e.g. BASE_NUMERIC_FEATURES for an ablation)."""
    if tune:
        pipeline, _ = tune_model(
            train, model_name, months, target_transform, clip, numeric_features=numeric_features
        )
        return pipeline
    pipeline = build_pipeline(MODEL_FACTORIES[model_name](), target_transform, numeric_features)
    y_train = _clip_target(train[target_column(months)], clip)
    pipeline.fit(train[feature_columns(numeric_features)], y_train)
    return pipeline


def fit_quantile_models(
    train: pd.DataFrame,
    months: int,
    quantiles: Sequence[float] = (0.1, 0.5, 0.9),
    target_transform: str = DEFAULT_TARGET_TRANSFORM,
    clip: tuple[float, float] | None = TARGET_CLIP,
) -> dict[float, object]:
    """One XGBoost quantile regressor per requested quantile, for prediction intervals."""
    y_train = _clip_target(train[target_column(months)], clip)
    models = {}
    for q in quantiles:
        regressor = XGBRegressor(objective="reg:quantileerror", quantile_alpha=q, **XGB_DEFAULTS)
        pipeline = build_pipeline(regressor, target_transform)
        pipeline.fit(train[FEATURE_COLUMNS], y_train)
        models[q] = pipeline
    return models


def predict_value_growth_interval(models: dict[float, object], current_players: pd.DataFrame) -> pd.DataFrame:
    """Predicted % change at each fitted quantile, plus the implied value range in EUR."""
    columns = pipeline_feature_columns(next(iter(models.values())))
    df = prepare_features(current_players).dropna(subset=columns)
    out = df[["player_id", "name", "sub_position", "age", "current_value_eur"]].copy()
    for q, pipeline in models.items():
        out[f"pct_q{int(round(q * 100)):02d}"] = pipeline.predict(df[columns])
    for col in [c for c in out.columns if c.startswith("pct_q")]:
        out[col.replace("pct_", "value_eur_")] = out["current_value_eur"] * (1 + out[col])
    return out


def modelable_rows(
    panel: pd.DataFrame, months: int, numeric_features: Sequence[str] = NUMERIC_FEATURES
) -> pd.DataFrame:
    """Prepared rows with every feature and this horizon's target present."""
    required = feature_columns(numeric_features) + [target_column(months), "snapshot_date"]
    return prepare_features(panel).dropna(subset=required)


def time_based_split(df: pd.DataFrame, test_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = df["snapshot_date"].quantile(1 - test_frac)
    return df[df["snapshot_date"] <= cutoff], df[df["snapshot_date"] > cutoff]


def train_horizon_models(
    panel: pd.DataFrame,
    months: int,
    test_frac: float = 0.2,
    tune: bool = False,
    model_names: Sequence[str] | None = None,
) -> dict:
    """Fit every model (or `model_names`) for one horizon and report held-out metrics.

    With `tune=True` each model's hyper-parameters are grid-searched with time-ordered
    CV inside the training split first, and the chosen values are reported as
    `best_params`."""
    df = modelable_rows(panel, months)
    train, test = time_based_split(df, test_frac)
    y_test = test[target_column(months)].clip(*TARGET_CLIP)

    results = {}
    for name in model_names or MODEL_FACTORIES:
        if tune:
            pipeline, best_params = tune_model(train, name, months)
        else:
            pipeline, best_params = fit_model(train, name, months), {}
        pred = pipeline.predict(test[FEATURE_COLUMNS])
        results[name] = {
            "pipeline": pipeline,
            "best_params": best_params,
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
    columns = pipeline_feature_columns(pipeline)
    df = prepare_features(current_players).dropna(subset=columns)
    predicted_pct = pipeline.predict(df[columns])
    context = [c for c in CONTEXT_COLUMNS if c in df]
    out = df[["player_id", "name", "sub_position", "age", "current_value_eur", *context]].copy()
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

    `actual` is clipped to TARGET_CLIP for the error and calibration metrics so a couple
    of extreme breakouts don't swamp them; the rank metrics are unaffected either way.

    `calibration_slope` is the OLS slope of actual on predicted: 1.0 means predicted
    sizes are right on average, > 1 means the model systematically undersizes moves,
    < 1 means it oversizes them."""
    actual_clipped = actual.clip(*TARGET_CLIP)
    errors = predicted - actual_clipped
    nonzero = actual != 0
    top_decile = predicted >= predicted.quantile(0.9)
    pred_var = predicted.var()
    slope = float(predicted.cov(actual_clipped) / pred_var) if pred_var > 0 else float("nan")
    return {
        "mae": float(errors.abs().mean()),
        "rmse": float(np.sqrt((errors**2).mean())),
        "pearson": float(predicted.corr(actual_clipped)),
        "spearman": float(spearmanr(predicted, actual).statistic),
        "direction_accuracy": float((np.sign(predicted[nonzero]) == np.sign(actual[nonzero])).mean()),
        "calibration_slope": slope,
        "mean_actual_top_decile": float(actual[top_decile].mean()),
        "mean_predicted_top_decile": float(predicted[top_decile].mean()),
        "mean_actual_all": float(actual.mean()),
    }


def walk_forward_backtest(
    panel: pd.DataFrame,
    months: int,
    cutoffs: Iterable[pd.Timestamp] | None = None,
    model_names: Sequence[str] = ("linear", "ridge", "lasso", "xgboost"),
    max_age_days: int = 365,
    target_transform: str = DEFAULT_TARGET_TRANSFORM,
    clip: tuple[float, float] | None = TARGET_CLIP,
    tune: bool = False,
    numeric_features: Sequence[str] = NUMERIC_FEATURES,
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
    usable = modelable_rows(panel, months, numeric_features)
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
            pipeline = fit_model(
                train, name, months, target_transform, clip, tune=tune, numeric_features=numeric_features
            )
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
    score_cols = ["mae", "rmse", "pearson", "spearman", "direction_accuracy", "calibration_slope"]
    summary = metrics.groupby("model")[score_cols].agg(["mean", "std", "min", "max"])
    summary["n_cutoffs"] = metrics.groupby("model").size()
    return summary.sort_values(("spearman", "mean"), ascending=False)


def walk_forward_interval_backtest(
    panel: pd.DataFrame,
    months: int,
    cutoffs: Iterable[pd.Timestamp] | None = None,
    quantiles: Sequence[float] = (0.1, 0.9),
    max_age_days: int = 365,
    target_transform: str = DEFAULT_TARGET_TRANSFORM,
) -> pd.DataFrame:
    """Walk-forward check of quantile prediction intervals: at each cutoff, what fraction of
    realised outcomes fell inside the [low, high] quantile band, and how wide was it?

    A well-calibrated 10-90 band should cover about 80% of outcomes."""
    usable = modelable_rows(panel, months)
    if cutoffs is None:
        cutoffs = default_backtest_cutoffs(panel, months)
    low_q, high_q = min(quantiles), max(quantiles)
    target_col = target_column(months)

    rows: list[dict[str, object]] = []
    for cutoff in cutoffs:
        cutoff = pd.Timestamp(cutoff)
        train = strict_training_panel(usable, cutoff, months)
        eval_rows = latest_snapshot_per_player(usable[usable["snapshot_date"] <= cutoff], max_age_days)
        if train.empty or eval_rows.empty:
            continue
        models = fit_quantile_models(train, months, (low_q, high_q), target_transform)
        preds = predict_value_growth_interval(models, eval_rows)
        preds = preds.merge(eval_rows[["player_id", target_col]], on="player_id", how="left")
        low = preds[f"pct_q{int(round(low_q * 100)):02d}"]
        high = preds[f"pct_q{int(round(high_q * 100)):02d}"]
        actual = preds[target_col]
        rows.append(
            {
                "cutoff": cutoff,
                "n_eval": len(preds),
                "coverage": float(((actual >= low) & (actual <= high)).mean()),
                "below_low": float((actual < low).mean()),
                "above_high": float((actual > high).mean()),
                "median_width": float((high - low).median()),
            }
        )
    return pd.DataFrame(rows)
