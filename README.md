# pl-transfer-value

Predict Premier League player transfer market value from performance and age
using regression, comparing a linear baseline against regularized (Ridge/Lasso)
and tree-based (XGBoost) models.

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/) and pinned in
`uv.lock`:

```
uv sync --all-groups
uv run jupyter lab
```

The Kaggle dataset is downloaded into `data/raw/` on first use via `kagglehub`.
The valuation-snapshot panel used by notebooks 02-05 is cached as parquet in
`data/processed/` (keyed on dataset version and panel schema version), so
re-opening a notebook reloads it in well under a second instead of rebuilding
it from the raw CSVs. Pass `build_snapshot_panel(cache=False)` to force a rebuild.

## Checking for new data

Notebook 02's forward predictions only become a true out-of-sample test once the
upstream Kaggle dataset is re-scraped with valuations dated after the panel's
last snapshot. To check (and optionally force a re-download):

```
uv run python -m src.data_refresh            # report dataset version and latest dates
uv run python -m src.data_refresh --refresh  # force kagglehub to fetch the newest version
```

## Notebooks

- `notebooks/01_eda.ipynb` — distribution of market value, value vs. age,
  missingness, correlation of value against performance stats, and which
  player characteristics add or subtract the most value.

  
- `notebooks/02_value_growth_prediction.ipynb` — predicts value *change* over
  the next 3/6/9/12 months per player, using a panel of historical valuation
  snapshots with trailing performance features (`src/panel.py`) and the same
  linear/Ridge/Lasso/XGBoost comparison (`src/modeling.py`). Models fit
  `log(future / current)` and, alongside age, form and transfer history, use
  the player's own valuation trajectory (last change, time since, value vs.
  career peak). Predictions come with a quantile-regression 10-90% range and
  the player's contract expiry as context.
- `notebooks/03_eye_test.ipynb` — uses the gap between actual value and a
  stats-only model's prediction as a proxy for value not explained by
  measurable output, then checks whether that gap correlates with big-six
  club affiliation, nationality, or position (a data-grounded look at the
  "eye test" idea from football media, not a direct measurement of it).
- `notebooks/04_fan_guide.ipynb` — plain-language tour of the headline
  findings from the other three notebooks (no stats jargon), for a reader
  who just wants the takeaways: prime age, what drives value, players
  projected to rise, and the big-six value premium.
- `notebooks/05_backtest.ipynb` — a walk-forward backtest repeated at every
  six months from mid-2017 to early 2025 (`walk_forward_backtest` in
  `src/modeling.py`): at each cutoff the model trains only on outcomes that had
  fully resolved by then, predicts 12 months forward, and is scored against what
  really happened. Every model ranks players by future growth with a Spearman
  correlation of ~0.55-0.6 at every cutoff and horizon, and the players ranked in
  the top tenth rose several times more than average in every window. The
  notebook also tests fixes for the model's tendency to undersize big moves:
  fitting `log(future / current)` instead of raw % change fixes calibration
  (slope ~1.0 vs ~0.75) and is now the project default, while Huber loss and
  looser clipping don't help. Quantile-regression 10-90% intervals cover ~75%
  of outcomes walk-forward, with the shortfall concentrated in the pandemic
  markdown of 2020. Time-ordered grid search of the hyper-parameters
  (`tune_model`) is also evaluated walk-forward: it changes MAE and Spearman
  by a few thousandths and worsens calibration, so the hard-coded defaults are
  kept and tuning stays opt-in. A feature ablation shows the value-trend
  features lift 12-month Spearman from ~0.56 to ~0.59 (and 0.58 to 0.66 at 3
  months), while cards and start-share add nothing.

- `notebooks/06_why_it_works.ipynb` — asks *why* the growth model works and
  when it doesn't, all walk-forward (`src/analysis.py`). The signal is not
  transfer anticipation (stayers are ranked better than movers). Market-wide
  drift is real and the model has been over-optimistic since the pandemic, but
  it is only ~5% of the error. The market corrects players it under-values vs.
  their stats but not the ones it over-values, except mildly above €20M.
  Survivorship flatters the headline a little: counting players who left the
  league, Spearman is ~0.565 rather than 0.59. The value-trend features work
  through momentum at every price, read by the model as a step. Unchanged
  valuations are common but near-random, so a hurdle model adds nothing.
  Relegation is a proportional, predictable bias rather than a tail risk.

## Tests

Unit tests cover the pure transformation logic in `src/panel.py` and
`src/modeling.py` (trailing-window math, horizon-target matching, feature
prep) against small synthetic DataFrames — they don't download the Kaggle
dataset, so they run in under a couple of seconds.

```
uv run pytest -x --tb=short
```
