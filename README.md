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
  linear/Ridge/Lasso/XGBoost comparison (`src/modeling.py`).
- `notebooks/03_eye_test.ipynb` — uses the gap between actual value and a
  stats-only model's prediction as a proxy for value not explained by
  measurable output, then checks whether that gap correlates with big-six
  club affiliation, nationality, or position (a data-grounded look at the
  "eye test" idea from football media, not a direct measurement of it).
- `notebooks/04_fan_guide.ipynb` — plain-language tour of the headline
  findings from the other three notebooks (no stats jargon), for a reader
  who just wants the takeaways: prime age, what drives value, players
  projected to rise, and the big-six value premium.
- `notebooks/05_backtest.ipynb` — a genuine walk-forward backtest: trains
  on data up to June 2025 only, predicts 12 months forward, and checks
  those predictions against real outcomes that have since resolved. Finds
  a real but modest signal (predicted vs. actual correlation ~0.4) and a
  meaningfully higher error than the same-period test split notebook 02
  reported — the model tends to call the right direction but undersizes
  genuine breakouts.

## Tests

Unit tests cover the pure transformation logic in `src/panel.py` and
`src/modeling.py` (trailing-window math, horizon-target matching, feature
prep) against small synthetic DataFrames — they don't download the Kaggle
dataset, so they run in under a couple of seconds.

```
uv run pytest -x --tb=short
```
