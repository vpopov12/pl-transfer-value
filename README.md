# pl-transfer-value

Predict Premier League player transfer market value from performance and age
using regression, comparing a linear baseline against regularized (Ridge/Lasso)
and tree-based (XGBoost) models.

## Notebooks

- `notebooks/01_eda.ipynb` — distribution of market value, value vs. age,
  missingness, correlation of value against performance stats, and which
  player characteristics add or subtract the most value.
- `notebooks/02_value_growth_prediction.ipynb` — predicts value *change* over
  the next 3/6/9/12 months per player, using a panel of historical valuation
  snapshots with trailing performance features (`src/panel.py`) and the same
  linear/Ridge/Lasso/XGBoost comparison (`src/modeling.py`).
