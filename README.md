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
- `notebooks/03_eye_test.ipynb` — uses the gap between actual value and a
  stats-only model's prediction as a proxy for value not explained by
  measurable output, then checks whether that gap correlates with big-six
  club affiliation, nationality, or position (a data-grounded look at the
  "eye test" idea from football media, not a direct measurement of it).
