# Frozen forecasts and how they will be judged

This file records, in advance, how the frozen forecasts in this directory will be
scored and what will count as success or failure. It was written on 18 September 2026,
before this project had seen a single outcome for any of them: the dataset they were
made from (Kaggle `davidcariboo/player-scores`, version 679) has no valuation after
12 June 2026. The commit that added this file is the timestamp.

The point of writing it down first is to remove the temptation to decide what the
result means after seeing it.

## What is frozen

`forward_2026-06-03_data679.csv` holds 2,801 predictions made from data running up to
3 June 2026: every Premier League player's latest valuation as of that date, predicted
3, 6, 9 and 12 months ahead by the default XGBoost model, with a 10th-90th percentile
band. The file can't be
revised: `src/forecast.py` refuses to overwrite it, and since its first outcomes were
due on 6 July 2026 it refuses even with `--force`.

Outcomes are matched exactly as in the backtest: the nearest Transfermarkt valuation,
in any league, within the horizon's tolerance window of the target date.

## What will be reported

For each horizon, whatever the numbers show:

- Spearman rank correlation between predicted and actual change (the primary metric)
- mean absolute error
- the share of outcomes inside the 10-90% band
- how the 10% of players with the highest forecasts did against the average player
- how many rows were resolved, and how many never re-valued

A horizon's score is **provisional** until every one of its rows is due, meaning its
matching window has closed in the data. Provisional numbers will be reported as such,
because the first outcomes to arrive are the players Transfermarkt re-values most
often, and notebook 06 shows those are ranked better than average.

## What the backtest says to expect

Walk-forward results for the same model, from notebook 05:

| Horizon | Spearman, mean | Worst cutoff | Best cutoff | MAE |
|---|---|---|---|---|
| 3 months | 0.679 | 0.588 | 0.803 | 0.194 |
| 6 months | 0.536 | 0.463 | 0.603 | 0.262 |
| 9 months | 0.553 | 0.482 | 0.615 | 0.350 |
| 12 months | 0.564 | 0.490 | 0.645 | 0.429 |

The 10-90% band covered 76% of outcomes on average against a target of 80%, and the
top-scored tenth beat the average player at all 17 cutoffs.

## How the result will be read

Each rule below was fixed before the result was known.

- **Consistent with the backtest:** a final Spearman inside that horizon's worst-to-best
  cutoff range. This is the expected result, and it will be described as confirming the
  backtest, not improving on it.
- **Worse than every backtest cutoff:** a final 12-month Spearman below 0.490. This will
  be reported plainly as the model failing to generalise to a period it never saw, and
  investigated. It will not be explained away as a hard year after the fact.
- **Better than every backtest cutoff:** above 0.645 at 12 months. This will be treated
  as most likely luck or an unusual mix of players, not as evidence the model got better.
- **The shortlist claim** fails if the top-scored tenth does *not* rise more than the
  average player at the 12-month horizon, whatever the Spearman says.
- **The band** is judged against the backtest's 76%, not the nominal 80%.

## Known weak spots, stated in advance

These are expected to score worse than the headline. That they do is not a surprise,
and they won't be excluded to flatter it:

- players whose latest valuation was more than six months old when the forecast was made
  (12% of the 12-month rows; the backtest ranks this group around 0.50)
- players who leave the Premier League (the backtest ranks them around 0.40)
- any market-wide shock in the window, which no player-level model can anticipate

## Every later forecast

Each new dataset version gets its own frozen file (`uv run python -m src.forecast`),
judged by these same rules unless this file is changed first, in a separate commit
made before that file's first outcome is due.
