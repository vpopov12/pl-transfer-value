#!/usr/bin/env bash
# Re-execute every notebook in order, in place, stopping at the first failure.
#
#   scripts/run_notebooks.sh                 # all notebooks
#   scripts/run_notebooks.sh 05 06           # only notebooks whose names start with 05 or 06
#
# Walk-forward backtests are cached (see enable_backtest_cache in src/modeling.py), so a
# re-run only recomputes what changed; a cold run after new data takes several minutes.
# If you might close the terminal meanwhile, run it detached:
#   nohup scripts/run_notebooks.sh > notebooks.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../notebooks"

selected=()
if [ "$#" -eq 0 ]; then
  selected=(*.ipynb)
else
  for prefix in "$@"; do
    matches=("$prefix"*.ipynb)
    [ -e "${matches[0]}" ] || { echo "no notebook starts with '$prefix'" >&2; exit 2; }
    selected+=("${matches[@]}")
  done
fi

for nb in "${selected[@]}"; do
  start=$(date +%s)
  printf '%-40s ' "$nb"
  if ! uv run jupyter nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=3600 "$nb" > "/tmp/run_notebooks_${nb%.ipynb}.log" 2>&1; then
    echo "FAILED (log: /tmp/run_notebooks_${nb%.ipynb}.log)"
    exit 1
  fi
  echo "ok  $(( $(date +%s) - start ))s"
done
