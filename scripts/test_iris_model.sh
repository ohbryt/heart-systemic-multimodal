#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -f "$PROJECT_DIR/.venv/bin/activate" ]; then
  source "$PROJECT_DIR/.venv/bin/activate"
fi

cd "$PROJECT_DIR"

echo "Running IRIS model end-to-end test..."

PYTHONPATH=. python3 scripts/run_iris_model.py --synthetic --n-genes 100 --epochs 5

for f in data/reports/iris_key_driver_scores.csv data/reports/iris_evaluation.csv; do
  if [ ! -f "$f" ]; then
    echo "FAIL: missing $f"
    exit 1
  fi
done

LINES=$(wc -l < data/reports/iris_key_driver_scores.csv)
if [ "$LINES" -lt 2 ]; then
  echo "FAIL: iris_key_driver_scores.csv is empty"
  exit 1
fi

if ! grep -q "auroc" data/reports/iris_evaluation.csv; then
  echo "FAIL: iris_evaluation.csv missing auroc metric"
  exit 1
fi

echo "PASS: IRIS model end-to-end test"
