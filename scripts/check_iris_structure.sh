#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Activate venv if present
if [ -f "$PROJECT_DIR/.venv/bin/activate" ]; then
  source "$PROJECT_DIR/.venv/bin/activate"
fi

cd "$PROJECT_DIR"

FAIL=0
for f in requirements.txt \
         src/__init__.py src/data/__init__.py src/models/__init__.py \
         src/training/__init__.py src/utils/__init__.py \
         src/utils/io_utils.py src/utils/http_utils.py src/utils/schema.py \
         configs/project.yaml configs/subagent_architecture.yaml \
         tests/__init__.py; do
  if [ ! -f "$f" ]; then
    echo "MISSING: $f"
    FAIL=1
  fi
done

python3 -c "import torch; import torch_geometric" 2>/dev/null || {
  echo "MISSING: PyTorch or PyTorch Geometric not importable"
  FAIL=1
}

if [ $FAIL -eq 1 ]; then
  echo "FAIL: structure check"
  exit 1
fi
echo "PASS: structure check"
