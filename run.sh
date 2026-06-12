#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"

"$PYTHON_BIN" preprocess.py
"$PYTHON_BIN" base_hier.py
# evaluate deepar and deepar-hier
"$PYTHON_BIN" evaluate.py
# output comparison result
"$PYTHON_BIN" compare.py
