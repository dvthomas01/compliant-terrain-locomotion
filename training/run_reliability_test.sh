#!/bin/bash
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source .venv/bin/activate; export PYTHONPATH="$PWD"
for S in 0 1 2; do
  echo "===== $(date '+%F %T') START fix_s${S} ====="
  python training/train_policy_b.py --run-name "policy_b_fix_s${S}" --render-freq 0 > "/tmp/fix_s${S}.log" 2>&1
  echo "===== $(date '+%F %T') DONE fix_s${S} ====="
done
echo "===== RELIABILITY TEST COMPLETE $(date '+%F %T') ====="
