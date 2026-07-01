#!/bin/bash
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source .venv/bin/activate
export PYTHONPATH="$PWD"
echo "waiting for uniform eval (pid 74720) to finish..."
while kill -0 74720 2>/dev/null; do sleep 60; done
echo "===== $(date '+%F %T') eval done; START clean_trans (89D) ====="
python training/train_clean_trans.py --run-name clean_trans --render-freq 0 > /tmp/clean_trans.log 2>&1
echo "===== $(date '+%F %T') START clean_trans_noh (49D) ====="
python training/train_clean_trans_noh.py --run-name clean_trans_noh --render-freq 0 > /tmp/clean_trans_noh.log 2>&1
echo "===== $(date '+%F %T') PILOT COMPLETE ====="
