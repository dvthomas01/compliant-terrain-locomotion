#!/bin/bash
# Add 2 more seeds of Policy A for 3-seed parity with B/B'.
# IMPORTANT: policy_a_v24 was produced by train_policy_v24.py (NOT train_policy_a.py,
# which is an older recipe). v24 shares B's recipe exactly (log_std_init=-1.5,
# command curriculum 0->0.2, ent_coef=0, k_max=0.3, use_tg) so only terrain + the
# by-design 49D/89D obs differ. SB3 unset PPO seed -> independent seed per run.
# v24 saves checkpoints/<run>/policy_v24_final.zip natively (matches the eval).
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source .venv/bin/activate
export PYTHONPATH="$PWD"

for S in 1 2; do
  run="policy_a_v24_s${S}"
  echo "===== $(date '+%F %T') START ${run} ====="
  python training/train_policy_v24.py --run-name "${run}" --render-freq 0 > "/tmp/${run}.log" 2>&1
  echo "===== $(date '+%F %T') DONE ${run} (exit $?) ====="
done
echo "===== POLICY A MULTISEED COMPLETE $(date '+%F %T') ====="
