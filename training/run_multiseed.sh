#!/bin/bash
# Multi-seed obs-ablation: 2 new seeds (1,2) x 4 policies = 8 sequential 15M runs.
# Seed-0 already exists (policy_b, policy_b_noh, policy_b_trans, policy_b_trans_noh).
# SB3's unset PPO seed makes each run an independent seed. Rendering disabled.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source .venv/bin/activate
export PYTHONPATH="$PWD"

SPECS=(
  "train_policy_b:policy_b"
  "train_policy_b_noh:policy_b_noh"
  "train_policy_b_trans:policy_b_trans"
  "train_policy_b_trans_noh:policy_b_trans_noh"
)

for S in 1 2; do
  for spec in "${SPECS[@]}"; do
    script="${spec%%:*}"; base="${spec##*:}"; run="${base}_s${S}"
    echo "===== $(date '+%F %T') START ${run} ====="
    python "training/${script}.py" --run-name "${run}" --render-freq 0 > "/tmp/ms_${run}.log" 2>&1
    echo "===== $(date '+%F %T') DONE ${run} (exit $?) ====="
  done
done
echo "===== ALL MULTISEED RUNS COMPLETE $(date '+%F %T') ====="
