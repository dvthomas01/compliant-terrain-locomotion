#!/bin/bash
# Workshop-hardening batch (reviewer feedback): go to 8 seeds for A/B/B' and add a
# trot-prior ablation. ~17 runs x ~30 min = ~8.5 h, sequential, overnight.
#
# Recipe is the canonical BASELINE (entropy 0.005 + competence-gated curriculum
# experiments were reverted) so new seeds pool with existing s0-s2.
#   A   = train_policy_v24.py      (rigid, 49D, PMTG)        -> policy_a_v24_s{3..7}
#   B   = train_policy_b.py        (compliance, 89D, PMTG)   -> policy_b_s{3..7}
#   B'  = train_policy_b_noh.py    (compliance, 49D, PMTG)   -> policy_b_noh_s{3..7}
#   A_notg = train_policy_a_notg.py (rigid, 49D, NO PMTG)    -> policy_a_notg[, _s1]
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source .venv/bin/activate
export PYTHONPATH="$PWD"

run() {  # $1=script  $2=run_name
  echo "===== $(date '+%F %T') START $2 ====="
  python "training/$1" --run-name "$2" --render-freq 0 > "/tmp/$2.log" 2>&1
  echo "===== $(date '+%F %T') DONE $2 (exit $?) ====="
}

# 1) trot-prior ablation first (quick to interpret)
run train_policy_a_notg.py policy_a_notg
run train_policy_a_notg.py policy_a_notg_s1

# 2) seeds 3-7 for the three main policies
for S in 3 4 5 6 7; do run train_policy_v24.py     "policy_a_v24_s${S}";  done
for S in 3 4 5 6 7; do run train_policy_b.py        "policy_b_s${S}";      done
for S in 3 4 5 6 7; do run train_policy_b_noh.py    "policy_b_noh_s${S}";  done

echo "===== EIGHTSEED BATCH COMPLETE $(date '+%F %T') ====="
