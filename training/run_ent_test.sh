#!/bin/bash
set -u
cd "/Users/damithomas/Documents/Locomotion on Compliant Terrain"
source .venv/bin/activate; export PYTHONPATH="$PWD"
for S in 0 1 2; do
  echo "===== $(date '+%F %T') START ent_s${S} ====="
  python training/train_policy_b.py --run-name "policy_b_ent_s${S}" --render-freq 0 > "/tmp/ent_s${S}.log" 2>&1
  echo "===== $(date '+%F %T') DONE ent_s${S} ====="
done
echo "===== ENT TEST COMPLETE $(date '+%F %T') ====="
