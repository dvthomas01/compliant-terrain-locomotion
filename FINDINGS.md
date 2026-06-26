# Findings — summary

> Updated 2026-06-25 (8-seed revision). **`results.md` is the authoritative,
> review-ready document.** This file is a one-screen summary.

## Headline

Across **8 seeds**, a rigid-trained trot-prior residual policy (A) is the **fastest
and most efficient** on rigid ground but **fails categorically** once contact
compliance passes the training edge. Compliance training extends the survivable
softness range with a clean **monotonic ordering: B′ (no history) > B (+history) >
A**. **Foot history provides no robustness benefit — and is associated with a
faster, less-conservative, less-robust gait.** No policy extrapolates beyond the
training range. A trot-prior ablation shows the **prior supplies propulsion**; the
policy contributes residual correction.

## Defensible (8-seed)

- **Robustness–performance tradeoff (monotonic).** Seeds surviving (fall<0.5):
  T5 — A 3/8, B 6/8, B′ **8/8**; T6 — A 1/8, B 5/8, B′ **7/8**.
- A fastest/most efficient on rigid (0.21 m/s, COT 1.78); B′ slowest (0.145).
- **T4 asymmetry is real:** B′ uniformly robust (0.01±0.03, 8/8) while A is bimodal
  (0.25±0.38). Mechanism: at T4, velocity & contact-variance order A>B>B′ — the
  conservative gait (B′) is the robust one.
- Limited extrapolation: B′ keeps a narrow margin at T7 (3/8 vs B 1/8, A 0/8); all collapse by T8–T9.
- **No trot prior ⇒ the policy stands, not walks** (vel ≈ 0.004 m/s, 0% progress) →
  prior supplies propulsion; compliance gains are to the *residual*, not gait discovery.
- Reliability: A 8/8 trains; B and B′ each have 1/8 stalled seed (two fixes didn't help).
- **Convergence ≠ robustness:** the stalled-curriculum seeds are NOT the transition
  failures (corr 0.15–0.36); training progress and held-out robustness decouple.

## Negative / retracted

- ❌ **"Foot history required for survival"** — refuted; B′ ≥ B at every transition.
- ❌ **"~25% COT win"** — single-seed luck + a since-fixed freezing-as-survival bug.

## Corrected error (transparency)

An earlier pass multi-seeded A with the wrong script (`train_policy_a.py`, stale
recipe) → spurious "A 1/8 reliable." Canonical A is `train_policy_v24.py`; all 8 A
seeds train normally. All eval used the correct A. See `results.md` §2.

## Caveats

8 seeds make the ordering credible but transition-zone seed-std is still 0.3–0.45;
the gait mechanism is correlational; A-noTG is only 2 seeds; compliance is
solver-level (not geometric). Artifacts: `results.md`,
`evaluation/ablation_results_multiseed.csv`, `evaluation/convergence_variance.csv`,
`analysis/variance_plot.png`.
