# Findings — summary

> Updated 2026-06-25. **`results.md` is the authoritative, review-ready document**
> (full setup, methodology, tables, limitations). This file is a one-screen summary.

## Headline

A rigid-trained controller (A) is the **fastest and most energy-efficient** on
rigid ground but **fails categorically** once terrain compliance passes the
training edge (T5–T6, ≈100% falls, all 3 seeds). Training on randomized contact
compliance (B/B′) **extends the survivable softness range** but the benefit is
**highly seed-dependent**, and **no policy generalizes beyond the training range**
(all fail on T8–T9). The honest contribution is a **robustness–performance
tradeoff + a reliability characterization**, not a clean "compliance wins."

## Survives seeds (defensible)

- A: walks on rigid (0 falls, 0.22 m/s, COT 1.75) → fails categorically at T5–T6.
- No policy generalizes past the training range (all 100% falls on T8–T9).
- Compliance training extends survivable softness (converged B/B′ survive T5–T6
  where A dies) — but seed-variable.
- Reliability is task-specific: rigid trains reliably (A 3/3); compliant-gait
  acquisition is bimodal (3/12 B-family seeds stall; two fixes didn't help).

## Retracted / negative (does NOT survive seeds)

- ❌ "Foot history required for survival" — B′ (no history) ≈ or > B under
  multi-seeding.
- ❌ "~25% COT efficiency win" — single-seed luck + a freezing-as-survival bug
  (now fixed).
- ❌ "Compliance reliably → robustness" — seed-std ≈ 0.45 at transition terrains;
  the mean effect depends on which seeds converged.

## Corrected error (for transparency)

An earlier pass multi-seeded Policy A with the wrong script (`train_policy_a.py`,
a stale recipe), producing a spurious "A is 1/8 reliable" result. The canonical A
recipe is **`train_policy_v24.py`** (shares B's recipe exactly). With it, all 3 A
seeds train normally. Retracted; see `results.md` §2.

## Key artifacts

- `results.md` — full writeup. · `evaluation/ablation_results_multiseed.csv` —
  held-out eval (3 seeds × 3 policies × T0–T9). ·
  `evaluation/convergence_variance.csv` — training reliability. ·
  `analysis/variance_plot.png` — per-seed fall rate + bimodal convergence.

## Main caveat

**n = 3 seeds** against large seed variance: B-vs-B′ differences are suggestive,
not statistically established. ≥8–10 seeds needed for distributional claims.
