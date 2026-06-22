# Findings — Compliance-Randomized Locomotion (honest reframe)

> Updated 2026-06-22. Supersedes the single-seed fine-grained claims in earlier
> figures (`analysis/ablation_plot*.png`). Multi-seeding revealed those were not
> representative; this document states only what survives seeds.

## Headline

Training a quadruped on randomized terrain compliance **does** buy robustness to
unseen compliant ground that a rigid-trained policy categorically lacks — **but
acquiring that robustness is high-variance across random seeds.** The variance,
not a clean efficiency/observation ablation, is the honest story.

## What survives seeds (robust claims)

1. **Policy A (rigid-trained) fails categorically on compliant terrain.** Fall
   rate jumps to 1.0 at T5 and stays there through T9 — deterministic collapse,
   no compliant traverse at any held-out softness. (Eval: A single seed; failure
   is deterministic, so a single seed is defensible but should be noted.)
2. **Robustness to compliant terrain is achievable.** Compliance-trained policies
   (B, B′) traverse T0–T6 with low fall rate on average — well past where A dies.
   The capability is real; the question is reliability.

## What does NOT survive seeds (retracted from the single-seed draft)

- ❌ "Compliance training **reliably** → robustness." Only ~3/4 of seeds reach the
  robust basin; ~1/4 stall.
- ❌ "Foot-history (B vs B′) helps **survival**." Both families straddle the same
  bimodal distribution; the difference is within seed noise.
- ❌ "~25% cost-of-transport efficiency gain." This was seed-0 luck compounded by a
  freezing artifact (now fixed: COT is computed over non-fallen episodes only and
  survival is progress-gated, so a frozen-but-upright robot no longer scores as
  surviving).

## The variance (core evidence)

Final curriculum `terrain_level` (3 = softest tier the policy sustains), measured
over the last 2M of 15M steps. Source: `evaluation/convergence_variance.csv`
(extracted from TensorBoard `locomotion/terrain_level`).

| Recipe | seed 0 | seed 1 | seed 2 | stalled (<1.4) |
|--------|--------|--------|--------|----------------|
| B baseline            | 2.79 | 2.02 | 2.16 | 0/3 |
| B′ no-history         | 2.52 | 0.77 | 2.78 | 1/3 |
| competence-gated curriculum | 2.02 | 2.83 | 1.34 | 1/3 |
| entropy 0.005         | 0.57 | 2.58 | 2.52 | 1/3 |

- **3 of 12 seeds fully stall** (never leave near-rigid terrain); even the
  "robust" seeds vary widely (2.0–2.8) in how soft they can handle.
- The pattern reproduces the held-out eval: training `terrain_level` predicts
  where a seed breaks (e.g. B seed 0, train 2.79, survives T5–T6; B seed 1,
  train 2.02, already falls at T4).
- **Two reliability interventions did not fix it.** A competence-gated curriculum
  (advance only on near-complete traverse, retreat only on a real fall) and a
  small entropy bonus (0.005) each still produced a stalled seed. The variance
  looks intrinsic to compliant-gait acquisition in this setup, not a tunable knob.

Figure: `analysis/variance_plot.png` (left: per-seed held-out fall rate; right:
the bimodal convergence across the four recipes).

## Recommended paper framing

Rigid training is a **categorical** failure mode on compliant terrain;
compliance randomization removes that failure mode but introduces a **reliability**
problem — gait acquisition is bimodal across seeds and resists two standard
stabilizers. Report the distribution, not a single seed. Honest, reproducible,
and a concrete open problem rather than an overstated win.

## Reproduce

```bash
source .venv/bin/activate && export PYTHONPATH="$PWD"
python analysis/plot_variance.py          # rebuilds analysis/variance_plot.png
#   reads evaluation/ablation_results_multiseed.csv  (held-out eval, N=100/terrain)
#   reads evaluation/convergence_variance.csv        (training terrain_level)
```
