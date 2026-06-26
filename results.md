# Results — Locomotion on Compliant Terrain

> Review-ready writeup. Updated 2026-06-25 (8-seed revision). Authoritative
> results document; `FINDINGS.md` is a short summary that defers to this file.
> All numbers come from committed CSVs and reproduce with §8.

---

## 1. Research question & contribution claim

Does training a quadruped (Unitree Go1, MuJoCo) on **randomized terrain
compliance** generalize to **unseen compliant terrain** better than rigid-only
training — and at what cost to performance, efficiency, and training reliability?

**Scoped contribution (defensible form):** We show that **contact-softness
domain randomization extends the survivable compliance range of a trot-prior
(PMTG) residual policy** relative to a rigid-trained baseline, and we characterize
the tradeoff against rigid-ground performance, energy efficiency, and training
reliability. We do **not** claim general "compliant-terrain generalization": the
benefit is bounded (no extrapolation past the training range) and seed-variable.

**Operational definition of "compliance" (read first).** "Compliant terrain" here
means **contact-solver softness** (MuJoCo `solref`/`solimp`): the ground reaction
force becomes softer/slower, **without geometric deformation, viscoelastic
patches, or topology change**. A robotics-hardware reader should not picture foam
or mud — picture a softer, slower contact response on flat ground.

## 2. The policies & the shared-recipe guarantee

| ID | Name | Terrain | Obs | Trot prior | Training script |
|----|------|---------|-----|-----------|-----------------|
| **A** | rigid baseline | rigid only | 49D (no history) | PMTG on | `train_policy_v24.py` |
| **B** | compliance + history | Level-1 compliance curriculum | 89D (+foot history) | PMTG on | `train_policy_b.py` |
| **B′** | compliance, no history | Level-1 compliance curriculum | 49D (no history) | PMTG on | `train_policy_b_noh.py` |
| **A‑noTG** | prior ablation | rigid only | 49D | **PMTG off** | `train_policy_a_notg.py` |

A, B, B′ share an **identical recipe**; only training terrain and (by design) the
observation differ. Verified at code level: `log_std_init=−1.5`, `ent_coef=0`,
penalty `k_max=0.3`, command curriculum 0→0.2 (track_frac 0.3), PMTG `use_tg=on`,
net `[256,128]`/ELU, shared `base_env` reward. (A‑noTG is identical to A except
`use_tg=off`.)

> **Provenance / corrected error (full disclosure).** `policy_a_v24` is produced
> by `train_policy_v24.py`, **not** the older `train_policy_a.py` (a stale,
> more-aggressive recipe). An early multi-seeding pass mistakenly used
> `train_policy_a.py`, which collapsed 7/8 seeds → a spurious "A is 1/8 reliable"
> result, now **retracted**. With the correct script, **all 8 A seeds train
> normally** (§5.3). **Eval provenance:** every number in this document was
> produced with the correct `v24` A policy; the wrong-script checkpoints were
> moved aside before evaluation, and B/B′ use entirely separate scripts and were
> never affected. `train_policy_a.py` now carries a deprecation header.

## 3. Setup (abbreviated)

Go1 (12-DOF), MuJoCo 3.9.0, 50 Hz policy / 500 Hz sim, position actuators
`kp=100,kd=0`. Control = residual joint targets (±0.5 rad) on a nominal stance +
an open-loop **PMTG trot** (`ctrl = NOMINAL + TG(φ) + 0.25·a`); TG off at command
0. Obs: A/B′ 49D (state + 2D gait-phase clock), B 89D (+ foot pos history 24D,
foot vel 12D, contact 4D). Reward: 9 terms (velocity tracking + penalties +
feet-air-time), penalties ramped `k:0.03→0.3`. PPO: 128 envs × 24 steps, 5 epochs,
γ=0.99, λ=0.95, clip 0.2, `ent_coef=0`, adaptive LR (KL 0.01), **15M steps/seed**,
custom timeout-bootstrapping GAE. Compliance curriculum (B/B′): 7-level Level-1
contact variation, advance on traverse / retreat on fall.

## 4. Evaluation protocol

Held-out suite **T0–T9** (`terrain_suite.py`): T0 rigid; T1–T6 inside the training
compliance range; **T7–T9 softer than trained (extrapolation)**. Floor `solref0`
0.01→0.30 s, `solimp0` 0.97→0.72; friction fixed (compliance is the only variable).
Fair shared-physics env (`EvalCompliantEnv`): identical deterministic physics per
terrain, emits 49D (A/B′) or 89D (B). **N=100 episodes/terrain/seed**, deterministic
actions, 1000-step cap. Metrics: `fall_rate` (ended before cap); `success_rate`
(not fallen **and** >2 m progress — frozen ≠ survived); `cost_of_transport` (median
over **non-fallen** episodes of Σ|τ·q̇|·dt/(m·g·d)); velocity; contact/height var.
**Seeds: 8 per main policy** (A, B, B′); 2 for the A‑noTG ablation.

## 5. Results

### 5.1 Held-out fall rate (8 seeds; mean ± seed-std, and # seeds surviving fall<0.5)

| Terrain | A (rigid) | B (+history) | B′ (no history) |
|---------|-----------|--------------|------------------|
| T0 rigid | 0.00±0.00 (8/8) | 0.00±0.00 (8/8) | 0.00±0.00 (8/8) |
| T1 firm | 0.07±0.09 (8/8) | 0.00±0.00 (8/8) | 0.01±0.01 (8/8) |
| T2 | 0.03 (8/8) | 0.00 (8/8) | 0.02 (8/8) |
| T3 | 0.14±0.33 (7/8) | 0.00 (8/8) | 0.03 (8/8) |
| **T4** | 0.25±0.38 (6/8) | 0.13±0.33 (7/8) | **0.01±0.03 (8/8)** |
| **T5** | 0.59±0.44 (3/8) | 0.29±0.42 (6/8) | **0.04±0.09 (8/8)** |
| **T6 (train edge)** | 0.80±0.32 (1/8) | 0.33±0.44 (5/8) | **0.16±0.32 (7/8)** |
| T7 (extrap) | 1.00 (0/8) | 0.88±0.32 (1/8) | 0.61±0.41 (3/8) |
| T8 (extrap) | 1.00 (0/8) | 1.00 (0/8) | 0.98±0.06 (0/8) |
| T9 (extrap soft) | 1.00 (0/8) | 1.00 (0/8) | 1.00 (0/8) |

**Monotonic ordering B′ > B > A holds across the whole transition zone (T4–T7).**

### 5.2 Performance & efficiency on rigid (T0, 8 seeds)

| Policy | velocity (m/s) | COT (median, non-fallen) | success_rate |
|--------|----------------|--------------------------|--------------|
| A | **0.211** | **1.78** | 1.00 (8/8) |
| B | 0.155 | 2.34 | 0.88 |
| B′ | 0.145 | 2.54 | 0.76 |

A is fastest + most efficient on rigid. **B′ success_rate < 1 on rigid is not
falling** (fall_rate=0): per-seed success is `[0.28, .85, 1, 1, 1, .95, 0, 1]` —
two B′ seeds adopt a gait so cautious they don't cover 2 m in 20 s on rigid
(tripped only by the strict progress gate). This is consistent with §5.4: the
conservative B′ phenotype trades rigid speed for compliant robustness.

### 5.3 Training reliability (8 seeds)

- **A: 8/8 seeds learn** (final ep_len ≈ 990–1000). Rigid locomotion is reliable.
- **B / B′:** final curriculum `terrain_level` (3 = softest tier); both have
  **1/8 fully-stalled** seeds and similar means → comparable reliability:
  - B  = [2.79, 2.02, 2.16, 2.60, 2.48, 2.76, 2.84, **1.23**], mean 2.36
  - B′ = [2.52, **0.77**, 2.78, 2.88, 2.73, 2.45, 2.40, 2.91], mean 2.43
  - Two earlier reliability fixes (competence-gated curriculum; entropy 0.005;
    3 seeds each) did **not** remove the stalled-seed mode.
- B trains slightly less stably than B′ (lower mean training ep_len), mild support
  for "the extra 40D makes B harder to train," not "B converges higher/lower."

### 5.4 The T4 asymmetry & the conservative-gait mechanism (held up at 8 seeds)

At T4, B′ is **uniformly robust and low-variance** while A and B are bimodal:

| | T4 fall (mean±std) | per-seed | T4 velocity | T4 contact-var |
|--|--------------------|----------|-------------|----------------|
| A | 0.25±0.38 | `[0,0,.01,.01,.01,.15,.89,.92]` | 0.234 | 0.718 |
| B | 0.13±0.33 | `[0,0,0,0,0,0,.05,1.0]` | 0.144 | 0.436 |
| B′ | **0.01±0.03** | `[0,0,0,0,0,0,.04,.08]` | **0.133** | **0.397** |

Velocity and foot-contact-variance both order **A > B > B′**, exactly inverse to
robustness. The most conservative, steadiest gait (B′) is the most T4-robust;
A's fast/aggressive gait is bimodal. Interpretation (one speculative sentence,
earned by data): the richer 89D observation appears to encourage **more active
adaptation** (faster, higher-contact-variance gait) that is less robust at the
onset of compliance than B′'s simpler, more conservative gait.

**The conservative gait is not free.** B′'s efficiency cost rises sharply with
softness: median COT goes from **2.5 on rigid (T0) to ~7.8 at T5** (computed over
all 8 surviving seeds), i.e. the robustness B′ buys on soft terrain is paid for in
energy. (T6 COT is high-variance — one seed at 66 — so we quote T5.)

### 5.5 Trot-prior (PMTG) ablation — quantifying the confound

A‑noTG (rigid, 49D, **no trot prior**, 2 seeds): on T0–T3 it **stands but does not
walk** — velocity ≈ 0.004 m/s, **0% success** (never travels 2 m), fall_rate ≈ 0.
With the prior, A walks at 0.21 m/s. **The PMTG prior supplies propulsion; the
learned policy contributes residual modulation, not gait discovery.** **Implication
for the compliance findings:** the robustness gains in §5.1 are therefore gains in
the *residual policy's ability to modulate the prior on soft contact*, not gains in
gait discovery — every result is "domain randomization on a trot-prior residual
policy," stated as scope, not hidden.

### 5.6 What predicts a seed's robustness? (Not curriculum progress — gait speed)

The intuitive hypothesis is "seeds that stalled low on the compliance curriculum
are the ones that fail at the transition." We tested it by joining per-seed
training `terrain_level` to per-seed eval fall rate (seed→checkpoint mapping
verified against the loaded weights, e.g. eval seed 7 ↔ `checkpoints/policy_b_s7`).

**The curriculum-progress hypothesis is refuted.** B's lowest-`terrain_level` seed
(s7 = 1.23) survives T4–T6 at fall = 0.00 (N=100 each) and only fails at T7; B's
actual T4 failure is a *mid*-curriculum seed (s1 = 2.02). B′'s stalled seed
(s1 = 0.77) likewise survives T4–T6. Pooled over B+B′ (16 seeds),
corr(`terrain_level`, # transition terrains survived) = **+0.25** (weak).

**A better predictor is gait speed.** corr(rigid velocity, # survived) = **−0.40**:
the *slower* seeds are the more robust (median split: slow seeds survive 3.1
transition terrains, fast seeds 2.5). Within B, the brittle seeds (s1,s2,s4) are
the fastest (0.18–0.20 m/s) and the robust seeds are slower (s7=0.095 is the
slowest and is robust). This is the **same conservative-gait axis** as the B′ > B
ordering (§5.1) and the T4 asymmetry (§5.4), now visible seed-by-seed.

**Why curriculum progress misleads:** the curriculum advances on *distance
traveled* (>50% of target), so a fast gait climbs to high `terrain_level` during
training yet can be brittle on held-out soft terrain, while a slow/conservative
gait stays low on the curriculum but generalizes. `terrain_level` conflates
"soft-terrain competence" with "speed"; gait conservatism is the axis that tracks
robustness.

**Caveat:** at 16 seeds, corr −0.40 is *moderate*, not decisive — gait conservatism
is a tendency, not a law, and velocity is itself a correlate (we have not run a
controlled speed-matched intervention).

## 6. Findings (graded)

**Defensible (8-seed distributional):**

1. **Robustness–performance tradeoff, monotonic B′ > B > A.** Rigid A is fastest
   (0.21 m/s) and most efficient (COT 1.78) but brittle (survives T5 in 3/8 seeds,
   T6 in 1/8). Compliance training extends the survivable range (B′ survives T5 in
   8/8, T6 in 7/8).
2. **Limited extrapolation, not none.** Just past the training edge (T7,
   solref0=0.15 vs the T6 edge 0.10), B′ retains partial robustness — **3/8 seeds
   survive** vs B 1/8, A 0/8 — so compliance training buys a *small* margin beyond
   the edge. But the margin is small and fragile: **all** policies collapse to
   ~100% falls by T8–T9. The headline "no free generalization" holds; the precise
   claim is "a narrow extrapolation margin for B′ at T7, full collapse by T8."
3. **Foot history does NOT help, and is associated with *reduced* robustness**
   (B′ ≥ B at every transition terrain). This **reverses** the original
   single-seed hypothesis (see negatives). Correlational evidence in §5.4 (gait
   conservatism); we do not claim a causal mechanism.
4. **A trains reliably; compliant-gait acquisition has a persistent ~1/8 stalled
   mode** unaffected by two interventions.
5. **The trot prior provides propulsion** (§5.5): without it, the policy stands.

**Negative / retracted results (now explicit):**

6. **"Foot history is required for survival on compliant terrain" — refuted.** We
   hypothesized history helps; under 8-seed evaluation, the no-history policy (B′)
   is *more* robust across T4–T7. The originally observed advantage was a
   single-seed artifact.
7. **"~25% COT efficiency win from compliance" — retracted** (single-seed luck +
   a since-fixed freezing-counts-as-survival bug).

## 7. Limitations & threats to validity

- **n = 8 seeds** (main policies) makes the monotonic B′ > B > A ordering and the
  T4 asymmetry credible, but transition-terrain seed-std is still large
  (0.3–0.45): individual cells are distributions, not point estimates. The gait
  mechanism (§5.4) is **correlational** (velocity/contact-var vs robustness across
  seeds), not a controlled intervention.
- **A‑noTG ablation is only 2 seeds** — sufficient to show "stands, not walks"
  (a near-deterministic outcome) but not a distributional claim.
- **Cost-of-transport is reported on rigid (T0) only** as the clean efficiency
  comparison. Compliant-terrain COT exists in the CSV but is computed over few
  non-fallen episodes (near-zero for A at T5+), so it is high-variance and **not**
  used for comparison — a deliberate scope decision, not an omission.
- **Solver-level compliance only (Level 1).** No geometric/viscoelastic terrain
  (Level 2 deferred: static box patches did not collide with the foot in MuJoCo —
  diagnosed, not root-caused). See §1 operational definition.
- **Single evaluation axis.** T0–T9 vary only floor softness; no slopes, friction,
  payloads, or pushes.
- **PMTG + command-curriculum confound** (§5.5): results describe a trot-prior
  residual policy, not pure-RL gait discovery (which did not emerge here).
- **Sim-only**; no hardware / sim-to-real claim.

## 8. Reproduce

```bash
source .venv/bin/activate && export PYTHONPATH="$PWD"
# Train (per seed; SB3 unset PPO seed => independent seed):
python training/train_policy_v24.py    --run-name policy_a_v24_sN  --render-freq 0   # A
python training/train_policy_b.py      --run-name policy_b_sN      --render-freq 0   # B
python training/train_policy_b_noh.py  --run-name policy_b_noh_sN  --render-freq 0   # B'
python training/train_policy_a_notg.py --run-name policy_a_notg[_sN] --render-freq 0 # A-noTG
# (or: bash training/run_eightseed_batch.sh  for the full 8-seed + ablation batch)

python evaluation/evaluate.py --n 100 --out evaluation/ablation_results_multiseed.csv
python analysis/plot_variance.py     # -> analysis/variance_plot.png
```

Artifacts: `evaluation/ablation_results_multiseed.csv` (8-seed held-out eval),
`evaluation/convergence_variance.csv` (training reliability),
`analysis/variance_plot.png` (fall-rate-vs-softness bands + convergence).

## 9. One-paragraph summary for the reviewer

On a Unitree Go1 in MuJoCo, a rigid-trained trot-prior residual policy is the
fastest and most energy-efficient on rigid ground but fails categorically once
contact compliance exceeds the training edge; training on randomized contact
softness extends the survivable range, with a clean monotonic ordering across 8
seeds — the **no-history** policy (B′) is the most robust, the history policy (B)
intermediate, and rigid (A) the most brittle. Foot history thus provides **no**
robustness benefit and is associated with a faster, less-conservative, less-robust
gait; no policy extrapolates beyond its training range; and a trot-prior ablation
shows the prior supplies propulsion while the policy contributes residual
correction. The honest contribution is a **robustness–performance tradeoff plus a
surprising negative result on observation history**, supported by 8-seed
distributions, a fair shared-physics evaluation, progress-gated metrics, and a
verified shared training recipe. Main residual caveats: large transition-zone seed
variance, a correlational (not interventional) gait mechanism, and solver-level
(not geometric) compliance.
```
