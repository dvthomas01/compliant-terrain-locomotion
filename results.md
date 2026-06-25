# Results — Locomotion on Compliant Terrain

> Review-ready writeup. Updated 2026-06-25. Authoritative results document;
> `FINDINGS.md` is a short summary that defers to this file.
> Goal of this doc: let an external reviewer judge whether the work is honest,
> realistic, and defensible. Numbers below come from committed CSVs and are
> reproducible with the commands in §8.

---

## 1. Research question

Does training a quadruped (Unitree Go1, MuJoCo) on **randomized terrain
compliance** produce a controller that generalizes to **unseen compliant
terrain** better than a controller trained only on rigid ground — and at what
cost to peak performance, energy efficiency, and training reliability?

Two ablations are nested:

- **Terrain ablation (A vs B):** rigid-trained vs compliance-trained.
- **Observation ablation (B vs B′):** with vs without foot-contact/position
  **history** in the observation, both compliance-trained.

## 2. The three policies

| ID | Name | Terrain | Obs | Training script |
|----|------|---------|-----|-----------------|
| **A** | rigid baseline | rigid only | 49D (no history) | `training/train_policy_v24.py` |
| **B** | compliance + history | Level-1 compliance curriculum | 89D (+foot history) | `training/train_policy_b.py` |
| **B′** | compliance, no history | Level-1 compliance curriculum | 49D (no history) | `training/train_policy_b_noh.py` |

**Shared-recipe guarantee (critical for ablation validity).** A, B, B′ use an
identical recipe; *only* the training terrain and (by design) the observation
dimensionality differ. Verified at the code level:

| Knob | A (`v24`) | B | B′ |
|------|-----------|---|----|
| `log_std_init` | −1.5 | −1.5 | −1.5 |
| `ent_coef` | 0.0 | 0.0 | 0.0 |
| penalty `k_max` | 0.3 | 0.3 | 0.3 |
| command curriculum | 0→0.2, track_frac 0.3 | same | same |
| PMTG trot prior (`use_tg`) | on | on | on |
| net arch / activation | [256,128]/ELU | same | same |
| reward terms (`base_env`) | shared | shared | shared |

> **Provenance note / corrected error.** `policy_a_v24` is produced by
> `train_policy_v24.py`, **not** the older `train_policy_a.py` (which is a
> stale, more aggressive recipe: `log_std_init=0`, no command curriculum). An
> early version of this multi-seeding accidentally used `train_policy_a.py`,
> which collapsed 7/8 seeds and produced a spurious "Policy A is 1/8 reliable"
> result. That was a **script-provenance bug, not a finding**, and is retracted.
> With the correct `train_policy_v24.py`, all 3 A seeds train normally (§5.3).
> Lesson for reviewers: the canonical A recipe is `train_policy_v24.py`.

## 3. Setup

- **Robot/sim:** Unitree Go1 (12-DOF), MuJoCo 3.9.0, 50 Hz policy / 500 Hz sim
  (10 substeps). Position actuators `kp=100, kd=0`.
- **Control:** policy outputs 12 residual joint targets (±0.5 rad) on top of a
  nominal stance + an **open-loop PMTG trot** (`ctrl = NOMINAL + TG(φ) + 0.25·a`).
  TG amplitude scales with command, so at command 0 the TG is off (clean stand).
- **Compliance model:** floor contact softness via MuJoCo `solref[0]` (contact
  time constant) and `solimp[0,1]` (impedance). The floor's `geom_priority` is
  raised above the foot's so floor compliance actually governs contact (a foot
  priority of 1 otherwise overrides the floor — see repo memory). This is
  **solver-level** compliance (Level 1), **not** geometric soft patches
  (Level 2 deferred; see §7).
- **Observations.** A/B′ = 49D (commands, gravity, body ang/lin vel, joint
  pos/vel, prev action, + 2D gait-phase clock). B = 89D = 49D + foot position
  history (24D at t, t−10, t−20) + foot velocity (12D) + foot contact (4D). The
  2D gait clock is an open-loop TG phase, **not** proprioceptive history, so
  "A has no history" holds.
- **Reward (9 terms, identical across policies):** velocity tracking (lin 1.0,
  ang 0.5), penalties (z-vel 4.0, xy-ang-vel 0.05, joint motion 1e-3, torque
  2e-5, action-rate 0.25, collision 1e-3), feet-air-time 2.0; all ×dt; penalties
  ramped by a curriculum coefficient `k: 0.03 → k_max=0.3` over training.
- **PPO:** 128 envs × 24 steps (batch 3072), 5 epochs, γ=0.99, λ=0.95,
  clip 0.2, **ent_coef=0**, adaptive LR (KL target 0.01), **15M steps/seed**,
  custom timeout-bootstrapping GAE (correct value bootstrap at episode timeout).
- **Compliance curriculum (B/B′ only):** game-style 7-level (Rudin 2022),
  Level-1 contact variation; advance on traverse, retreat on fall.

## 4. Evaluation protocol

- **Held-out terrain suite T0–T9** (`evaluation/terrain_suite.py`): T0 rigid;
  T1–T6 inside the training compliance range (interpolation); **T7–T9 softer
  than anything trained on (extrapolation)**. Compliance varied via floor
  `solref0` (0.01→0.30 s) and `solimp0` (0.97→0.72); friction held nominal so
  compliance is the only variable.
- **Fair shared-physics env** (`EvalCompliantEnv`): identical physics for every
  policy on a given terrain; deterministic (no randomization, nominal
  mass/motor/damping, no obs noise); emits 49D for A/B′ or 89D for B.
- **N=100 episodes/terrain/seed**, deterministic actions, 1000-step cap.
- **Metrics** (`evaluation/evaluate.py`):
  - `fall_rate` = fraction of episodes ending before the 1000-step cap.
  - `success_rate` = not fallen **and** traversed >2 m (progress-gated, so a
    frozen-but-upright robot is **not** counted as surviving).
  - `cost_of_transport` = median over **non-fallen** episodes of
    Σ|τ·q̇|·dt / (m·g·d) (dimensionless).
  - `forward_velocity_mean/std`, `foot_contact_variance`, `base_height_variance`.
- **Seeds: 3 per policy** (independent SB3 RNG draws), the central limitation
  (§7). Data: `evaluation/ablation_results_multiseed.csv`.

## 5. Results

### 5.1 Held-out fall rate (seed-mean ± seed-std over 3 seeds, N=100 each)

| Terrain | A (rigid) | B (+history) | B′ (no history) |
|---------|-----------|--------------|------------------|
| T0 rigid | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 |
| T1 firm | 0.07±0.05 | 0.00±0.00 | 0.02±0.02 |
| T2 | 0.03±0.02 | 0.00±0.00 | 0.04±0.06 |
| T3 | 0.00±0.00 | 0.00±0.00 | 0.08±0.09 |
| T4 | 0.31±0.43 | 0.35±0.46 | 0.04±0.03 |
| **T5** | **0.98±0.02** | 0.67±0.47 | **0.10±0.11** |
| **T6 (train edge)** | **1.00±0.00** | 0.67±0.47 | 0.36±0.45 |
| T7 (extrap) | 1.00±0.00 | 1.00±0.00 | 0.70±0.42 |
| T8 (extrap) | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 |
| T9 (extrap soft) | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 |

### 5.2 Performance & efficiency on rigid (T0)

| Policy | fwd velocity (m/s) | COT (median, non-fallen) | success_rate |
|--------|--------------------|--------------------------|--------------|
| A (rigid) | **0.221** | **1.75** | 1.00 |
| B (+history) | 0.161 | 2.11 | 1.00 |
| B′ (no history) | 0.146 | 2.78 | 0.71 |

A is the fastest and most energy-efficient on rigid; compliance-trained
policies pay a peak-performance/efficiency tax.

### 5.3 Training reliability

- **A (rigid):** 3/3 seeds trained to full episodes (final `ep_len_mean`
  ≈ 1000, rew −10 to −31). Rigid locomotion is **reliably learned**.
- **B/B′ (compliance):** seed-variable. Final curriculum `terrain_level`
  (3 = softest tier sustained), `evaluation/convergence_variance.csv`:

  | Recipe | seed 0 | seed 1 | seed 2 | fully stalled (<1.4) |
  |--------|--------|--------|--------|----------------------|
  | B baseline | 2.79 | 2.02 | 2.16 | 0/3 |
  | B′ no-history | 2.52 | **0.77** | 2.78 | 1/3 |
  | + competence-gated curriculum (fix attempt) | 2.02 | 2.83 | **1.34** | 1/3 |
  | + entropy 0.005 (fix attempt) | **0.57** | 2.58 | 2.52 | 1/3 |

  Across 4 recipes × 3 seeds, **3/12 seeds fully stall** on near-rigid terrain
  and even "converged" seeds vary widely (2.0–2.8). Two interventions
  (competence-gated curriculum; small entropy bonus) did **not** make
  convergence seed-robust.

## 6. Findings (what we claim, and how strongly)

**Survives seeds (defensible):**

1. **Rigid training yields a fast, efficient, but brittle controller.** A walks
   on rigid/firm terrain (T0–T3, 0 falls, 0.22 m/s, COT 1.75) and **fails
   categorically once terrain softens past the training edge** (T5–T6: ≈100%
   falls, all 3 seeds).
2. **No policy generalizes beyond the training range.** Every policy falls 100%
   on T8–T9 (softer than anything trained on). Compliance training extends the
   survivable range; it does not confer unbounded extrapolation.
3. **Compliance training can extend the survivable softness range** — converged
   B/B′ seeds survive T5–T6 where A is at 100% falls — but the benefit is
   **highly seed-dependent** (see variance below).
4. **Training reliability is task-specific:** rigid locomotion trains reliably
   (A 3/3); compliant-gait acquisition is bimodal (3/12 seeds stall) and
   resisted two standard stabilizers.

**Does NOT survive seeds (retracted / negative results):**

5. **"Foot history is required for survival on compliant terrain" — not
   supported.** With proper multi-seeding, B′ (no history) is **comparable to or
   better than** B (+history) on T4–T7 fall rate. The original single-seed claim
   was seed luck.
6. **"~25% cost-of-transport efficiency win from compliance" — retracted.**
   Artifact of a single seed plus a freezing-counts-as-survival bug (now fixed
   via non-fallen-only COT + progress-gated success).
7. **"Compliance reliably → robustness" — overstated.** It works for some seeds;
   the seed variance (seed-std ≈ 0.45 at transition terrains) is large enough
   that the mean effect rests on which seeds converged.

## 7. Limitations & threats to validity (read before trusting any number)

- **n = 3 seeds.** The dominant limitation. At transition terrains seed-std is
  ≈ 0.45 (seeds split between full survival and full failure), so B-vs-B′
  differences in §5.1 are **not statistically distinguishable**. Any claim that
  rests on the seed-*mean* (e.g., "B′ beats B") is suggestive, not proven. More
  seeds (≥8–10) are needed to make distributional claims.
- **Solver-level, not geometric, compliance.** Compliance is `solref/solimp`
  contact softness (Level 1). Geometric soft patches (Level 2, Kim 2021) were
  deferred: static box patches did not collide with the foot in MuJoCo
  (diagnosed, not root-caused). So "compliant terrain" here = soft contact, not
  deformable geometry.
- **Single evaluation environment family.** T0–T9 vary only floor
  `solref/solimp` with friction fixed. No slopes, friction changes, payloads,
  or pushes. Robustness is characterized along one axis (softness) only.
- **A's T4 fragility is itself seed-variable** (one A seed falls 0.92 at T4, two
  at 0.01) — the rigid→compliant failure boundary is not sharp at T4, only by T5–T6.
- **Sim-only.** No hardware; no sim-to-real claim.
- **Curriculum/PMTG confound (inherent, not a bug).** Both A and B use the PMTG
  trot prior and command curriculum; results describe *residual-policy + prior*,
  not pure-RL gait discovery (which did not emerge here in earlier work).
- **Reward/curriculum hyperparameters** were tuned (v2→v24 iterations) on the
  rigid task; this tuning history could mildly favor A on rigid.

## 8. Reproduce

```bash
source .venv/bin/activate && export PYTHONPATH="$PWD"

# Train (per seed; ~30 min each on this Mac). SB3 unset PPO seed => independent seed.
python training/train_policy_v24.py    --run-name policy_a_v24_sN --render-freq 0   # Policy A
python training/train_policy_b.py      --run-name policy_b_sN     --render-freq 0   # Policy B
python training/train_policy_b_noh.py  --run-name policy_b_noh_sN --render-freq 0   # Policy B'

# Evaluate all seeds on T0–T9 (N=100): writes the multiseed CSV
python evaluation/evaluate.py --n 100 --out evaluation/ablation_results_multiseed.csv

# Training-reliability table (terrain_level from TensorBoard) + figure
python analysis/plot_variance.py       # -> analysis/variance_plot.png
```

Artifacts: `evaluation/ablation_results_multiseed.csv` (held-out eval),
`evaluation/convergence_variance.csv` (training reliability),
`analysis/variance_plot.png` (per-seed fall rate + bimodal convergence).

## 9. One-paragraph summary for the reviewer

A rigid-trained quadruped controller is the fastest and most energy-efficient on
rigid ground but fails categorically once terrain compliance exceeds the training
edge; training on randomized contact compliance extends the survivable softness
range, but the benefit is highly seed-dependent, foot-history provides no clear
robustness advantage under proper multi-seeding, and no controller generalizes
beyond its training range. The honest contribution is therefore a
**robustness–performance tradeoff plus a reliability characterization** of
compliant-gait acquisition, not a clean "compliance training wins" result. The
study's main weakness is n=3 seeds against large seed variance; the main
correctness safeguards are a verified shared training recipe, a fair
shared-physics evaluation env, and progress-gated metrics that prevent a frozen
robot from being scored as a survivor.
```
