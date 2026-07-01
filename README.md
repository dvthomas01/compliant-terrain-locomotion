# Compliant-Terrain Quadruped Locomotion: A Robustness Ablation

A controlled study of what makes a reinforcement-learning locomotion policy robust to
ground it never trained on. A simulated Unitree Go1 quadruped is trained in MuJoCo and
evaluated on a held-out sweep of contact softness, from rigid to spongy, to ask a
simple question: does training across varied ground compliance produce a controller
that generalizes to unseen soft terrain, and what actually carries that robustness?

## Premise

Legged robots trained in simulation tend to overfit the exact physics they saw in
training. Real ground is not uniformly rigid, so a policy that only ever walked on a
hard floor can fall the moment the surface gives underfoot. Two interventions are
commonly proposed to fix this: **training across randomized ground compliance**, and
**feeding the policy a short history of its own foot contacts** so it can sense and
adapt to the surface. This project tests both, in isolation, with a clean ablation.

## The experiment

Ground softness is treated as a controlled variable. In MuJoCo, contact compliance is
a solver setting (the stiffness and damping of the contact constraint), so the floor
can be swept from rigid to spongy without changing anything else, including how it
looks. Policies are evaluated across ten softness levels, `T0` (rigid) through `T9`
(softest), with 100 episodes per level per seed and 8 seeds per policy.

Every policy rides on a fixed open-loop **trot prior** (a residual trajectory
generator); the neural network outputs small corrections on top of a gait that is
already propulsive. This isolates the contribution of learning from the contribution
of the gait prior. Three policies are compared under an **identical training recipe**,
differing only in training terrain and observation:

| Policy | Training terrain | Observation |
|--------|------------------|-------------|
| **A**  | rigid only | current state, no foot history |
| **B**  | compliance curriculum | current state **+ foot-contact history** |
| **B'** | compliance curriculum | current state, no foot history |

Holding everything else fixed means any difference at evaluation traces to exactly one
of two variables: the training terrain, or the proprioceptive history.

## What we found

- **A monotonic robustness–performance tradeoff.** On rigid ground the rigid policy A
  is the fastest and most energy-efficient, but it fails sharply once the floor passes
  its training edge. The compliance-trained policies give up peak performance and, in
  return, keep their footing on softer ground, in a strict order: **B' > B > A**. Of 8
  seeds, at softness level `T5` the number keeping a fall rate below one half is A 3,
  B 6, B' 8; at `T6` it is A 1, B 5, B' 7.
- **Foot-contact history did not help.** The no-history policy B' is at least as robust
  as the history-equipped B at every transition level. The extra observation was
  associated with a faster, less-settled gait that fell sooner.
- **Robustness is a gait property, not a training knob.** It is not explained by how far
  a seed climbed the curriculum, nor by gait speed (a speed-matched intervention shows
  brittle seeds still fall at matched velocity). It tracks the **steadiness of foot
  contact**, which is measurable but, in this study, not yet reliably inducible.
- **The gains are bounded.** Compliance training widens the survivable-softness band but
  does not grant open-ended extrapolation; past the training edge every policy collapses.

## Repository layout

```
environments/    MuJoCo Go1 simulation: base_env, compliance_env, rigid_env, robot assets
training/        PPO recipe: the three policies, ppo_config, timeout bootstrap, smoke test
evaluation/      Held-out terrain suite, evaluation harness, result CSVs
analysis/        Results figures
requirements.txt Pinned Python dependencies (Python 3.13)
```

Trained checkpoints, TensorBoard logs, and raw rollout renders are large and are not
tracked in git; the scripts below regenerate them.

## Setup

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Replicating the study

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD"

# 1. Sanity-check the environment and the trot prior
python training/smoke_compliance.py
python training/verify_tg.py

# 2. Train the three policies (about 15M steps each)
python training/train_policy_v24.py      # Policy A  (rigid)
python training/train_policy_b.py        # Policy B  (compliance + history)
python training/train_policy_b_noh.py    # Policy B' (compliance, no history)

# 3. Evaluate across the T0–T9 softness suite (100 episodes/seed/level)
python evaluation/evaluate.py            # -> evaluation/ablation_results_multiseed.csv

# 4. Generate the results figure
python analysis/plot_variance.py         # -> analysis/variance_plot.png
```

The `training/run_*.sh` scripts drive the multi-seed batches used to produce the
8-seed results.

## Tech stack

MuJoCo 3.9 for simulation, Stable-Baselines3 PPO (with a timeout-bootstrapping fix for
fixed-length locomotion episodes), PyTorch 2.12, and Python 3.13.

## Attribution

The Unitree Go1 model is from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie); its license is
in `environments/assets/unitree_go1/LICENSE`.
