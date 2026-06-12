# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ablation study comparing two quadruped locomotion policies trained in MuJoCo:

- **Policy A (rigid baseline):** Trained on flat rigid ground only.
- **Policy B (compliance-randomized):** Trained on a curriculum spanning contact parameter variation through geometrically soft patches.

The research question: does training on varied terrain compliance produce a policy that generalizes better to unseen compliant terrain — and at what cost to peak performance and energy efficiency?

Both policies share identical architecture, reward function, and hyperparameters. **Only the training terrain differs.** Every implementation choice must preserve this isolation.

## Environment Setup

```bash
# Activate the project venv (Python 3.13)
source .venv/bin/activate

# Launch TensorBoard
tensorboard --logdir runs/
```

The `.venv/` virtualenv is the only Python environment for this project. Never use system Python or conda.

## Installed Packages

| Package | Version | Purpose |
|---------|---------|---------|
| mujoco | 3.9.0 | Simulator |
| stable-baselines3 | 2.8.0 | PPO implementation |
| gymnasium | 1.2.3 | Env interface |
| torch | 2.12.0 | Neural networks (MPS available on this Mac) |
| tensorboard | 2.20.0 | Training curves |
| matplotlib | 3.10.x | Plots |
| tqdm | 4.68.0 | Progress bars |
| rich | 15.0.0 | Terminal output |
| pandas | 3.0.x | Results tables |

## Robot Model

`mujoco_menagerie/unitree_go1/` — cloned from google-deepmind/mujoco_menagerie.

- Entry point for simulation: `mujoco_menagerie/unitree_go1/scene.xml`
- 12 actuators: FR/FL/RR/RL × hip/thigh/calf (`nq=19`, `nv=18`, `nu=12`)
- Always call `mujoco.mj_forward(model, data)` after reset before reading body positions

## Tech Stack

- **Simulator:** MuJoCo 3.9.0 (`mujoco` Python package) — chosen specifically because `solimp`/`solref` directly parameterizes contact stiffness/damping needed for Level 1 compliance.
- **RL Framework:** Stable-Baselines3 PPO (with a custom timeout bootstrapping fix — standard SB3 PPO is incorrect for fixed-length locomotion episodes).
- **Robot:** Unitree Go1 (12-DOF quadruped, 3 joints/leg: HAA, HFE, KFE).
- **Python 3.13**, NumPy, PyTorch 2.12 with MPS (Apple Silicon GPU).

## Planned Repository Structure

Implemented per `master_implementation_plan.md` Section 12:

```
quadruped-compliance/
├── environments/
│   ├── base_env.py             # MuJoCo base environment
│   ├── rigid_env.py            # Policy A (rigid terrain only)
│   ├── compliance_env.py       # Policy B (Level 1 + Level 2)
│   └── terrain/
│       ├── level1_params.py    # Contact parameter sampling
│       ├── level2_patches.py   # Soft patch terrain construction
│       └── curriculum.py       # Game-inspired curriculum
├── policies/
│   ├── policy_a_obs.py         # 47D observation builder
│   ├── policy_b_obs.py         # 87D observation builder (with EE history)
│   └── reward.py               # 9-term reward + penalty curriculum
├── training/
│   ├── train_policy_a.py
│   ├── train_policy_b.py
│   ├── ppo_config.py
│   └── timeout_bootstrap.py    # Critical: custom GAE with timeout bootstrapping
├── evaluation/
│   ├── evaluate.py             # N=100 episodes per terrain type
│   ├── metrics.py              # COT, fall rate, velocity, contact variance
│   └── terrain_suite.py        # T0–T9 held-out evaluation terrains
└── analysis/
    ├── plot_results.py
    └── ablation.py
```

## Key Architectural Decisions

### Observation Spaces (Different by Design)
- **Policy A: 47D** — current state only (velocity commands, gravity, angular/linear velocity, joint positions/velocities, previous action). No history. Forces Policy A to specialize for rigid terrain.
- **Policy B: 87D** — adds foot position history (24D at t, t-10, t-20), foot velocity (12D), foot contact states (4D). History is *required* per Kim & Lee 2021 — without it, the policy learns to stand still rather than walk on compliant terrain.

**Do NOT include** solimp/solref values or terrain labels in observations — the policy must infer compliance from movement history.

### Control
- Policy outputs 12 joint position targets, clipped to ±0.5 rad from nominal.
- PD controller converts to torques (Kp=20, Kd=0.5).
- 50 Hz policy, 500 Hz simulation (10 sub-steps per policy step).

### Reward Function (9 terms, identical for both policies)
All terms multiplied by `dt` for rate-independence. Penalty terms multiplied by curriculum coefficient `k_t`:
- Tracking: linear velocity (1.0·dt), angular velocity (0.5·dt)
- Penalties: z-velocity (4.0·dt), xy angular velocity (0.05·dt), joint motion (0.001·dt), torques (2×10⁻⁵·dt), action rate (0.25·dt), collisions (0.001·dt)
- Feet air time (2.0·dt) — essential, prevents shuffling

### Penalty Curriculum (Critical for Training Stability)
Start penalties at k₀=0.03, increase via `k = min(1.0, k / 0.997)` per PPO iteration. Without this, the robot learns to stand still (zero motion = minimum penalty).

### Timeout Bootstrapping (Critical Correctness Fix)
Standard SB3 PPO does not correctly bootstrap value at episode timeout — it treats timeout as failure. Must implement custom GAE that distinguishes `timeout=True` (robot survived, bootstrap) from `done=True, timeout=False` (robot fell, no bootstrap). See Section 6.2 of `master_implementation_plan.md` for the exact implementation.

### Terrain Curriculum (Policy B Only)
Game-inspired curriculum (Rudin 2022): 7 levels (0=rigid, 1-3=Level 1 contact variation, 4-6=Level 2 soft patches). Advance on success (distance > 50% of target), retreat on failure. Loop back at max level to prevent catastrophic forgetting.

### Parameter Sampling
Use **log-uniform** for all multiplicative physics parameters (friction, mass, damping, motor strength). Uniform sampling severely undersamples low values for these parameters.

## Evaluation

10 held-out terrain types (T0–T9), 100 episodes each. Primary metrics:
- `forward_velocity_mean/std` — performance
- `fall_rate` — robustness
- `cost_of_transport` — energy efficiency (dimensionless: Σ|τ·q̇| / (m·g·d))
- `foot_contact_variance` — gait regularity
- `base_height_variance` — vertical stability

**Expected pattern:** Policy A ≥ Policy B on T0 (rigid); Policy B >> Policy A on T5–T9 (compliant).

## PPO Hyperparameters

| Parameter | Value |
|-----------|-------|
| n_envs | 128 (MacBook CPU) |
| n_steps | 24 per env per update |
| batch_size | 3072 (128×24) |
| n_epochs | 5 |
| γ | 0.99 |
| λ_GAE | 0.95 |
| clip_range | 0.2 |
| entropy_coef | 0.01 |
| initial_lr | 1×10⁻³ (adaptive, KL target=0.01) |
| total_steps | 15M per policy |

## Source Papers

Summaries in `papers/md summaries/`:
- `lee2020_analysis.md` — observation space template, PMTG, proprioceptive history
- `kumar2021_rma_analysis.md` — penalty curriculum (k₀=0.03, decay=0.997), domain randomization ranges
- `rudin2022_legged_gym_analysis.md` — PPO hyperparameters, timeout bootstrapping fix, game curriculum, reward function
- `peng2018_dynamics_randomization_analysis.md` — log-uniform sampling, ablation structure
- `tan2018_sim_to_real_locomotion_analysis.md` — contact parameter ranges (Table I), compact observation principle
- `kim2021_nonrigid_terrain_analysis.md` — Level 2 terrain model, EE history requirement
