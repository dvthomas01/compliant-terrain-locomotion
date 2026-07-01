# Fast Breaks Fast: Robustness in Compliant-Terrain Locomotion

An ablation study on what makes a simulated quadruped generalize to ground softer
than it ever trained on. A Unitree Go1 is trained in MuJoCo with a residual policy on
top of a trot prior, then evaluated on a held-out sweep of contact softness from rigid
to spongy. The headline result is a monotonic robustness ordering, and a set of
intuitive predictions that did not survive testing.

**Read the write-up:** [`blog/index.html`](blog/index.html) is a self-contained
static blog post with the full story, figures, and side-by-side simulation renders.

## Key findings

- **A robustness–performance tradeoff, monotonic in training compliance.** The
  rigid-trained policy is fastest and most energy-efficient on rigid ground but fails
  once the floor passes its training edge. Compliance-trained policies degrade later,
  in a strict order: no-history > with-history > rigid.
- **Foot-contact history did not help.** The no-history policy is at least as robust
  as the history-equipped one at every transition terrain.
- **Robustness is a gait property, not a knob.** It is not explained by curriculum
  progress or by gait speed. It tracks the steadiness of foot contact, which is
  measurable but, in this study, not yet reliably inducible.

## Repository layout

```
blog/            Self-contained static blog post (see blog/README.md)
environments/    MuJoCo Go1 sim: base_env, compliance_env, rigid_env, robot assets
training/        PPO recipe: canonical policies, ppo_config, timeout bootstrap, smoke test
evaluation/      Held-out terrain suite, evaluation, comparison renders, result CSVs
analysis/        Figure and render generation for the blog and paper
requirements.txt Pinned Python dependencies (Python 3.13)
```

The three policies compared share an identical recipe and differ only in training
terrain and observation:

- **Policy A** — trained on rigid ground only (`training/train_policy_v24.py`).
- **Policy B** — compliance curriculum, with foot-contact history (`train_policy_b.py`).
- **Policy B'** — compliance curriculum, no history (`train_policy_b_noh.py`).

## Setup

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the results

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD"

# Sanity check the environment and gait prior
python training/smoke_compliance.py
python training/verify_tg.py

# Evaluate trained policies across the T0–T9 softness suite
python evaluation/evaluate.py

# Regenerate the blog figures and comparison renders
python analysis/plot_blog_figures.py
python analysis/plot_schematics.py
python analysis/render_trot_diagram.py
python evaluation/render_blog.py
```

Training a policy from scratch (about 15M steps):

```bash
python training/train_policy_v24.py      # Policy A (rigid)
python training/train_policy_b.py        # Policy B (compliance + history)
python training/train_policy_b_noh.py    # Policy B' (compliance, no history)
```

Trained checkpoints, TensorBoard logs, and raw rollout renders are large and are not
tracked in git; the scripts above regenerate them.

## Viewing the blog locally

```bash
cd blog
python3 -m http.server 8000
# open http://localhost:8000
```

## Tech stack

MuJoCo 3.9 for simulation, Stable-Baselines3 PPO for training, PyTorch 2.12, and
Python 3.13. The blog is plain HTML, CSS, and a small amount of JavaScript, with no
build step.

## Attribution

The Unitree Go1 model is from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie); its license is
in `environments/assets/unitree_go1/LICENSE`.
