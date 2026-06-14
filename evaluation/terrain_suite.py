"""Held-out evaluation terrains T0-T9 and a fair shared-physics eval environment.

The ablation compares Policy A (rigid, 49D) and Policy B (compliance-randomized, 89D)
on the SAME terrain instances. EvalCompliantEnv applies a FIXED, deterministic contact
compliance per terrain (no curriculum, no randomization, nominal mass/motor/damping, no
obs noise) and emits 49D obs for A or 89D for B — identical physics, only the observation
differs, as each policy requires.

T0 = rigid (foot-governed, == Policy A's training ground). T1-T6 span the TRAINING range
(interpolation). T7-T9 go BEYOND it — softer than anything trained on (extrapolation).
Compliance is varied via floor solref[0] (contact time constant; larger = softer) and
solimp[0,1] (impedance; lower = softer). Friction held nominal so compliance is isolated.
"""

import mujoco
import numpy as np
from gymnasium import spaces

from environments.base_env import Go1BaseEnv
from environments.compliance_env import CompliantTerrainEnv, _FLOOR_PRIORITY_L1

# name, solref0 (s), solimp0 == solimp1.  rigid=True -> foot-governed (== Policy A ground).
# Training range (curriculum d<=1): solref0 up to ~0.10, solimp0 down to ~0.85.
TERRAINS = [
    {"name": "T0_rigid",      "rigid": True},
    # --- T1-T6: within training range (interpolation) ---
    {"name": "T1_firm",       "solref0": 0.01, "solimp0": 0.97},
    {"name": "T2",            "solref0": 0.025, "solimp0": 0.95},
    {"name": "T3",            "solref0": 0.04, "solimp0": 0.93},
    {"name": "T4",            "solref0": 0.06, "solimp0": 0.90},
    {"name": "T5",            "solref0": 0.08, "solimp0": 0.87},
    {"name": "T6_train_edge", "solref0": 0.10, "solimp0": 0.85},
    # --- T7-T9: BEYOND training range (extrapolation, softer) ---
    {"name": "T7_extrap",     "solref0": 0.15, "solimp0": 0.82},
    {"name": "T8_extrap",     "solref0": 0.20, "solimp0": 0.78},
    {"name": "T9_extrap_soft","solref0": 0.30, "solimp0": 0.72},
]


class EvalCompliantEnv(CompliantTerrainEnv):
    """Compliance env pinned to one fixed terrain, emitting 49D (A) or 89D (B) obs."""

    def __init__(self, terrain: dict, obs_mode: str, **kwargs):
        self._terrain = terrain
        super().__init__(max_level=0, obs_mode=obs_mode, **kwargs)   # parent owns obs_mode; curriculum inert

    def reset(self, *, seed=None, options=None):
        # bypass the per-env curriculum update in CompliantTerrainEnv.reset
        return super(CompliantTerrainEnv, self).reset(seed=seed, options=options)

    def _reset_terrain(self) -> None:
        self._apply_eval_terrain()
        foot_xy = self._foot_pos_base()[:, :2]
        self._foot_hist[:] = foot_xy
        self._prev_foot_world = self._foot_world()

    def _apply_eval_terrain(self) -> None:
        m, gid = self._model, self._floor_geom_id
        # nominal robot params + clean obs (isolate terrain compliance as the only variable)
        m.body_mass[:] = self._body_mass0;  m.body_inertia[:] = self._body_inertia0
        m.actuator_gear[:] = self._gear0;    m.dof_damping[:] = self._dof_damping0
        self._obs_noise_std = 0.0;           self._imu_bias = 0.0
        if self._terrain.get("rigid"):
            m.geom_priority[gid] = self._floor_priority0
            m.geom_solref[gid]   = self._floor_solref0
            m.geom_solimp[gid]   = self._floor_solimp0
            m.geom_friction[gid] = self._floor_friction0
        else:
            m.geom_priority[gid] = _FLOOR_PRIORITY_L1     # floor governs (> foot priority 1)
            m.geom_solref[gid]   = [float(self._terrain["solref0"]), 1.0]
            solimp = self._floor_solimp0.copy()
            solimp[0] = solimp[1] = float(self._terrain["solimp0"])
            m.geom_solimp[gid]   = solimp
            m.geom_friction[gid] = self._floor_friction0  # nominal friction
        mujoco.mj_forward(m, self._data)

    def step(self, action):
        obs, r, term, trunc, info = super().step(action)
        tau = self._data.actuator_force
        qd  = self._data.qvel[6:]
        info["power"] = float(np.sum(np.abs(tau * qd)))   # Σ|τ·q̇| for cost-of-transport
        info["x_pos"] = float(self._data.qpos[0])
        return obs, r, term, trunc, info

    @property
    def robot_mass(self) -> float:
        return float(self._model.body_mass.sum())
