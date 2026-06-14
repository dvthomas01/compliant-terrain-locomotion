"""Level-2 transition environment: rigid -> soft -> rigid within one episode.

The robot walks on rigid ground, crosses a soft zone (a row of spring-tiles at x in
[1.8, 3.0] that physically sink under load), then returns to rigid ground. Unlike Level-1
(uniform per-episode compliance), the ground CHANGES underfoot mid-episode — the only
regime that exercises ONLINE compliance adaptation, which is what the 89D foot-history
observation is for (Kim & Lee 2021). Tests B_trans (89D) vs B_trans_noh (49D).

Spring tiles are MOVING bodies (box on a vertical slide-joint spring) — they collide with
the foot, unlike the static box patches that did not (see memory; validated Phase A). The
base floor is the rigid plane (firm sections). Per-env game curriculum softens the tiles:
level 0 = near-rigid (stiffness ~20000), level max = soft (~1500), log-interpolated.
"""

import numpy as np
import mujoco

from environments.compliance_env import CompliantTerrainEnv, _COMPLIANCE_XML  # noqa: F401
import os

_TRANSITION_XML = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "assets", "unitree_go1", "scene_transition.xml")
_N_TILES        = 4
_MAX_LEVEL_T    = 4
_K_FIRM         = 20000.0     # near-rigid soft zone (level 0 -> no real transition)
_K_SOFT         = 1500.0      # softest tiles (level max) -> deep sink, strong transition
_TILE_DAMPING   = 60.0
_SOFT_ZONE_END  = 3.0         # x past which the robot has crossed the soft zone
_TARGET_CROSS   = _SOFT_ZONE_END + 0.3


class TransitionEnv(CompliantTerrainEnv):
    def __init__(self, *args, max_level: int = _MAX_LEVEL_T, obs_mode: str = "B", **kwargs):
        kwargs["xml_path"] = _TRANSITION_XML
        super().__init__(*args, max_level=max_level, obs_mode=obs_mode, **kwargs)
        self._tile_jids = np.array([mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, f"tile{i}_j")
                                    for i in range(_N_TILES)])
        self._tile_qadr = np.array([self._model.jnt_qposadr[j] for j in self._tile_jids])
        self._tile_dofs = np.array([self._model.jnt_dofadr[j] for j in self._tile_jids])

    def _reset_terrain(self) -> None:
        # firm base floor (rigid plane, foot-governed) is unchanged; only the soft-zone tiles vary.
        d = self._level / self._max_level if self._max_level else 0.0
        rng = self.np_random
        k_lo, k_hi = _K_SOFT, _K_FIRM
        # difficulty d: d=0 -> firm (k_hi), d=1 -> soft (k_lo). per-tile log-uniform jitter.
        for j in self._tile_jids:
            logk = np.log(k_hi) + d * (np.log(k_lo) - np.log(k_hi))
            k = float(np.exp(logk + rng.uniform(-0.15, 0.15)))
            self._model.jnt_stiffness[j] = k
        self._model.dof_damping[self._tile_dofs] = _TILE_DAMPING
        # tiles start at rest (slide 0); zero their velocities
        self._data.qpos[self._tile_qadr] = 0.0
        self._data.qvel[self._tile_dofs] = 0.0
        mujoco.mj_forward(self._model, self._data)
        # initialise foot history for the 89D obs
        foot_xy = self._foot_pos_base()[:, :2]
        self._foot_hist[:] = foot_xy
        self._prev_foot_world = self._foot_world()

    def reset(self, *, seed=None, options=None):
        # per-env curriculum: advance only if the robot CROSSED the soft zone last episode
        if getattr(self, "_step_count", 0) > 0:
            crossed = float(self._data.qpos[0]) > _TARGET_CROSS
            self._level = min(self._level + 1, self._max_level) if crossed else max(0, self._level - 1)
        return super(CompliantTerrainEnv, self).reset(seed=seed, options=options)

    def step(self, action):
        obs, r, term, trunc, info = super(CompliantTerrainEnv, self).step(action)
        info["terrain_level"] = float(self._level)
        info["tile_sink"] = float(-self._data.qpos[self._tile_qadr].min())   # deepest tile sink (m)
        info["x_pos"] = float(self._data.qpos[0])
        return obs, r, term, trunc, info
