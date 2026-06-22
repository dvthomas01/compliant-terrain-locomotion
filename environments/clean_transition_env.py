"""Clean Level-2 transition env (v2): rigid->soft->rigid with the 3 confounds removed.

Fixes vs TransitionEnv: (1) the WHOLE strip is flush slide-joint tiles at z=0 — no infinite
plane, no raised 5cm step (rigid sections are stiff tiles; soft zone sinks BELOW grade).
(2) per-tile damping c = ALPHA*sqrt(k) so the damping ratio ζ is CONSTANT across the
stiffness sweep (isolates stiffness, not bounciness). (3) slide range -0.15 so soft tiles
don't saturate. This makes "does foot-history help adapt to a stiffness transition" a clean,
well-powered test. Trains B_trans2 (89D) vs B_trans2_noh (49D).
"""
import os
import numpy as np
import mujoco

from environments.compliance_env import CompliantTerrainEnv

_CLEAN_XML = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "assets", "unitree_go1", "scene_transition_clean.xml")
_SOFT_JOINTS  = ["soft0_j", "soft1_j", "soft2_j", "soft3_j", "soft4_j"]
_RIGID_JOINTS = ["rig_app_j", "rig_exit_j"]
_MAX_LEVEL_T2 = 4
_K_FIRM       = 80000.0     # level 0: near-rigid soft zone (barely sinks)
_K_SOFT       = 3000.0      # level max: clearly soft (sinks several cm, no saturation)
_K_RIGID      = 1_000_000.0 # the rigid approach/exit tiles
_ALPHA        = 2.8         # c = ALPHA*sqrt(k) -> constant damping ratio ζ≈0.7 (no bounce confound)
_SOFT_ZONE_END = 3.0
_TARGET_CROSS  = _SOFT_ZONE_END + 0.3


class CleanTransitionEnv(CompliantTerrainEnv):
    def __init__(self, *args, max_level: int = _MAX_LEVEL_T2, obs_mode: str = "B", **kwargs):
        kwargs["xml_path"] = _CLEAN_XML
        super().__init__(*args, max_level=max_level, obs_mode=obs_mode, **kwargs)
        def jids(names):
            return np.array([mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names])
        self._soft_j  = jids(_SOFT_JOINTS)
        self._rigid_j = jids(_RIGID_JOINTS)
        self._soft_qadr  = np.array([self._model.jnt_qposadr[j] for j in self._soft_j])
        self._soft_dof   = np.array([self._model.jnt_dofadr[j] for j in self._soft_j])
        self._all_qadr   = np.array([self._model.jnt_qposadr[j] for j in np.concatenate([self._soft_j, self._rigid_j])])
        self._all_dof    = np.array([self._model.jnt_dofadr[j] for j in np.concatenate([self._soft_j, self._rigid_j])])
        # rigid tiles: stiff + constant-ζ damping (set once)
        for j in self._rigid_j:
            self._model.jnt_stiffness[j] = _K_RIGID
            self._model.dof_damping[self._model.jnt_dofadr[j]] = _ALPHA * np.sqrt(_K_RIGID)

    def _reset_terrain(self) -> None:
        d = self._level / self._max_level if self._max_level else 0.0
        rng = self.np_random
        logk = np.log(_K_FIRM) + d * (np.log(_K_SOFT) - np.log(_K_FIRM))   # firm -> soft
        for j in self._soft_j:
            k = float(np.exp(logk + rng.uniform(-0.15, 0.15)))
            self._model.jnt_stiffness[j] = k
            self._model.dof_damping[self._model.jnt_dofadr[j]] = _ALPHA * np.sqrt(k)   # constant ζ
        self._data.qpos[self._all_qadr] = 0.0
        self._data.qvel[self._all_dof] = 0.0
        mujoco.mj_forward(self._model, self._data)
        foot_xy = self._foot_pos_base()[:, :2]
        self._foot_hist[:] = foot_xy
        self._prev_foot_world = self._foot_world()

    def reset(self, *, seed=None, options=None):
        if getattr(self, "_step_count", 0) > 0:
            crossed = float(self._data.qpos[0]) > _TARGET_CROSS
            self._level = min(self._level + 1, self._max_level) if crossed else max(0, self._level - 1)
        return super(CompliantTerrainEnv, self).reset(seed=seed, options=options)

    def step(self, action):
        obs, r, term, trunc, info = super(CompliantTerrainEnv, self).step(action)
        info["terrain_level"] = float(self._level)
        info["tile_sink"] = float(-self._data.qpos[self._soft_qadr].min())
        info["x_pos"] = float(self._data.qpos[0])
        return obs, r, term, trunc, info
