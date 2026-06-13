"""Policy B environment: compliance-randomized terrain.

SCOPE (2026-06-13): the ablation runs on **Level-1 contact-parameter compliance only**
(train with max_level=3). **Level-2 geometric soft patches are DEFERRED** — a documented
scope choice, not an omission. Level 1 IS terrain compliance (contact stiffness/damping),
so Policy A (rigid) vs Policy B (randomized compliance) on held-out compliance is a complete
test of the research question. The Level-2 code below (`_place_patches`, scene patch geoms)
is left in place but inert at max_level=3, pending a fix to a MuJoCo box-collision bug:
static box patches do not generate foot contacts in the full model (mj_geomDistance reports
overlap, pipeline yields zero contacts; diagnosed, not root-caused). See memory
policy-b-level2-deferred-and-collision-bug.

Original design (Level 1 + Level 2):

Shares Go1BaseEnv's reward, action, TG, and the 49D base observation block with
Policy A — ONLY the observation history and the training terrain differ, so the
A-vs-B ablation stays clean.

Observation (89D) = 49D base (identical to Policy A) + 24D foot-position history
(base-frame xy at t, t-10, t-20) + 12D foot velocity (base-frame) + 4D contact.
History is required for compliance inference (Kim & Lee 2021). NO solimp/solref or
terrain labels in the obs — compliance must be inferred from movement.

Terrain compliance is applied via MuJoCo contact parameters. CRITICAL: the Go1 foot
geom has priority=1, so it overrides the floor's solref/solimp UNLESS the floor's
priority is raised above it. At curriculum level 0 (rigid) we leave the floor
foot-governed (identical to Policy A); at level >0 we raise the floor priority to 2
and apply the sampled compliance. (Level 2 soft patches: priority 3 — Phase 2.)

Per-episode Level-1 randomization scales with a difficulty d = level / n_levels in
{0, 1/3, 2/3, 1}: d=0 is exactly rigid/nominal; ranges widen smoothly toward the
full Tan 2018 / Lee 2020 ranges at d=1. Game curriculum (Rudin 2022): advance on
success (distance > 50% target), retreat on failure, per-env.
"""

import os

import mujoco
import numpy as np
from gymnasium import spaces

from environments.base_env import Go1BaseEnv

_COMPLIANCE_XML = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "assets", "unitree_go1", "scene_compliance.xml")

_HIST_OFFSETS = (0, 10, 20)          # policy-steps back for foot-position history
_HIST_LEN     = max(_HIST_OFFSETS) + 1
_N_FEET       = 4

# Curriculum (Rudin 2022 game curriculum): 0 = rigid (foot-governed, == Policy A);
# 1-3 = Level 1 contact-parameter variation (widening); 4-6 = Level 1 (full) + Level 2
# soft patches of increasing softness (light/medium/pillow foam, Kim & Lee 2021).
_MAX_LEVEL         = 6
_FLOOR_PRIORITY_L1 = 2               # > foot priority (1) so the floor governs contact
_TARGET_DISTANCE   = 0.2 * 1000 * 0.02   # cmd 0.2 m/s × 1000 steps × dt = 4.0 m

_N_PATCHES    = 3
_PATCH_BURY_Z = -10.0
# Active patches sit slightly PROUD of the floor (top at +_PATCH_RAISE). A patch flush
# at z=0 is masked by the coincident rigid floor plane (the stiffer contact bears the
# load); raising it makes the patch the sole contact there so its compliance governs.
# 3 cm is well within the ~12 cm TG foot-lift, so it is not a climbing obstacle.
_PATCH_RAISE  = 0.03
# Level-2 patch compliance by stage (level-3): solimp[0,1] range and solref[0] range.
_PATCH_STAGES = {
    1: ((0.93, 0.97), (0.03, 0.07)),   # light foam  (~1 cm sinkage)
    2: ((0.90, 0.95), (0.02, 0.05)),   # medium foam (~2-3 cm)
    3: ((0.85, 0.92), (0.01, 0.03)),   # pillow-like (~4-5 cm)
}


class CompliantTerrainEnv(Go1BaseEnv):
    def __init__(self, *args, max_level: int = _MAX_LEVEL, **kwargs):
        kwargs.setdefault("xml_path", _COMPLIANCE_XML)
        super().__init__(*args, **kwargs)
        self._max_level = int(max_level)
        self._level     = 0           # per-env curriculum level (self-managed)

        # geom / model handles
        self._floor_geom_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self._patch_geom_ids = [
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, f"soft_patch_{i}")
            for i in range(_N_PATCHES)
        ]
        self._patch_half_z = float(self._model.geom_size[self._patch_geom_ids[0]][2])
        # nominal values to scale from (copied once, never mutated)
        self._floor_priority0 = int(self._model.geom_priority[self._floor_geom_id])
        self._floor_solref0   = self._model.geom_solref[self._floor_geom_id].copy()
        self._floor_solimp0   = self._model.geom_solimp[self._floor_geom_id].copy()
        self._floor_friction0 = self._model.geom_friction[self._floor_geom_id].copy()
        self._body_mass0      = self._model.body_mass.copy()
        self._body_inertia0   = self._model.body_inertia.copy()
        self._gear0           = self._model.actuator_gear.copy()
        self._dof_damping0    = self._model.dof_damping.copy()

        # per-episode obs corruption (Level 1)
        self._obs_noise_std = 0.0
        self._imu_bias      = 0.0

        # foot-history ring buffer (filled in _reset_terrain)
        self._foot_hist = np.zeros((_HIST_LEN, _N_FEET, 2), dtype=np.float64)
        self._prev_foot_world = np.zeros((_N_FEET, 3), dtype=np.float64)

    # ------------------------------------------------------------------
    # Observation: 49D base + 24 foot-pos history + 12 foot vel + 4 contact
    # ------------------------------------------------------------------
    def _obs_space(self) -> spaces.Box:
        return spaces.Box(low=-np.inf, high=np.inf, shape=(89,), dtype=np.float32)

    def _foot_pos_base(self) -> np.ndarray:
        """Foot site positions in the base frame, (4,3)."""
        rot   = self._data.xmat[self._trunk_id].reshape(3, 3)
        trunk = self._data.xpos[self._trunk_id]
        out   = np.empty((_N_FEET, 3))
        for i, sid in enumerate(self._foot_site_ids):
            out[i] = rot.T @ (self._data.site_xpos[sid] - trunk)
        return out

    def _foot_world(self) -> np.ndarray:
        return np.array([self._data.site_xpos[sid] for sid in self._foot_site_ids])

    def _get_obs(self) -> np.ndarray:
        base = self._base_obs()

        # roll the history buffer forward and write current foot xy
        foot_xy = self._foot_pos_base()[:, :2]
        self._foot_hist = np.roll(self._foot_hist, 1, axis=0)
        self._foot_hist[0] = foot_xy
        hist = np.concatenate([self._foot_hist[o].reshape(-1) for o in _HIST_OFFSETS])  # 24

        # foot velocity in base frame via finite difference of world positions
        cur_world = self._foot_world()
        rot       = self._data.xmat[self._trunk_id].reshape(3, 3)
        vel_world = (cur_world - self._prev_foot_world) / self._ctrl_dt
        foot_vel  = (rot.T @ vel_world.T).T.reshape(-1)                                 # 12
        self._prev_foot_world = cur_world

        contact = self._feet_contact.astype(np.float64)                                # 4

        obs = np.concatenate([base, hist, foot_vel, contact])
        # Level-1 observation corruption (per-episode IMU bias + per-step noise)
        if self._obs_noise_std > 0.0 or self._imu_bias != 0.0:
            obs = obs + self._imu_bias + self.np_random.normal(0.0, self._obs_noise_std, obs.shape)
        return obs.astype(np.float32)

    # ------------------------------------------------------------------
    # Terrain: apply curriculum compliance + (re)initialise history buffers
    # ------------------------------------------------------------------
    def _reset_terrain(self) -> None:
        self._apply_compliance(self._level)
        # initialise history with the current pose so t-10/t-20 are well-defined at t=0
        foot_xy = self._foot_pos_base()[:, :2]
        self._foot_hist[:] = foot_xy
        self._prev_foot_world = self._foot_world()

    def _apply_compliance(self, level: int) -> None:
        # Level 1 difficulty (base floor) saturates at level 3; Level 2 stage = level-3.
        d1    = min(level, 3) / 3.0
        stage = max(0, level - 3)
        m, gid = self._model, self._floor_geom_id

        if d1 <= 0.0:                                 # rigid — identical to Policy A
            m.geom_priority[gid] = self._floor_priority0
            m.geom_solref[gid]   = self._floor_solref0
            m.geom_solimp[gid]   = self._floor_solimp0
            m.geom_friction[gid] = self._floor_friction0
            m.body_mass[:]       = self._body_mass0
            m.body_inertia[:]    = self._body_inertia0
            m.actuator_gear[:]   = self._gear0
            m.dof_damping[:]     = self._dof_damping0
            self._obs_noise_std  = 0.0
            self._imu_bias       = 0.0
            self._place_patches(stage)                # stage 0 here → bury all
            mujoco.mj_forward(m, self._data)
            return

        d = d1
        rng = self.np_random
        def loguniform(lo, hi):
            return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        def lerp(a, b):
            return a + (b - a) * d

        # raise floor priority so the floor (not the foot) governs contact compliance
        m.geom_priority[gid] = _FLOOR_PRIORITY_L1
        # contact compliance: widen from rigid toward soft as d grows
        m.geom_solref[gid] = [loguniform(0.01, lerp(0.01, 0.1)), 1.0]
        solimp = self._floor_solimp0.copy()          # (5,): d0, d1, width, midpoint, power
        solimp[0] = rng.uniform(lerp(0.99, 0.85), 0.99)
        solimp[1] = rng.uniform(lerp(0.99, 0.85), 0.99)
        m.geom_solimp[gid] = solimp
        # friction (floor governs): widen around the 0.8 nominal
        fr = self._floor_friction0.copy()
        fr[0] = loguniform(lerp(0.8, 0.3), lerp(0.8, 2.0))
        fr[1] = loguniform(lerp(0.01, 0.01), lerp(0.01, 0.1)) if d > 0 else fr[1]
        m.geom_friction[gid] = fr
        # multiplicative body / motor / damping randomization (scaled by d)
        mass_mult = loguniform(lerp(1.0, 0.8), lerp(1.0, 1.2))
        m.body_mass[:]    = self._body_mass0 * mass_mult
        m.body_inertia[:] = self._body_inertia0 * mass_mult
        m.actuator_gear[:] = self._gear0 * loguniform(lerp(1.0, 0.8), lerp(1.0, 1.2))
        m.dof_damping[:]   = self._dof_damping0 * loguniform(lerp(1.0, 0.5), lerp(1.0, 2.0))
        # obs corruption
        self._obs_noise_std = rng.uniform(0.0, 0.05 * d)
        self._imu_bias      = rng.uniform(-0.05 * d, 0.05 * d)

        # Level 2 soft patches (levels 4-6)
        self._place_patches(stage)
        mujoco.mj_forward(m, self._data)

    def _place_patches(self, stage: int) -> None:
        """Position/soften active patches along the +x path; bury the rest.

        stage 0 → all patches buried (rigid / Level-1-only). stage 1-3 → 1-3 patches
        placed in x∈[1.0, 3.5] (within the ~4 m traversal so at least one is crossed),
        top face flush at z=0, softness sampled from the stage band (priority 3 governs)."""
        m, rng = self._model, self.np_random
        # bury everything first
        for pid in self._patch_geom_ids:
            m.geom_pos[pid] = [0.0, 0.0, _PATCH_BURY_Z]
        if stage <= 0:
            return
        (si_lo, si_hi), (sr_lo, sr_hi) = _PATCH_STAGES[stage]
        n = int(rng.integers(1, _N_PATCHES + 1))            # 1..3 patches
        xs = np.sort(rng.uniform(1.0, 3.5, size=n))
        for k in range(n):
            pid = self._patch_geom_ids[k]
            m.geom_pos[pid] = [float(xs[k]), float(rng.uniform(-0.15, 0.15)),
                               _PATCH_RAISE - self._patch_half_z]   # top face at +_PATCH_RAISE
            solimp = m.geom_solimp[pid].copy()
            solimp[0] = rng.uniform(si_lo, si_hi)
            solimp[1] = rng.uniform(si_lo, si_hi)
            m.geom_solimp[pid] = solimp
            m.geom_solref[pid] = [float(np.exp(rng.uniform(np.log(sr_lo), np.log(sr_hi)))), 1.0]

    # ------------------------------------------------------------------
    # Per-env game curriculum: advance on success, retreat on failure
    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        # update level from the episode that just ended (distance vs target)
        if getattr(self, "_step_count", 0) > 0:
            dist = float(self._data.qpos[0])          # forward distance travelled (x)
            if dist > 0.5 * _TARGET_DISTANCE:
                self._level = min(self._level + 1, self._max_level)
            else:
                self._level = max(0, self._level - 1)
        return super().reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info["terrain_level"] = float(self._level)   # per-env curriculum level (for logging)
        return obs, reward, terminated, truncated, info
