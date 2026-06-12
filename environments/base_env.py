"""
Go1BaseEnv: Gymnasium wrapper around the Unitree Go1 in MuJoCo.

Shared infrastructure for both Policy A (rigid) and Policy B (compliance).
Subclasses override _obs_space() and _get_obs() to add history / extra terms.
_reset_terrain() is a no-op hook for subclasses to randomize physics params.
"""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_XML = os.path.join(_HERE, "assets", "unitree_go1", "scene.xml")

# Standing keyframe joint angles: hip=0, thigh=0.9, calf=-1.8 for all 4 legs
NOMINAL_JOINT_POS = np.array([
    0.0,  0.9, -1.8,   # FR: hip, thigh, calf
    0.0,  0.9, -1.8,   # FL
    0.0,  0.9, -1.8,   # RR
    0.0,  0.9, -1.8,   # RL
], dtype=np.float64)

ACTION_SCALE = 0.5   # max joint deviation from nominal (rad)

FOOT_NAMES = ("FR", "FL", "RR", "RL")

# Reward weights — all multiplied by ctrl_dt inside step() for rate-independence
_W_LIN_VEL     = 1.5    # raised from 1.0: forward progress matters more
_W_ANG_VEL     = 0.5
_W_LIN_VEL_Z   = 4.0
_W_ANG_VEL_XY   = 0.15   # raised from 0.05; now outside k-curriculum (always active)
_W_ORIENTATION  = 1.0    # penalises static tilt via projected gravity x/y components
_W_BACKWARD_VEL = 2.0    # penalises negative forward velocity (outside k-curriculum)
_W_JOINT_MOT   = 0.001
_W_TORQUES     = 2e-5
_W_ACTION_RATE = 0.25
_W_COLLISION   = 0.001
_W_FEET_AIR    = 5.0    # raised from 2.0: stronger gait-cycle incentive
_W_BASE_HEIGHT = 2.0    # penalises crouching below nominal stance height
_W_ALIVE       = 0.5    # constant per-step bonus for surviving upright
_W_PITCH       = 25.0   # v6: dense pitch penalty beyond the trot lean (outside k-curriculum)
_W_FWD_PROGRESS = 8.0   # v7: dense monotonic "move forward" reward (outside k-curriculum); v13 raised 4→8
_W_OVERSPEED   = 3.0    # v8: linear penalty for exceeding the command (kills the sprint)
_FEET_AIR_REF_SPEED = 0.2  # v10: feet-air-time reward scales with command up to this speed

# v6 tracking / pitch shaping
# _SIGMA_SQ_VEL tightened 0.25 → 0.04 (σ≈0.2): at the 0.2 m/s target the old σ≈0.5
# Gaussian was nearly flat (0.58 m/s still scored 0.56×), giving no deceleration gradient.
#   vx=0.20 → 1.000   vx=0.30 → 0.607   vx=0.40 → 0.135   vx=0.58 → 0.027   vx=0.00 → 0.368
# _W_PITCH sized for the narrow 0.26→0.40 penalty band (termination held at 0.40):
#   pitch 0.40 → -33% of max tracking   0.35 → -13%   0.30 → -3%   ≤0.26 → 0 (protects healthy lean)
_SIGMA_SQ_VEL    = 0.04
_PITCH_FREE_ZONE = 0.26   # rad (~15°): no penalty inside the nominal forward trot lean

_TARGET_BASE_HEIGHT  = 0.27    # Go1 nominal trunk z in standing keyframe (metres)
_TERMINATION_PENALTY = -20.0   # one-off penalty added when robot falls or tips over

# Termination thresholds
_MIN_HEIGHT = 0.15   # m
_MAX_ROLL   = 0.8    # rad (~46°)
_MAX_PITCH  = 0.4    # rad (~23°)
_MAX_STEPS  = 1000   # ~20 s at 50 Hz


class Go1BaseEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        xml_path: str = DEFAULT_XML,
        render_mode: str | None = None,
        target_lin_vel: tuple[float, float] = (0.5, 0.0),
        target_ang_vel: float = 0.0,
    ):
        super().__init__()
        self.render_mode = render_mode
        self._target_lin_vel = np.array(target_lin_vel, dtype=np.float64)
        self._target_ang_vel = float(target_ang_vel)
        self.penalty_scale = 0.03   # k_t; updated by PenaltyCurriculum during training

        self._model = mujoco.MjModel.from_xml_path(xml_path)
        self._data  = mujoco.MjData(self._model)

        self._ctrl_dt    = 0.02  # 50 Hz
        self._n_substeps = round(self._ctrl_dt / self._model.opt.timestep)

        self._trunk_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, "trunk"
        )
        self._foot_geom_ids = np.array([
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, n)
            for n in FOOT_NAMES
        ])
        self._foot_site_ids = np.array([
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, n)
            for n in FOOT_NAMES
        ])
        self._robot_body_ids = set(range(1, self._model.nbody))  # excludes world (0)

        # Action: 12 normalized offsets in [-1, 1]
        # Actual joint target = NOMINAL_JOINT_POS + action * ACTION_SCALE
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(12,), dtype=np.float32
        )
        self.observation_space = self._obs_space()

        self._renderer: mujoco.Renderer | None = None
        if render_mode == "rgb_array":
            self._renderer = mujoco.Renderer(self._model, height=480, width=640)

        # Episode state — initialised properly in reset()
        self._prev_action    = np.zeros(12, dtype=np.float64)
        self._prev_joint_vel = np.zeros(12, dtype=np.float64)
        self._feet_contact   = np.zeros(4, dtype=bool)
        self._feet_air_time  = np.zeros(4, dtype=np.float64)
        self._step_count     = 0

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    def _obs_space(self) -> spaces.Box:
        """Return the observation space. Override in Policy B subclass."""
        return spaces.Box(low=-np.inf, high=np.inf, shape=(47,), dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        """Build observation. Override in Policy B subclass to add history."""
        return self._base_obs().astype(np.float32)

    def _reset_terrain(self) -> None:
        """Called at end of reset(). Override to randomize contact physics."""
        pass

    # ------------------------------------------------------------------
    # Shared 47-D observation block (used by both Policy A and B)
    # ------------------------------------------------------------------

    def _base_obs(self) -> np.ndarray:
        lin_vel, ang_vel = self._base_velocity_body_frame()
        gravity          = self._gravity_body_frame()
        joint_pos_rel    = self._data.qpos[7:] - NOMINAL_JOINT_POS
        joint_vel        = self._data.qvel[6:].copy()
        return np.concatenate([
            self._target_lin_vel,   # 2  — commanded forward/lateral speed
            gravity,                # 3  — gravity direction in body frame
            ang_vel,                # 3  — base angular velocity (body frame)
            lin_vel,                # 3  — base linear velocity (body frame)
            joint_pos_rel,          # 12 — joint angles relative to home stance
            joint_vel,              # 12 — joint velocities
            self._prev_action,      # 12 — last action (normalised)
        ])  # 47 total

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self._model, self._data, 0)
        self._add_reset_noise()
        mujoco.mj_forward(self._model, self._data)

        self._prev_action    = np.zeros(12, dtype=np.float64)
        self._prev_joint_vel = np.zeros(12, dtype=np.float64)
        self._feet_contact   = self._foot_contacts()
        self._feet_air_time  = np.zeros(4, dtype=np.float64)
        self._step_count     = 0

        self._reset_terrain()

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float64)
        self._data.ctrl[:] = NOMINAL_JOINT_POS + action * ACTION_SCALE

        for _ in range(self._n_substeps):
            mujoco.mj_step(self._model, self._data)

        self._step_count += 1

        # --- feet air-time tracking ---
        new_contact = self._foot_contacts()
        just_landed = (~self._feet_contact) & new_contact
        self._feet_air_time[~new_contact] += self._ctrl_dt
        self._feet_air_time[new_contact]   = 0.0
        feet_air_bonus = float(np.sum(self._feet_air_time[just_landed] - 0.5))
        self._feet_contact = new_contact

        reward, info = self._compute_reward(action, feet_air_bonus)

        terminated = self._is_terminated()
        truncated  = self._step_count >= _MAX_STEPS

        if terminated:
            reward += _TERMINATION_PENALTY
            info["r_termination"] = _TERMINATION_PENALTY
            # Diagnose which limit was hit (uses values already in info dict)
            h = info["base_height"]
            r = info["roll"]
            p = info["pitch"]
            info["term_height"] = float(h < _MIN_HEIGHT)
            info["term_roll"]   = float(abs(r) > _MAX_ROLL)
            info["term_pitch"]  = float(abs(p) > _MAX_PITCH)
        else:
            info["r_termination"] = 0.0
            info["term_height"]   = 0.0
            info["term_roll"]     = 0.0
            info["term_pitch"]    = 0.0

        self._prev_action    = action.copy()
        self._prev_joint_vel = self._data.qvel[6:].copy()

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self._renderer is None:
            return None
        cam          = mujoco.MjvCamera()
        cam.type     = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = self._data.xpos[self._trunk_id]
        cam.distance  = 2.5
        cam.azimuth   = 150
        cam.elevation = -20
        self._renderer.update_scene(self._data, camera=cam)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------
    # Reward (9 terms, from Rudin 2022 + RMA penalty curriculum)
    # ------------------------------------------------------------------

    def _compute_reward(
        self, action: np.ndarray, feet_air_bonus: float
    ) -> tuple[float, dict]:
        dt  = self._ctrl_dt
        k   = self.penalty_scale
        lin_vel, ang_vel = self._base_velocity_body_frame()
        roll, pitch      = self._roll_pitch()

        # --- stability gate (softer exponent: -2 vs old -5) ---
        # At a normal 10° forward lean (0.17 rad): exp(-2·0.03) ≈ 0.94 vs exp(-5·0.03) ≈ 0.86
        # Still cuts reward during aggressive nose-dives, but doesn't punish a healthy gait lean
        stability_scale = float(np.exp(-2.0 * (pitch ** 2 + roll ** 2)))

        # --- tracking (stability-gated, not curriculum-scaled) ---
        lin_err          = self._target_lin_vel - lin_vel[:2]
        r_lin_vel_ungated = float(np.exp(-np.dot(lin_err, lin_err) / _SIGMA_SQ_VEL)) * _W_LIN_VEL * dt
        r_lin_vel         = r_lin_vel_ungated * stability_scale
        ang_err           = self._target_ang_vel - ang_vel[2]
        r_ang_vel         = float(np.exp(-(ang_err ** 2) / 0.25)) * _W_ANG_VEL * dt

        # --- stability penalties (NOT curriculum-scaled — active from step 1) ---
        r_ang_vel_xy  = -float(np.dot(ang_vel[:2], ang_vel[:2]))          * _W_ANG_VEL_XY  * dt
        gravity_body  = self._gravity_body_frame()
        r_orientation = -(gravity_body[0] ** 2 + gravity_body[1] ** 2)   * _W_ORIENTATION * dt

        # --- backward velocity penalty (NOT curriculum-scaled) ---
        # Zero when moving forward or standing still; negative proportional to backward speed
        backward_speed = max(0.0, -float(lin_vel[0]))
        r_backward_vel = -_W_BACKWARD_VEL * backward_speed * dt

        # --- pitch penalty (v6, NOT curriculum-scaled — dense gradient before the cliff) ---
        # The multiplicative stability gate goes toothless when tracking → 0 (overshoot + pitched);
        # this absolute quadratic penalty bites in exactly that regime. Free zone protects the
        # healthy ~0.24 rad trot lean; the penalty escalates toward the 0.40 rad termination limit.
        excess_pitch   = max(0.0, abs(pitch) - _PITCH_FREE_ZONE)
        r_pitch_pen    = -_W_PITCH * (excess_pitch ** 2) * dt

        # --- forward-progress reward (v7, NOT curriculum-scaled) ---
        # Dense, monotonic "always move forward" signal the tracking Gaussian cannot give:
        # the Gaussian hands out exp(-0.2²/σ²) ≈ 37% reward for standing still at v=0, so once
        # k→1 made motion costly the policy froze. This term pays for forward speed up to the
        # target. Clipped at the target → adds NO overshoot incentive (the Gaussian still
        # penalises going faster than 0.2 m/s).
        fwd_progress   = min(max(0.0, float(lin_vel[0])), float(self._target_lin_vel[0]))
        r_fwd_progress = _W_FWD_PROGRESS * fwd_progress * dt

        # --- overspeed penalty (v8, NOT curriculum-scaled) ---
        # The clipped progress reward is flat above the command and the σ²=0.04 Gaussian is
        # dead (gradient≈0) far above it, so v7 had NO force decelerating a sprint back to target.
        # This linear penalty gives a constant deceleration gradient at any speed above command.
        # Uses the live command (self._target_lin_vel) so it tracks the velocity curriculum.
        overspeed      = max(0.0, float(lin_vel[0]) - float(self._target_lin_vel[0]))
        r_overspeed    = -_W_OVERSPEED * overspeed * dt

        # --- efficiency penalties (curriculum-scaled by k_t) ---
        r_lin_vel_z   = -(lin_vel[2] ** 2)                                         * _W_LIN_VEL_Z   * dt
        joint_vel     = self._data.qvel[6:]
        joint_accel   = (joint_vel - self._prev_joint_vel) / dt
        r_joint_mot   = -float(np.dot(joint_accel, joint_accel)
                               + np.dot(joint_vel, joint_vel))                     * _W_JOINT_MOT   * dt
        torques       = self._data.actuator_force
        r_torques     = -float(np.dot(torques, torques))                           * _W_TORQUES     * dt
        act_diff      = action - self._prev_action
        r_action_rate = -float(np.dot(act_diff, act_diff))                         * _W_ACTION_RATE * dt
        r_collision   = -float(self._count_bad_collisions())                       * _W_COLLISION   * dt

        # --- structural terms (not curriculum-scaled) ---
        # v10: gate feet-air-time by command speed. At command 0 the robot should plant all 4 feet
        # and stand (no stepping reward → no roll-tipping); the reward ramps to full at the target so
        # a proper stepping gait is rewarded only when actually moving. Scales live with the curriculum.
        feet_air_gate = min(1.0, abs(float(self._target_lin_vel[0])) / _FEET_AIR_REF_SPEED)
        r_feet_air    = feet_air_bonus * _W_FEET_AIR * dt * feet_air_gate
        height_err    = float(self._data.qpos[2]) - _TARGET_BASE_HEIGHT
        r_base_height = -(height_err ** 2) * _W_BASE_HEIGHT * dt
        r_alive       = _W_ALIVE * dt

        total = (
            r_lin_vel + r_ang_vel + r_feet_air + r_base_height + r_alive
            + r_ang_vel_xy + r_orientation + r_backward_vel + r_pitch_pen
            + r_fwd_progress + r_overspeed   # stability + direction: always active
            + k * (r_lin_vel_z + r_joint_mot                  # efficiency: k-scaled
                   + r_torques + r_action_rate + r_collision)
        )

        info = {
            # per-term rewards
            "r_lin_vel":          r_lin_vel,
            "r_lin_vel_ungated":  r_lin_vel_ungated,
            "r_ang_vel":          r_ang_vel,
            "r_lin_vel_z":        r_lin_vel_z,
            "r_ang_vel_xy":       r_ang_vel_xy,
            "r_orientation":      r_orientation,
            "r_backward_vel":     r_backward_vel,
            "r_pitch_pen":        r_pitch_pen,
            "r_fwd_progress":     r_fwd_progress,
            "r_overspeed":        r_overspeed,
            "r_joint_mot":        r_joint_mot,
            "r_torques":          r_torques,
            "r_action_rate":      r_action_rate,
            "r_collision":        r_collision,
            "r_feet_air":         r_feet_air,
            "r_base_height":      r_base_height,
            "r_alive":            r_alive,
            # locomotion diagnostics
            "penalty_scale":       k,
            "stability_scale":     stability_scale,
            "command_lin_vel_x":   float(self._target_lin_vel[0]),
            "base_lin_vel_x":      float(lin_vel[0]),
            "forward_vel":         float(lin_vel[0]),   # alias kept for backward compat
            "tracking_error_x":    float(self._target_lin_vel[0] - lin_vel[0]),
            "base_height":         float(self._data.qpos[2]),
            "pitch":               pitch,
            "roll":                roll,
            "pitch_vel":           float(ang_vel[1]),
            "roll_vel":            float(ang_vel[0]),
            "contact_FR":          float(self._feet_contact[0]),
            "contact_FL":          float(self._feet_contact[1]),
            "contact_RR":          float(self._feet_contact[2]),
            "contact_RL":          float(self._feet_contact[3]),
            "feet_in_contact":     float(np.sum(self._feet_contact)),
            "step_count":          self._step_count,
        }
        return float(total), info

    # ------------------------------------------------------------------
    # Physics helpers
    # ------------------------------------------------------------------

    def _base_velocity_body_frame(self) -> tuple[np.ndarray, np.ndarray]:
        """Trunk linear and angular velocity rotated into the body frame."""
        rot = self._data.xmat[self._trunk_id].reshape(3, 3)
        lin = rot.T @ self._data.qvel[:3]
        ang = rot.T @ self._data.qvel[3:6]
        return lin, ang

    def _gravity_body_frame(self) -> np.ndarray:
        """Unit gravity vector expressed in the trunk body frame."""
        rot = self._data.xmat[self._trunk_id].reshape(3, 3)
        return rot.T @ np.array([0.0, 0.0, -1.0])

    def _foot_contacts(self) -> np.ndarray:
        """Binary (4,) array — True if each foot geom has an active contact."""
        in_contact = np.zeros(4, dtype=bool)
        foot_set   = set(self._foot_geom_ids.tolist())
        for i in range(self._data.ncon):
            c = self._data.contact[i]
            for fi, fid in enumerate(self._foot_geom_ids):
                if c.geom1 == fid or c.geom2 == fid:
                    in_contact[fi] = True
        return in_contact

    def _count_bad_collisions(self) -> int:
        """Count contacts where a non-foot robot geom touches anything."""
        foot_set = set(self._foot_geom_ids.tolist())
        count    = 0
        for i in range(self._data.ncon):
            c = self._data.contact[i]
            for geom_id in (c.geom1, c.geom2):
                if geom_id in foot_set:
                    continue
                body_id = int(self._model.geom_bodyid[geom_id])
                if body_id in self._robot_body_ids:
                    count += 1
                    break  # count each contact once
        return count

    def _is_terminated(self) -> bool:
        height = float(self._data.qpos[2])
        roll, pitch = self._roll_pitch()
        return (height < _MIN_HEIGHT
                or abs(roll)  > _MAX_ROLL
                or abs(pitch) > _MAX_PITCH)

    def _roll_pitch(self) -> tuple[float, float]:
        """Roll and pitch from the trunk free-joint quaternion (w, x, y, z)."""
        qw, qx, qy, qz = self._data.qpos[3:7]
        roll  = float(np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2)))
        pitch = float(np.arcsin(np.clip(2*(qw*qy - qz*qx), -1.0, 1.0)))
        return roll, pitch

    def _add_reset_noise(self) -> None:
        """Small noise on joint positions and all velocities at episode start."""
        self._data.qpos[7:] += self.np_random.uniform(-0.05, 0.05, 12)
        self._data.qvel[:]   = self.np_random.uniform(-0.05, 0.05, 18)

    # ------------------------------------------------------------------
    # Convenience: live viewer for interactive inspection
    # ------------------------------------------------------------------

    def launch_viewer(self) -> None:
        """
        Open the MuJoCo interactive viewer and run the simulation live.
        Blocks until the window is closed.
        Call from a script (not during training):
            env = Go1BaseEnv()
            env.reset()
            env.launch_viewer()
        """
        import mujoco.viewer
        mujoco.mj_resetDataKeyframe(self._model, self._data, 0)
        mujoco.mj_forward(self._model, self._data)
        with mujoco.viewer.launch_passive(self._model, self._data) as v:
            while v.is_running():
                mujoco.mj_step(self._model, self._data)
                v.sync()
