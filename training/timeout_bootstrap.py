"""
Training utilities: penalty curriculum and adaptive learning rate.

Timeout bootstrapping note:
    SB3 2.x + gymnasium already handles this correctly. SubprocVecEnv
    automatically injects info["TimeLimit.truncated"] and
    info["terminal_observation"] from the gymnasium step return, and
    on_policy_algorithm bootstraps the value at timeout. No custom fix needed
    as long as the env returns truncated=True (not terminated=True) on timeout.
    Our Go1BaseEnv does this correctly at step_count >= MAX_STEPS.
"""

import os
from collections import defaultdict

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat


class PenaltyCurriculum:
    """
    Exponential penalty scale schedule from Kumar et al. 2021 (RMA).

    k starts at k0=0.03 and grows toward k_max each PPO iteration.
    The reward is:  R = tracking_terms + k * penalty_terms
    Starting at 3% penalty weight lets the robot learn to walk first,
    then gradually optimise for energy efficiency and smoothness.

    k_max caps the schedule below 1.0 (v7): runs v2–v6 all peaked at low k and
    then collapsed as k→1.0 — the efficiency penalties crushed locomotion before
    it stabilised. Capping k keeps those penalties from ever dominating.
    """

    def __init__(self, k0: float = 0.03, decay: float = 0.997, k_max: float = 1.0):
        self.k = k0
        self._decay = decay
        self._k_max = k_max

    def step(self) -> float:
        """Advance one PPO iteration. Returns the new scale."""
        self.k = min(self._k_max, self.k / self._decay)
        return self.k

    @property
    def value(self) -> float:
        return self.k


class CommandCurriculum:
    """
    Success-gated forward-command schedule (v8).

    Starts the commanded forward velocity at `start` (0 = learn to STAND/balance first,
    which is statically stable and easy to reach), then advances toward `target` by `step`
    each PPO iteration ONLY when the rolling mean episode length clears `advance` — i.e. the
    robot is stable at the current command. Retreats by `step` if ep_len falls below
    `retreat` (game-curriculum style). This makes a stable gait *reachable*: each command
    increase is a small perturbation of an already-balanced policy rather than a from-scratch
    discovery — addressing the v5–v7 failure where the policy never reached the stable basin.
    """

    def __init__(self, start: float = 0.0, target: float = 0.2, step: float = 0.02,
                 advance: float = 400.0, retreat: float = 150.0, track_frac: float = 0.0):
        self.cmd = start
        self._start = start
        self._target = target
        self._step = step
        self._advance = advance
        self._retreat = retreat
        self._track_frac = track_frac   # v12: also require vel_x >= track_frac*cmd to advance (0 = survival-only)

    def update(self, ep_len_mean: float, vel_mean: float = 0.0) -> float:
        tracking_ok = (self._track_frac <= 0.0) or (vel_mean >= self._track_frac * self.cmd)
        if ep_len_mean >= self._advance and tracking_ok and self.cmd < self._target:
            self.cmd = min(self._target, self.cmd + self._step)
        elif ep_len_mean < self._retreat and self.cmd > self._start:
            self.cmd = max(self._start, self.cmd - self._step)
        return self.cmd


class CommandCurriculumCallback(BaseCallback):
    """Advances CommandCurriculum once per PPO iteration from the rolling ep_len, and
    pushes the new forward command to every vectorised env (sets _target_lin_vel)."""

    def __init__(self, curriculum: CommandCurriculum, verbose: int = 0):
        super().__init__(verbose)
        self._curriculum = curriculum
        self._vel_sum = 0.0
        self._vel_count = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "base_lin_vel_x" in info:
                self._vel_sum += info["base_lin_vel_x"]
                self._vel_count += 1
        return True

    def _on_rollout_end(self) -> bool:
        import numpy as np
        buf = self.model.ep_info_buffer
        ep_len_mean = float(np.mean([e["l"] for e in buf])) if buf else 0.0
        vel_mean = (self._vel_sum / self._vel_count) if self._vel_count > 0 else 0.0
        cmd = self._curriculum.update(ep_len_mean, vel_mean)
        self.training_env.set_attr("_target_lin_vel", np.array([cmd, 0.0], dtype=np.float64))
        self.logger.record("train/command_vel", cmd)
        self._vel_sum = 0.0
        self._vel_count = 0
        return True


class PenaltyCurriculumCallback(BaseCallback):
    """
    SB3 callback that advances PenaltyCurriculum once per PPO iteration
    and propagates the new k_t to every vectorised environment.
    """

    def __init__(self, curriculum: PenaltyCurriculum, verbose: int = 0):
        super().__init__(verbose)
        self._curriculum = curriculum

    def _on_rollout_end(self) -> bool:
        k = self._curriculum.step()
        self.training_env.set_attr("penalty_scale", k)
        self.logger.record("train/penalty_scale", k)
        return True

    def _on_step(self) -> bool:
        return True


class AdaptiveLRCallback(BaseCallback):
    """
    KL-divergence based adaptive learning rate from Rudin et al. 2022 Algorithm 1.

    After each PPO update, reads the logged approx_kl and adjusts lr:
      - KL > 2 × target  →  lr ÷ 1.5  (step was too large, slow down)
      - KL < 0.5 × target →  lr × 1.5  (step was conservative, speed up)
      - Otherwise         →  lr unchanged
    """

    def __init__(
        self,
        kl_target: float = 0.01,
        lr_min: float = 1e-5,
        lr_max: float = 1e-2,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self._kl_target = kl_target
        self._lr_min = lr_min
        self._lr_max = lr_max
        self._current_lr: float = 1e-3

    def _on_training_start(self) -> None:
        lr = self.model.learning_rate
        self._current_lr = float(lr(1.0) if callable(lr) else lr)

    def _on_rollout_start(self) -> bool:
        # approx_kl is logged during train(), so it's available at next rollout start
        approx_kl = self.logger.name_to_value.get("train/approx_kl")
        if approx_kl is not None:
            if approx_kl > 2.0 * self._kl_target:
                self._current_lr = max(self._lr_min, self._current_lr / 1.5)
            elif approx_kl < 0.5 * self._kl_target:
                self._current_lr = min(self._lr_max, self._current_lr * 1.5)
            lr = self._current_lr  # capture only the float, not self (avoids cloudpickle error)
            self.model.lr_schedule = lambda _: lr
            self.logger.record("train/lr_adaptive", self._current_lr)
        return True

    def _on_step(self) -> bool:
        return True


class VideoRenderCallback(BaseCallback):
    """
    Periodically records the current policy and saves a GIF to disk.

    Runs `n_episodes` complete episodes (reset → fall/timeout) per GIF so
    you can see the full start-to-fall behaviour rather than an arbitrary
    mid-episode slice. Each episode is capped at `max_ep_steps` as a safety
    limit. Playback is slowed to 40 ms/frame (25 fps) — real-time at 50 Hz
    would be too fast to follow.

    Set `live_viewer=True` to also pop open the interactive MuJoCo window
    for ~5 seconds each time (blocks training briefly but lets you orbit/zoom).
    """

    def __init__(
        self,
        eval_env_factory,
        render_freq: int = 500_000,
        n_episodes: int = 3,        # complete episodes to record per GIF
        max_ep_steps: int = 1000,   # safety cap per episode (= env max)
        save_dir: str = "renders",
        live_viewer: bool = False,
        vecnorm=None,               # training VecNormalize: normalise eval obs with its running stats
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self._env_factory  = eval_env_factory
        self._render_freq  = render_freq
        self._n_episodes   = n_episodes
        self._max_ep_steps = max_ep_steps
        self._save_dir     = save_dir
        self._live_viewer  = live_viewer
        self._vecnorm      = vecnorm
        self._last_render  = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_render >= self._render_freq:
            self._record()
            self._last_render = self.num_timesteps
        return True

    def _record(self) -> None:
        from PIL import Image
        import mujoco.viewer as mjviewer

        os.makedirs(self._save_dir, exist_ok=True)
        env = self._env_factory()
        frames = []
        ep_lengths = []

        for ep in range(self._n_episodes):
            obs, _ = env.reset(seed=ep)
            ep_len = 0
            for _ in range(self._max_ep_steps):
                policy_obs = self._vecnorm.normalize_obs(obs) if self._vecnorm is not None else obs
                action, _ = self.model.predict(policy_obs, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(action)
                frame = env.render()
                if frame is not None:
                    frames.append(Image.fromarray(frame))
                ep_len += 1
                if terminated or truncated:
                    break   # episode done — start the next one
            ep_lengths.append(ep_len)

        env.close()

        step_m   = self.num_timesteps / 1_000_000
        label    = f"{step_m:.1f}M".replace(".0M", "M")
        gif_path = os.path.join(self._save_dir, f"policy_{label}.gif")

        if frames:
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=40,    # 40 ms/frame (25 fps) — easier to watch than real-time 50 Hz
                loop=0,
            )
            avg_len = sum(ep_lengths) / len(ep_lengths)
            print(
                f"\n[render] {gif_path}  "
                f"({len(frames)} frames across {self._n_episodes} episodes, "
                f"avg ep_len={avg_len:.0f} steps, {self.num_timesteps:,} total steps)"
            )

        if self._live_viewer:
            live_env = self._env_factory()
            live_env.reset(seed=0)
            with mjviewer.launch_passive(live_env._model, live_env._data) as v:
                import time
                t0 = time.time()
                while v.is_running() and time.time() - t0 < 5.0:
                    obs = live_env._get_obs()
                    if self._vecnorm is not None:
                        obs = self._vecnorm.normalize_obs(obs)
                    action, _ = self.model.predict(obs, deterministic=True)
                    live_env.step(action)
                    v.sync()
            live_env.close()


class RewardLoggingCallback(BaseCallback):
    """
    Logs per-term reward contributions and locomotion diagnostics to TensorBoard.

    Reads the info dict returned by Go1BaseEnv._compute_reward() every step,
    accumulates per-rollout sums, then records averages at rollout end.
    Reward terms appear under rewards/ and locomotion metrics under locomotion/.
    """

    _REWARD_KEYS = (
        "r_lin_vel", "r_lin_vel_ungated", "r_ang_vel", "r_lin_vel_z", "r_ang_vel_xy",
        "r_orientation", "r_backward_vel", "r_pitch_pen", "r_roll_pen", "r_fwd_progress", "r_overspeed",
        "r_joint_mot", "r_torques", "r_action_rate", "r_collision", "r_feet_air",
        "r_base_height", "r_alive", "r_termination",
    )
    _LOCO_KEYS = (
        "forward_vel", "base_lin_vel_x", "command_lin_vel_x", "tracking_error_x",
        "base_height", "penalty_scale", "stability_scale",
        "pitch", "roll", "pitch_vel", "roll_vel",
        "contact_FR", "contact_FL", "contact_RR", "contact_RL", "feet_in_contact",
        "term_height", "term_roll", "term_pitch",
    )

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._sums: dict[str, float] = defaultdict(float)
        self._count = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            for key in self._REWARD_KEYS + self._LOCO_KEYS:
                if key in info:
                    self._sums[key] += info[key]
        self._count += len(self.locals.get("infos", []))
        return True

    def _on_rollout_end(self) -> bool:
        if self._count == 0:
            return True
        for key in self._REWARD_KEYS:
            if key in self._sums:
                self.logger.record(f"rewards/{key}", self._sums[key] / self._count)
        for key in self._LOCO_KEYS:
            if key in self._sums:
                self.logger.record(f"locomotion/{key}", self._sums[key] / self._count)
        self._sums.clear()
        self._count = 0
        return True

    def _on_training_start(self) -> None:
        self._sums.clear()
        self._count = 0


class CheckpointAtStepsCallback(BaseCallback):
    """Save model checkpoints at specific total timestep milestones."""

    def __init__(self, steps: list[int], save_dir: str, prefix: str, verbose: int = 0):
        super().__init__(verbose)
        self._steps = sorted(steps)
        self._save_dir = save_dir
        self._prefix = prefix
        self._next_idx = 0

    def _on_step(self) -> bool:
        while (
            self._next_idx < len(self._steps)
            and self.num_timesteps >= self._steps[self._next_idx]
        ):
            milestone = self._steps[self._next_idx]
            label = f"{milestone // 1_000_000}M"
            path = os.path.join(self._save_dir, f"{self._prefix}_{label}")
            self.model.save(path)
            print(f"\n[checkpoint] {path}.zip  ({self.num_timesteps:,} steps)")
            self._next_idx += 1
        return True


class DistributionHistogramCallback(BaseCallback):
    """
    Writes per-rollout TensorBoard histograms of base_lin_vel_x and pitch.

    v6 diagnostic: the velocity histogram should be unimodal around the 0.2 m/s
    target. A bimodal shape (a stand-still cluster near 0 plus a sprint cluster)
    means the incentive structure is still broken — exactly the failure the
    tightened tracking Gaussian and dense pitch penalty are meant to remove.
    """

    _KEYS = ("base_lin_vel_x", "pitch")

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._vals: dict[str, list[float]] = {k: [] for k in self._KEYS}
        self._writer = None

    def _on_training_start(self) -> None:
        for fmt in self.logger.output_formats:
            if isinstance(fmt, TensorBoardOutputFormat):
                self._writer = fmt.writer

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            for key in self._KEYS:
                if key in info:
                    self._vals[key].append(float(info[key]))
        return True

    def _on_rollout_end(self) -> bool:
        if self._writer is not None:
            for key, vals in self._vals.items():
                if vals:
                    self._writer.add_histogram(
                        f"hist/{key}", np.asarray(vals), self.num_timesteps
                    )
        for key in self._KEYS:
            self._vals[key].clear()
        return True
