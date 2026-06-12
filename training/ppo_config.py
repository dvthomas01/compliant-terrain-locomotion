"""
Consolidated PPO hyperparameters for Policy A and B training.
All values sourced from the master_implementation_plan.md Section 11.
"""

from dataclasses import dataclass, field


@dataclass
class PPOConfig:
    # --- Parallel environments ---
    n_envs: int = 128                  # MacBook CPU equivalent of Rudin's GPU setup

    # --- PPO core (Rudin et al. 2022) ---
    n_steps: int = 24                  # steps per env per update (minimum viable for GAE)
    n_epochs: int = 5                  # PPO epochs per update
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    learning_rate: float = 1e-3        # initial; adjusted by AdaptiveLRCallback
    max_grad_norm: float = 1.0

    # --- Adaptive learning rate (Rudin et al. 2022 Algorithm 1) ---
    kl_target: float = 0.01
    lr_min: float = 1e-5
    lr_max: float = 1e-2

    # --- Penalty curriculum (Kumar et al. 2021 RMA) ---
    # k starts at 0.03, grows toward 1.0 each PPO iteration.
    # Without this the robot learns to stand still (zero motion = min penalty).
    penalty_k0: float = 0.03
    penalty_decay: float = 0.999       # k_new = min(1.0, k / decay)
    # 0.999 → k reaches 1.0 at ~10.8M steps (72% of 15M run)
    # 0.997 was wrong: k hit 1.0 at 3.6M steps (24%), crushing the reward signal too early

    # --- Training schedule ---
    total_timesteps: int = 15_000_000
    checkpoint_steps: list = field(
        default_factory=lambda: [5_000_000, 10_000_000, 15_000_000]
    )

    # --- Logging ---
    tensorboard_log: str = "runs/"

    @property
    def batch_size(self) -> int:
        """Total samples per PPO update = n_envs × n_steps."""
        return self.n_envs * self.n_steps
