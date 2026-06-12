# Paper Analysis: Rudin et al. 2022
## Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning
**Authors:** Nikita Rudin, David Hoeller, Philipp Reist, Marco Hutter
**Venue:** Conference on Robot Learning (CoRL) 2022
**arXiv:** 2109.11978
**Code:** https://leggedrobotics.github.io/legged_gym/

---

## SECTION 1 — PROBLEM STATEMENT

**One-sentence summary:** Training locomotion policies for legged robots takes days to weeks with standard simulators — this paper shows that running thousands of parallel robots on a single GPU reduces this to minutes, enabling rapid iteration on the full sim-to-real pipeline.

Prior work on quadruped locomotion RL required training times of 12 hours (Lee et al. 2020), 82-88 hours (combined learning + optimal control approaches), and up to 120 hours (full learning approaches). OpenAI's Rubik's cube policy took months. This creates a fundamental bottleneck: when a single training run takes a day, hyperparameter tuning becomes prohibitively expensive, reward shaping is hard to iterate on, and the field moves slowly.

The root cause: standard simulators (MuJoCo, Bullet, RaiSim) are CPU-based. The simulation step, reward calculation, and observation computation all run on CPU. GPU is only used for the neural network forward and backward pass. The CPU-GPU data transfer bottleneck makes it impossible to saturate GPU utilization during training. With only hundreds of agents running in parallel, data collection is the limiting factor.

This paper fills the gap: by using NVIDIA's Isaac Gym — which runs simulation, observation, and reward computation entirely on GPU — and co-designing the training algorithm with the massively parallel regime in mind, they reduce training time by 2-3 orders of magnitude. Flat terrain in 4 minutes. Complex terrain in 20 minutes.

---

## SECTION 2 — CORE IDEA & INTUITION

The central insight is: **GPU parallelism is the bottleneck-breaker, but only if you redesign the data collection loop, the curriculum, and PPO hyperparameters to work at the regime of thousands of simultaneous agents.**

It's not just "run more robots." Three things have to change together:

1. **Move everything to GPU:** Simulation, observation, reward, inference — all on GPU. No PCIe transfers during the inner loop. Isaac Gym makes this possible; standard simulators don't.

2. **Redesign PPO for massive parallelism:** With 4096 robots, each robot only needs to take 24 steps per policy update (vs. hundreds in standard training) to produce enough samples for a meaningful batch. But this requires careful handling of episode timeouts — the critic needs bootstrapping at timeout to maintain correct value estimates.

3. **Game-inspired curriculum:** With thousands of robots, you get natural statistics about policy performance across all terrain difficulties simultaneously. Instead of a particle filter or a separate curriculum generator, you just track which terrain level each robot has reached and use that directly as the curriculum signal. Robots that succeed move up; robots that fail move down. No overhead, no tuning.

The reason simpler approaches fail: naively running more robots on CPU doesn't help because communication overhead dominates. And standard PPO curriculum approaches don't scale to thousands of parallel agents because they require trajectory rollout statistics that are expensive to collect per terrain type.

---

## SECTION 3 — METHODOLOGY

### 3a. Pipeline / Architecture

```
TRAINING PIPELINE (all on GPU)
================================
Environment setup:
  - 4096 robots simulated in parallel in a single Isaac Gym instance
  - Single terrain mesh with all terrain types tiled side-by-side
  - Each robot assigned a terrain type + difficulty level
  - Robots physically moved on mesh at reset — no mesh regeneration

Per training iteration (24 steps per robot per update):
  For t in range(nsteps=24):
    1. Inference: policy(observation) → joint_position_targets  [GPU]
    2. Actuator network: targets → torques  [GPU]
    3. PhysX simulation step  [GPU]
    4. Observation + reward computation  [GPU]
    5. Curriculum update: check if robot crossed terrain boundary
       - crossed: level += 1
       - moved < 50% of target distance: level -= 1
    6. Reset fallen robots (height < threshold OR body contact)
       - timeout reset: bootstrap critic target with V(s_{T+1})
       - failure reset: terminal state, V = 0
  
  Batch: 4096 robots × 24 steps = 98,304 transitions [all on GPU]
  
Policy update (PPO):
  - 5 epochs, minibatch size 24,576 (= 4096 × 6)
  - Clip range 0.2
  - Adaptive learning rate based on KL divergence
  - Entropy coefficient 0.01

Total: 1500 iterations × 98,304 = 147M transitions in ~20 minutes

DEPLOYMENT PIPELINE
====================
  - Read observations from robot sensors
  - Query terrain heights from LiDAR elevation map (108 values)
  - Feed to policy → joint position targets
  - Send directly to motor controllers
  - No filtering, no post-processing
```

**Control frequency:** 50 Hz policy, simulation at 200 Hz (4 simulation sub-steps per policy step at dt=0.005s).

### 3b. Key Algorithms & Formulas

**Batch size:**
```
B = n_robots × n_steps = 4096 × 24 = 98,304
```
- `n_robots`: number of parallel environments = 4096
- `n_steps`: steps per robot per policy update = 24 (minimum viable for GAE)
- Key insight: with 4096 robots, each robot only needs 24 steps per update to achieve a 98K batch. Minimum viable `n_steps` ≈ 25 (0.5s of simulated time at 50Hz) — below this, GAE estimates become unreliable.

**Timeout bootstrapping (critical for correctness):**
```
For normal reset (failure): target_value = reward
For timeout reset: target_value = reward + γ * V(s_{T+1})
```
- Standard PPO assumes episodes end at failure — critic target = terminal reward
- With time-limited episodes (20s max), many resets are timeouts, not failures
- Without bootstrapping: critic underestimates value at non-terminal timeouts → 10-20% reward loss
- This fix is NOT in standard Stable-Baselines PPO but IS in Spinning Up

**Adaptive learning rate:**
```
kl = KL(π_new, π_old)
if kl > 2 * kl*:
    α = max(1e-5, α / 1.5)    # too big a step: reduce lr
elif kl < 0.5 * kl*:
    α = min(1e-2, α * 1.5)    # too small a step: increase lr
# target kl* = 0.01
```
- Keeps policy updates in a reasonable trust region without hard clipping alone
- Allows faster learning when updates are conservative, prevents instability when aggressive

**Game-inspired curriculum:**
```
For each robot i at reset:
    if robot crossed terrain boundary:
        level[i] += 1           # robot succeeded: harder terrain
    elif distance_moved < 0.5 * target_distance * episode_time:
        level[i] -= 1           # robot failed: easier terrain
    
    if level[i] == max_level:
        level[i] = random.choice(all_levels)  # loop back for diversity
```
- No external tuning parameters
- Difficulty adapts per terrain type independently
- Uses robot population statistics directly as curriculum signal
- Loop-back at max level prevents catastrophic forgetting of easier terrains

**Reward function (from Table 2 in appendix):**
```
R = Σ_i w_i * r_i * dt    (all terms multiplied by timestep dt)
```

| Term | Weight | Formula | Notes |
|------|--------|---------|-------|
| Linear velocity tracking | 1·dt | φ(v*_xy - v_xy) | φ(x) = exp(-\|\|x\|\|²/0.25) — exponential tracking |
| Angular velocity tracking | 0.5·dt | φ(ω*_z - ω_z) | Same exponential form |
| Linear velocity penalty | 4·dt | -v²_z | Penalize vertical bobbing |
| Angular velocity penalty | 0.05·dt | -\|\|ω_xy\|\|² | Penalize roll/pitch rate |
| Joint motion | 0.001·dt | -\|\|q̈_j\|\|² - \|\|q̇_j\|\|² | Penalize acceleration + velocity |
| Joint torques | 0.00002·dt | -\|\|τ_j\|\|² | Energy efficiency |
| Action rate | 0.25·dt | -\|\|q̇*_j\|\|² | Penalize rapid target changes |
| Collisions | 0.001·dt | -n_collision | Count knee/shank/vertical contacts |
| Feet air time | 2·dt | Σ(t_air,f - 0.5) | Encourage longer strides |

**Tracking reward exponential form:**
```
φ(x) = exp(-||x||² / 0.25)
```
- Smooth, bounded reward for velocity tracking
- At x=0 (perfect tracking): φ=1.0
- At ||x||=0.5: φ≈0.37
- Numerically stable, no saturation issues unlike linear rewards

**Feet air time reward:**
```
r_feet = Σ_{f=0}^{3} (t_air,f - 0.5)
```
- `t_air,f`: time a foot has been in the air during current stance
- Positive when foot is airborne for > 0.5s
- Negative when foot is kept on ground too long
- This reward is what produces visually appealing longer strides — without it, the policy drags feet

### 3c. Design Decisions & Tradeoffs

**Why 4096 robots specifically?**
From Figure 4: there is a sweet spot between 2048-4096 robots with batch size ~100K. Below 2048: data diversity drops (too few samples from too few locations). Above 8192: performance degrades sharply because each robot only gets ~12 steps per update — too short for meaningful GAE. 4096 is empirically optimal for this task.

**Why 24 steps minimum?**
Below ~25 steps (0.5s at 50Hz), GAE estimates become unreliable because the advantage calculation requires enough temporal context to distinguish good from bad actions. This is a fundamental constraint of on-policy methods — it sets the floor on `n_steps` regardless of parallelism.

**Why not use particle filter curriculum (as in Lee et al. 2020)?**
With 4096 parallel robots, you already have natural statistics about policy performance across all terrain difficulties. The particle filter adds computational overhead and tuning parameters that are unnecessary when performance statistics are available directly from the robot population. The game-inspired approach uses zero additional compute.

**Single terrain mesh vs. per-robot terrain:**
Standard approach: generate unique terrain per robot. Problem at scale: regenerating thousands of meshes is expensive. Solution: one large tiled mesh with all terrain types at all difficulty levels placed side by side. Robots are moved to appropriate terrain squares at reset. This enables arbitrary terrain diversity with zero regeneration overhead.

**Actuator model simplification:**
Lee et al. 2020 used a complex actuator network with joint error history at multiple timesteps. This paper uses an LSTM that only sees current measurements. Slightly less accurate but simpler. The paper acknowledges this tradeoff: "A potential drawback of this set-up is that the policy does not have the temporal information of the actuators as in previous work."

**No motion primitives / no gait specification:**
Unlike Lee et al. (PMTG with foot trajectory generators) and RMA (PMTG optional), this paper uses raw joint position targets with no locomotion priors. The policy is free to discover any gait. Result: always converges to trotting anyway, but occasionally produces artifacts (dragging leg, wrong base height). Reward tuning is required to eliminate these.

**Friction range for sim-to-real:**
Each robot gets friction uniformly sampled from [0.5, 1.25]. This is narrower than RMA's [0.05, 4.5]. The paper focuses on sim-to-real via perceptive terrain sensing rather than physics parameter robustness — a different philosophy from RMA.

### 3d. Reward Function

Full reward from Table 2 (Appendix):

The exponential tracking reward φ(x) = exp(-||x||²/0.25) is the most important design choice. It provides:
- Smooth gradients everywhere (unlike sparse binary rewards)
- Natural saturation — perfect tracking gives 1.0, far from target asymptotes to 0
- No manual tuning of "good enough" threshold

The feet air time term (`Σ(t_air,f - 0.5)`) is what produces natural-looking gaits. Without it, the policy learns to shuffle with minimal foot lifting — technically stable but visually unnatural and less efficient on rough terrain.

All terms are multiplied by `dt` — this makes the reward rate-independent and scales naturally with control frequency changes.

---

## SECTION 4 — IMPLEMENTATION GUIDE

### 4a. Step-by-step pseudocode

```
# ============================================================
# SETUP
# ============================================================

initialize:
    n_robots = 4096
    n_steps = 24        # steps per robot per PPO update
    n_epochs = 5
    batch_size = n_robots * n_steps  # = 98,304
    mini_batch_size = n_robots * 6   # = 24,576

    # Create single large terrain mesh: all types × all difficulties tiled
    terrain_mesh = create_tiled_terrain_mesh(
        types=['flat', 'rough', 'slopes', 'stairs', 'obstacles'],
        difficulties=range(0, max_level),
        tile_size=8.0  # 8m × 8m tiles
    )

    # Assign each robot to a terrain type and starting level
    robot_terrain_type = [i % n_terrain_types for i in range(n_robots)]
    robot_terrain_level = [0] * n_robots  # all start at level 0

    # Initialize all robots on appropriate terrain squares
    reset_all_robots(robot_terrain_type, robot_terrain_level)

    # PPO
    policy = MLP(obs_dim, [512, 256, 128], action_dim=12)
    value_fn = MLP(obs_dim, [512, 256, 128], 1)
    lr = 1e-3  # adaptive
    kl_target = 0.01

# ============================================================
# TRAINING LOOP
# ============================================================

for iteration in range(1500):
    
    # ---- DATA COLLECTION (all GPU) ----
    batch_obs, batch_acts, batch_rews, batch_dones, batch_timeouts = [], ...
    
    for step in range(n_steps=24):
        obs = get_observations_all_robots()  # [n_robots, obs_dim] on GPU
        
        with torch.no_grad():
            actions, log_probs, values = policy(obs)
        
        # Send actions to actuator network → torques → simulation step
        next_obs, rewards, dones, timeouts = env_step(actions)
        
        batch_obs.append(obs)
        batch_acts.append(actions)
        batch_rews.append(rewards)
        batch_dones.append(dones)
        batch_timeouts.append(timeouts)
        
        # Update curriculum for robots that reset
        for i in range(n_robots):
            if dones[i]:
                if timeouts[i]:
                    # Timeout: check locomotion progress
                    if distance_moved[i] > 0.5 * target_distance[i]:
                        robot_terrain_level[i] = min(robot_terrain_level[i]+1, max_level)
                        # Locomotion success: advance to harder terrain
                        # Check if reached border of terrain tile
                    else:
                        robot_terrain_level[i] = max(0, robot_terrain_level[i]-1)
                    if robot_terrain_level[i] == max_level:
                        robot_terrain_level[i] = random.randint(0, max_level)
                reset_robot(i, robot_terrain_type[i], robot_terrain_level[i])
    
    # ---- COMPUTE ADVANTAGES (GAE) ----
    # Critical: bootstrap value at timeout, not at failure
    last_values = value_fn(next_obs)
    
    advantages = torch.zeros(n_robots, n_steps)
    returns = torch.zeros(n_robots, n_steps)
    gae = torch.zeros(n_robots)
    
    for t in reversed(range(n_steps)):
        if t == n_steps - 1:
            next_value = last_values
        else:
            next_value = batch_values[t+1]
        
        # KEY: only bootstrap on timeout, not on failure
        # dones[t] = True on both timeout AND failure
        # timeouts[t] = True only on timeout
        mask = 1 - batch_dones[t].float()
        timeout_mask = batch_timeouts[t].float()
        
        # For timeout: next_value is valid (robot didn't actually fail)
        # For failure: next_value = 0 (episode truly ended)
        effective_next_value = (mask + timeout_mask) * next_value
        
        delta = batch_rews[t] + gamma * effective_next_value - batch_values[t]
        gae = delta + gamma * lambda_ * mask * gae  # don't propagate through failures
        advantages[:, t] = gae
        returns[:, t] = advantages[:, t] + batch_values[t]
    
    # ---- PPO UPDATE ----
    for epoch in range(n_epochs=5):
        for minibatch in split_randomly(batch, mini_batch_size=24576):
            obs_b, act_b, adv_b, ret_b, logp_old_b = minibatch
            
            logp_new, entropy, values_new = policy.evaluate(obs_b, act_b)
            
            ratio = torch.exp(logp_new - logp_old_b)
            surr1 = ratio * adv_b
            surr2 = torch.clamp(ratio, 1-0.2, 1+0.2) * adv_b
            policy_loss = -torch.min(surr1, surr2).mean()
            
            value_loss = 0.5 * (values_new - ret_b).pow(2).mean()
            entropy_loss = -0.01 * entropy.mean()
            
            loss = policy_loss + value_loss + entropy_loss
            loss.backward()
            optimizer.step()
    
    # ---- ADAPTIVE LEARNING RATE ----
    kl = compute_kl(policy, old_policy)
    if kl > 2 * kl_target:
        lr = max(1e-5, lr / 1.5)
    elif kl < 0.5 * kl_target:
        lr = min(1e-2, lr * 1.5)
    update_optimizer_lr(optimizer, lr)
```

### 4b. Python implementation skeleton

```python
import torch
import torch.nn as nn
import numpy as np

# ============================================================
# NETWORK ARCHITECTURES
# ============================================================

class ActorCritic(nn.Module):
    """Simple MLP actor-critic for locomotion.
    No motion primitives — raw joint position targets as output."""
    
    def __init__(self, obs_dim, action_dim=12, hidden_sizes=[512, 256, 128]):
        super().__init__()
        
        # Shared feature extractor (optional — can separate actor/critic)
        self.actor = self._build_mlp(obs_dim, hidden_sizes, action_dim)
        self.critic = self._build_mlp(obs_dim, hidden_sizes, 1)
        
        # Log standard deviation for stochastic policy
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def _build_mlp(self, in_dim, hidden_dims, out_dim):
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ELU()])
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        return nn.Sequential(*layers)
    
    def forward(self, obs):
        """Return action mean, value estimate."""
        action_mean = self.actor(obs)
        value = self.critic(obs).squeeze(-1)
        return action_mean, value
    
    def get_action(self, obs):
        """Sample action and return log probability."""
        action_mean, value = self.forward(obs)
        std = self.log_std.exp()
        dist = torch.distributions.Normal(action_mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, value
    
    def evaluate(self, obs, action):
        """Evaluate log prob, entropy, value for stored action."""
        action_mean, value = self.forward(obs)
        std = self.log_std.exp()
        dist = torch.distributions.Normal(action_mean, std)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value


# ============================================================
# OBSERVATION SPACE
# ============================================================

def get_observation(robot_state, terrain_heights):
    """
    Full observation space.
    Proprioceptive: base velocities, gravity, joints, previous actions
    Exteroceptive: 108 terrain height measurements around base
    Total: ~235 dimensions
    """
    obs = torch.cat([
        robot_state.base_linear_vel,          # 3: v_x, v_y, v_z
        robot_state.base_angular_vel,          # 3: ω_x, ω_y, ω_z
        robot_state.projected_gravity,         # 3: gravity direction in base frame
        robot_state.commands,                  # 3: vx_cmd, vy_cmd, ωz_cmd
        robot_state.joint_positions - robot_state.default_joint_positions,  # 12
        robot_state.joint_velocities,          # 12
        robot_state.previous_actions,          # 12
        terrain_heights,                       # 108: 12x9 grid around base
    ], dim=-1)
    # Total: 3+3+3+3+12+12+12+108 = 156 dims
    # (exact depends on implementation choices)
    return obs


# ============================================================
# REWARD FUNCTION
# ============================================================

def compute_reward(state, next_state, action, prev_action, dt=0.02):
    """
    9-term reward function from Table 2.
    All terms multiplied by dt for rate-independence.
    φ(x) = exp(-||x||²/0.25) for tracking terms.
    """
    
    def phi(x, sigma=0.25):
        """Exponential tracking reward."""
        return torch.exp(-x.pow(2).sum(-1) / sigma)
    
    v = next_state.base_linear_vel   # [n_robots, 3]
    omega = next_state.base_angular_vel  # [n_robots, 3]
    v_cmd = state.commands[:, :2]    # commanded vx, vy
    omega_cmd = state.commands[:, 2:3]  # commanded ωz
    
    # 1. Linear velocity tracking (x,y)
    r_lin_vel = phi(v[:, :2] - v_cmd) * 1.0 * dt
    
    # 2. Angular velocity tracking (yaw)
    r_ang_vel = phi(omega[:, 2:3] - omega_cmd) * 0.5 * dt
    
    # 3. Penalize vertical velocity
    r_lin_vel_z = -(v[:, 2].pow(2)) * 4.0 * dt
    
    # 4. Penalize roll/pitch rate
    r_ang_vel_xy = -(omega[:, :2].pow(2).sum(-1)) * 0.05 * dt
    
    # 5. Penalize joint motion (acceleration + velocity)
    joint_acc = (state.joint_velocities - next_state.joint_velocities) / dt
    r_joint_motion = -(joint_acc.pow(2).sum(-1) + next_state.joint_velocities.pow(2).sum(-1)) * 0.001 * dt
    
    # 6. Penalize joint torques
    r_torques = -(next_state.joint_torques.pow(2).sum(-1)) * 0.00002 * dt
    
    # 7. Penalize rapid action changes (action rate)
    r_action_rate = -((action - prev_action).pow(2).sum(-1)) * 0.25 * dt
    
    # 8. Penalize collisions (knees, shanks, vertical foot contacts)
    r_collision = -(next_state.n_collisions.float()) * 0.001 * dt
    
    # 9. Reward feet air time (encourages longer strides)
    # t_air: time each foot has been in air during current stance phase
    r_feet_air = ((next_state.feet_air_time - 0.5).sum(-1)) * 2.0 * dt
    
    total_reward = (r_lin_vel + r_ang_vel + r_lin_vel_z + r_ang_vel_xy +
                    r_joint_motion + r_torques + r_action_rate +
                    r_collision + r_feet_air)
    
    return total_reward


# ============================================================
# GAME-INSPIRED CURRICULUM
# ============================================================

class GameInspiredCurriculum:
    """Automatically adjusts terrain difficulty based on robot performance.
    No external tuning parameters. Works with thousands of parallel robots."""
    
    def __init__(self, n_robots, n_terrain_types, max_level=10):
        self.n_robots = n_robots
        self.n_terrain_types = n_terrain_types
        self.max_level = max_level
        
        # Each robot tracks its terrain type and current level
        self.terrain_type = torch.arange(n_robots) % n_terrain_types
        self.terrain_level = torch.zeros(n_robots, dtype=torch.long)
        
        # Track locomotion progress for curriculum decision
        self.episode_start_pos = torch.zeros(n_robots, 2)
        
    def update(self, robot_positions, robot_commands, dones, timeouts):
        """Update curriculum levels for robots that just reset."""
        reset_robots = dones.nonzero(as_tuple=True)[0]
        
        for i in reset_robots:
            if timeouts[i]:
                # Timeout: evaluate locomotion progress
                distance_moved = torch.norm(robot_positions[i] - self.episode_start_pos[i])
                target_distance = torch.norm(robot_commands[i, :2]) * 20.0  # 20s episode
                
                if distance_moved > 0.5 * target_distance:
                    # Success: advance difficulty
                    self.terrain_level[i] = min(self.terrain_level[i] + 1, self.max_level)
                else:
                    # Failure: reduce difficulty
                    self.terrain_level[i] = max(0, self.terrain_level[i] - 1)
                
                # Loop back at max level to avoid catastrophic forgetting
                if self.terrain_level[i] == self.max_level:
                    self.terrain_level[i] = torch.randint(0, self.max_level, (1,))
            
            # Failure reset: don't change level (robot fell — not a fair terrain assessment)
            # Update start position for new episode
            self.episode_start_pos[i] = robot_positions[i]
    
    def get_terrain_assignment(self, robot_idx):
        """Get terrain type and level for a robot."""
        return self.terrain_type[robot_idx], self.terrain_level[robot_idx]
    
    def get_progress_stats(self):
        """Return distribution of robots across levels — training progress indicator."""
        level_counts = torch.bincount(self.terrain_level, minlength=self.max_level+1)
        return level_counts / self.n_robots


# ============================================================
# TIMEOUT BOOTSTRAPPING (critical for correctness)
# ============================================================

def compute_gae_with_timeout_bootstrap(rewards, values, dones, timeouts,
                                        gamma=0.99, lambda_=0.95):
    """
    Compute GAE advantages with correct handling of episode timeouts.
    
    KEY DISTINCTION:
    - Normal termination (fall/crash): future value = 0. Don't bootstrap.
    - Timeout termination: robot didn't actually fail. Bootstrap with V(s_{T+1}).
    
    Without this: critic underestimates value at timeout → policy too conservative
    Effect: ~10-20% reward improvement (from Figure 10 in paper)
    
    Args:
        rewards: [n_robots, n_steps]
        values: [n_robots, n_steps]  
        dones: [n_robots, n_steps] — True for BOTH timeout AND failure
        timeouts: [n_robots, n_steps] — True ONLY for timeout
        last_values: [n_robots] — V(s_{T+1}) for last timestep
    """
    n_robots, n_steps = rewards.shape
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(n_robots)
    
    # Get value of next state after last step
    last_values = values[:, -1]  # or compute from policy
    
    for t in reversed(range(n_steps)):
        if t == n_steps - 1:
            next_val = last_values
        else:
            next_val = values[:, t + 1]
        
        # For timeout: next state is valid → bootstrap
        # For failure: episode truly done → no bootstrap
        # For non-terminal: normal GAE → bootstrap
        non_terminal = 1.0 - dones[:, t].float()
        is_timeout = timeouts[:, t].float()
        
        # Bootstrap if: not done (non_terminal) OR it's a timeout
        bootstrap_mask = non_terminal + is_timeout * dones[:, t].float()
        effective_next_val = bootstrap_mask * next_val
        
        delta = rewards[:, t] + gamma * effective_next_val - values[:, t]
        gae = delta + gamma * lambda_ * non_terminal * gae
        advantages[:, t] = gae
    
    returns = advantages + values
    return advantages, returns


# ============================================================
# PPO WITH ADAPTIVE LEARNING RATE
# ============================================================

def ppo_update(policy, optimizer, batch, clip_eps=0.2, value_coeff=1.0,
               entropy_coeff=0.01, n_epochs=5, mini_batch_size=24576,
               kl_target=0.01):
    """PPO update with adaptive learning rate."""
    
    obs_b, act_b, adv_b, ret_b, logp_old_b = batch
    
    # Normalize advantages
    adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)
    
    total_kl = 0
    n_updates = 0
    
    for epoch in range(n_epochs):
        # Random permutation for minibatches
        perm = torch.randperm(len(obs_b))
        
        for start in range(0, len(obs_b), mini_batch_size):
            idx = perm[start:start+mini_batch_size]
            mb_obs = obs_b[idx]
            mb_act = act_b[idx]
            mb_adv = adv_b[idx]
            mb_ret = ret_b[idx]
            mb_logp_old = logp_old_b[idx]
            
            logp_new, entropy, values_new = policy.evaluate(mb_obs, mb_act)
            
            # PPO clipped objective
            ratio = torch.exp(logp_new - mb_logp_old)
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = value_coeff * (values_new - mb_ret).pow(2).mean()
            
            # Entropy bonus (encourage exploration)
            entropy_loss = -entropy_coeff * entropy.mean()
            
            loss = policy_loss + value_loss + entropy_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            
            # Track KL for adaptive lr
            with torch.no_grad():
                kl = (mb_logp_old - logp_new).mean()
                total_kl += kl.item()
                n_updates += 1
    
    # Adaptive learning rate update
    avg_kl = total_kl / n_updates
    return avg_kl


def adaptive_lr_update(optimizer, avg_kl, lr, kl_target=0.01):
    """Update learning rate based on KL divergence."""
    if avg_kl > 2 * kl_target:
        lr = max(1e-5, lr / 1.5)
    elif avg_kl < 0.5 * kl_target:
        lr = min(1e-2, lr * 1.5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
```

### 4c. Key Hyperparameters

| Parameter | Value | Controls | Sensitivity |
|-----------|-------|---------|------------|
| n_robots | 4096 | Parallelism level | High — optimal range 2048-4096 for this task. Below 1024: data diversity drops. Above 8192: performance degradation |
| n_steps | 24 | Steps per robot per update | High — minimum ~25 for viable GAE estimates at 50Hz |
| batch_size | 98,304 (4096×24) | Total samples per update | Medium — larger is better but scales training time |
| mini_batch_size | 24,576 (4096×6) | Gradient update chunk | Low — larger mini-batches are better for GPU utilization in this regime |
| n_epochs | 5 | PPO reuse factor | Low |
| clip_eps | 0.2 | PPO trust region | Low — standard |
| γ (discount) | 0.99 | Effective horizon | Medium — 0.99 means ~100 step horizon at 50Hz = ~2s |
| λ_GAE | 0.95 | Advantage smoothing | Low |
| kl_target | 0.01 | Adaptive lr target | Medium |
| entropy_coeff | 0.01 | Exploration bonus | Low — standard |
| Control frequency | 50 Hz | Policy rate | High — determines minimum n_steps for GAE |
| Simulation sub-steps | 4 | Stability vs. speed | Medium — below 4 (dt>0.005s), actuator network becomes unstable |
| Friction range | [0.5, 1.25] | Sim-to-real physics range | Medium — narrower than RMA, focused on normal surfaces |
| Push magnitude | ±1 m/s | Disturbance robustness | Medium |
| Push interval | every 10s | Push frequency | Low |
| Max episode length | 20s (1000 steps at 50Hz) | Exploration horizon | Low |
| Terrain tile size | 8m × 8m | Terrain scale | Low — chosen for ANYmal C kinematics |
| Max stair height | 20cm | Hardest terrain difficulty | Medium — close to kinematic limits of ANYmal |
| Feet air time target | 0.5s | Stride length | Medium — tune for natural gait |

### 4d. Data Requirements

**Observation space (~156 dimensions):**
- Base linear velocity: 3
- Base angular velocity: 3
- Projected gravity vector: 3
- Velocity commands: 3 (vx, vy, ωz)
- Joint positions (relative to default): 12
- Joint velocities: 12
- Previous actions: 12
- Terrain height grid: 108 (12×9 grid, each = distance from terrain to base height)

**Observation noise (Table 4) — applied during training for robustness:**
- Joint positions: ±0.01 rad
- Joint velocities: ±1.5 rad/s
- Base linear velocity: ±0.01 m/s
- Base angular velocity: ±0.2 rad/s
- Projected gravity: ±0.05 rad/s²
- Terrain heights: ±0.1 m (significant — reflects LiDAR noise in real deployment)

**Action space:** 12 desired joint positions (sent to actuator network → torques)

**Terrain types:** flat, randomly rough (±0.1m), slopes (0-25°), stairs (5-20cm steps), discrete obstacles (up to ±0.2m)

### 4e. Dependencies

| Tool | Purpose | Notes |
|------|---------|-------|
| Isaac Gym | GPU-accelerated physics simulation | **Required for full speedup.** Not available on macOS. Linux + NVIDIA GPU only. |
| MuJoCo | Alternative for CPU training | What you'll use on macOS. Full training works but much slower. |
| PyTorch | Neural network training | Must be CUDA version for GPU training |
| legged_gym | Reference implementation | Open-sourced by authors at the URL in the paper |
| ANYmal URDF | Robot model | Or use Unitree A1/Go1 URDF from Unitree GitHub |

**Critical note for your project:** Isaac Gym runs on Linux with NVIDIA GPU only — NOT macOS. For development on MacBook, use MuJoCo with CPU. For actual training runs, use a cloud GPU instance (Lambda Labs, Vast.ai, or RunPod). The curriculum design and reward function are fully portable to MuJoCo.

---

## SECTION 5 — BENCHMARK RESULTS

### 5a. Evaluation environments
- Simulation: 5 terrain types at increasing difficulty levels
- Real world: ANYmal C hardware deployment on stairs, obstacles
- Multiple robot architectures: ANYmal B, ANYmal C, ANYmal C with arm, Unitree A1, Cassie

### 5b. Metrics
- **Training time:** Primary metric — wall-clock minutes to convergence
- **Total episode reward:** Policy quality
- **Success rate:** Fraction of robots completing terrain traversal (forward velocity command, base must not contact ground)
- **Stair traversal success rate:** At increasing step heights up to 20cm

### 5c. Quantitative results

**Training time comparison:**
| Approach | Training time |
|---------|--------------|
| Blind locomotion (Lee et al. 2020) | 12 hours |
| Perceptive locomotion (Miki et al. 2021) | 120 hours |
| **This work (flat terrain)** | **< 4 minutes** |
| **This work (complex terrain)** | **< 20 minutes** |

**Speedup: 2-3 orders of magnitude.**

**Optimal configuration:** 4096 robots, batch size 98,304 — best performance/time tradeoff (Figure 4c).

**Simulation success rates (Figure 5):**
- Stairs: ~100% at step heights up to 20cm (max training level)
- Discrete obstacles: decreasing from ~100% at 5cm to ~60% at 20cm
- Slopes: ~100% below 25°, decreasing above (robot slides down gracefully rather than walking)

**Effect of timeout bootstrapping:** +10-20% total reward on both flat and rough terrain (Figure 10).

**GPU scaling:** Near-linear training time reduction up to ~4000 robots, plateaus above (Figure 4b).

### 5d. Qualitative findings
- Policy converges to trotting gait without being specified
- Occasional artifacts without reward tuning: dragging leg, wrong base height
- Generalization to untrained configurations: A1 (4× lighter, different kinematics) works with minor adaptation
- Cassie (bipedal) requires one additional "single foot stance" reward term but otherwise same setup

### 5e. Comparison to baselines
No direct performance comparison to Lee et al. or RMA in this paper — the contribution is about training speed, not peak performance. The paper acknowledges: "The purpose of this work is not to obtain the absolute best-performing policy with the highest robustness."

### 5f. Ablation studies
- **n_robots effect (Figure 4):** Sharp performance drop above ~8192 robots (too few steps per robot). Gradual decrease below ~1024 (low data diversity). Optimal at 2048-4096.
- **Timeout bootstrapping (Figure 10):** +10-20% total reward. Critic loss significantly reduced. Not implementing this is a known but often-ignored mistake.
- **Batch size effect (Figure 4a):** Larger batch = better performance but linearly longer training. 98K is optimal tradeoff.

---

## SECTION 6 — WHEN TO USE THIS METHOD

### 6a. Ideal conditions
- You have access to an NVIDIA GPU with ≥9GB VRAM (for 4096 robots with rendering)
- You need to iterate quickly on reward functions, hyperparameters, or curriculum design
- You want a clean open-source implementation to build from (legged_gym is well-maintained)
- Your robot model is importable as a URDF into Isaac Gym

### 6b. Practical sweet spot
Any project where you expect to run multiple training experiments. The 20-minute training time means you can do in an afternoon what would take weeks with standard approaches. For a summer project with limited hardware access, this is transformative.

### 6c. Scale considerations
- Designed for medium quadrupeds (ANYmal scale, 30-40kg). A1 (12kg) works with minor tuning.
- Terrain tiles are 8m × 8m — sized for ANYmal. Scale up for larger robots.
- GPU VRAM: 6GB minimum (without rendering), 9GB recommended (with rendering) for 4096 robots

### 6d. Computational profile
- Isaac Gym on Linux + NVIDIA GPU: 20 minutes for complex terrain
- MuJoCo on CPU (MacBook): likely 2-4 hours for equivalent training
- Inference: fast MLP, runs at 50Hz on any hardware including robot's onboard CPU

---

## SECTION 7 — WHEN NOT TO USE THIS METHOD

### 7a. Failure modes
- Without reward tuning: dragging legs, abnormal base heights (artifacts visible even after training)
- On extreme slopes (>25°): robot slides, doesn't walk uphill reliably
- With imperfect terrain height maps (real deployment): robustness decreases
- On deformable terrain: standard Isaac Gym doesn't simulate deformable ground

### 7b. Known limitations (from authors)
- Not aimed at maximum robustness — Lee et al. teacher-student achieves better robustness
- Terrain height map quality limits sim-to-real transfer at high speeds
- Maximum commanded speed reduced from 0.75m/s (sim) to 0.6m/s (real) due to map imperfections

### 7c. Hidden assumptions
- Terrain is rigid and static (major assumption for your project!)
- Height map accurately reflects terrain geometry
- Friction range [0.5, 1.25] covers deployment conditions
- No deformable contact (foam, pillows, sand not modeled)

### 7d. Simpler baselines sometimes better
- For a single known terrain: MPC or hand-tuned controller may be more reliable
- For offline development without GPU: MuJoCo with standard PPO is equivalent algorithmically

---

## SECTION 8 — EXPECTED RESULTS

### 8a. What good output looks like
- Stable trotting gait in ~500 curriculum iterations
- Robots visibly spreading across terrain difficulty levels in Figure 3 style
- Nearly 100% stair success at step heights ≤ 20cm
- Policy transfers to real hardware with minimal modification

### 8b. Characteristic artifacts (even when working correctly)
- Slight dragging of one leg unless feet air time reward is well-tuned
- Trotting gait always (no walking, no galloping) — the policy converges here reliably
- Conservative base height on uneven terrain

### 8c. What failure looks like
- Robots all stuck at level 0 after 500+ iterations: curriculum not triggering, likely reward collapse (check penalty term balance)
- Policy converges but doesn't transfer to harder terrain: max_level too high for current training time, or timeout bootstrapping missing
- NaN loss: observation normalization missing, or learning rate too high

### 8d. Realistic numbers
- Training time: 4 min (flat), 20 min (complex) on RTX A6000
- On MacBook CPU with MuJoCo: ~2-4 hours per equivalent training run
- Stair success at 20cm steps: ~100%
- Slope success at 25°: ~100%

---

## SECTION 9 — CONNECTIONS TO OTHER WORK

### 9a. Key predecessors
- Lee et al. 2020 (Science Robotics) — architecture inspiration, adaptive curriculum concept
- Hwangbo et al. 2019 — learned actuator network
- Isaac Gym (Makoviychuk et al. NeurIPS 2021) — the enabling technology
- Schulman et al. 2017 — PPO base algorithm
- Schulman et al. 2016 — GAE for advantage estimation

### 9b. Key successors
- Miki et al. 2022 — combined perceptive locomotion using legged_gym framework
- Many 2022-2023 papers use legged_gym as their starting point

### 9c. Position in field
**Infrastructure milestone paper.** This paper didn't introduce novel algorithms — it showed that training infrastructure determines research velocity more than algorithmic novelty. By open-sourcing legged_gym and demonstrating 20-minute training, it became the standard starting point for all subsequent legged locomotion research. If you're building a quadruped RL project in 2024-2025, you start from legged_gym.

---

## SECTION 10 — PROJECT APPLICATION MAPPING

**Your project:** Train PPO quadruped on Level 1 (contact parameter variation) and Level 2 (soft patches) terrain in MuJoCo on MacBook. Compare randomized vs. rigid baseline.

### 10a. How this paper applies
This paper is your **training infrastructure reference** and **curriculum design reference**. You are not using Isaac Gym (macOS incompatible) but you are using the same curriculum logic, reward function, and PPO hyperparameters. The game-inspired curriculum is the most directly transferable contribution.

### 10b. What is directly usable vs. what needs adaptation

| Component | Usable | Notes |
|-----------|--------|-------|
| Game-inspired curriculum | ✅ Direct | Fully portable to MuJoCo. This is the key thing to take from this paper. |
| 9-term reward function (Table 2) | ✅ Direct | Use as-is. Exponential tracking φ(x) is clean and well-tuned. |
| Observation space | ✅ Adapt | Drop terrain heights for your Level 1 (flat ground with varied params). Add back for Level 2. |
| Timeout bootstrapping | ✅ Direct | Critical — implement this in your PPO or use a library that does it (Spinning Up, not Stable-Baselines) |
| Adaptive learning rate | ✅ Direct | Copy Algorithm 1 exactly. Simple and effective. |
| Hyperparameters (Table 3) | ✅ Adapt | Scale n_robots down for CPU: use 128-512 on MacBook |
| Isaac Gym / GPU pipeline | ❌ Replace | Use MuJoCo + CPU. Accept slower training, use cloud for full runs. |
| Observation noise (Table 4) | ✅ Direct | Apply during training for robustness. |

### 10c. The curriculum applied to your compliance terrain

Your compliance curriculum is a direct translation of the game-inspired curriculum:

```
Level 0: Flat rigid ground (your Policy A baseline)
Level 1: Rigid ground + friction variation [0.5, 1.25]
Level 2: Rigid ground + full contact param randomization (Level 1 terrain)
Level 3: Soft patches (light compliance, solimp step 1)
Level 4: Soft patches (medium compliance, solimp step 2)
Level 5: Mixed rigid + soft patches (Level 2 terrain)
Level 6: Full compliance spectrum (hardest)

Curriculum rule:
  - Robot crosses terrain boundary → level += 1
  - Robot moves < 50% target distance → level -= 1
  - At max level → loop back to random level
```

This gives you a principled, auto-tuning difficulty schedule for your compliance curriculum without requiring the particle filter complexity of Lee et al.

### 10d. MuJoCo-specific adaptations

1. **Scale n_robots down:** On MacBook CPU, use 128-256 parallel environments (gym.vector.AsyncVectorEnv or similar). The same curriculum logic applies at any parallelism level.

2. **Remove terrain height observations for Level 1:** Your Level 1 terrains are flat (contact params only, no geometry change). Drop the 108 terrain height measurements. Add them back for Level 2 when geometry changes.

3. **Soft terrain height approximation:** For Level 2, you don't have true terrain deformation. Instead, add a "compliance estimate" to the observation — the current solimp/solref values. This replaces terrain height as the signal that helps the policy identify what surface it's on.

4. **Accept slower training:** 20 min (Isaac Gym GPU) → ~2-4 hours (MuJoCo CPU, 128 envs). For development on MacBook this is fine. For final training runs, rent a GPU instance.

### 10e. The single most important thing to take from this paper

**The timeout bootstrapping fix.** This is buried in Section 2.2.2 but it's the difference between a working PPO implementation and one that's subtly broken. Your training episodes have a max length (timeouts). Standard Stable-Baselines PPO doesn't handle this correctly — the critic underestimates values at timeouts, leading to a policy that's too conservative. Either implement the bootstrapping yourself (as shown in the pseudocode) or use Spinning Up's PPO which handles it correctly.

### 10f. Minimal viable implementation

```
Minimum implementation to use from this paper:
1. 9-term reward function from Table 2 with exponential tracking φ(x)
2. Game-inspired curriculum: track per-robot level, advance/retreat on episode end
3. Timeout bootstrapping: distinguish timeout vs. failure in GAE computation
4. Adaptive learning rate: Algorithm 1 (copy exactly)
5. Observation noise (Table 4): add during training
6. Add feet_air_time reward with target 0.5s to get natural gait

Do NOT worry about:
- GPU parallelism (use MuJoCo CPU for development)
- Isaac Gym (Linux GPU only)
- Actuator LSTM (use simple PD controller model)
- Terrain height grid (for Level 1; add for Level 2)
```

---

## IMPLEMENTATION CHEAT SHEET

### 3 most important things to understand

1. **The game-inspired curriculum is the paper's most practical contribution for your project.** It replaces the complex particle filter from Lee et al. with a dead-simple rule: robots that succeed get harder terrain, robots that fail get easier terrain. This runs with zero overhead, zero tuning, and scales from 128 to 4096 robots with no modification. Use this for your compliance curriculum.

2. **Timeout bootstrapping is a correctness requirement, not an optimization.** Standard Stable-Baselines PPO is wrong for episodic environments with fixed time limits. Your robot will time out frequently (not always fall). Without bootstrapping, your critic systematically underestimates values at episode boundaries, making your policy unnecessarily conservative. The fix is 5 lines of code. Do it.

3. **The feet air time reward is what makes the gait look natural.** Without it, the policy learns to shuffle with feet barely leaving the ground — technically stable but energetically inefficient and poor at rough terrain. Set the target to 0.5s and weight to 2.0×dt.

### 5-step implementation recipe

1. **Port the 9-term reward** from Table 2 to your MuJoCo environment. Use the exponential tracking form φ(x) = exp(-||x||²/0.25) — it's smooth, bounded, and requires no tuning.
2. **Implement the game-inspired curriculum** — per-robot level tracking, advance on success, retreat on failure, loop-back at max. Map your compliance levels (rigid → Level 1 params → soft patches) to curriculum levels.
3. **Fix timeout bootstrapping** — either use Spinning Up's PPO or add the 5-line bootstrap fix to your GAE calculation. Verify by checking that critic loss is lower with it than without.
4. **Add observation noise** from Table 4 during training for robustness to sensor noise.
5. **Tune the feet air time reward weight** until the gait looks natural (robot lifts feet clearly between strides).

### 3 most common implementation mistakes

1. **Using Stable-Baselines PPO without timeout bootstrapping** — subtly wrong for time-limited episodes. Switch to Spinning Up PPO or implement bootstrapping manually. This is the most common silent failure mode in PPO locomotion training.

2. **Setting n_steps too low** — below ~25 steps per robot per update (0.5s at 50Hz), GAE estimates are unreliable and training becomes noisy. Don't try to "go faster" by reducing this below 24-25.

3. **Not applying observation noise during training** — the policy becomes brittle to sensor noise in simulation and fails to transfer. Apply the noise levels from Table 4 from day 1, not as an afterthought.

### Most important hyperparameter to tune first
**Feet air time reward weight (currently 2.0·dt)** — this single parameter controls whether your robot produces a natural, terrain-adaptive gait or a shuffling motion. If your robot is dragging feet, increase this weight. If it's bouncing excessively, decrease it. Tune this before touching any other reward weight.

### Should I use this method for my project?
**Yes — use the curriculum design and reward function directly, accept slower training on CPU.** The game-inspired curriculum is the most implementable and most impactful contribution for your specific project. Use the 9-term reward function and Table 3/4 hyperparameters as your default setup. Skip Isaac Gym (macOS incompatible) and run on MuJoCo CPU for development, cloud GPU for final training runs. The legged_gym open-source code is an excellent implementation reference even if you don't use Isaac Gym.
