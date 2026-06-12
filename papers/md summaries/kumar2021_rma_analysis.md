# Paper Analysis: Kumar et al. 2021
## RMA: Rapid Motor Adaptation for Legged Robots
**Authors:** Ashish Kumar, Zipeng Fu, Deepak Pathak, Jitendra Malik
**Venue:** Robotics: Science and Systems (RSS) 2021
**arXiv:** 2107.04034

---

## SECTION 1 — PROBLEM STATEMENT

**One-sentence summary:** Train a quadruped locomotion policy that can adapt in real-time to unseen terrain conditions — deformable ground, slippery surfaces, changing payloads, varying friction — using only proprioceptive history, with no real-world data collection or fine-tuning at deployment.

The core failure mode of prior approaches was the following: domain randomization produces a single policy that must be robust across all training conditions simultaneously. This forces the policy to be conservative — it cannot specialize to the current terrain because it doesn't know what terrain it is on. The result is degraded performance everywhere compared to a policy that could identify its environment and act accordingly. The paper calls this explicitly: "domain randomization trades optimality for robustness leading to an over conservative policy."

The second failure mode is adaptation methods that require real-world data collection before the robot can walk reliably. If a robot is placed on rocky terrain for the first time and needs 3-5 minutes of rollouts to adapt, it will fall and break itself during that collection phase. This makes online adaptation via policy gradient or Bayesian optimization impractical for physical robots.

The gap this paper fills: how to train a policy that is simultaneously optimal (not over-conservative) AND adaptive to the current environment, entirely in simulation, without needing real-world rollouts to identify the terrain. The key insight is that terrain identification can itself be learned in simulation and deployed zero-shot, because the mapping from proprioceptive history to extrinsic parameters is a learnable function.

---

## SECTION 2 — CORE IDEA & INTUITION

The central insight is: **the robot's own movement history contains enough signal to identify what kind of ground it is standing on — and a neural network can learn to decode this signal entirely in simulation.**

When you command a robot to move its joints in a specific way, the actual resulting movement differs based on the environment. On slippery ground, commanded foot forces produce less traction than expected. On soft ground, the robot sinks slightly and joint angles deviate from targets. With a heavy payload, torques are higher than normal. These deviations are visible in the history of joint positions, velocities, and actions.

RMA makes this explicit with two components:

1. **Base policy + extrinsics:** Train a policy that takes privileged environment information (friction, terrain height, payload mass, motor strength) and compresses it into a small latent vector called the *extrinsics* (z_t, 8-dimensional). The policy conditions on this vector to know how to adapt. This is trained with PPO in simulation.

2. **Adaptation module:** Train a second network that estimates z_t from the last 0.5 seconds of (state, action) history. It learns to decode terrain type from how the robot actually moved versus how it was supposed to move. This is trained with supervised learning in simulation — no real-world data needed.

At deployment, the adaptation module runs at 10 Hz estimating z_t, the base policy runs at 100 Hz conditioning on the latest z_t estimate. The robot adapts in under 1 second.

The reason this works when simpler approaches don't: the extrinsics vector z_t is a low-dimensional compression that only captures *how behavior should change*, not the raw physical parameters. This makes the supervised learning problem easier, avoids identifiability issues (multiple physical configurations can produce the same optimal behavior), and is robust to imperfect estimation because the system is end-to-end optimized for the right action, not for correct physics parameter recovery.

---

## SECTION 3 — METHODOLOGY

### 3a. Pipeline / Architecture

```
PHASE 1 — BASE POLICY TRAINING (PPO in simulation)
===================================================
Inputs at each timestep t:
  x_t ∈ R^30    current robot state (joint pos/vel, roll/pitch, foot contacts)
  a_{t-1} ∈ R^12  previous action (joint position targets)
  e_t ∈ R^17    privileged environment vector (not available at deployment)
                  = [mass×1, COM position×3, motor strength×12, friction×1,
                     local terrain height×1]

Forward pass:
  z_t = μ(e_t)               # environment factor encoder: R^17 → R^8
                               # compresses privileged info to extrinsics
  a_t = π(x_t, a_{t-1}, z_t)  # base policy: R^(30+12+8) → R^12
                               # outputs 12 desired joint positions

Training: PPO jointly optimizes π and μ
  Objective: maximize J(π) = E[Σ γ^t r_t]
  Penalty curriculum: start with small penalty weights, exponentially increase

PHASE 2 — ADAPTATION MODULE TRAINING (supervised, on-policy)
=============================================================
Goal: learn φ such that φ(x_{t-k:t-1}, a_{t-k:t-1}) ≈ z_t = μ(e_t)
  k = 50 steps = 0.5 seconds at 100 Hz

Training data collection:
  - Roll out base policy using RANDOMLY INITIALIZED φ (not ground truth z_t)
  - This generates exploratory/imperfect trajectories (key for robustness)
  - For each visited state, record (history, ground truth z_t from μ(e_t))

Training objective:
  L = MSE(ẑ_t, z_t) = ||φ(x_{t-k:t-1}, a_{t-k:t-1}) - μ(e_t)||²

Repeat iteratively until convergence (on-policy DAgger-style)

DEPLOYMENT
==========
Process 1 at 100 Hz:
  a_t = π(x_t, a_{t-1}, ẑ_async)   # uses most recent ẑ from Process 2

Process 2 at 10 Hz:
  ẑ_async = φ(x_{t-50:t-1}, a_{t-50:t-1})  # updates extrinsics estimate

No synchronization required — asynchronous by design
No fine-tuning, no real-world data collection
```

### 3b. Key Algorithms & Formulas

**Environment factor encoding:**
```
z_t = μ(e_t)
```
- `e_t ∈ R^17`: raw environment parameters — mass (1), COM position (3), motor strength (12), friction (1), terrain height (1)
- `z_t ∈ R^8`: compressed latent extrinsics
- `μ`: 3-layer MLP, hidden dims [256, 128]
- Intuition: 17 parameters → 8 latent dims. The compression forces the encoder to only capture information that matters for behavior. Parameters with identical behavioral effects collapse to the same z_t.

**Base policy:**
```
a_t = π(x_t, a_{t-1}, z_t)
```
- `x_t ∈ R^30`: [joint positions(12), joint velocities(12), roll(1), pitch(1), foot contact binary(4)]
- `a_{t-1} ∈ R^12`: previous joint position targets
- `z_t ∈ R^8`: extrinsics
- Input dimension: 30 + 12 + 8 = 50
- Architecture: 3-layer MLP, hidden dim 128
- Output: 12 desired joint positions (converted to torques via PD controller)

**PD controller:**
```
τ = Kp(q̂ - q) + Kd(q̂˙ - q̇)
```
- `q̂`: desired joint positions (policy output)
- `q`: actual joint positions
- `q̂˙ = 0`: target velocity is always zero
- `Kp = 55, Kd = 0.8`: fixed gains

**Adaptation module:**
```
ẑ_t = φ(x_{t-50:t-1}, a_{t-50:t-1})
```
- Input: 50 timesteps of (state, action) pairs = 50 × (30+12) = 50 × 42 sequence
- Architecture:
  - Step 1: embed each (x_t, a_t) pair via 2-layer MLP → 32-dim representation
  - Step 2: 3-layer 1D CNN over the 50-step sequence
    - Layer 1: [in=32, out=32, kernel=8, stride=4]
    - Layer 2: [in=32, out=32, kernel=5, stride=1]
    - Layer 3: [in=32, out=32, kernel=5, stride=1]
  - Step 3: flatten CNN output → linear projection to R^8

**Training objective for adaptation module:**
```
L = (1/T·N_env) Σ_k ||ẑ_t - z_t||²
```
- MSE between predicted extrinsics and ground truth extrinsics
- Ground truth z_t = μ(e_t) computed from simulation privileged info

**RL return objective:**
```
J(π) = E_{τ~p(τ|π)} [Σ_{t=0}^{T-1} γ^t r_t]
γ = 0.998, λ_GAE = 0.95
```

**Penalty curriculum:**
```
k_t = k_0 * 0.997^t
k_0 = 0.03 (initial small multiplier for penalty terms)
```
Applied to reward terms 3–10 (all penalty terms). Penalty coefficients start nearly zero and grow over training iterations. This prevents the robot from learning to stay still to avoid penalties before it learns to walk.

### 3c. Design Decisions & Tradeoffs

**Why predict z_t (extrinsics) rather than e_t (raw parameters)?**
This is the most important design decision. Predicting raw physics parameters (SysID baseline in Table II) performs worse (56.5% vs 73.5% success). Reasons:
1. Some parameters are unidentifiable from motion alone — two different mass/friction combinations can produce identical trajectories.
2. The extrinsics z_t only encodes what matters for behavior, not what's physically true. End-to-end optimization ensures z_t contains actionable information.
3. Lower dimensionality (8 vs 17) makes the supervised learning problem easier.

**Why train adaptation module on RANDOM φ trajectories, not expert?**
If you train φ on perfect expert trajectories (base policy + ground truth z_t), the adaptation module never sees the kind of imperfect behavior it will produce at deployment. It needs to handle the distribution shift from its own prediction errors. DAgger-style on-policy training with randomly initialized φ exposes the adaptation module to its own mistakes during training.

**Why asynchronous 100Hz/10Hz design?**
The adaptation module is a CNN over 50 timesteps — too slow for 100Hz on the A1's onboard compute (Intel NUC equivalent). Decoupling at 10Hz allows the base policy to run fast while the adaptation module updates slowly. This works because z_t changes slowly in the real world — terrain type doesn't flip every 10ms.

**Why 50-step history (0.5 seconds)?**
Long enough to detect compliance changes (foot sinking takes several steps to manifest), short enough to be reactive (< 1 second adaptation time). The paper does not ablate this — 50 steps is a design choice not deeply justified.

**No PMTG / no reference trajectories:**
Unlike Lee et al. 2020, RMA uses raw joint position targets as output with no foot trajectory generator or motion primitives. This makes the policy more flexible (can learn any gait) but harder to train (no locomotion prior). The curriculum + bioenergetic rewards compensate for this.

**Fractal terrain generator:**
The paper uses a fractal terrain generator (octaves=2, lacunarity=2.0, gain=0.25, z-scale=0.27). This creates natural-looking uneven terrain without specifying explicit terrain types. Combined with physics randomization, the robot sees a wide variety of ground conditions during training.

### 3d. Reward Function

```
R = Σ_i w_i * r_i
```

| Term | Weight | Formula | Purpose |
|------|--------|---------|---------|
| Forward velocity | 20 | min(v_x, 0.35) | Move forward up to 0.35 m/s |
| Lateral + rotation | 21 | -\|\|v_y\|\|² - \|\|ω_yaw\|\|² | Penalize sideways drift and turning |
| Work | 0.002 | -\|τᵀ·(q_t - q_{t-1})\| | Penalize energy expenditure (bioenergetic) |
| Ground impact | 0.02 | -\|\|f_t - f_{t-1}\|\|² | Penalize sudden foot force changes |
| Smoothness | 0.001 | -\|\|τ_t - τ_{t-1}\|\|² | Penalize jerky torque changes |
| Action magnitude | 0.07 | -\|\|a_t\|\|² | Penalize large joint displacements |
| Joint speed | 0.002 | -\|\|q̇_t\|\|² | Penalize fast joint velocities |
| Orientation | 1.5 | -\|\|θ_roll,pitch\|\|² | Penalize body tilting |
| Z acceleration | 2.0 | -\|\|v_z\|\|² | Penalize vertical bobbing |
| Foot slip | 0.8 | -\|\|diag(g_t)·v_{f,t}\|\|² | Penalize feet sliding when in contact |

**Critical notes:**
- Terms 3-10 are multiplied by penalty curriculum coefficient k_t starting at 0.03
- Forward reward is the only non-penalty term — everything else penalizes bad behavior
- Foot slip term (diag(g_t)·v_f): only penalizes foot velocity when the foot is in contact (g_t is binary contact indicator). This teaches the robot to plant feet firmly without sliding.
- Work term uses signed torque × displacement, capturing actual mechanical work
- Target speed 0.35 m/s is lower than Lee et al. (0.6 m/s) — this is the A1 robot which is smaller

**Curriculum formula:**
```
k_0 = 0.03
k_{t+1} = k_t^{0.997}  (exponential increase toward 1.0)
```
The penalty terms gradually reach their full weight as training progresses. Early in training, only the forward reward matters — the robot learns to move first.

---

## SECTION 4 — IMPLEMENTATION GUIDE

### 4a. Step-by-step pseudocode

```
# ============================================================
# PHASE 1: BASE POLICY TRAINING
# ============================================================

initialize:
    base_policy π = MLP(50 → [128, 128] → 12)
    env_encoder μ = MLP(17 → [256, 128] → 8)
    k = 0.03  # penalty curriculum multiplier
    
for iteration in range(15000):
    batch = []
    
    for episode in range(N_env):
        state = env.reset()  # randomize e_t: mass, friction, motor strength, etc.
        a_prev = zeros(12)
        
        for t in range(1000):  # max episode length
            # Get current state and privileged environment info
            x_t = get_state(robot)     # 30-dim: joint pos/vel, roll/pitch, contacts
            e_t = get_privileged(env)  # 17-dim: mass, COM, motor strength, friction, terrain height
            
            # Encode environment to extrinsics
            z_t = μ(e_t)   # 8-dim latent
            
            # Compute action
            a_t = π(x_t, a_prev, z_t)  # 12 desired joint positions
            
            # Compute reward with penalty curriculum
            r_forward = min(v_x, 0.35) * 20
            r_penalty = k * (work + ground_impact + smoothness + ...)
            r_t = r_forward + r_lateral_rot + r_penalty
            
            batch.append((x_t, e_t, z_t, a_prev, a_t, r_t))
            
            # Early termination: height < 0.28m OR |roll| > 0.4 OR |pitch| > 0.2
            x_{t+1} = env.step(a_t)
            a_prev = a_t
            
            if terminated: break
        
    # PPO update on both π and μ
    update_PPO(batch, π, μ, clip_ratio=0.2, lr=5e-4,
               minibatches=4, epochs=4, γ=0.998, λ=0.95)
    
    # Update penalty curriculum
    k = k * 0.997

# ============================================================
# PHASE 2: ADAPTATION MODULE TRAINING
# ============================================================

initialize:
    adaptation_module φ = CNN_MLP(50×42 → 8)  # randomly initialized
    replay_buffer = []

for iteration in range(1000):
    
    for episode in range(N_env):
        state = env.reset()
        a_prev = zeros(12)
        history = deque(maxlen=50)  # (x_t, a_t) pairs
        
        for t in range(1000):
            x_t = get_state(robot)
            e_t = get_privileged(env)
            
            # Adaptation module ESTIMATES extrinsics (imperfect at start)
            if len(history) == 50:
                ẑ_t = φ(history)  # predict from history
            else:
                ẑ_t = zeros(8)    # no history yet
            
            # Base policy uses ESTIMATED extrinsics (not ground truth)
            a_t = π(x_t, a_prev, ẑ_t)
            
            # Ground truth extrinsics for supervision
            z_t_true = μ(e_t)
            
            # Store (history, ground_truth_z) for training
            replay_buffer.append((list(history), z_t_true.detach()))
            
            history.append((x_t, a_t))
            x_{t+1} = env.step(a_t)
            a_prev = a_t
    
    # Supervised update on φ
    for minibatch in sample(replay_buffer, batch_size=80000, minibatches=4):
        hist_batch, z_true_batch = minibatch
        ẑ_pred = φ(hist_batch)
        loss = MSE(ẑ_pred, z_true_batch)
        optimizer.step(loss)

# ============================================================
# DEPLOYMENT
# ============================================================

history = deque(maxlen=50)
ẑ_async = zeros(8)  # initial estimate
t = 0

# Process 1: runs at 100 Hz
def control_loop():
    global ẑ_async
    x_t = read_sensors()  # joint encoders + IMU + foot contacts
    a_t = π(x_t, a_prev, ẑ_async)
    send_to_PD_controller(a_t)

# Process 2: runs at 10 Hz
def adaptation_loop():
    global ẑ_async
    hist = list(history)  # last 50 (x,a) pairs
    ẑ_async = φ(hist)
```

### 4b. Python implementation skeleton

```python
import numpy as np
import torch
import torch.nn as nn
from collections import deque

# ============================================================
# NETWORK ARCHITECTURES
# ============================================================

class EnvironmentFactorEncoder(nn.Module):
    """Encodes privileged environment parameters e_t into extrinsics z_t.
    Only used during Phase 1 training — not deployed on robot."""
    
    def __init__(self, e_dim=17, z_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(e_dim, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, z_dim)
        )
    
    def forward(self, e_t):
        # e_t: [batch, 17] → z_t: [batch, 8]
        return self.net(e_t)


class BasePolicy(nn.Module):
    """Base policy: conditions on state, previous action, and extrinsics.
    Outputs desired joint positions."""
    
    def __init__(self, x_dim=30, a_dim=12, z_dim=8):
        super().__init__()
        input_dim = x_dim + a_dim + z_dim  # 30 + 12 + 8 = 50
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ELU(),
            nn.Linear(128, 128), nn.ELU(),
            nn.Linear(128, a_dim)
        )
    
    def forward(self, x_t, a_prev, z_t):
        # Concatenate all inputs
        inp = torch.cat([x_t, a_prev, z_t], dim=-1)  # [batch, 50]
        return self.net(inp)  # [batch, 12]


class AdaptationModule(nn.Module):
    """Estimates extrinsics from (state, action) history.
    Deployed at 10 Hz on robot. Replaces privileged encoder at runtime."""
    
    def __init__(self, x_dim=30, a_dim=12, z_dim=8, history_len=50):
        super().__init__()
        sa_dim = x_dim + a_dim  # 42 per timestep
        embed_dim = 32
        
        # Embed each (state, action) pair to fixed-dim representation
        self.sa_embed = nn.Sequential(
            nn.Linear(sa_dim, embed_dim), nn.ELU(),
            nn.Linear(embed_dim, embed_dim), nn.ELU()
        )
        
        # 1D CNN over time dimension to capture temporal correlations
        # Input: [batch, embed_dim, history_len] = [batch, 32, 50]
        self.cnn = nn.Sequential(
            # Layer 1: kernel=8, stride=4 → output length = (50-8)/4 + 1 = 12
            nn.Conv1d(embed_dim, embed_dim, kernel_size=8, stride=4),
            nn.ELU(),
            # Layer 2: kernel=5, stride=1 → output length = 12-5+1 = 8
            nn.Conv1d(embed_dim, embed_dim, kernel_size=5, stride=1),
            nn.ELU(),
            # Layer 3: kernel=5, stride=1 → output length = 8-5+1 = 4
            nn.Conv1d(embed_dim, embed_dim, kernel_size=5, stride=1),
            nn.ELU(),
        )
        
        # Compute flattened CNN output size: 32 channels × 4 time steps = 128
        cnn_out_dim = embed_dim * 4  # TODO: compute exactly based on history_len
        
        # Project to extrinsics dimension
        self.project = nn.Linear(cnn_out_dim, z_dim)
    
    def forward(self, history):
        """
        Args:
            history: [batch, history_len, x_dim + a_dim] = [batch, 50, 42]
        Returns:
            z_hat: [batch, z_dim] = [batch, 8]
        """
        batch, T, sa_dim = history.shape
        
        # Embed each (state, action) timestep independently
        h_flat = history.view(batch * T, sa_dim)
        embedded = self.sa_embed(h_flat)  # [batch*T, embed_dim]
        embedded = embedded.view(batch, T, -1)  # [batch, T, embed_dim]
        
        # Transpose for CNN: [batch, embed_dim, T]
        embedded = embedded.transpose(1, 2)
        
        # Apply CNN over time
        cnn_out = self.cnn(embedded)  # [batch, embed_dim, reduced_T]
        
        # Flatten and project
        flat = cnn_out.flatten(1)  # [batch, embed_dim * reduced_T]
        return self.project(flat)  # [batch, z_dim]


# ============================================================
# OBSERVATION SPACE
# ============================================================

def get_state_xt(robot):
    """Robot state x_t: 30-dimensional.
    Available from physical sensors — no privileged info."""
    return np.concatenate([
        robot.joint_positions,    # 12: θ_i for all joints
        robot.joint_velocities,   # 12: θ̇_i for all joints
        [robot.roll],             # 1: base roll from IMU
        [robot.pitch],            # 1: base pitch from IMU
        robot.foot_contacts,      # 4: binary foot contact indicators
    ])  # total: 30


def get_environment_et(sim):
    """Privileged environment vector e_t: 17-dimensional.
    Only available in simulation during training."""
    return np.concatenate([
        [sim.payload_mass],           # 1: additional mass on robot
        sim.com_offset,               # 3: center of mass offset [x, y, z]
        sim.motor_strength,           # 12: per-motor strength multipliers
        [sim.friction_coefficient],   # 1: foot-terrain friction
        [sim.local_terrain_height],   # 1: max terrain height under feet
    ])  # total: 17


def get_sa_history(state_history, action_history):
    """Concatenate state and action history for adaptation module input.
    Returns array of shape [history_len, 42]."""
    # state_history: list of 30-dim arrays
    # action_history: list of 12-dim arrays
    sa_pairs = [np.concatenate([s, a]) for s, a in
                zip(state_history, action_history)]
    return np.array(sa_pairs)  # [50, 42]


# ============================================================
# REWARD FUNCTION
# ============================================================

def compute_reward(state, next_state, action, prev_action,
                   joint_torques, foot_forces, prev_foot_forces, k_penalty):
    """10-term bioenergetically-inspired reward.
    k_penalty: penalty curriculum multiplier (starts at 0.03, increases to 1.0)"""
    
    v_x = next_state.base_linear_velocity[0]   # forward velocity
    v_y = next_state.base_linear_velocity[1]   # lateral velocity
    omega_yaw = next_state.base_angular_velocity[2]
    roll = next_state.roll
    pitch = next_state.pitch
    v_z = next_state.base_linear_velocity[2]
    q = state.joint_positions
    q_next = next_state.joint_positions
    q_dot = state.joint_velocities
    g = state.foot_contacts  # binary: [4]
    v_feet = state.foot_velocities  # [4, 3]
    
    # Term 1: Forward velocity (no penalty multiplier)
    r_forward = min(v_x, 0.35) * 20
    
    # Term 2: Lateral movement and rotation (no penalty multiplier)
    r_lateral = -(v_y**2 + omega_yaw**2) * 21  # TODO: confirm sign/formula
    
    # Terms 3-10: penalized with curriculum multiplier k_penalty
    # Term 3: Work = |τᵀ·Δq|
    delta_q = q_next - q
    r_work = -np.abs(np.dot(joint_torques, delta_q)) * 0.002
    
    # Term 4: Ground impact = ||Δf||²
    r_impact = -np.sum((foot_forces - prev_foot_forces)**2) * 0.02
    
    # Term 5: Torque smoothness = ||Δτ||²
    r_smooth = -np.sum((joint_torques - prev_action)**2) * 0.001
    # Note: using prev_action as proxy for prev_torques
    
    # Term 6: Action magnitude = ||a||²
    r_action_mag = -np.sum(action**2) * 0.07
    
    # Term 7: Joint speed = ||q̇||²
    r_joint_speed = -np.sum(q_dot**2) * 0.002
    
    # Term 8: Orientation = ||roll, pitch||²
    r_orientation = -(roll**2 + pitch**2) * 1.5
    
    # Term 9: Z acceleration = ||v_z||²
    r_z_accel = -(v_z**2) * 2.0
    
    # Term 10: Foot slip = ||diag(g)·v_f||²
    # Only penalize foot velocity when foot is in contact
    foot_slip = np.array([g[i] * np.linalg.norm(v_feet[i]) for i in range(4)])
    r_foot_slip = -np.sum(foot_slip**2) * 0.8
    
    # Apply curriculum to penalty terms
    penalty_sum = (r_work + r_impact + r_smooth + r_action_mag +
                   r_joint_speed + r_orientation + r_z_accel + r_foot_slip)
    
    return r_forward + r_lateral + k_penalty * penalty_sum


# ============================================================
# ENVIRONMENT RANDOMIZATION
# ============================================================

def randomize_environment():
    """Sample environment parameters for a new episode.
    Ranges from Table I in the paper."""
    return {
        'friction': np.random.uniform(0.05, 4.5),
        'Kp': np.random.uniform(50, 60),
        'Kd': np.random.uniform(0.4, 0.8),
        'payload_kg': np.random.uniform(0, 6),
        'com_offset_m': np.random.uniform(-0.15, 0.15, size=3),
        'motor_strength': np.random.uniform(0.90, 1.10, size=12),
        # Resample mid-episode with probability 0.004 per step during training
        'resample_prob': 0.004
    }


# ============================================================
# TRAINING LOOPS
# ============================================================

def train_phase1(π, μ, env, n_iterations=15000):
    """Phase 1: joint PPO training of base policy and environment encoder."""
    optimizer = torch.optim.Adam(
        list(π.parameters()) + list(μ.parameters()),
        lr=5e-4, betas=(0.9, 0.999), eps=1e-8
    )
    k_penalty = 0.03  # penalty curriculum
    
    for iteration in range(n_iterations):
        # Collect batch of 80,000 transitions
        batch = collect_trajectories(π, μ, env, batch_size=80000,
                                      resample_prob=0.004)
        
        # PPO update: 4 minibatches, 4 epochs
        # clip ratio: [0.8, 1.2] (ε = 0.2)
        # value loss coefficient: 0.5
        ppo_update(π, μ, optimizer, batch,
                   clip_eps=0.2, value_coeff=0.5,
                   γ=0.998, λ=0.95,
                   n_minibatches=4, n_epochs=4)
        
        # Update penalty curriculum
        k_penalty *= 0.997
        
        if iteration % 1000 == 0:
            print(f"Iter {iteration}, k_penalty={k_penalty:.4f}")
    
    return π, μ


def train_phase2(φ, π, μ, env, n_iterations=1000):
    """Phase 2: supervised training of adaptation module with on-policy data."""
    optimizer = torch.optim.Adam(φ.parameters(), lr=5e-4)
    
    for iteration in range(n_iterations):
        # Collect on-policy data with RANDOM/CURRENT φ
        # This generates imperfect trajectories → robustness
        batch_data = collect_adaptation_data(φ, π, μ, env,
                                              batch_size=80000,
                                              history_len=50)
        
        # Supervised update: minimize MSE(ẑ, z_true)
        for minibatch in split_minibatches(batch_data, n=4):
            hist, z_true = minibatch
            ẑ = φ(hist)
            loss = nn.functional.mse_loss(ẑ, z_true)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    return φ


class RMADeployment:
    """Asynchronous deployment of RMA on physical robot."""
    
    def __init__(self, π, φ, history_len=50):
        self.π = π
        self.φ = φ
        self.history = deque(maxlen=history_len)
        self.z_hat = torch.zeros(8)  # current extrinsics estimate
        self.a_prev = torch.zeros(12)
    
    def control_step(self, x_t):
        """Run at 100 Hz: base policy with latest extrinsics estimate."""
        x_tensor = torch.tensor(x_t, dtype=torch.float32).unsqueeze(0)
        a_prev_tensor = self.a_prev.unsqueeze(0)
        z_hat_tensor = self.z_hat.unsqueeze(0)
        
        with torch.no_grad():
            a_t = self.π(x_tensor, a_prev_tensor, z_hat_tensor)
        
        self.history.append(np.concatenate([x_t, self.a_prev.numpy()]))
        self.a_prev = a_t.squeeze(0)
        return a_t.squeeze(0).numpy()
    
    def adaptation_step(self):
        """Run at 10 Hz: update extrinsics estimate from history."""
        if len(self.history) < 50:
            return  # not enough history yet
        
        hist = torch.tensor(np.array(self.history), dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            self.z_hat = self.φ(hist).squeeze(0)
```

### 4c. Key Hyperparameters

| Parameter | Value | Controls | Sensitivity |
|-----------|-------|---------|------------|
| Extrinsics dim (z_dim) | 8 | Bottleneck capacity for environment encoding | Medium — too small: can't capture variation. Too large: harder to estimate from history. 8 is chosen empirically |
| History length k | 50 steps = 0.5s | How much past behavior to use for terrain ID | Medium — longer = more context, slower adaptation. 0.5s is a good balance |
| Adaptation frequency | 10 Hz | Update rate of extrinsics estimate | Low — works because terrain type changes slowly |
| Base policy frequency | 100 Hz | Control rate | High — must be fast enough for stable PD control |
| Penalty curriculum start | k₀ = 0.03 | Initial weight of penalty terms | High — if too large, robot learns to stay still. If too small, robot never develops energy efficiency |
| Curriculum decay | 0.997 per iter | Speed of penalty weight increase | Medium — too fast → training instability. Too slow → gait never becomes natural |
| Friction range (train) | [0.05, 4.5] | Diversity of friction encountered | High — must include very low values (0.05) for slippery surface generalization |
| Motor strength range | [0.9, 1.1] | Actuator variation | Low — captures normal wear/tear without extreme damage |
| Payload range | [0, 6] kg | Load variation | Medium — 6kg = 50% of A1 body weight |
| Max forward speed | 0.35 m/s | Forward reward saturation | Medium — lower than Lee et al. because A1 is smaller |
| Episode length | 1000 steps = 10s | Learning horizon | Low |
| PPO clip ratio | [0.8, 1.2] | Trust region size | Medium — standard PPO |
| Resample probability | 0.004/step (train) | How often environment changes mid-episode | Low — ensures policy sees varied conditions per episode |
| γ (discount) | 0.998 | Effective horizon | Medium — very high gamma for long-term stability |

### 4d. Data Requirements

**State x_t (30 dimensions):**
- Joint positions: 12 (all leg joints, in radians)
- Joint velocities: 12 (rad/s)
- Roll: 1 (from IMU)
- Pitch: 1 (from IMU)
- Foot contact binary: 4 (from foot pressure sensors — OR inferred from force estimation)

**Note:** No yaw, no linear velocity, no heading command. Simpler observation space than Lee et al. 2020. The A1 robot's onboard state estimate is used directly.

**Environment e_t (17 dimensions, simulation only):**
- Payload mass: 1
- COM offset: 3
- Motor strength per joint: 12
- Friction coefficient: 1 (scalar, not per-foot)
- Terrain height: 1 (max height under 4 feet, discretized to 0.1m)

**Adaptation module input:**
- History of (x_t, a_t) pairs: 50 × 42 = 2100 values
- Sampled at 100 Hz → covers 0.5 seconds

### 4e. Dependencies

| Tool | Purpose |
|------|---------|
| RaiSim (paper) / MuJoCo (your project) | Physics simulation |
| PyTorch | Training |
| PPO implementation | Phase 1 training — use stable-baselines3 or raisimgymtorch |
| Adam optimizer | Both phases |
| Unitree A1 URDF | Robot model — available from Unitree GitHub |
| NumPy | Array operations |

---

## SECTION 5 — BENCHMARK RESULTS

### 5a. Evaluation settings
**Simulation:** Table I ranges for training/testing, 3 random policy seeds × 1000 episodes
**Real world indoor:** 5 trials per task, A1 robot, various surfaces
**Real world outdoor:** Natural terrain (sand, mud, grass, pebbles, stairs, construction debris)

### 5b. Metrics
- **Success Rate (%):** Task-specific completion
- **TTF (Time to Fall):** Normalized by max episode length [0,1] — higher is better
- **Reward:** Forward + lateral reward per step
- **Distance:** Normalized distance covered
- **Torque:** L2 norm of joint torques
- **Smoothness (Jerk):** L2 norm of Δτ
- **Ground Impact:** L2 norm of Δf (foot force changes)

### 5c. Quantitative results

**Simulation (Table II):**

| Method | Success % | TTF | Reward | Distance |
|--------|-----------|-----|--------|----------|
| Domain Randomization (Robust) | 62.4 | 0.80 | 4.62 | 1.13 |
| SysID | 56.5 | 0.74 | 4.82 | 1.17 |
| AWR | 41.7 | 0.65 | 4.17 | 0.95 |
| RMA w/o Adapt | 52.1 | 0.75 | 4.72 | 1.15 |
| **RMA** | **73.5** | **0.85** | **5.22** | **1.34** |
| Expert (upper bound) | 76.2 | 0.86 | 5.23 | 1.35 |

RMA achieves 73.5% success vs 62.4% for domain randomization — a 11.1 percentage point improvement. Crucially, RMA without adaptation drops to 52.1%, proving the adaptation module is the key contribution.

**Real world indoor (Figure 3):**

| Task | RMA | RMA w/o Adapt | A1 Controller |
|------|-----|--------------|--------------|
| Uneven Foam (success %) | 80 | 0 | 20 |
| Mattress (success %) | 100 | 0 | 100 |
| Upward Incline (success %) | 100 | 20 | 100 |
| Step Down 15cm (success %) | 100 | 0 | 60 |
| Step Up 6cm (success %) | 100 | 40 | 80 |
| Step Up 8cm (success %) | 60 | 0 | 20 |
| Payload 12kg (success %) | ~80 | ~0 | ~0 |

**Outdoor:** 100% on sand/mud/grass/vegetation, 70% on stairs, 80% on pebble pile.

### 5d. Qualitative findings
- Extrinsics vector z_t visibly changes when robot enters slippery/compliant surface (Figure 4)
- Post-adaptation, torque magnitudes increase to compensate for slip — robot actively pushes harder
- Gait timing recovers after adaptation phase (~1-2 seconds)
- Without adaptation module, robot fails completely on foam (0% success) — domain randomization alone insufficient

### 5e. Comparison to baselines
- RMA > Domain Randomization: +11.1% success. DR is over-conservative.
- RMA > SysID: +17% success. Predicting raw physics params is harder and less useful than predicting behavioral extrinsics.
- RMA ≈ Expert: only 2.7% gap. The adaptation module nearly recovers ground-truth performance.
- AWR worst performer: requires real-world rollouts AND is slow to converge.

### 5f. Ablation results
- **Without adaptation module:** 52.1% vs 73.5% — adaptation module worth +21.4% success
- **SysID vs extrinsics prediction:** 56.5% vs 73.5% — predicting z_t is +17% over predicting e_t
- **Friction generalization:** At friction=0.0125 (near frictionless), RMA maintains higher success than all baselines (Figure S4)

---

## SECTION 6 — WHEN TO USE THIS METHOD

### 6a. Ideal conditions
- You need a single policy that handles diverse terrain without per-environment tuning
- The robot's sensors are sufficient to detect terrain changes (proprioception only)
- You want adaptation in < 1 second without real-world data collection
- You're deploying on relatively affordable hardware (A1, Go1 level)

### 6b. Practical sweet spot
Any quadruped deployment scenario where you cannot predict what terrain the robot will encounter. The adaptation module handles the identification automatically.

### 6c. Scale considerations
- Designed for small-medium quadrupeds (A1: 12 kg). Paper does not test on larger platforms.
- History length (50 steps at 100Hz) is designed for robots where terrain ID is visible within 0.5 seconds of stepping.

### 6d. Computational profile
- Phase 1: 24 hours on desktop + 1 GPU (1.2 billion simulation steps)
- Phase 2: 3 hours (80 million simulation steps)
- Inference: base policy at 100Hz (MLP — very fast), adaptation at 10Hz (CNN over 50 steps — moderate)
- No GPU needed at deployment — runs on A1's onboard CPU

---

## SECTION 7 — WHEN NOT TO USE THIS METHOD

### 7a. Failure modes
- Sudden catastrophic terrain changes (falling off a ledge) — no anticipation
- Multiple simultaneous leg obstructions (rocks catching multiple legs)
- Very long adaptation time required for extreme terrain (deep sand requires more than 0.5s to identify)
- Terrain changes faster than 10 Hz (theoretically possible but practically rare)

### 7b. Known limitations (from authors)
- Blind controller — needs vision for long-range path planning and avoiding fatal terrain
- Still limited by proprioception: cannot see what's coming, only react to what has happened

### 7c. Hidden assumptions
- Terrain type is identifiable from 0.5 seconds of joint/contact history
- Environment parameters are stationary within an episode (or change slowly)
- The training distribution of e_t covers the real-world deployment range
- Testing friction range [0.04, 6.0] must be covered by training range [0.05, 4.5] — very low friction (icy) is in testing but barely in training

### 7d. Simpler baselines sometimes better
- On flat, predictable concrete: A1's own controller performs comparably to RMA w/o adaptation
- For controlled single-terrain deployment: fine-tuned domain randomization can match RMA

---

## SECTION 8 — EXPECTED RESULTS

### 8a. What good output looks like
- Stable walking at ~0.35 m/s on flat and varied terrain
- Visible gait adaptation within 1-2 seconds of stepping onto new terrain type
- Extrinsics vector z_t components shift noticeably when terrain changes
- Higher torques on soft/slippery terrain (robot compensates actively)

### 8b. Characteristic artifacts
- Conservative initial gait on unknown terrain (before adaptation kicks in)
- Slightly slower speed vs. hardcoded controller on familiar terrain
- 0.5-second lag before adaptation to step changes in terrain

### 8c. What failure looks like
- Adaptation module not converging: z_t stays near zero regardless of terrain → same as "w/o Adapt" baseline → falls on foam/slippery surfaces
- Phase 1 collapse: robot stays still (penalty terms dominating) → check k₀ value, reduce to 0.01
- Simulation instability: z_t produces NaN or extreme values → check e_t normalization

### 8d. Realistic numbers
- Flat floor success: ~100% (matches A1 controller)
- Foam/deformable surface: ~80-100% (vs 0% w/o adaptation)
- Staircase (never seen in training): ~70%
- Extreme friction (μ=0.0125): still 40-50% success vs ~10% for baselines

---

## SECTION 9 — CONNECTIONS TO OTHER WORK

### 9a. Key predecessors
- Peng et al. 2018 (Dynamics Randomization ICRA) — domain randomization foundation
- Lee et al. 2020 (Science Robotics) — privileged learning for quadruped locomotion
- Peng et al. 2020 (AMP/DeepMimic) — RL locomotion with animal motion imitation
- Yu et al. 2017 (UPOSI) — online system identification (RMA builds directly on this)

### 9b. Key successors
- Miki et al. 2022 — extends RMA-style adaptation with perceptive sensors
- Walk These Ways (Margolis 2022) — builds on RMA for gait diversity
- Various follow-up works on learning adaptation modules for different robot morphologies

### 9c. Position in field
**Milestone paper at RSS 2021.** RMA is the paper that established the two-phase privileged learning + adaptation module paradigm as the standard for sim-to-real locomotion. It is the direct predecessor to almost all modern quadruped RL deployment work. The "extrinsics estimation from history" idea is now used broadly.

---

## SECTION 10 — PROJECT APPLICATION MAPPING

**Your project:** Train PPO quadruped policy in MuJoCo to traverse Level 1 (contact parameter variation) and Level 2 (soft patches) terrain. Compare randomized vs. rigid baseline.

### 10a. How this paper applies
RMA is your primary reference for **why compliance randomization should work** and **how adaptation happens mechanistically**. The key finding directly relevant to your project: the domain randomization baseline (your Policy A analog) achieves 62.4% success while RMA achieves 73.5% — showing that randomization alone is insufficient but that implicit terrain identification from history is powerful.

For your project, you do NOT need to implement the full adaptation module. Instead, RMA teaches you:
1. What environment parameters to randomize (your Level 1 parameters map directly to RMA's e_t)
2. How to structure the observation space to allow implicit terrain identification
3. Why the "robust but conservative" domain randomization policy will exist and be measurably different from an adaptive policy

### 10b. What is directly usable vs. what needs adaptation

| Component | Usable | Notes |
|-----------|--------|-------|
| Environment parameter ranges (Table I) | ✅ Direct | Friction [0.05, 4.5] is your Level 1 range template |
| Reward function (10 terms) | ✅ Direct | Bioenergetic reward is more natural than Lee et al. — simpler to tune |
| Observation space x_t (30-dim) | ✅ Direct | Simpler than Lee et al. — no leg phases, no command direction |
| Penalty curriculum (k₀=0.03, decay 0.997) | ✅ Direct | Critical — prevents collapse to stationary policy |
| Adaptation module | ❌ Optional | Adds Phase 2 complexity. Only implement if you want to go beyond the ablation |
| Extrinsics concept | ✅ Conceptual | Helps you design what your Policy B's observation should include |

### 10c. Integration points for your project

1. **Your Level 1 parameters = RMA's e_t:** Friction [0.05, 4.5], damping, solimp/solref map directly to RMA's randomizable parameters. Use RMA's training ranges as your starting ranges.

2. **Your Policy B observation:** Consider adding recent history (last 10-20 steps of joint states + actions) to Policy B's observation. This is a simplified version of RMA's adaptation module — the policy itself can learn implicit terrain ID from history without a separate module.

3. **Reward function:** RMA's 10-term bioenergetic reward is arguably better for your project than Lee et al.'s 7-term reward because it has simpler tuning (no foot clearance term requiring terrain scan). Use RMA's reward.

4. **Penalty curriculum:** This is critical. Your training will collapse to a stationary policy if penalty terms start at full weight. Implement k₀=0.03 exponentially increasing. This is not mentioned in Lee et al. but is crucial per RMA.

### 10d. The key insight for your robustness-efficiency tradeoff question

RMA's result in Figure S4 shows that at friction=0.8 (normal floor), RMA and domain randomization perform nearly identically. At friction=0.05 (near-frictionless), RMA dramatically outperforms DR. This is the pattern you should expect in your project:
- **Policy A (rigid baseline):** High performance on rigid ground, catastrophic failure on soft/slippery
- **Policy B (randomized):** Moderate performance on rigid (DR conservative), better performance on compliant

The quantitative gap between your Policy A and Policy B on rigid ground is your "cost of robustness" — directly analogous to RMA's finding that DR is more conservative.

### 10e. The single most important thing RMA teaches you about your project

**The domain randomization baseline is NOT your experimental control — it is your experimental condition.** RMA shows that training with physics parameter randomization produces a policy that implicitly learns terrain identification from history. Your Policy B is therefore not just "a policy that saw more terrain" — it is a policy that learned a different internal representation. The interesting analysis question is: what did it learn to represent, and how does that manifest in its behavior on novel terrain types?

### 10f. Minimal viable implementation

```
For your project, extract from RMA:
1. Environment parameter ranges (Table I) → your Level 1 randomization ranges
2. 10-term bioenergetic reward → use this instead of Lee et al. 7-term
3. Penalty curriculum (k₀=0.03, decay 0.997) → prevent training collapse
4. Simple 30-dim observation space → faster to get working than Lee et al. 121-dim
5. Append 10-step (state, action) history to Policy B's observation
   → cheapest way to enable implicit terrain ID without full adaptation module
```

---

## IMPLEMENTATION CHEAT SHEET

### 3 most important things to understand

1. **The adaptation module is what makes RMA novel, but you don't need to implement it.** The key lesson for your project is that randomizing e_t during training produces a base policy that is already terrain-aware when given extrinsics. Your Policy B is doing the same thing — just without the explicit adaptation module. The implicit version (history concatenation) is sufficient for your ablation study.

2. **The penalty curriculum is non-negotiable.** If you implement RMA's reward without the k₀=0.03 exponentially increasing curriculum, your robot will learn to stay still. The penalty terms (work, smoothness, action magnitude) collectively dominate the forward reward early in training. This curriculum is buried in the supplementary but it's the most practically important implementation detail.

3. **Predicting z_t (behavior-relevant compression) is strictly better than predicting e_t (raw physics).** This has a direct implication for your project: if you want to give Policy B an advantage, encode your terrain compliance parameters (solimp, friction, damping) into a small latent before providing it to the policy. Don't feed raw physics parameters — compress them through a learned encoder.

### 5-step implementation recipe

1. **Implement 30-dim observation space** (joint pos/vel, roll/pitch, foot contacts) — simpler and faster to train than Lee et al.
2. **Use RMA's 10-term reward** with penalty curriculum starting at k₀=0.03, multiplied by 0.997 per iteration.
3. **Train Policy A** (rigid ground only) until stable walking. Save as rigid baseline.
4. **Train Policy B** with per-episode randomization of {friction, Kp, Kd, damping, solimp} from Table I ranges, plus Level 2 soft patches. Optionally concatenate 10-step (state, action) history to observation.
5. **Evaluate both policies** on held-out terrain spectrum. Your key metric: does Policy B's implicit terrain detection (visible in gait adaptation) overcome the conservative penalty of domain randomization?

### 3 most common implementation mistakes

1. **Not implementing penalty curriculum** — robot learns to be stationary, never walks. Start k₀ at 0.03 and increase slowly. This is the single most common reason RMA-style training fails.

2. **Setting friction lower bound too high** — if training minimum friction is 0.5 instead of 0.05, the policy never learns to handle slippery/compliant terrain and fails at test time. RMA trains down to 0.05 — match this range.

3. **Training adaptation module on expert trajectories** — if you implement Phase 2, you MUST use on-policy data from the randomly initialized adaptation module, not from the base policy with ground truth z_t. Otherwise the adaptation module will fail on its own prediction errors at deployment.

### Most important hyperparameter to tune first
**Penalty curriculum initial weight k₀** — this single value determines whether your training converges to walking (k₀ ≈ 0.03) or to stationary behavior (k₀ = 1.0). Set it to 0.03 and use the 0.997 exponential schedule. Do not adjust anything else until you have confirmed stable walking.

### Should I use this method for my project?
**Yes, for the reward function and training curriculum — and partially for the architecture.** Use RMA's 10-term bioenergetic reward and penalty curriculum directly — they are simpler and more robust than Lee et al.'s reward for your use case. The extrinsics concept informs how to design your Policy B observation (add history). The adaptation module itself is optional — implement it only if you want to demonstrate terrain identification as a stretch goal beyond the core ablation study.
