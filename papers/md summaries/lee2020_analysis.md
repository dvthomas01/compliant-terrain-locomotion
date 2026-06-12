# Paper Analysis: Lee et al. 2020
## Learning Quadrupedal Locomotion over Challenging Terrain
**Authors:** Joonho Lee, Jemin Hwangbo, Lorenz Wellhausen, Vladlen Koltun, Marco Hutter
**Venue:** Science Robotics Vol. 5, Issue 47 (2020)
**arXiv:** 2010.11251

---

## SECTION 1 — PROBLEM STATEMENT

**One-sentence summary:** Train a blind quadruped locomotion controller using only proprioceptive sensors that generalizes zero-shot to natural terrain conditions — including deformable ground, dynamic footholds, and overground obstructions — that were never seen during training.

Prior work on legged locomotion used explicit state machines that hard-coded motion primitives and reflexes, triggering them via estimated contact states, slip detection, and threshold-based logic. These systems escalated in engineering complexity as new scenarios were added, were brittle when encountering unmodeled conditions (mud, snow, vegetation), and required separate tuning for each new terrain type.

Prior RL-based approaches had made progress but were limited to laboratory flat or mildly textured ground. End-to-end training of a proprioceptive policy directly via RL on rough terrain failed: the reward signal was too sparse, and the policy could not learn to balance and locomote within reasonable training budgets. Exteroceptive approaches using cameras and LiDAR failed in conditions like snow, water, and thick vegetation where sensors are physically obstructed or give unreliable data.

The gap this paper fills: how to train a controller that is robust enough to handle the full complexity of outdoor natural terrain when the only inputs available are joint encoders and an IMU, and when simulation cannot faithfully reproduce the physical complexity of real terrain.

---

## SECTION 2 — CORE IDEA & INTUITION

The central insight is: **you don't need to simulate hard terrain to learn to traverse it — you need a controller that can feel its way through anything using proprioception history.**

The aha idea has three parts that work together:

1. **Privileged learning:** Train a teacher policy that cheats — it gets ground truth terrain geometry, contact forces, friction coefficients, and disturbance information from the simulator. This teacher learns fast because it can see everything. Then use that teacher to supervise a student that only gets proprioceptive measurements. The student learns to infer what the teacher knew by reading its own body signals over time.

2. **Temporal convolutional memory:** Instead of a snapshot-based MLP, use a TCN over a 2-second window of proprioceptive history. The robot implicitly learns to detect slip (friction change shows up in joint dynamics), terrain deformation (unexpected contact forces), and upcoming obstacles (leg collision changes the joint state pattern) from its own movement history — without explicit contact estimation modules.

3. **Adaptive terrain curriculum:** Rather than training on fixed terrains or randomly sampled terrains, use particle filtering to always train on terrains that are at the edge of the policy's current ability — hard enough to force learning, easy enough not to waste episodes on total failures.

The reason simpler approaches fail: a single-snapshot MLP cannot detect slip or deformation because those phenomena require comparing current state to expected state over time. And direct RL on rough terrain fails because the reward signal comes too rarely — the teacher bootstraps the student past this barrier.

---

## SECTION 3 — METHODOLOGY

### 3a. Pipeline / Architecture

**Full pipeline from input to deployed controller:**

```
TRAINING PHASE
==============
Step 1 — Teacher Training
  Input:  robot state o_t + privileged info x_t
  o_t:    command, gravity vector, base angular vel, base linear vel,
          joint pos/vel, leg phases/frequencies, joint pos error history,
          foot target history
  x_t:    terrain profile (9 scan points per foot at 10cm radius),
          foot contact states/forces, friction coefficients,
          external disturbance forces
  Network: MLP encoder (x_t → latent l_t) + MLP (l_t + o_t → action a_t)
  Training: TRPO with adaptive terrain curriculum
  Output: trained teacher weights + terrain curriculum

Step 2 — Student Training
  Input:  proprioceptive history H = {h_{t-1}, ..., h_{t-N}} (N=100, 2s window)
  h_t:    subset of o_t: command, gravity, base angular/linear vel,
          joint pos/vel, leg phases/frequencies
          (excludes: joint pos error history, foot target history)
  Network: TCN encoder (H → latent l_t) + MLP (l_t + o_t → action a_t)
  Training: Supervised imitation of teacher (DAgger)
  Loss:   action imitation + latent vector imitation
  Output: trained student weights (proprioceptive only)

DEPLOYMENT PHASE
================
  Input (50 Hz):  command direction + proprioceptive stream
  Neural network: TCN encoder + MLP → leg frequencies f_i + foot residuals Δr_fi
  Foot trajectory generator: F(φ_i) → base foot position per leg
  Target foot position: r_fi_T = F(φ_i) + Δr_fi
  Inverse kinematics: foot positions → joint position targets
  Joint PD controller (400 Hz): tracks joint position targets
  Output: joint torques applied to physical robot
```

**Two-loop control architecture:**
- **50 Hz outer loop:** neural network policy runs, outputs foot position residuals and leg frequency offsets
- **400 Hz inner loop:** PD controller tracks joint position targets from IK

### 3b. Key Algorithms & Formulas

**Leg phase definition:**
```
φ_i = (φ_{i,0} + (f_0 + f^i) * t) mod 2π
```
- `φ_i`: phase of leg i ∈ [0, 2π)
- `φ_{i,0}`: initial phase offset, sampled from U(0, 2π)
- `f_0`: base frequency, fixed at 1.25 Hz (trot gait)
- `f^i`: policy-output frequency offset for leg i
- Contact phase: φ ∈ [0, π). Swing phase: φ ∈ [π, 2π)
- Intuition: the policy modulates frequency to control gait timing adaptively

**Foot trajectory generator (FTG):**
```
F(φ_i) = cubic Hermite spline in z, constant in xy
         returns foot position target in horizontal frame H_i
```
During swing (k ∈ [0,2), k = 2(φ_i - π)/π):
```
z = h(-2k³ + 3k²) - 0.5    for k ∈ [0,1]
z = h(2k³ - 9k² + 12k - 4) - 0.5  for k ∈ [1,2]
z = -0.5                            otherwise (stance)
```
- `h`: maximum foot height parameter = 0.2 m
- Spline ensures zero velocity at connection points → smooth lift and landing
- Policy adds residuals on top of this base trajectory

**Traversability labeling:**
```
ν(s_t, a_t, s_{t+1}) = 1  if v_pr(s_{t+1}) > 0.2 m/s
                       = 0  if v_pr < 0.2 OR termination
```
- `v_pr`: inner product of base velocity and commanded direction
- 0.2 m/s threshold ≈ 1/3 of maximum speed

**Traversability score:**
```
Tr(c_T, π) = E_{ξ~π}[ν(s_t, a_t, s_{t+1} | c_T)] ∈ [0, 1]
```
- `c_T`: terrain parameter vector
- Expected traversability over trajectories generated by current policy

**Terrain desirability:**
```
Td(c_T, π) = Pr(Tr(c_T, π) ∈ [0.5, 0.9])
```
- Target range [0.5, 0.9] means "challenging but learnable"
- Below 0.5: too hard. Above 0.9: too easy.

**Particle filter importance weight update:**
```
w^k ∝ Pr(y^k_j | c^k_{T,j}) = #{trajectories where Tr ∈ [0.5, 0.9]} / N_traj
```
Resampling: probability of selecting particle k = w^k / Σ w^i

**Student loss function:**
```
L = (ā_t(o_t, x_t) − a_t(o_t, H))² + (l̄_t(o_t, x_t) − l_t(H))²
```
- First term: action imitation — student matches teacher's action
- Second term: latent imitation — student's TCN encoding matches teacher's terrain embedding
- Quantities with bar (ā, l̄) are teacher outputs used as targets
- Critical: the latent loss forces the student to build an internal terrain representation, not just mimic outputs

**Cost of transport (COT):**
```
COT = Σ_{12 actuators} [τ θ̇]+ / (mgv)
```
- `[·]+`: positive part (only motoring, not regenerative)
- `mg`: total weight
- `v`: locomotion speed
- Dimensionless energy efficiency metric

**Saliency map:**
```
M_i = Σ_{j∈channels} |d((r_{f,T})_z) / dH_{i,j}|
```
- Measures sensitivity of foot height target to each timestep in the proprioceptive history
- Used for interpretability — identifies which past moments the policy attends to

### 3c. Design Decisions & Tradeoffs

**TCN over RNN:** The paper explicitly compares TCN vs GRU. TCN wins on training speed (9ms vs 152ms per SGD update) and performance on step traversal. GRU is between TCN-20 and TCN-100. TCN is chosen because it has transparent receptive field control, parallelizable training, and known hyperparameter robustness.

**Direction command only, not velocity:** Unlike most work, the policy receives only target direction (unit vector) not target speed. This is intentional — speed is terrain-dependent and difficult to specify on rough ground. The tradeoff: less precise speed control on flat ground.

**Horizontal frame H_i for foot targets:** Rather than expressing foot positions in the base frame, they use a frame attached below each hip joint with yaw-aligned-to-base but decoupled roll/pitch. This stabilizes training by reducing the effect of base attitude instability during early training on the action distribution.

**PMTG architecture:** Neural network adds residuals to a base FTG motion, not raw joint targets. This gives the policy a strong locomotion prior, reducing the search space and making it easier to learn. The tradeoff: the policy is constrained to trot-like gaits — it cannot discover radically different gaits.

**Learned actuator model:** Rather than simulating the PD controller analytically, they train a neural network to reproduce SEA (Series Elastic Actuator) dynamics from joint position error and velocity history. This captures actuator lag and nonlinearity more accurately than a simple model.

**Privileged information scope:** x_t contains 9 terrain scan points per foot (36 total), contact forces/states per foot and limb segment, friction coefficient per foot, and external disturbance forces. The terrain scan is local (10cm radius circles) — not a global map. This keeps x_t bounded in size and focuses the teacher on foot-level contact dynamics.

### 3d. Reward Function

Full reward: `R = 0.05*r_lv + 0.05*r_av + 0.04*r_b + 0.01*r_fc + 0.02*r_bc + 0.025*r_s + 2e-5*r_τ`

| Term | Weight | Formula | Purpose |
|------|--------|---------|---------|
| r_lv (linear vel) | 0.05 | exp(-2(v_pr - 0.6)²) if v_pr < 0.6, else 1.0 | Forward progress in commanded direction |
| r_av (angular vel) | 0.05 | exp(-1.5(ω_pr - 0.6)²) if < 0.6, else 1.0 | Turning speed tracking |
| r_b (base motion) | 0.04 | exp(-1.5*v_o²) + exp(-1.5*||ω_xy||²) | Penalize lateral drift and base roll/pitch rate |
| r_fc (foot clearance) | 0.01 | Fraction of swing feet above local terrain scan height | Avoid foot trapping during swing |
| r_bc (body collision) | 0.02 | -\|body contacts excluding feet\| | Penalize shin/thigh/body contact with ground |
| r_s (smoothness) | 0.025 | -||second finite diff of foot targets|| | Penalize jerky foot trajectories |
| r_τ (torque) | 2e-5 | -Σ\|τ_i\| | Penalize joint torques, save actuators |

**Key design notes:**
- Linear velocity reward saturates at 0.6 m/s — this is the max speed of the prior baseline controller. The policy is not pushed beyond this.
- Foot clearance reward requires privileged terrain scan — only available to teacher. Student must infer needed clearance from history.
- Smoothness reward penalizes second-order differences — this is important for sim-to-real because jerky targets saturate actuators and cause hardware damage.
- Torque reward weight (2e-5) is very small — safety term, not dominant.

---

## SECTION 4 — IMPLEMENTATION GUIDE

### 4a. Step-by-step pseudocode

```
# ============================================================
# PHASE 1: TEACHER TRAINING
# ============================================================

initialize:
    N_particles = 10 per terrain type (hills, steps, stairs, slippery hills)
    sample initial terrain params c_T uniformly from C (Table S2)
    initialize TRPO optimizer
    initialize replay memory

for iteration i in range(10000):  # teacher training
    for each particle k in N_particles:
        for each trajectory m in N_traj=6:
            # Generate terrain using c_T^k
            terrain = generate_terrain(c_T^k)
            # Initialize robot at random position on terrain
            state = env.reset(terrain)
            episode_data = []
            for t in range(max_episode_length=400):
                o_t = get_robot_state()        # proprioceptive measurements
                x_t = get_privileged_info()    # terrain scan, contact, friction
                l_t = MLP_encoder(x_t)         # embed privileged info
                a_t = MLP_policy(l_t, o_t)     # compute action
                # action = [f^i for 4 legs, Δr_fi for 4 legs×3D = 16D total]
                # compute foot targets: r_fi_T = FTG(φ_i) + Δr_fi
                # run IK → joint targets → PD controller → step simulation
                reward = compute_reward(state, next_state)  # 7-term reward
                label = traversability_label(v_pr)          # Eq. 2
                episode_data.append((state, action, reward, label))
                if termination: break
            
            # Compute traversability for this particle
            Tr_k = mean(labels in episode_data)
            
        # Compute desirability weight for particle k
        w_k = (Tr_k in [0.5, 0.9]) ? 1 : 0  # simplified; Eq. 7 in paper
    
    # Update policy with TRPO on collected trajectories
    update_policy_TRPO(all_episode_data)
    
    # Every N_evaluate=10 iterations: update particle filter
    if i % N_evaluate == 0:
        normalize_weights(w_k)
        resample_particles()  # SIR resampling
        random_walk_particles()  # p_transition=0.8 chance to move each param

# ============================================================
# PHASE 2: STUDENT TRAINING (DAgger)
# ============================================================

initialize:
    student_TCN = TCN_100_architecture()  # 100-step receptive field = 2s
    # Copy MLP layers from teacher to student (marked * in Table S5)
    copy_teacher_MLP_to_student(teacher, student_TCN)
    replay_buffer = []

for iteration i in range(4000):  # student training
    # Roll out trajectories using STUDENT policy
    for episode in range(num_episodes):
        state = env.reset(terrain_from_curriculum)
        history = deque(maxlen=100)  # 2s of h_t at 50Hz
        for t in range(max_episode_length):
            o_t = get_robot_state()
            h_t = extract_proprioceptive_subset(o_t)
            # h_t excludes: joint error history, foot target history
            history.append(h_t)
            H = array(history)  # shape [N, 60]
            
            # Student forward pass
            l_t_student = TCN_encoder(H)
            a_t_student = MLP(l_t_student, o_t)
            
            # Teacher computes targets for same state
            x_t = get_privileged_info()
            l_t_teacher = MLP_encoder(x_t)   # teacher latent
            a_t_teacher = MLP_policy(l_t_teacher, o_t)  # teacher action
            
            # Store (H, o_t, a_t_teacher, l_t_teacher) as training data
            replay_buffer.append((H, o_t, a_t_teacher, l_t_teacher))
            
            # Execute student action
            env.step(a_t_student)
    
    # Supervised update on replay buffer
    for minibatch in sample(replay_buffer, batch_size=20000):
        H, o_t, a_bar, l_bar = minibatch
        l_pred = TCN_encoder(H)
        a_pred = MLP(l_pred, o_t)
        
        # Two-term loss: action imitation + latent imitation
        loss = MSE(a_bar, a_pred) + MSE(l_bar, l_pred)
        backprop(loss)

# ============================================================
# DEPLOYMENT (50 Hz outer loop)
# ============================================================

history = deque(maxlen=100)
for each timestep at 50 Hz:
    o_t = read_sensors()  # joint encoder + IMU
    h_t = extract_proprioceptive_subset(o_t)
    history.append(h_t)
    
    l_t = TCN_encoder(array(history))
    [f_i×4, Δr_fi×12] = MLP(l_t, o_t)  # 16D action
    
    # Update leg phases
    for each leg i:
        φ_i += (f_0 + f_i) * dt  # dt = 0.02s
    
    # Compute foot targets
    for each leg i:
        r_fi_T = FTG(φ_i) + Δr_fi  # in horizontal frame H_i
    
    # Convert to joint targets
    q_targets = inverse_kinematics(r_fi_T)
    
    # Send to 400Hz PD controller
    send_joint_targets(q_targets)
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

class MLPEncoder(nn.Module):
    """Encodes privileged information x_t into latent vector l_t.
    Teacher only — not used at deployment."""
    def __init__(self, x_dim, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim, 72), nn.Tanh(),
            nn.Linear(72, 64), nn.Tanh(),
            nn.Linear(64, latent_dim), nn.Tanh()
        )
    
    def forward(self, x_t):
        return self.net(x_t)  # returns l_t


class TeacherMLP(nn.Module):
    """Teacher policy: takes robot state + privileged latent → action.
    Architecture: MLP encoder + shared MLP policy head."""
    def __init__(self, o_dim, latent_dim, action_dim=16):
        super().__init__()
        self.encoder = MLPEncoder(x_dim=TODO, latent_dim=latent_dim)
        # Shared layers (marked * in Table S5 — copied to student)
        self.shared = nn.Sequential(
            nn.Linear(o_dim + latent_dim, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh(),
            nn.Linear(128, 64), nn.Tanh(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, o_t, x_t):
        l_t = self.encoder(x_t)
        return self.shared(torch.cat([o_t, l_t], dim=-1)), l_t


class TCNBlock(nn.Module):
    """Single dilated causal convolution block."""
    def __init__(self, in_channels, out_channels, kernel_size=5, dilation=1):
        super().__init__()
        # Causal padding: only pad on the left (past), not future
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size, dilation=dilation, padding=0)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x shape: [batch, channels, time]
        x_padded = nn.functional.pad(x, (self.padding, 0))
        return self.relu(self.conv(x_padded))


class TCNEncoder(nn.Module):
    """Temporal convolutional encoder for proprioceptive history.
    Replaces MLP encoder in student. Receptive field = N timesteps.
    Architecture: dilated causal conv layers interleaved with strided layers."""
    def __init__(self, h_dim=60, latent_dim=64, history_len=100):
        super().__init__()
        # Dilated layers expand receptive field exponentially
        # Strided layers reduce temporal dimension
        self.layers = nn.Sequential(
            TCNBlock(h_dim, h_dim, dilation=1),
            nn.Conv1d(h_dim, h_dim, kernel_size=5, stride=2),  # stride reduces time
            TCNBlock(h_dim, h_dim, dilation=2),
            nn.Conv1d(h_dim, h_dim, kernel_size=5, stride=2),
            TCNBlock(h_dim, h_dim, dilation=4),
            nn.Conv1d(h_dim, h_dim, kernel_size=5, stride=2),
            nn.Flatten(),
            nn.Linear(TODO, 64), nn.Tanh()  # TODO: compute flattened dim
        )
        self.proj = nn.Linear(64, latent_dim)
    
    def forward(self, H):
        # H shape: [batch, history_len, h_dim] → transpose to [batch, h_dim, time]
        x = H.transpose(1, 2)
        x = self.layers(x)
        return self.proj(x)  # returns l_t estimate


class StudentPolicy(nn.Module):
    """Student policy: proprioceptive history + current state → action.
    Shared MLP layers are initialized from teacher weights."""
    def __init__(self, o_dim, h_dim=60, latent_dim=64, action_dim=16, history_len=100):
        super().__init__()
        self.tcn = TCNEncoder(h_dim, latent_dim, history_len)
        # These layers are COPIED from teacher after teacher training
        self.shared = nn.Sequential(
            nn.Linear(o_dim + latent_dim, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh(),
            nn.Linear(128, 64), nn.Tanh(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, o_t, H):
        l_t = self.tcn(H)
        action = self.shared(torch.cat([o_t, l_t], dim=-1))
        return action, l_t


# ============================================================
# FOOT TRAJECTORY GENERATOR
# ============================================================

def foot_trajectory_generator(phi_i, h=0.2):
    """FTG: maps leg phase to foot position target in horizontal frame H_i.
    Returns z-axis offset (vertical). xy handled separately.
    
    Args:
        phi_i: leg phase in [0, 2π)
        h: max foot height (0.2m default)
    Returns:
        foot z position target in horizontal frame
    """
    if 0 <= phi_i < np.pi:  # contact phase
        return -0.5 * h
    else:  # swing phase
        k = 2 * (phi_i - np.pi) / np.pi  # k in [0, 2)
        if 0 <= k < 1:
            z = h * (-2*k**3 + 3*k**2) - 0.5
        else:  # k in [1, 2)
            z = h * (2*k**3 - 9*k**2 + 12*k - 4) - 0.5
        return z


# ============================================================
# OBSERVATION SPACE
# ============================================================

def get_observation_ot(robot):
    """Full robot state o_t (available both in sim and on real robot).
    Total dimension: 2+1+3+3+3+24+8+4+1+24+24+24 = 121 dims approximately.
    See Table S4 for exact breakdown."""
    obs = np.concatenate([
        robot.command_direction,        # 2D unit vector (cos ψ, sin ψ)
        [robot.command_turning],        # 1D: {-1, 0, 1}
        robot.gravity_vector,           # 3D: direction of gravity in base frame
        robot.base_angular_velocity,    # 3D: ω in base frame
        robot.base_linear_velocity,     # 3D: v in base frame (from state estimator)
        robot.joint_positions,          # 12D: θ_i for all joints
        robot.joint_velocities,         # 12D: θ̇_i for all joints
        robot.leg_phases_encoded,       # 8D: [cos(φ_i), sin(φ_i)] for 4 legs
        robot.leg_frequencies,          # 4D: f^i for 4 legs
        [robot.base_frequency],         # 1D: f_0
        robot.joint_position_errors_t1, # 12D: pos error at t-0.01s
        robot.joint_position_errors_t2, # 12D: pos error at t-0.02s
        robot.joint_velocities_history, # 12D: velocities at t-0.01s
        robot.previous_foot_targets,    # 24D: (r_f,T)_{t-1,t-2} for 4 feet
    ])
    return obs

def get_privileged_info_xt(sim):
    """Privileged info x_t — only available in simulation for teacher.
    Total dimension: 36+12+4+4+4+4+4+3 = 71 dims approximately.
    See Table S4 for exact breakdown."""
    priv = np.concatenate([
        sim.terrain_heights_per_foot,   # 36D: 9 scan points × 4 feet
        sim.terrain_normals_per_foot,   # 12D: 3D normal × 4 feet
        sim.foot_contact_forces,        # 4D: normal force per foot
        sim.foot_contact_states,        # 4D: binary per foot
        sim.thigh_contact_states,       # 4D: binary per thigh
        sim.shank_contact_states,       # 4D: binary per shank
        sim.friction_coefficients,      # 4D: per foot
        sim.external_disturbance_force, # 3D: force on base
    ])
    return priv

def get_proprioceptive_ht(robot):
    """Subset of o_t used as history input to student TCN.
    Excludes: joint error history, foot target history (these have temporal
    redundancy with the TCN's own memory)."""
    h = np.concatenate([
        robot.command_direction,        # 2D
        [robot.command_turning],        # 1D
        robot.gravity_vector,           # 3D
        robot.base_angular_velocity,    # 3D
        robot.base_linear_velocity,     # 3D
        robot.joint_positions,          # 12D
        robot.joint_velocities,         # 12D
        robot.leg_phases_encoded,       # 8D
        robot.leg_frequencies,          # 4D
    ])  # total: ~48-60D
    return h


# ============================================================
# REWARD FUNCTION
# ============================================================

def compute_reward(state, next_state, action, prev_action, prev_prev_action, sim):
    """Full 7-term reward function for teacher training."""
    v_pr = np.dot(next_state.base_linear_vel[:2],
                  next_state.command_direction)  # projected velocity
    
    # Linear velocity reward (saturates at 0.6 m/s)
    if v_pr >= 0.6:
        r_lv = 1.0
    elif next_state.command_is_stop:
        r_lv = 0.0
    else:
        r_lv = np.exp(-2.0 * (v_pr - 0.6)**2)
    
    # Angular velocity reward
    omega_pr = next_state.base_angular_vel[2] * next_state.command_turning
    r_av = 1.0 if omega_pr >= 0.6 else np.exp(-1.5 * (omega_pr - 0.6)**2)
    
    # Base motion reward (penalize lateral drift + roll/pitch rate)
    v_lateral = np.linalg.norm(
        next_state.base_linear_vel[:2] - v_pr * next_state.command_direction)
    r_b = (np.exp(-1.5 * v_lateral**2) +
           np.exp(-1.5 * np.linalg.norm(next_state.base_angular_vel[:2])**2))
    
    # Foot clearance reward (requires privileged terrain info)
    swing_legs = [i for i, phi in enumerate(state.leg_phases) if phi >= np.pi]
    clear_feet = sum(1 for i in swing_legs
                     if state.foot_positions[i][2] > max(sim.terrain_scan[i]))
    r_fc = clear_feet / len(swing_legs) if swing_legs else 1.0
    
    # Body collision reward (penalize non-foot body contacts)
    r_bc = -len(sim.body_contacts - sim.foot_contacts)
    
    # Smoothness reward (second-order finite difference of foot targets)
    # prev_action and prev_prev_action needed
    r_s = -np.linalg.norm(action[4:] - 2*prev_action[4:] + prev_prev_action[4:])
    
    # Torque reward
    r_tau = -np.sum(np.abs(sim.joint_torques))
    
    return (0.05*r_lv + 0.05*r_av + 0.04*r_b + 0.01*r_fc +
            0.02*r_bc + 0.025*r_s + 2e-5*r_tau)


# ============================================================
# TRAVERSABILITY CURRICULUM
# ============================================================

class TerrainCurriculumParticleFilter:
    """Particle filter that maintains terrain parameter distribution
    with traversability in [0.5, 0.9]."""
    
    def __init__(self, n_particles=10, terrain_types=4):
        self.n_particles = n_particles
        # TODO: initialize particles uniformly from C (Table S2)
        self.particles = self._sample_initial_particles()
        self.weights = np.ones(n_particles * terrain_types) / (n_particles * terrain_types)
        self.replay_memory = []
    
    def _sample_initial_particles(self):
        # TODO: sample from parameter ranges in Table S2
        pass
    
    def compute_traversability(self, c_T, trajectories):
        """Compute traversability for terrain params c_T from episode data."""
        labels = [1 if v_pr > 0.2 else 0 for v_pr, terminated in trajectories]
        return np.mean(labels)
    
    def update_weights(self, traversabilities):
        """Update importance weights based on traversability desirability."""
        for k, tr in enumerate(traversabilities):
            self.weights[k] = 1.0 if 0.5 <= tr <= 0.9 else 0.0
        # Normalize
        total = self.weights.sum()
        if total > 0:
            self.weights /= total
        else:
            self.weights = np.ones_like(self.weights) / len(self.weights)
    
    def resample_and_walk(self, p_transition=0.8):
        """SIR resampling + random walk for exploration."""
        # Resample with replacement proportional to weights
        indices = np.random.choice(len(self.particles),
                                   size=len(self.particles),
                                   p=self.weights)
        self.particles = self.particles[indices]
        # Random walk: each parameter moves to adjacent discrete value
        for k in range(len(self.particles)):
            for param_idx in range(self.particles[k].shape[0]):
                if np.random.random() < p_transition:
                    self.particles[k][param_idx] += np.random.choice([-1, 0, 1])
                    # TODO: clip to valid range from Table S2
        self.weights = np.ones(len(self.particles)) / len(self.particles)
    
    def sample_terrain_params(self):
        """Sample terrain parameters proportional to current weights."""
        idx = np.random.choice(len(self.particles), p=self.weights)
        return self.particles[idx]


# ============================================================
# STUDENT TRAINING LOOP (DAgger)
# ============================================================

def train_student(teacher, student, env, curriculum, n_iterations=4000):
    optimizer = torch.optim.Adam(student.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
    replay_buffer = []
    
    for iteration in range(n_iterations):
        # Collect trajectories with STUDENT acting, TEACHER supervising
        c_T = curriculum.sample_terrain_params()
        obs = env.reset(c_T)
        history = deque(maxlen=100)
        
        for t in range(400):  # max episode length
            o_t = get_observation_ot(obs)
            h_t = get_proprioceptive_ht(obs)
            history.append(h_t)
            H = torch.tensor(np.array(history), dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                # Student acts
                a_student, l_student = student(torch.tensor(o_t).unsqueeze(0), H)
                # Teacher supervises
                x_t = get_privileged_info_xt(env.sim)
                a_teacher, l_teacher = teacher(torch.tensor(o_t).unsqueeze(0),
                                               torch.tensor(x_t).unsqueeze(0))
            
            replay_buffer.append((H.squeeze(0), torch.tensor(o_t),
                                   a_teacher.squeeze(0), l_teacher.squeeze(0)))
            
            obs, _, done, _ = env.step(a_student.squeeze(0).numpy())
            if done: break
        
        # Supervised update
        if len(replay_buffer) >= 20000:
            batch = random.sample(replay_buffer, 20000)
            H_b, o_b, a_bar, l_bar = [torch.stack([x[i] for x in batch])
                                       for i in range(4)]
            
            a_pred, l_pred = student(o_b, H_b)
            loss = nn.functional.mse_loss(a_pred, a_bar) + \
                   nn.functional.mse_loss(l_pred, l_bar)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        scheduler.step()
```

### 4c. Key Hyperparameters

| Parameter | Value Used | Controls | Sensitivity |
|-----------|-----------|---------|------------|
| History length N (TCN-N) | 100 (2.0s) | Receptive field of proprioceptive memory | High — TCN-1 fails on steps, TCN-100 succeeds. Diminishing returns above 100 |
| Base frequency f_0 | 1.25 Hz | Trot gait speed baseline | Medium — must match robot's natural trot frequency |
| Max foot height h | 0.2 m | FTG swing amplitude | Medium — too low → foot trapping, too high → unstable |
| Traversability target range | [0.5, 0.9] | Curriculum difficulty band | Medium — expanding it makes curriculum less targeted |
| N_particles | 10 per terrain type | Particle filter resolution | Low — more particles = smoother distribution but slower |
| p_transition | 0.8 | Random walk probability in parameter space | Medium — controls exploration rate in curriculum |
| N_traj | 6 | Trajectories per particle for traversability estimate | Low — more = better estimate but slower updates |
| N_evaluate | 10 | Policy iterations between curriculum updates | Low |
| Discount factor (γ) | 0.995 | TRPO effective horizon | Medium — high gamma emphasizes long-term stability |
| KL threshold (TRPO) | 0.01 | Trust region size | Medium — too large → training instability |
| Student learning rate | 5e-4 initial, exp decay | Imitation learning convergence | Medium — standard for Adam |
| Teacher batch size | 80,000 | TRPO sample efficiency | Low — more samples = lower variance policy gradient |
| Student batch size | 20,000 | DAgger update size | Low |
| Friction coefficient randomization | N(0.7, 0.2), clipped above 0.1 | Domain randomization range | High — this is key for robustness to slippery terrain |
| External disturbance | Applied randomly during training | Robustness | Medium |

### 4d. Data Requirements

**Simulation inputs:**
- Full rigid-body simulator (RaiSim used; MuJoCo equivalent for your project)
- Robot URDF/MJCF with accurate inertia, joint limits, actuator models
- Learned actuator model (captures SEA dynamics) OR learned PD controller model
- Terrain generation: procedural heightfield for hills/steps/stairs

**Observation dimensions (from Table S4):**
- o_t: ~121 dimensions (see Table S4 exact breakdown)
- x_t: ~71 dimensions (privileged)
- h_t (history input): ~48-60 dimensions
- History H: h_t × 100 timesteps

**No preprocessing required** — raw sensor values are used. Joint positions/velocities come directly from encoders. Base velocity/orientation from state estimator (EKF on IMU + kinematics).

**Control frequency:** 50 Hz policy, 400 Hz PD inner loop. History sampled at 50 Hz (every 0.02s).

### 4e. Dependencies & Libraries

| Tool | Purpose | Notes |
|------|---------|-------|
| RaiSim (paper) / MuJoCo (your project) | Physics simulation | MuJoCo 3.x is drop-in equivalent |
| PyTorch | Neural network training | Standard |
| TRPO implementation | Teacher policy optimization | Use stable-baselines3 or custom; or substitute PPO |
| TensorFlow C++ API | Onboard inference on robot | Only for hardware deployment |
| NumPy | Numerical operations | Standard |
| State estimator | Base velocity + orientation on real robot | EKF from Bloesch et al. 2013 |

**For your project:** Replace RaiSim with MuJoCo. Replace TRPO with PPO (simpler, nearly equivalent). Keep the rest of the architecture.

---

## SECTION 5 — BENCHMARK RESULTS

### 5a. Evaluation environments
- Natural outdoor environments: mountain trails, creek, mud, vegetation, snow, forest
- Controlled indoor: loose debris, discrete steps (6.6–20.1 cm), slippery whiteboard
- DARPA Subterranean Challenge Urban Circuit (competitive field deployment)
- Comparison robot: ANYmal-B and ANYmal-C (different kinematics/actuators, same policy)

### 5b. Metrics
- **Average locomotion speed (m/s):** Direct measure of task performance
- **Cost of Transport (COT):** Energy efficiency per unit weight per unit distance. Dimensionless, allows comparison across speed regimes
- **Step success rate (%):** Fraction of trials where robot crosses step with all 4 legs, over 10 trials per height
- **Heading error (degrees):** Angle between commanded and actual direction
- **Lateral deviation from 50N push:** Robustness quantification

### 5c. Quantitative results (Table 1 + Fig 3)

**Natural terrain speed and efficiency:**
| Terrain | Ours speed (m/s) | Baseline speed (m/s) | Ours COT | Baseline COT |
|---------|-----------------|---------------------|----------|-------------|
| Moss | 0.452 | 0.199 | 0.423 | 0.625 |
| Mud | 0.338 | 0.197 | 0.692 | 0.931 |
| Vegetation | 0.248 | — (failed) | 1.23 | — |

**Step traversal success rate (10 trials each):**
- At 17.1 cm step: Ours ~100%, Baseline 0.2 m/s ~50%
- At 20.1 cm step: Ours ~75%, Baseline ~0%
- Baseline fails completely with 10 kg payload at any step height

**Disturbance robustness:**
- TCN-100 deviation under 50N lateral force: 35.5% lower than TCN-1

### 5d. Qualitative findings
- Emergent foot-trapping reflex: policy lifts foot higher than normal clearance (up to 22.5cm vs 12.9cm normal) when foot collision is detected in proprioceptive history
- Hind legs adapt trajectory after front legs traverse step — coordinated whole-body adaptation
- Controller responds to mid-shin collision (not foot contact) — generalizes beyond foot-triggered reflexes
- TCN embedding provably encodes terrain geometry: decoder can reconstruct terrain elevation, friction coefficient, contact states from latent vector

### 5e. Comparison to baseline
Baseline is a model-based controller (Jenelten et al. 2019 + Bellicoso et al. 2018 — the prior state of the art for ANYmal). Ours wins in every metric by large margins. Baseline exhibited catastrophic failures in mud, vegetation, and with payload. Ours had zero failures in equivalent conditions.

### 5f. Ablation studies

| Component removed | Cost |
|------------------|------|
| Memory (TCN-1 vs TCN-100) | Step success drops ~50% at 18cm, disturbance deviation +35.5% |
| Privileged training (direct TCN-20 RL) | Complete failure — 0 speed on slope, 0% step success |
| Adaptive curriculum | ~30% lower reward plateau, shorter episode lengths |
| Latent loss in student (action only) | Lower step success rate, slope/disturbance similar |

**Critical finding:** Privileged training is non-negotiable. Without it, the proprioceptive policy cannot learn at all on rough terrain.

---

## SECTION 6 — WHEN TO USE THIS METHOD

### 6a. Ideal conditions
- Quadruped locomotion where only proprioceptive sensors are available or reliable
- Deployment conditions that differ significantly from training (zero-shot transfer required)
- Rough, unstructured terrain where exteroceptive sensors would fail (snow, water, vegetation, mud)
- Systems where fine-tuning on the target domain is not feasible

### 6b. Practical sweet spot
Any scenario where you want a single trained controller that works across a wide variety of terrain without per-environment tuning. The key use case is "train once, deploy everywhere."

### 6c. Scale considerations
- Works well on quadrupeds of ANYmal scale (~30-44 kg)
- Transfers across robot generations (ANYmal-B and C used same policy)
- Scales to 10 kg payload (22.7% of body weight) without retraining

### 6d. Computational profile
- Teacher training: ~12 hours on i7-8700K + RTX 2080
- Student training: ~4 hours additional
- Adaptive curriculum overhead: 2.9s per update (negligible)
- Inference: runs at 400 Hz on Intel i7-5600U (onboard CPU) — NOT GPU-dependent at inference
- Training is GPU-accelerated but inference is CPU-viable

---

## SECTION 7 — WHEN NOT TO USE THIS METHOD

### 7a. Failure modes
- Commanded to walk off a cliff — blind controller cannot anticipate fatal terrain
- Very high-speed locomotion (beyond 0.6 m/s) — reward saturates at this speed
- Novel robot morphologies — policy is trained for specific leg geometry/kinematics
- Gaits other than trot — PMTG architecture constrains to trot pattern

### 7b. Known limitations (from authors)
- Only trot gait is learned — walking, bounding, galloping not exhibited
- Blind locomotion is inherently conservative — must feel terrain with body
- Cannot avoid hazards that require vision (cliffs, drops)
- Authors suggest future work: hybrid proprioceptive-exteroceptive controller

### 7c. Hidden assumptions
- Robot has functional joint encoders and IMU — no explicit contact sensors needed
- State estimator provides reliable base velocity and orientation
- Terrain is rigid in simulation (despite real-world deformable terrain success — this is the zero-shot transfer that works but is not designed for)
- Actuator model is available or learnable for sim-to-real transfer

### 7d. Simpler baselines sometimes better
- On flat, predictable ground: a simple MLP or even a hand-tuned trot controller may achieve higher speed and efficiency without the training cost
- For controlled laboratory experiments with known terrain: model-based MPC likely more precise

---

## SECTION 8 — EXPECTED RESULTS

### 8a. What good output looks like
- Stable trot gait on flat ground at ~0.4-0.6 m/s
- Omnidirectional locomotion with heading error < 10°
- Visible adaptation on rough terrain: variable foot clearance, adjusted timing
- No falls on steps up to ~17cm

### 8b. Characteristic artifacts
- Only trot gait — no walk/bound/gallop
- Conservative foot clearance even on flat ground (the policy hedges)
- Slightly reduced speed vs. specialized baseline on flat terrain (generalist cost)

### 8c. What failure looks like
- Policy never learns to balance (episode terminates immediately): privileged training not working, or reward weights wrong
- Policy learns flat-ground trot but falls on any terrain variation: curriculum not challenging enough, or TCN history too short
- Policy falls consistently when commanded laterally: angular velocity reward not activating, or base motion reward too dominant

### 8d. Realistic numbers
- Flat terrain speed: ~0.4-0.6 m/s
- Step success at 16cm: ~100%. At 20cm: ~75%
- COT on flat: ~0.42 (lower is better)
- Heading error: < 10° in all directions

---

## SECTION 9 — CONNECTIONS TO OTHER WORK

### 9a. Key predecessors
- Hwangbo et al. 2019 (Science Robotics) — learned actuator model, flat-ground RL locomotion
- Peng et al. 2018 ICRA — dynamics randomization for sim-to-real
- Chen et al. 2019 (Learning by Cheating) — privileged learning concept from autonomous driving
- Bai et al. 2018 — TCN empirical evaluation
- Iscen et al. 2018 — PMTG architecture
- Wang et al. 2019 (POET) — open-ended terrain curriculum concept

### 9b. Key successors
- Kumar et al. 2021 (RMA) — extends with online adaptation module for runtime terrain ID
- Rudin et al. 2022 — massively parallel version, game-inspired curriculum
- Miki et al. 2022 (Science Robotics) — adds exteroceptive perception on top of this stack

### 9c. Position in field
**Milestone paper.** This is the paper that demonstrated zero-shot generalization of a simulation-trained locomotion policy to fully natural outdoor environments. It defined the teacher-student privileged learning paradigm for legged locomotion that all subsequent work builds on. Required reading for anyone working on legged locomotion with RL.

---

## SECTION 10 — PROJECT APPLICATION MAPPING

**Your project:** Train a PPO quadruped locomotion policy in MuJoCo to traverse terrain patches with different compliance levels (Level 1: contact parameter variation; Level 2: soft deformable patches). Evaluate whether terrain randomization improves robustness vs. rigid-ground baseline.

### 10a. How this paper applies
This paper is your primary architecture and training reference. Your observation space, reward function, network structure, and training philosophy should all be derived from this paper. The core difference: you are not doing the two-stage privileged learning (that requires a more complex implementation), and you are using PPO instead of TRPO.

### 10b. What is directly usable vs. what needs adaptation

| Component | Usable as-is | Needs adaptation |
|-----------|-------------|-----------------|
| Reward function (all 7 terms) | ✅ Directly | Foot clearance term needs terrain height scan — use simplified version or drop |
| Observation space o_t | ✅ Directly | Adapt to your robot's DOF count |
| Action space (leg freq + foot residuals) | ✅ Directly | Use PMTG or simplify to joint targets |
| Terrain curriculum concept | ✅ Adapt | Replace particle filter with simpler randomization schedule |
| TCN student architecture | ⚠️ Optional | For your project, an MLP with short history (or full TCN) both work |
| Privileged teacher training | ❌ Skip | Your project focuses on PPO baseline comparison, not teacher-student |
| Adaptive curriculum | ⚠️ Simplify | Replace with manual difficulty schedule or uniform randomization |

### 10c. Integration points

1. **Observation space:** Use Table S4 as your exact template. For your compliance project, you can optionally add terrain compliance estimate to privileged info during teacher training.

2. **Reward function:** Use the 7-term reward verbatim. For Level 1 (contact parameter variation), all terms apply directly. For Level 2 (soft patches), the foot clearance term may need modification if your terrain surface height is dynamic (sinking under foot).

3. **Terrain randomization:** Where this paper randomizes terrain geometry (hills/steps/stairs), you randomize contact parameters (friction, damping, solimp/solref) AND terrain geometry (soft patches). Same curriculum logic applies.

4. **Network architecture:** Use their MLP architecture (simpler option) since you are not doing privileged learning. Two hidden layers of 256 units, ReLU or Tanh. If you want memory, add a short history (N=20 is sufficient based on their ablations) via simple concatenation or TCN.

### 10d. Modifications for your use case

- **No privileged learning needed:** Train Policy A (rigid baseline) and Policy B (compliance randomization) both as direct PPO policies with MLP. The teacher-student framework is optional and adds complexity you don't need for an ablation study.

- **Compliance observation:** Consider adding terrain compliance parameters (current solimp/solref values) to Policy B's observation space during training, to test whether explicit compliance awareness helps over implicit history-based detection.

- **Foot clearance reward:** The paper's r_fc requires a terrain height scan. For your Level 1 terrains (flat with varied contact params), this term reduces to trivially satisfied. For Level 2 (soft patches), soft terrain doesn't raise the floor geometrically, so this term is less relevant. You can drop it or weight it at 0 for your project.

### 10e. Domain-specific concerns

- **Soft terrain + reward:** Your foam/pillow patches don't raise the terrain height geometrically, but they do absorb foot energy. The robot will "sink" slightly. The body height reward (r_b) will naturally penalize excessive sinking but you may need to tune the base height target.

- **Contact loss on soft terrain:** On very compliant surfaces, foot contact may be intermittent. This will corrupt the foot contact state signals and disrupt gait timing. Monitor this carefully — it may require adjusting the FTG contact phase threshold.

- **Solimp instability:** Very low stiffness solimp values can cause simulation instability. Start from Level 1 values that are conservative (time constant 0.05-0.1) and increase compliance gradually. Test each new terrain configuration in isolation before adding to curriculum.

### 10f. Minimal viable implementation for your project

```
Minimum implementation to test your core question:
1. MuJoCo quadruped (Unitree Go1 or ant model)
2. PPO with MLP policy (256-256 hidden, ReLU)
3. Observation: o_t from Table S4, minus privileged terms (60-80D total)
4. Action: joint position targets (12D) OR PMTG residuals (16D)
5. Reward: r_lv + r_b + r_tau (3 terms, drop clearance/collision for simplicity)
6. Policy A: train 10M steps on rigid flat ground
7. Policy B: train 10M steps with per-episode randomization of
             {friction, damping, solimp, solref} from Table S2 ranges
             + Level 2 soft patch terrain
8. Evaluate both on held-out terrain spectrum, record fall rate + speed
```

This minimal version already answers your core research question. Add curriculum complexity only if Policy B fails to learn.

---

## IMPLEMENTATION CHEAT SHEET

### 3 most important things to understand

1. **Privileged learning is why this works, but you don't need to replicate it.** The paper's key insight is that direct RL on rough terrain fails — the teacher bootstraps the student past sparse rewards. For your project, you avoid this by using Level 1 (contact param variation) which is easy enough for direct PPO. Don't implement teacher-student unless your training diverges.

2. **The TCN history is why the policy handles novel terrain at test time.** The controller infers compliance, friction, and contact anomalies from the pattern of proprioceptive signals over 2 seconds. For your project, a shorter history (N=20, 0.4s) is sufficient based on their ablation showing TCN-20 handles most cases.

3. **The reward function has a specific structure you should not simplify too aggressively.** The smoothness term (r_s) and torque term (r_τ) are critical for sim-to-real. Even if you're staying in sim, they produce more stable, natural gaits. Keep them from day one.

### 5-step implementation recipe

1. **Set up MuJoCo environment** with your quadruped model, implement o_t observation space from Table S4 (minus privileged columns), and implement the 7-term reward function.
2. **Train PPO baseline (Policy A)** on flat rigid ground until walking is stable (~5-10M steps). Verify heading tracking, speed, and no falls.
3. **Add Level 1 terrain randomization** — randomize friction, damping, solimp/solref per episode. Verify Policy B still converges to walking.
4. **Add Level 2 soft patches** — place compliant geom patches in the terrain. Train Policy B on the full compliance spectrum.
5. **Run evaluation** — test both policies on N=100 episodes per terrain type, record fall rate, forward velocity, COT, and gait stability variance.

### 3 most common implementation mistakes

1. **Skipping the smoothness reward** — the policy learns jerky foot trajectories that look fine in sim but cause training instability and would fail on hardware. Always include r_s.

2. **Using a single-timestep MLP (no history)** when testing on compliance-varied terrain — the policy cannot detect slip or compliance changes without temporal context. Use at minimum a 20-step concatenated history even if you don't implement a full TCN.

3. **Setting solimp too soft too fast** — numerical instability in MuJoCo from very compliant contacts will corrupt your training data with NaN rewards or absurd velocities. Start compliance values conservative, verify stability in simulation first, then increase softness.

### Most important hyperparameter to tune first
**History length N** — this single parameter controls whether your policy can detect compliance changes at all. Start with N=20 (0.4s). If Policy B fails to generalize to soft patches at test time, increase to N=100 before touching anything else.

### Should I use this method for my project?
**Yes** — this paper is your primary architecture reference. Use the observation space (Table S4), reward function (Section S4), and PMTG control structure directly. Simplify by using PPO instead of TRPO and skipping teacher-student in favor of direct training. Your entire Level 1 and Level 2 training pipeline should be a simplified implementation of this paper's framework with contact parameter randomization substituted for geometry randomization.
