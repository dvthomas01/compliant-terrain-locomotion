# Paper Analysis: Tan et al. 2018
## Sim-to-Real: Learning Agile Locomotion For Quadruped Robots
**Authors:** Jie Tan, Tingnan Zhang, Erwin Coumans, Atil Iscen, Yunfei Bai, Danijar Hafner, Steven Bohez, Vincent Vanhoucke
**Venue:** Robotics: Science and Systems (RSS) 2018
**arXiv:** 1804.10332
**Affiliation:** Google Brain

---

## SECTION 1 — PROBLEM STATEMENT

**One-sentence summary:** Train quadruped locomotion policies in simulation using PPO and transfer them to a real robot by combining simulation fidelity improvements (accurate actuator model + latency simulation), dynamics randomization over contact and physical parameters, and compact observation space design.

This paper sits at a unique intersection: it applies dynamics randomization specifically to **quadruped locomotion** (as opposed to Peng et al. 2018's manipulation task) and provides the most thorough empirical breakdown of *which* components of the sim-to-real pipeline matter most. Prior work on locomotion sim-to-real either required real-world data collection, used simplified simulators that didn't model actuator dynamics or latency, or trained with rich observation spaces that overfit the simulator.

The gap it fills: a complete, systematic demonstration that PPO-trained quadruped locomotion policies can transfer zero-shot to a real robot by combining (1) simulator improvement, (2) physics parameter randomization, and (3) compact observation design — with quantitative ablations isolating each component's contribution.

---

## SECTION 2 — CORE IDEA & INTUITION

The central insight is: **the reality gap has three separable causes that require three separable fixes, and fixing any one without the others is insufficient.**

1. **Simulator inaccuracy** (actuator model + latency): The default Bullet position controller is overdamped and uses instantaneous feedback — neither matches real hardware. Without fixing this, no amount of randomization helps. Figure 6 shows that perturbation training on a bad simulator still fails completely on the real robot.

2. **Physics parameter uncertainty** (dynamics randomization): Even a good simulator has uncertain parameters (inertia, friction, motor strength). Randomizing these during training produces a policy that handles the real robot's actual (unknown) parameters.

3. **Observation overfitting** (compact observation space): A high-dimensional observation space lets the policy exploit simulator-specific artifacts. Reducing observations to only what matters (IMU readings) shrinks the policy's ability to overfit and increases the probability that the observation distribution in sim and reality overlap.

The key non-obvious finding: **these three components are complementary, not substitutable.** You cannot compensate for a bad actuator model with more randomization. You cannot compensate for a high-dimensional observation space with better randomization. Each addresses a different failure mode.

---

## SECTION 3 — METHODOLOGY

### 3a. Pipeline / Architecture

```
FULL PIPELINE
=============

Step 1 — System Identification
  - Disassemble robot: measure link masses, dimensions, CoM
  - Estimate inertia: assume uniform density per link
  - Measure motor friction: experimental characterization
  - Measure latency: spike PWM signal, measure time-to-detect

Step 2 — Simulator Improvement
  a) Accurate actuator model:
     - Replace Bullet's constraint-based PD with DC motor model
     - Torque: τ = Kt * I
     - Current: I = (V_pwm - V_emf) / R
     - Back EMF: V_emf = Kt * q_dot
     - Nonlinear saturation: piecewise-linear torque-current lookup
     - PD control: V_pwm = V * (kp*(q_des - q_curr) + kd*(0 - q_dot_curr))
  
  b) Latency simulation:
     - Store observation history {(t_i, O_i)}
     - At query time n: interpolate between O_i, O_{i+1} at t_latency
     - Measured: microcontroller 3ms, TX2 neural net inference 15-19ms
     - Total latency: ~18-22ms per control step

Step 3 — Policy Training (PPO)
  - Sample dynamics parameters μ ~ ρ_μ per episode (Table I)
  - Run episode with hybrid policy: a(t,o) = ā(t) + π(o)
  - Collect 25 parallel rollouts per PPO update (up to 1000 steps)
  - Train for 7M simulation steps

Step 4 — Deployment
  - Load policy weights directly
  - Run at 150-200Hz on Nvidia Jetson TX2
  - No fine-tuning, no calibration
```

**Hybrid policy architecture:**
```
a(t, o) = ā(t) + π(o)

ā(t): open loop component (user-specified periodic signal OR zero)
π(o): feedback MLP (2 hidden layers, PPO-trained)

For galloping (learned from scratch): ā(t) = 0
For trotting (user-guided): ā(t) = [0.3sin(4πt), 0.35sin(4πt)+2]

Action space: leg space (swing s, extension e) per leg
  → Motor angles: θ1 = e+s, θ2 = e-s
```

### 3b. Key Algorithms & Formulas

**DC motor actuator model:**
```
τ = Kt * I                           # torque from current
I = (V_pwm - V_emf) / R             # Ohm's law
V_emf = Kt * q̇                      # back EMF
V_pwm = V * (kp*(q̄ - q_n) + kd*(0 - q̇_n))  # PD control
```
- `Kt`: torque constant (from motor spec sheet)
- `R`: armature resistance (from motor spec sheet)
- `V`: battery voltage (randomized: [14.0, 16.8]V in training)
- Key difference from Bullet's default: uses **current** joint state at time n for PD, not end-of-step constraint. This matches real hardware behavior.
- Piecewise linear saturation: τ_actual = f_piecewise(I) to model current saturation

**Latency simulation:**
```
history = {(t_0, O_0), (t_1, O_1), ..., (t_{n-1}, O_{n-1})}
t_query = n*Δt - t_latency

# Find adjacent observations
i such that: t_i ≤ t_query ≤ t_{i+1}

# Linear interpolation
α = (t_query - t_i) / (t_{i+1} - t_i)
O_effective = (1-α)*O_i + α*O_{i+1}
```
- `t_latency`: measured from hardware (3ms micro + 15-19ms TX2 = ~18-22ms)
- Effect: policy receives stale observations, matching real control loop behavior

**Reward function:**
```
r = (p_n - p_{n-1}) · d - w*Δt*|τ_n · q̇_n|
```
- `(p_n - p_{n-1}) · d`: forward progress in desired direction d
- `w*Δt*|τ·q̇|`: energy expenditure (mechanical power = torque × velocity)
- `w = 0.008`: weight balancing speed and energy
- Episode terminates after 1000 steps OR |base tilt| > 0.5 radians

**Hybrid policy:**
```
a(t, o) = ā(t) + π(o)
```
- `ā(t)`: periodic open-loop signal (optional)
- `π(o)`: MLP feedback, bounds [±0.25 rad] for guided trotting, [±0.5 rad] for learned galloping
- Allows spectrum from fully specified (large ā, small π bounds) to fully learned (ā=0, wide π bounds)

**Observation space (compact, 4D for trotting):**
```
o = [roll, pitch, ω_roll, ω_pitch]
```
- Roll and pitch from IMU
- Angular velocities of base along roll and pitch axes
- Notably: NO motor angles, NO yaw (drifts), NO velocities (noisy)
- For galloping (12D): adds 8 motor angles

**Action space (leg space):**
```
θ1 = e + s    (motor 1 angle from extension + swing)
θ2 = e - s    (motor 2 angle from extension - swing)
```
- `s`: swing component (forward/backward leg rotation)
- `e`: extension component (leg length)
- Bounds: swing [−0.5, 0.5] rad, extension [π/2−0.5, π/2+0.5] rad
- Advantage: convex valid region, no self-collision configurations, natural parameterization

### 3c. Design Decisions & Tradeoffs

**Compact observation vs. full observation:**
This is the most surprising finding of the paper. A 4D observation (IMU only) outperforms 12D (IMU + motor angles) on the real robot, despite 12D being better in simulation. The explanation: with more observations, the policy finds simulator-specific correlations. The sim observation distribution becomes sparse in high dimensions, and real observations fall outside the training distribution. Compact observations force the policy to rely only on robust, physically meaningful signals.

**Leg space vs. motor space for actions:**
Motor space has scattered valid configurations due to self-collision constraints. Leg space provides a convex valid region — any point in [swing_bounds × extension_bounds] is physically valid. This dramatically simplifies the learning problem.

**Uniform sampling vs. log-uniform:**
Unlike Peng et al. 2018, this paper uses uniform sampling for all parameters. The ranges are conservative (80%-120% for mass, 50%-150% for inertia) because system identification was performed first. With narrower ranges, the distinction between uniform and log-uniform matters less.

**Why perturbation forces ≈ parameter randomization:**
The paper notes these two approaches have similar effects: "uncertainties of the physical parameters can be viewed as extra forces/torques applied to the system." Random perturbation forces (130-220N, every 1.2s) produce similar robustness benefits to parameter randomization. The paper uses both together.

**Piecewise linear torque saturation:**
Real DC motors saturate — at high current, the torque-current relationship becomes nonlinear. Without this, the simulated motor can produce more torque than physically possible, leading to behaviors that work in sim but fail on hardware.

### 3d. Reward Function

Two-term reward — the simplest viable formulation for locomotion:

```
r = forward_progress - energy_penalty
r = (p_n - p_{n-1}) · d - 0.008 * Δt * |τ_n · q̇_n|
```

**Forward progress term:** Dot product of displacement with desired direction d. This is direction-agnostic speed maximization — the robot is rewarded proportionally to how fast it moves in direction d.

**Energy term:** |τ · q̇| = mechanical power. Taking the absolute value and multiplying by Δt gives mechanical work per step. This discourages high-torque, high-velocity leg thrashing.

**w = 0.008:** The paper reports robustness to a "wide range of w" — no extensive tuning needed. This makes the reward practically easy to use.

**Episode termination:** |base tilt| > 0.5 radians (≈ 28.6°). Generous termination condition — allows some recovery behavior.

---

## SECTION 4 — IMPLEMENTATION GUIDE

### 4a. Step-by-step pseudocode

```
# ============================================================
# ACTUATOR MODEL IMPLEMENTATION
# ============================================================

def compute_torque(q_des, q_curr, q_dot_curr, battery_voltage, Kt, R, kp, kd):
    """DC motor torque with piecewise linear saturation."""
    
    # PD control computes PWM voltage
    V_pwm = battery_voltage * (kp * (q_des - q_curr) + kd * (0 - q_dot_curr))
    
    # Back EMF voltage
    V_emf = Kt * q_dot_curr
    
    # Armature current (Ohm's law)
    I = (V_pwm - V_emf) / R
    
    # Torque (ideal: τ = Kt * I)
    # Apply piecewise linear saturation
    torque = piecewise_linear_saturation(Kt * I)
    
    return torque

def piecewise_linear_saturation(ideal_torque):
    """Approximate nonlinear torque-current saturation with piecewise linear function."""
    # Fit from real motor measurements
    # TODO: fit breakpoints from your motor's spec sheet or measurement
    breakpoints = [(0, 0), (5, 4.8), (10, 9.2), (20, 15.0)]  # (current, torque) pairs
    return np.interp(abs(ideal_torque), 
                     [b[0] for b in breakpoints],
                     [b[1] for b in breakpoints]) * np.sign(ideal_torque)


# ============================================================
# LATENCY SIMULATION
# ============================================================

class LatencyBuffer:
    """Stores observation history for latency simulation.
    At query time, returns linearly interpolated past observation."""
    
    def __init__(self, latency_seconds, dt):
        self.latency = latency_seconds  # e.g., 0.02 (20ms)
        self.dt = dt
        self.history = []  # list of (timestamp, observation) pairs
    
    def push(self, t, observation):
        """Store current observation with timestamp."""
        self.history.append((t, observation.copy()))
        # Keep only enough history to cover latency
        max_history = int(self.latency / self.dt) + 2
        if len(self.history) > max_history:
            self.history.pop(0)
    
    def get_delayed_observation(self, t_current):
        """Return observation from t_current - latency (interpolated)."""
        t_query = t_current - self.latency
        
        if len(self.history) < 2:
            return self.history[-1][1] if self.history else None
        
        # Find adjacent observations
        for i in range(len(self.history) - 1):
            t_i, obs_i = self.history[i]
            t_i1, obs_i1 = self.history[i + 1]
            if t_i <= t_query <= t_i1:
                # Linear interpolation
                alpha = (t_query - t_i) / (t_i1 - t_i + 1e-10)
                return (1 - alpha) * obs_i + alpha * obs_i1
        
        # Before first or after last: clamp
        return self.history[0][1] if t_query < self.history[0][0] else self.history[-1][1]


# ============================================================
# DYNAMICS RANDOMIZATION (Table I)
# ============================================================

def sample_dynamics_parameters():
    """Sample physical parameters uniformly within Table I ranges.
    All ranges are UNIFORM (not log-uniform) as stated in paper."""
    return {
        # Robot body properties
        'mass_multiplier': np.random.uniform(0.80, 1.20),        # ±20% of measured mass
        'inertia_multiplier': np.random.uniform(0.50, 1.50),     # ±50% (less certain)
        'motor_strength_multiplier': np.random.uniform(0.80, 1.20),  # wear and tear
        
        # Contact and friction
        'contact_friction': np.random.uniform(0.5, 1.25),        # rubber on carpet range
        'motor_friction': np.random.uniform(0.0, 0.05),          # Nm, additive
        
        # Timing and electronics
        'control_step_seconds': np.random.uniform(0.003, 0.020), # 3-20ms latency
        'total_latency_seconds': np.random.uniform(0.0, 0.040),  # 0-40ms
        'battery_voltage': np.random.uniform(14.0, 16.8),        # V
        
        # Sensor noise
        'imu_bias': np.random.uniform(-0.05, 0.05),              # radian, constant per episode
        'imu_noise_std': np.random.uniform(0.0, 0.05),           # radian per step
    }

def apply_dynamics(sim, params):
    """Apply sampled parameters to MuJoCo/Bullet simulation."""
    # Mass
    for i in range(sim.model.nbody):
        sim.model.body_mass[i] *= params['mass_multiplier']
    
    # Inertia
    for i in range(sim.model.nbody):
        sim.model.body_inertia[i] *= params['inertia_multiplier']
    
    # Contact friction
    for i in range(sim.model.ngeom):
        sim.model.geom_friction[i, 0] = params['contact_friction']  # lateral friction
    
    # Motor strength (scale all joint actuator gear ratios)
    for i in range(sim.model.nu):
        sim.model.actuator_gear[i] *= params['motor_strength_multiplier']
    
    sim.model.forward()


# ============================================================
# OBSERVATION FUNCTIONS
# ============================================================

def get_compact_observation(robot, imu_bias=0.0, imu_noise_std=0.0):
    """Compact 4D observation: IMU roll, pitch, angular velocities only.
    This is what worked best for real-world transfer (Section VI-B).
    
    DO NOT include:
    - Motor angles (overfit to simulator, noisy)
    - Yaw (IMU drift makes it unreliable)
    - Motor velocities (noisy)
    - Base linear velocity (not available on real robot without mocap)
    """
    roll = robot.imu.roll + imu_bias + np.random.normal(0, imu_noise_std)
    pitch = robot.imu.pitch + imu_bias + np.random.normal(0, imu_noise_std)
    omega_roll = robot.imu.angular_velocity_roll + np.random.normal(0, imu_noise_std)
    omega_pitch = robot.imu.angular_velocity_pitch + np.random.normal(0, imu_noise_std)
    return np.array([roll, pitch, omega_roll, omega_pitch])

def get_full_observation(robot, imu_bias=0.0, imu_noise_std=0.0):
    """12D observation: IMU + motor angles.
    Better in simulation but worse on real robot due to overfitting."""
    compact = get_compact_observation(robot, imu_bias, imu_noise_std)
    motor_angles = robot.joint_positions  # 8 motor angles
    return np.concatenate([compact, motor_angles])


# ============================================================
# LEG SPACE ACTION CONVERSION
# ============================================================

def leg_space_to_motor_angles(swing, extension):
    """Convert leg space (swing, extension) to motor angles.
    θ1 = e + s  (both motors rotate in same direction for swing)
    θ2 = e - s  (motors rotate in opposite directions for extension)"""
    theta1 = extension + swing
    theta2 = extension - swing
    return theta1, theta2

def motor_angles_to_leg_space(theta1, theta2):
    """Inverse conversion."""
    swing = (theta1 - theta2) / 2.0
    extension = (theta1 + theta2) / 2.0
    return swing, extension

# Action bounds in leg space
SWING_BOUNDS = (-0.5, 0.5)      # radians
EXTENSION_BOUNDS = (np.pi/2 - 0.5, np.pi/2 + 0.5)  # radians


# ============================================================
# HYBRID POLICY
# ============================================================

class HybridPolicy(nn.Module):
    """Combines open-loop reference signal with learned feedback.
    a(t, o) = ā(t) + π(o)"""
    
    def __init__(self, obs_dim, action_dim, hidden_sizes, open_loop_fn=None,
                 feedback_bounds=0.5):
        super().__init__()
        
        # Feedback component π(o)
        layers = []
        prev = obs_dim
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev, h), nn.Tanh()])
            prev = h
        layers.append(nn.Linear(prev, action_dim))
        layers.append(nn.Tanh())  # bound to [-1, 1]
        self.feedback_net = nn.Sequential(*layers)
        
        self.feedback_bounds = feedback_bounds  # scale tanh output
        self.open_loop_fn = open_loop_fn  # callable: t → action
    
    def forward(self, obs, t=None):
        """Compute action = open_loop(t) + feedback_bounds * π(obs)."""
        feedback = self.feedback_net(obs) * self.feedback_bounds
        
        if self.open_loop_fn is not None and t is not None:
            open_loop = torch.tensor(self.open_loop_fn(t), dtype=torch.float32)
            return open_loop + feedback
        else:
            return feedback


def trot_reference_signal(t, freq=2.0):
    """Open-loop trot reference signal (user-specified).
    Two diagonal leg pairs 180° out of phase."""
    s_bar = 0.3 * np.sin(2 * np.pi * freq * t)
    e_bar = 0.35 * np.sin(2 * np.pi * freq * t) + 2.0
    
    # Pair 1: front-left + rear-right (in phase)
    # Pair 2: front-right + rear-left (180° out of phase)
    actions = np.zeros(8)  # 4 legs × 2 components (swing, extension)
    actions[0] = s_bar;  actions[1] = e_bar   # FL swing, extension
    actions[2] = -s_bar; actions[3] = e_bar   # FR (180° out of phase)
    actions[4] = -s_bar; actions[5] = e_bar   # RL
    actions[6] = s_bar;  actions[7] = e_bar   # RR
    return actions


# ============================================================
# PPO TRAINING LOOP (simplified)
# ============================================================

def train_ppo(env, policy, n_steps=7_000_000, n_parallel=25):
    """PPO with dynamics randomization per episode."""
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    
    steps_done = 0
    while steps_done < n_steps:
        
        # Collect rollouts from 25 parallel environments
        all_obs, all_acts, all_rews, all_dones = [], [], [], []
        
        for env_i in range(n_parallel):
            # Sample new dynamics for this episode
            params = sample_dynamics_parameters()
            apply_dynamics(env, params)
            
            obs = env.reset()
            latency_buf = LatencyBuffer(params['total_latency_seconds'], dt=1/200.0)
            
            episode_obs, episode_acts, episode_rews = [], [], []
            t = 0
            
            for step in range(1000):  # max episode length
                t_sim = step * params['control_step_seconds']
                
                # Get delayed observation (latency simulation)
                latency_buf.push(t_sim, obs)
                obs_delayed = latency_buf.get_delayed_observation(t_sim)
                
                # Add IMU noise and bias
                obs_noisy = add_imu_noise(obs_delayed, params)
                
                # Policy forward pass
                with torch.no_grad():
                    action, log_prob, value = policy.get_action(
                        torch.tensor(obs_noisy, dtype=torch.float32).unsqueeze(0), t=t_sim
                    )
                
                # Convert leg space to motor commands
                motor_cmds = leg_space_to_motor_angles(action[:4], action[4:])
                
                # Step environment
                obs_next, done, info = env.step(motor_cmds)
                
                # Reward: forward progress - energy
                reward = compute_reward(obs, obs_next, info['torques'], info['velocities'])
                
                episode_obs.append(obs_noisy)
                episode_acts.append(action.numpy())
                episode_rews.append(reward)
                
                obs = obs_next
                t += 1
                steps_done += 1
                
                if done:
                    break
            
            all_obs.extend(episode_obs)
            all_acts.extend(episode_acts)
            all_rews.extend(episode_rews)
        
        # PPO update (standard)
        ppo_update(policy, optimizer, all_obs, all_acts, all_rews)


def compute_reward(obs, obs_next, torques, velocities, w=0.008, dt=0.005):
    """Two-term reward: forward progress - energy cost."""
    # Forward progress (assume base position available in obs)
    displacement = obs_next['base_position'] - obs['base_position']
    desired_direction = np.array([1, 0, 0])  # forward
    forward_progress = np.dot(displacement, desired_direction)
    
    # Energy expenditure: |τ · q̇| × Δt
    energy = np.sum(np.abs(torques * velocities)) * dt
    
    return forward_progress - w * energy


# ============================================================
# PERTURBATION FORCE (random pushes during training)
# ============================================================

class PerturbationForce:
    """Apply random external forces to robot base every 1.2s during training."""
    
    def __init__(self, period_steps=200, magnitude_range=(130, 220), duration_steps=10):
        self.period = period_steps      # every 200 steps = 1.2s at 150Hz
        self.mag_range = magnitude_range  # Newtons
        self.duration = duration_steps  # 10 steps = 0.06s
        self.force = None
        self.remaining = 0
    
    def get_force(self, step):
        """Return current perturbation force (if any)."""
        if step % self.period == 0:
            # New perturbation
            magnitude = np.random.uniform(*self.mag_range)
            direction = np.random.randn(3)
            direction = direction / np.linalg.norm(direction)
            self.force = magnitude * direction
            self.remaining = self.duration
        
        if self.remaining > 0:
            self.remaining -= 1
            return self.force
        else:
            return np.zeros(3)
```

### 4b. Python implementation skeleton — key classes shown above.

### 4c. Key Hyperparameters

| Parameter | Value | Controls | Sensitivity |
|-----------|-------|---------|------------|
| Observation dim (trotting) | 4 | Sim-to-real transferability | HIGH — 4D transfers well, 12D does not |
| Observation dim (galloping) | 12 | Same | Medium — galloping is more robust to obs size |
| Policy hidden layers | (125, 89) trotting; (185, 95) galloping | Capacity | Low |
| Energy weight w | 0.008 | Speed-energy tradeoff | Low — paper reports robustness to wide range |
| Episode length | 1000 steps | Training horizon | Low |
| Termination threshold | 0.5 rad base tilt | Safety in training | Medium — too tight → too many resets |
| Parallel rollouts | 25 | Training throughput | Low |
| Total training steps | 7M | Training duration | Medium — 4-5 hours per task |
| Perturbation force | 130-220N, every 200 steps, lasts 10 steps | Disturbance robustness | Medium |
| Feedback bounds (guided) | ±0.25 rad | How much feedback can override reference | Medium |
| Feedback bounds (scratch) | ±0.5 rad | Full learned range | Low |
| Total latency | 0-40ms randomized | Latency robustness | High — without latency sim, policies fail on hardware |
| Control step | 3-20ms randomized | Timing jitter | High |
| Contact friction | [0.5, 1.25] | Terrain variation | Medium |
| Battery voltage | [14.0, 16.8]V | Motor power variation | Low |
| IMU bias | [-0.05, 0.05] rad | Sensor drift | Medium |
| IMU noise std | [0, 0.05] rad | Sensor noise | Medium |

### 4d. Data Requirements

**Compact observation (4D — recommended for sim-to-real):**
- Roll (from IMU): 1
- Pitch (from IMU): 1
- Angular velocity (roll axis): 1
- Angular velocity (pitch axis): 1

**Full observation (12D — better in sim, worse on real):**
- All 4D above + 8 motor angles

**Action space (8D):** 4 legs × (swing, extension) in leg space

**Physics parameters to identify before building simulator:**
- Link masses (weigh each link separately)
- Motor friction (measure with controlled experiments)
- System latency (spike PWM test: measure time-to-detect in motor response)
- Motor constants Kt and R (from actuator data sheet)
- PD gains kp, kd (from manufacturer's microcontroller implementation)

### 4e. Dependencies

| Tool | Purpose |
|------|---------|
| PyBullet (paper) / MuJoCo (your project) | Physics simulation |
| PPO (custom or stable-baselines3) | Training algorithm |
| PyTorch / TensorFlow | Neural network |
| Nvidia Jetson TX2 (deployment) | Onboard inference |
| IMU | Base orientation measurements |
| Motor encoders | Joint angle measurements |

---

## SECTION 5 — BENCHMARK RESULTS

### 5a-5b. Evaluation
- Real Minitaur quadruped (Ghost Robotics)
- Two gaits: trotting (guided) and galloping (from scratch)
- Quantitative metric: expected return (Eq. 2) in sim and real world
- 100 policies trained, top 3 deployed, 3 runs each = 9 real-world data points per condition

### 5c. Quantitative results

**Gait performance (Table III):**

| Gait | Speed (m/s) | Avg. Mechanical Power (W) |
|------|-----------|--------------------------|
| Trotting (handcrafted) | 0.56 | 92.72 |
| **Trotting (learned)** | **0.60** | **71.78** (−23%) |
| Galloping (handcrafted) | 1.21 | 290.00 |
| **Galloping (learned)** | **1.18** | **188.79** (−35%) |

Learned gaits match or exceed handcrafted speed while using significantly less energy.

**Simulator improvement ablation (Figure 6):**

| Condition | Sim Return | Real Return |
|-----------|-----------|-------------|
| Baseline sim (no actuator, no latency) | ~2.5 | ~0 |
| Baseline sim + perturbations | ~2.5 | ~0.5 |
| Improved sim + perturbations | ~2.5 | ~2.5 |

Key finding: even with perturbation training, bad simulator → failure on real robot.

**Observation space ablation (Figure 9):**

| Observation | Sim Return | Real Return |
|-------------|-----------|-------------|
| Small (4D), no rand | ~3.5 | ~1.5 |
| Small (4D), randomized | ~3.0 | ~3.0 ← best real-world |
| Large (12D), no rand | ~5.5 | ~1.0 |
| Large (12D), randomized | ~5.0 | ~1.5 |

Key finding: smaller observation space achieves better real-world performance despite worse simulation performance.

**Robustness-optimality tradeoff (Figure 8):**
Randomized training: lower mean return but lower variance across test environments.
No randomization: higher peak return but drops significantly at out-of-distribution parameters.
This exactly validates your project's core hypothesis.

### 5d. Qualitative findings
- Galloping emerges automatically without any gait specification — most random seeds converge to galloping
- Some seeds produce trotting, pacing, or unusual gaits
- Perturbation forces and parameter randomization have "similar effects" and are grouped together
- Combination of small observation space + randomization achieves best real-world results

---

## SECTION 6 — WHEN TO USE THIS METHOD

### 6a. Ideal conditions
- Small to medium quadruped (Minitaur scale: direct-drive, ~2kg)
- Task is locomotion on flat terrain (primary use case)
- No joint velocity sensors (or they're too noisy to use)
- Latency is a significant concern (embedded hardware, UART communication)

### 6b. Practical sweet spot
Any quadruped sim-to-real project where you want a complete checklist of what to implement: (1) measure and model your actuators, (2) measure and model latency, (3) randomize all Table I parameters, (4) use compact observation.

### 6c. Scale considerations
Minitaur has 8 direct-drive motors. The framework scales directly to any quadruped with motor encoders + IMU. The compact observation principle is generalizable.

### 6d. Computational profile
- 4-5 hours per gait policy (7M steps, 25 parallel)
- Inference: small MLP, runs at 150-200Hz on Jetson TX2
- Total parameters: ~10K-30K (small MLP) — very fast inference

---

## SECTION 7 — WHEN NOT TO USE THIS METHOD

### 7a. Failure modes
- Complex terrain (paper acknowledges flat-ground only)
- If simulator fidelity cannot be improved (no actuator data, unmeasurable latency)
- Tasks requiring visual input

### 7b. Known limitations (from authors)
- Flat ground only — extending to complex terrain is stated as future work
- Fixed direction locomotion — dynamic speed and direction changes are future work
- Requires system identification effort before training

### 7c. Hidden assumptions
- Actuator dynamics are reasonably approximated by DC motor model
- Latency is measurable and approximately constant
- IMU provides stable roll/pitch (IMU bias is small)
- Contact friction range [0.5, 1.25] covers deployment surface

---

## SECTION 8 — EXPECTED RESULTS

### 8a. What good output looks like
- Galloping emerges automatically at ~1.2 m/s
- Guided trotting at ~0.5-0.6 m/s
- Learned gaits use ~25-35% less energy than handcrafted equivalents
- Comparable sim and real performance with improved simulator

### 8b. Characteristic artifacts
- Some seeds produce unusual gaits (pacing, hopping)
- Feedback component is small relative to open-loop when guided (small bounds)
- Conservative behavior with large observation noise or wide parameter ranges

### 8c. What failure looks like
- All real-world trials fail: check actuator model and latency simulation
- Simulation performance good, real failure: observation space too large (overfitting)
- Conservative, slow gait despite high reward: randomization ranges too wide

### 8d. Realistic numbers
- Trotting: 0.50-0.60 m/s, ~71W average mechanical power
- Galloping: 1.18-1.34 m/s (sim/real), ~189W
- Training time: 3-5 hours on CPU cluster (25 parallel)

---

## SECTION 9 — CONNECTIONS TO OTHER WORK

### 9a. Key predecessors
- Peng et al. 2018 (ICRA) — dynamics randomization for manipulation (your paper 4)
- Schulman et al. 2017 — PPO (algorithm used here)
- Rajeswaran et al. 2016 (EPOpt) — ensemble-based robustness (predecessor to DR)

### 9b. Key successors
- Lee et al. 2020 (Science Robotics) — extends to rough terrain, teacher-student
- RMA 2021 — extends to online adaptation module
- Rudin et al. 2022 — massively parallel version
- This paper is directly cited as the contact parameter randomization precedent in all three

### 9c. Position in field
**Milestone paper for quadruped sim-to-real.** This is the paper that demonstrated the complete pipeline for zero-shot transfer of quadruped locomotion from simulation to real hardware, specifically applying dynamics randomization to locomotion (Peng et al. 2018 did manipulation). It is the direct predecessor to Lee et al. 2020's terrain robustness work and is cited by all subsequent quadruped RL papers as the DR-for-locomotion baseline.

---

## SECTION 10 — PROJECT APPLICATION MAPPING

**Your project:** PPO quadruped in MuJoCo, Level 1 (contact param randomization) and Level 2 (soft patches). Compare Policy A (rigid) vs. Policy B (randomized).

### 10a. How this paper applies
This is your **most direct precedent** for contact parameter randomization in quadruped locomotion specifically. Where Peng et al. 2018 did manipulation with RDPG, this paper does quadruped locomotion with PPO — exactly your setup. Table I gives you validated parameter ranges for contact friction, motor parameters, and timing that have been proven to work on a real quadruped.

### 10b. What is directly usable

| Component | Usable | Notes |
|-----------|--------|-------|
| Table I parameter ranges | ✅ Direct template | Contact friction [0.5, 1.25] is your Level 1 anchor |
| Compact observation principle | ✅ Critical | Explains why you should keep obs space small |
| Two-term reward (forward - energy) | ✅ Simple baseline | Simplest viable reward for locomotion |
| Actuator model (DC motor) | ✅ If using real hardware | Not needed for sim-only project |
| Latency simulation | ✅ If using real hardware | Add for robustness even in sim-only |
| Perturbation forces | ✅ Direct | Add random pushes every ~1.2s during training |
| Hybrid policy (open loop + feedback) | ⚠️ Optional | Useful if you want to specify gait style |
| PPO with 25 parallel envs | ✅ Direct | Your primary algorithm |

### 10c. Most critical contribution for your project

**The observation space finding** (Section VI-B / Figure 9) is the most important result for your compliance terrain project. It shows:

- Large observation (12D): better in simulation, worse on real robot
- Small observation (4D): worse in simulation, better on real robot

For your project, the analog is: **your Policy B will perform more conservatively in simulation than Policy A (which can exploit exact simulator contact dynamics), but Policy B generalizes better to unseen terrain types because its observation space interpretation is more physically grounded.**

This is why your robustness-efficiency tradeoff result is not just expected — it's the *known and documented cost* of building a generalizable policy. You can cite this paper when explaining why Policy B is more conservative on rigid ground.

### 10d. Table I as your Level 1 template

Your Level 1 contact parameter randomization should be anchored by Table I:

| Your MuJoCo parameter | Maps to Table I | Range |
|----------------------|----------------|-------|
| geom_friction (lateral) | contact friction | [0.5, 1.25] |
| body_mass | mass multiplier | [0.8, 1.2]× default |
| dof_damping | motor friction (additive) | [0, 0.05] Nm |
| actuator_gear (motor strength) | motor strength | [0.8, 1.2]× |
| sim timestep variation | control step | [3, 20]ms |
| observation noise | IMU noise | std [0, 0.05] |

Your solimp/solref (Level 1 compliance) parameters are **beyond Table I** — they're your novel extension. Use Table I as your baseline Level 1 and add solimp/solref as your additional compliance dimension.

### 10e. The robustness-optimality tradeoff finding

Figure 8 is your empirical preview of what your results should look like. The paper shows:
- Randomized: lower mean, lower standard deviation
- No randomization: higher peak, higher variance

Translate this directly to your project: Policy A (rigid only) will have higher peak performance on rigid ground but high variance across terrain types. Policy B (randomized) will have lower peak on rigid but consistent performance across your compliance spectrum. This is a known, documented, citable finding — you're not discovering something new, you're measuring how large the effect is specifically for compliance randomization on a compliance spectrum, which is novel.

### 10f. Minimal viable implementation

```
From Tan et al. 2018 specifically, extract:
1. Table I ranges → your Level 1 parameter bounds (use exactly as given)
2. 4D compact observation → strong argument for keeping your obs space small
3. Two-term reward → simple starting reward if RMA/Rudin rewards seem complex
4. Perturbation forces → add random pushes every 200 steps even in sim-only
5. Uniform sampling → all parameters sampled uniformly within their ranges
6. The ablation structure (Figure 8, 9) → template for your results section
```

---

## IMPLEMENTATION CHEAT SHEET

### 3 most important things to understand

1. **Simulator fidelity is a prerequisite for randomization to work.** Figure 6 makes this unambiguous: perturbation training on a bad simulator does not help. For your project, you're staying in simulation so this translates to: make sure your MuJoCo contact model is physically reasonable before adding Level 1 randomization. Test your baseline rigid terrain policy first — if it can't walk on rigid ground, adding compliance randomization won't fix it.

2. **Compact observation space is not a limitation — it's a design choice that improves generalization.** Your instinct might be to give the policy more information. This paper shows the opposite is true for generalization: 4D outperforms 12D on the real robot. For your compliance terrain project: don't give the policy direct access to solimp/solref values. Force it to infer terrain compliance from its own movement history (IMU + joint state patterns). This makes Policy B's generalization genuine rather than trivially coded.

3. **Table I is your parameter range specification.** Don't guess at ranges — use these measured, validated values as your anchor for Level 1 randomization. Your novel addition (solimp/solref variation) sits on top of this baseline.

### 5-step implementation recipe

1. **Build a reliable baseline** — train Policy A on rigid ground only, verify stable trotting at reasonable speed, then lock this as your control condition.
2. **Add Table I randomization** as your Level 1 — per-episode sampling of friction [0.5, 1.25], mass [0.8, 1.2]×, motor strength [0.8, 1.2]×, plus observation noise and timestep jitter.
3. **Add solimp/solref variation** as your Level 1 extension — this is your novel parameter range beyond what Tan et al. demonstrated.
4. **Add Level 2 soft patches** — compliant geometry on top of the parameter randomization.
5. **Replicate the Figure 8-style analysis** for your evaluation — plot performance mean and variance across your terrain spectrum for both Policy A and B.

### 3 most common implementation mistakes

1. **Using wide parameter ranges without system identification** — Table I uses conservative ranges (±20% for mass) because masses were actually measured. If you don't measure your robot model's parameters, widen the ranges but risk training instability. Start with Table I ranges and expand only if needed.

2. **Not adding perturbation forces** — these are cheap to implement (random force to base every 200 steps) and substantially improve robustness. They're not just a sim-to-real technique — they help generalization within simulation too.

3. **Evaluating only binary success/failure** — the paper explicitly critiques binary success rate as inadequate and uses continuous expected return instead. Your evaluation should use continuous metrics (forward velocity, COT, fall frequency) not just "did it fall or not."

### Most important hyperparameter to tune first
**Observation dimension** — before tuning any reward weight or randomization range, decide whether your policy gets motor angles or just IMU readings. The paper shows this is the single most impactful design decision for generalization. Start with compact (4D or equivalent) and only add motor angles if the task is genuinely unlearnable without them.

### Should I use this method for my project?
**Yes — use Table I directly as your Level 1 parameter specification, and use the observation space finding as your design justification.** This paper is your primary citation for contact parameter randomization specifically applied to quadruped locomotion with PPO. Reference it when defining your Level 1 ranges (contact friction [0.5, 1.25] is directly from this paper), and use Figure 8 as the template for your robustness-optimality tradeoff analysis. The actuator model and latency simulation are relevant only if you extend to real hardware.
