# Paper Analysis: Kim & Lee 2021
## Quadruped Locomotion on Non-Rigid Terrain using Reinforcement Learning
**Authors:** Taehei Kim, Sung-Hee Lee
**Affiliation:** KAIST Graduate School of Cultural Technology
**Venue:** arXiv preprint (2021)
**arXiv:** 2107.02955

---

## SECTION 1 — PROBLEM STATEMENT

**One-sentence summary:** Train a quadruped locomotion policy using PPO on a physically compliant terrain modeled as a matrix of spring-loaded tiles that elastically sink under foot contact, and identify the critical observation terms that make this non-rigid terrain locomotion possible.

Nearly all prior RL-based quadruped locomotion research assumes rigid terrain. Real-world environments include non-rigid surfaces — sand, foam, shaky bridges, trampolines — that elastically or plastically deform under foot contact. A controller trained only on rigid terrain fails on soft ground because foot contact timing, expected force feedback, and base height relative to the ground are all disrupted by sinkage.

The gap this paper fills: the first DRL framework (as of publication) that explicitly trains on and demonstrates locomotion on elastically deformable tiled terrain using a Laikago quadruped in PyBullet simulation. The key contribution is identifying which observation terms are necessary to make this work — specifically, that **end-effector position history and end-effector velocity are critical and not obvious to include**.

---

## SECTION 2 — CORE IDEA & INTUITION

The central insight is: **a robot walking on soft terrain needs to remember where its feet have been, not just where they are right now, because current foot position alone is ambiguous on compliant ground.**

On rigid terrain, foot contact is binary and predictable — the foot hits the floor at a known height and stops. On elastic terrain, the tile under each foot sinks an amount that depends on the current spring stiffness. The robot cannot directly observe the stiffness or the sinkage depth. What it can observe is the history of its end-effector positions over recent phases — and that history carries implicit information about how much each tile sank.

Two specific observation terms make the difference:

1. **End-effector position history** (positions at the start of 3 previous phases + current): Provides temporal memory of foot placement patterns. Different terrain stiffness produces different foot position trajectories. The policy learns to read these patterns and infer terrain compliance implicitly.

2. **End-effector velocity**: Without this, the policy learns to stand still and balance but never moves forward. The velocity signal is what enables the policy to initiate and sustain locomotion rather than just maintain stability.

The terrain model itself is also novel: tiles with spring-loaded prismatic joints, where stiffness is varied to produce 2-5cm sinkage depths. This is not granular simulation or FEM — it's a tractable spring-mass model that approximates the key effect of compliant ground (unexpected height changes under foot contact) without prohibitive simulation cost.

---

## SECTION 3 — METHODOLOGY

### 3a. Pipeline / Architecture

```
NON-RIGID TERRAIN MODEL
========================
- PyBullet simulation
- Ground = N×N grid of tiles, each 20cm × 20cm
- Each tile connected to flat rigid base via prismatic joint + spring
- Spring stiffness tuned so average sinkage = target depth when robot stands
  - 2cm: springs set so 4 loaded tiles average 2cm sinkage
  - 3cm, 4cm, 5cm: same principle, decreasing stiffness
- Tiles move PASSIVELY: robot pushes tile down, spring pushes back
- No active terrain modeling — pure physical simulation

ROBOT MODEL
============
- Laikago quadruped (Unitree)
- 25 kg body mass
- 55cm base length, ~50cm leg length
- 12 DOF: 2-DOF shoulder (abduction + hip) + 1-DOF knee per leg

CONTROL ARCHITECTURE (Phase-Based)
====================================
- One locomotion cycle = 4 phases
- Each phase = 0.75 seconds
- Fixed phase order: front-left → rear-right → front-right → rear-left
- In each phase: ONE foot swings, THREE feet on ground
- Policy produces ONE action vector per phase (not per timestep)
- Action is executed over the entire 0.75s phase via Bezier curve interpolation

TRAINING
=========
- PPO (Stable Baselines implementation)
- OpenAI Gym environment
- 5 terrain types in training: rigid + 2, 3, 4, 5cm sinking
- Terrain changes randomly every N (2 or 8) meters along the path
- Curriculum: always start at 2cm, then add harder terrains
- Goal: 2 meters ahead, repeatedly reset when reached

DEPLOYMENT
===========
- Policy evaluated across terrain types
- Visualization of base height trajectories and foot landing positions
```

### 3b. Key Algorithms & Formulas

**Action space (27D — Bezier curve control points):**
```
Action = 9 control points × 3D coordinates = 27D

Three cubic Bezier curves per phase:
  1. Base position trajectory: 3 control points × 3D = 9D
  2. Base orientation trajectory: 3 control points × 3D Euler = 9D
  3. Swing foot trajectory: 3 control points × 3D = 9D

Note: First control point of each curve = current value (not part of action)
Total: 3 remaining control points × 3 curves × 3D = 27D
```

**Action bounds (key for training stability):**
```
Base position: CP2, CP3 ∈ [-6, 6]cm from current position
               CP4 (last) ∈ [-4, 8]cm from current position
Base orientation: all CPs ∈ [-0.3, 0.3]rad in each Euler angle
Foot height: all CPs ∈ [-15, 15]cm from current height
Foot lateral: ∈ [-15, 15]cm from default position
Foot frontal (front feet): ∈ [-15, 15]cm from default
Foot frontal (rear feet): ∈ [δ-15, δ+15]cm, δ = -2cm (rear legs slightly back)
```

**Motion generation (Bezier execution):**
```
Each 0.75s phase → 180 Bezier curve sample points
Control frequency: 180 points / 0.75s ≈ 240Hz → actual ~250Hz = 4ms per step
At each control step:
  1. Sample next point on Bezier curve
  2. Solve IK for joint angles matching (base pose, foot position)
  3. Send joint angles to PyBullet position controller
  4. PyBullet generates torques internally
```

**Reward function (5 terms):**
```
R = Rd + Ro + Rs + Rt + Rr

1. Goal distance reward:
   Rd = αd * (||ρg|| - ||ρg,p||)
   αd = 10 if moving toward goal, αd = 1 if moving away
   ρg = goal position in base frame
   ρg,p = goal position at START of previous phase

2. Goal orientation reward (heading):
   Ro = max(0, 0.02 × (10 - |φg|))
   φg = azimuth angle of goal from base (degrees)
   → Positive only when heading error < 10°

3. Minimum height reward:
   Rs = 0.1 if hb > 25cm, else 0

4. Torque minimizing reward:
   Rt = max(0, 0.004 × (140 - τave))
   τave = average magnitude of joint torque during phase

5. Roll angle reward:
   Rr = max(0, 2 × (0.1 - |φ|))
   φ = roll angle in radians
   → Positive only when |roll| < 0.1 rad = 5.7°
```

**Observation space (102D):**
```
1.  Base height above terrain under each of 4 hip joints (4D)
    - Measured relative to tile height directly below each joint
    - Captures terrain sinkage indirectly

2.  Gravity direction in base frame (3D)
    - Encodes base orientation robustly

3.  Base linear velocity (3D) + angular velocity (3D) = 6D

4.  Base pitch angle (1D)

5.  ** CRITICAL ** End-effector position history — 4 time steps (48D)
    - Position of each of 4 feet in base frame
    - At: start of 3 previous phases + current time
    - = 4 time steps × 4 feet × 3D = 48D
    - Captures 3 seconds of gait history (3 phases × 0.75s)

6.  ** CRITICAL ** End-effector positions at 4ms and 8ms ago (24D)
    - Position of each foot at t-4ms and t-8ms
    - = 2 time points × 4 feet × 3D = 24D
    - Fine-grained recent history: detects if foot is stuck/trapped

7.  ** CRITICAL ** End-effector velocity (12D)
    - Velocity of each of 4 feet = 4 × 3D = 12D
    - Required to enable forward locomotion (vs. just balancing)

8.  Goal direction azimuth angle (1D)

9.  Goal position in base frame, frontal and lateral only (2D)

10. Current phase index (1D)

Total: 4+3+6+1+48+24+12+1+2+1 = 102D
```

**Termination conditions:**
```
Episode terminated (reward = -10) if:
  - Base height < 20cm (near joint limit / fallen)
  - |pitch| > 15° (inclined too much)
  - Any non-foot link contacts ground
```

**PPO hyperparameters:**
```
Policy network: 2 hidden layers [256, 128], tanh activation
Value network: 2 hidden layers [256, 128], tanh activation
Discount factor γ: 0.95
Learning rate: 2×10^{-4}
Minibatch size: 4096
PPO epochs: 10
Algorithm: PPO from Stable Baselines
```

### 3c. Design Decisions & Tradeoffs

**Phase-based control (one action per 0.75s) vs. per-timestep control:**
The paper uses a very long action interval — one policy query per 0.75 seconds, executed as a Bezier curve over 180 sub-steps. This is coarser-grained than most locomotion RL work (which queries at 20-100Hz). The motivation is computational tractability — the policy produces smooth reference trajectories rather than noisy per-step commands.

The cost: the paper explicitly identifies this as a limitation. Because the policy only plans once per 0.75s, it cannot reactively adjust mid-phase when a tile sinks unexpectedly. The end-effector history partially compensates for this by giving the policy information about the gait pattern over previous phases, but within-phase responsiveness is limited.

**Spring-loaded prismatic joint terrain model:**
The terrain uses a physics-based spring model, not a visual deformation trick or pure parameter change. Each tile is an independent rigid body connected to a fixed base via a spring-loaded prismatic joint. When a foot contacts a tile, the tile physically moves down, compressing the spring. This produces realistic foot sinkage forces and height changes.

This is more expensive than changing `solimp/solref` in MuJoCo (your Level 1 approach), but cheaper than full granular simulation. For your project, this model is directly relevant — your Level 2 soft patches should use MuJoCo's equivalent: compliant contact geoms or flex bodies that physically sink under load.

**Fixed phase order:**
The policy always swings feet in the same order (FL → RR → FR → RL). This constraint simplifies learning but produces an asymmetric gait — the paper notes the robot relies on the left side more than the right due to this. Removing the constraint would allow gait discovery but significantly complicate training.

**Bezier curve action space:**
Bezier curves guarantee smooth trajectories by construction — the curve is C1 continuous at control point boundaries. For locomotion, smooth foot trajectories reduce impact forces and improve stability. The tradeoff is a complex 27D action space with non-intuitive semantics for each dimension.

**End-effector positions at 4ms and 8ms intervals (Section V-C):**
This is a separate fine-grained history component from the phase-level history. It detects foot-trapping events: when a foot is stuck on a tile threshold and not moving normally at the millisecond level, this signal captures that differently than the 0.75s phase-level history. Without it, the robot gets stuck on tile edges.

### 3d. Reward Function Analysis

The five-term reward is simple but carefully designed for the non-rigid terrain problem:

**Goal distance reward (αd = 10 forward, 1 backward):** The asymmetric scaling (10× reward for forward progress, 1× for backward) strongly biases the policy toward forward locomotion without making backward movement completely inaccessible for recovery.

**Goal orientation reward:** Bounded to 10° heading error maximum. Prevents the policy from learning to walk sideways toward a goal.

**Minimum height reward:** Binary threshold at 25cm. This is primarily a safety signal — it prevents the robot from learning to crawl or collapse onto soft terrain. On 5cm sinking terrain, the nominal base height would be ~30-35cm (from Table I), so 25cm gives reasonable margin.

**Torque minimization reward (threshold 140 Nm):** A ceiling rather than a continuous penalty — rewards are only given when average torque is below threshold. This prevents the policy from using excessive joint torques to fight against terrain compliance.

**Roll stabilization reward:** Bounded to 0.1 rad = 5.7°. Non-rigid terrain causes more lateral instability than rigid — tiles sink unequally under left and right feet. This term explicitly combats the resulting roll tendency.

---

## SECTION 4 — IMPLEMENTATION GUIDE

### 4a. Step-by-step pseudocode

```
# ============================================================
# TERRAIN CONSTRUCTION (MuJoCo equivalent)
# ============================================================

def create_elastic_terrain(grid_n=10, tile_size=0.20, sinking_depth=0.05):
    """Create NxN grid of spring-loaded tiles in MuJoCo/PyBullet.
    
    PyBullet implementation:
    - Create flat rigid base plane
    - For each tile (i,j): create box body of tile_size × tile_size × 0.02m
    - Connect each tile to base via prismatic joint (vertical axis only)
    - Add spring constraint: force = -stiffness * displacement
    
    Stiffness calibration:
    - Simulate robot standing still on terrain
    - Measure average sinkage under 4 feet
    - Adjust stiffness until average sinkage = target depth
    """
    tiles = []
    for i in range(grid_n):
        for j in range(grid_n):
            x = i * tile_size
            y = j * tile_size
            tile = create_box(position=[x, y, 0], 
                              size=[tile_size, tile_size, 0.02])
            
            # Prismatic joint: tile can only move in z direction
            joint = create_prismatic_joint(parent=base_plane, child=tile,
                                           axis=[0, 0, 1],
                                           limits=[-sinking_depth*2, 0])
            
            # Spring force: restore tile to z=0
            spring_stiffness = calibrate_stiffness(robot_mass=25.0,
                                                    target_sinkage=sinking_depth,
                                                    n_loaded_tiles=4)
            add_spring_constraint(joint, stiffness=spring_stiffness)
            tiles.append(tile)
    return tiles

def calibrate_stiffness(robot_mass, target_sinkage, n_loaded_tiles):
    """Find spring stiffness so that average sinkage = target when robot stands.
    F = mass * g / n_tiles = k * x
    k = mass * g / (n_tiles * target_sinkage)"""
    g = 9.81
    return (robot_mass * g) / (n_loaded_tiles * target_sinkage)


# ============================================================
# OBSERVATION CONSTRUCTION
# ============================================================

class NonRigidTerrainObservation:
    """102D observation space for non-rigid terrain locomotion.
    
    Critical components (from ablation studies):
    1. End-effector position history (48D) — REQUIRED
    2. End-effector at 4ms, 8ms ago (24D) — needed for foot trap detection
    3. End-effector velocity (12D) — REQUIRED for forward locomotion
    """
    
    def __init__(self, history_phases=4, phase_dt=0.75, control_dt=0.004):
        self.history_phases = history_phases  # 4 time steps = 3 previous + current
        self.phase_dt = phase_dt              # 0.75s per phase
        self.control_dt = control_dt          # 4ms control interval
        
        # Storage
        self.phase_ee_history = []   # foot positions at phase starts
        self.recent_ee = []          # foot positions at t-4ms, t-8ms
    
    def record_phase_start(self, ee_positions_base_frame):
        """Record end-effector positions at start of each phase."""
        self.phase_ee_history.append(ee_positions_base_frame.copy())
        if len(self.phase_ee_history) > self.history_phases:
            self.phase_ee_history.pop(0)
    
    def record_control_step(self, ee_positions_base_frame):
        """Record fine-grained end-effector positions at each 4ms control step."""
        self.recent_ee.insert(0, ee_positions_base_frame.copy())
        if len(self.recent_ee) > 3:  # keep last 3 (current, -4ms, -8ms)
            self.recent_ee.pop()
    
    def get_observation(self, robot):
        """Construct full 102D observation vector."""
        
        # 1. Base height above terrain under each hip joint (4D)
        # Height of hip joints relative to tile surface below each joint
        base_height_per_hip = []
        for hip in robot.hip_joints:
            hip_pos = hip.world_position
            tile_height = get_tile_height_at(hip_pos[0], hip_pos[1])
            base_height_per_hip.append(hip_pos[2] - tile_height)
        
        # 2. Gravity direction in base frame (3D)
        gravity_base = robot.base_rotation_matrix.T @ np.array([0, 0, -9.81])
        gravity_base = gravity_base / np.linalg.norm(gravity_base)
        
        # 3. Base velocities (6D): linear (3D) + angular (3D)
        base_linear_vel = robot.base_linear_velocity_base_frame
        base_angular_vel = robot.base_angular_velocity_base_frame
        
        # 4. Base pitch angle (1D)
        pitch = robot.euler_angles[1]
        
        # 5. ** CRITICAL ** End-effector position history (48D)
        # Positions of 4 feet at 4 time steps (3 prev phases + current)
        ee_history_flat = []
        current_ee = robot.get_foot_positions_base_frame()  # 4×3
        
        # Pad with current position if not enough history
        history_to_use = self.phase_ee_history + [current_ee]
        while len(history_to_use) < self.history_phases:
            history_to_use.insert(0, current_ee)
        
        for ee_step in history_to_use[-self.history_phases:]:
            ee_history_flat.extend(ee_step.flatten())  # 4 feet × 3D = 12D per step
        # Total: 4 steps × 12D = 48D
        
        # 6. End-effector positions at 4ms and 8ms ago (24D)
        recent_history = []
        if len(self.recent_ee) >= 2:
            recent_history = [self.recent_ee[1], self.recent_ee[2]]  # t-4ms, t-8ms
        else:
            recent_history = [current_ee, current_ee]
        recent_flat = []
        for ee_step in recent_history:
            recent_flat.extend(ee_step.flatten())  # 2 steps × 4 feet × 3D = 24D
        
        # 7. ** CRITICAL ** End-effector velocities (12D)
        # Velocity of each foot in base frame
        ee_velocities = robot.get_foot_velocities_base_frame().flatten()  # 4 × 3 = 12D
        
        # 8. Goal direction azimuth angle (1D)
        goal_pos_base = robot.world_to_base(robot.current_goal)
        goal_azimuth = np.arctan2(goal_pos_base[1], goal_pos_base[0]) * 180 / np.pi
        
        # 9. Goal position in base frame — frontal and lateral only (2D)
        goal_frontal_lateral = goal_pos_base[:2]
        
        # 10. Current phase index (1D)
        current_phase = robot.current_phase_index  # 0, 1, 2, or 3
        
        # Concatenate all components
        obs = np.concatenate([
            base_height_per_hip,        # 4D
            gravity_base,               # 3D
            base_linear_vel,            # 3D
            base_angular_vel,           # 3D
            [pitch],                    # 1D
            ee_history_flat,            # 48D ***CRITICAL***
            recent_flat,                # 24D
            ee_velocities,              # 12D ***CRITICAL***
            [goal_azimuth],             # 1D
            goal_frontal_lateral,       # 2D
            [current_phase],            # 1D
        ])
        
        assert len(obs) == 102, f"Expected 102D, got {len(obs)}D"
        return obs


# ============================================================
# BEZIER CURVE ACTION SYSTEM
# ============================================================

def cubic_bezier(P0, P1, P2, P3, t):
    """Evaluate cubic Bezier curve at parameter t ∈ [0, 1]."""
    return ((1-t)**3 * P0 + 
            3*(1-t)**2*t * P1 + 
            3*(1-t)*t**2 * P2 + 
            t**3 * P3)

class BezierMotionGenerator:
    """Generates smooth motion trajectories from Bezier curve action.
    
    Action (27D) defines control points for 3 curves:
    - Base position: 3 × 3D control points
    - Base orientation: 3 × 3D Euler angles
    - Swing foot: 3 × 3D position
    
    Execution: sample 180 points over 0.75s, solve IK, send joint angles.
    """
    
    def __init__(self, phase_duration=0.75, n_samples=180, control_dt=0.004):
        self.phase_duration = phase_duration
        self.n_samples = n_samples
        self.control_dt = control_dt
    
    def execute_phase(self, action_27d, robot, env):
        """Execute one locomotion phase from a 27D Bezier action."""
        # Unpack control points
        base_pos_cps = action_27d[:9].reshape(3, 3)    # 3 control points × 3D
        base_ori_cps = action_27d[9:18].reshape(3, 3)  # 3 × Euler angles
        foot_cps = action_27d[18:27].reshape(3, 3)     # 3 × 3D foot position
        
        # Current state = first control point (P0)
        P0_pos = robot.base_position
        P0_ori = robot.base_euler_angles
        P0_foot = robot.swing_foot_position
        
        # Full control point arrays: [P0, P1, P2, P3]
        pos_curve = np.vstack([P0_pos, P0_pos + base_pos_cps])
        ori_curve = np.vstack([P0_ori, P0_ori + base_ori_cps])
        foot_curve = np.vstack([P0_foot, P0_foot + foot_cps])
        
        # Sample 180 points and execute
        observations = []
        for i in range(self.n_samples):
            t = i / (self.n_samples - 1)
            
            target_base_pos = cubic_bezier(*pos_curve, t)
            target_base_ori = cubic_bezier(*ori_curve, t)
            target_foot_pos = cubic_bezier(*foot_curve, t)
            
            # Solve IK for joint angles
            joint_angles = robot.solve_ik(target_base_pos, target_base_ori,
                                           target_foot_pos)
            
            # Send to position controller (PyBullet generates torques internally)
            env.set_joint_positions(joint_angles)
            env.step_simulation(dt=self.control_dt)
            
            observations.append(env.get_state())
        
        return observations


# ============================================================
# REWARD FUNCTION
# ============================================================

def compute_reward(robot, goal_pos, prev_goal_pos, phase_torques):
    """5-term reward for non-rigid terrain locomotion."""
    
    # 1. Goal distance reward
    current_dist = np.linalg.norm(goal_pos)
    prev_dist = np.linalg.norm(prev_goal_pos)
    delta_dist = current_dist - prev_dist  # negative = moving closer
    
    if current_dist < prev_dist:
        alpha_d = 10.0  # moving toward goal: 10× reward
    else:
        alpha_d = 1.0   # moving away: 1× penalty
    R_d = alpha_d * (prev_dist - current_dist)  # positive when approaching
    
    # 2. Goal orientation reward
    goal_azimuth_deg = np.degrees(np.arctan2(goal_pos[1], goal_pos[0]))
    R_o = max(0, 0.02 * (10 - abs(goal_azimuth_deg)))
    
    # 3. Minimum height reward
    base_height = robot.base_height_above_terrain
    R_s = 0.1 if base_height > 0.25 else 0.0
    
    # 4. Torque minimization reward
    tau_ave = np.mean(np.abs(phase_torques))
    R_t = max(0, 0.004 * (140.0 - tau_ave))
    
    # 5. Roll angle reward
    roll_rad = robot.roll_angle
    R_r = max(0, 2.0 * (0.1 - abs(roll_rad)))
    
    return R_d + R_o + R_s + R_t + R_r


# ============================================================
# TRAINING LOOP
# ============================================================

def train(env, policy, value_fn, n_steps_total=int(1e7)):
    """Main training loop with terrain curriculum."""
    
    # Terrain types: rigid + 2cm, 3cm, 4cm, 5cm sinking
    terrain_types = ['rigid', '2cm', '3cm', '4cm', '5cm']
    
    # PPO from Stable Baselines — configure
    from stable_baselines3 import PPO
    model = PPO(
        policy='MlpPolicy',
        env=env,
        n_steps=2048,
        batch_size=4096,      # minibatch_size from paper
        n_epochs=10,          # PPO epochs
        gamma=0.95,           # discount
        learning_rate=2e-4,
        ent_coef=0.0,
        verbose=1
    )
    
    # Phase 1: Train on 2cm terrain first (curriculum)
    env.set_terrain('2cm')
    model.learn(total_timesteps=int(3e6))
    
    # Phase 2: Mixed terrain training (changes every 2 meters)
    env.set_terrain_mode('random_every_2m', terrains=terrain_types)
    model.learn(total_timesteps=int(7e6))
    
    return model
```

### 4b. Key Hyperparameters

| Parameter | Value | Controls | Sensitivity |
|-----------|-------|---------|------------|
| Phase duration | 0.75s | Action granularity | High — too short: jerky; too long: unresponsive |
| Bezier samples per phase | 180 | Control frequency (~240Hz) | Low — just needs to be high enough for smooth execution |
| Policy hidden dims | [256, 128] | Capacity | Low |
| Activation | tanh | Gradient flow | Low |
| γ (discount) | 0.95 | Effective horizon | Medium — low because phase-level episodes are long |
| Learning rate | 2×10⁻⁴ | Convergence speed | Low — standard |
| Minibatch size | 4096 | GPU efficiency | Low |
| PPO epochs | 10 | Sample reuse | Low |
| Tile size | 20cm × 20cm | Terrain resolution | Medium — must be larger than foot contact area |
| Target sinking depths | 2, 3, 4, 5cm | Compliance range | High — 5cm is near robot's ability limit |
| Termination height | 20cm | Episode length | Medium |
| Termination pitch | ±15° | Episode length | Medium |
| EE history depth | 4 phases = 3s | Memory of terrain patterns | HIGH — removing even 1 phase causes failure |
| Fine history | 4ms, 8ms | Foot trap detection | Medium — needed for edge cases |
| Goal distance | 2m per sub-goal | Training structure | Low |
| Terrain change interval | every 2m or 8m | Curriculum structure | Medium — 8m allows per-stiffness adaptation |

### 4c. Data Requirements

**Observation space (102D):**
Full breakdown provided in Section 3b above. Key components:
- Base height relative to terrain: 4D (requires terrain height query per hip joint)
- EE position history: 48D (requires storing 3 previous phase-start foot positions)
- Recent EE history: 24D (requires storing last 2 control-step foot positions)
- EE velocities: 12D (compute from consecutive foot positions)

**Terrain model:**
- PyBullet: prismatic joints with spring constraints
- MuJoCo equivalent: flex bodies OR low-stiffness contact geoms (solimp/solref)
- Stiffness calibration: simulate static stance, adjust until average sinkage = target

**Robot model:** Laikago URDF (available from Unitree GitHub)

### 4d. Dependencies

| Tool | Purpose |
|------|---------|
| PyBullet | Physics simulation (paper uses this) |
| MuJoCo | Alternative — better for your project (already your choice) |
| OpenAI Gym | RL environment interface |
| Stable Baselines (v1) | PPO implementation |
| Stable Baselines3 | Updated version — use this |
| NumPy | Array operations |
| Laikago/Unitree Go1 URDF | Robot model |

---

## SECTION 5 — BENCHMARK RESULTS

### 5a-5b. Evaluation
- Simulated Laikago in PyBullet
- 4 terrain scenarios: Tv2, Tv8, T²c, T⁵c
- Metrics: base height trajectory smoothness, target landing heights per foot, qualitative locomotion success

### 5c. Quantitative results (Table I)

**Base height statistics across scenarios:**

| Scenario | μ(h_b) cm | σ(h_b) cm |
|----------|-----------|-----------|
| Tv2 (avg) | 33.1 | 3.1 |
| T²v2 (2cm tiles in mixed) | 33.7 | 3.2 |
| T⁵v2 (5cm tiles in mixed) | 32.8 | 3.2 |
| T²v8 (2cm, change every 8m) | 33.0 | 3.2 |
| T⁵v8 (5cm, change every 8m) | 32.6 | 3.1 |
| T²c (constant 2cm) | 35.8 | 3.3 |
| T⁵c (constant 5cm) | 33.5 | 2.9 |

All scenarios show consistent base height (~33cm) and similar standard deviation (~3cm), demonstrating stable locomotion across terrain types.

**Key finding on policy specialization:**
- T²c and T⁵c learn different landing heights per foot → specialized policies per stiffness
- Tv2 (changes every 2m): learns conservative policy matching T⁵c → treats all terrain as worst case
- Tv8 (changes every 8m): learns different landing heights for T² and T⁵ → achieves partial terrain identification from history

### 5d. Qualitative findings
- Policy cannot adapt within 2m distance of terrain change (Tv2 is conservative)
- Policy CAN adapt when terrain persists for 8m (Tv8 learns different strategies per stiffness)
- Without end-effector history: training fails completely
- Without end-effector velocity: robot learns to balance but not walk
- Robot learns higher foot clearance on softer terrain (implicit adaptation)
- Phase-based control is the main limitation: cannot react mid-phase to unexpected sinkage

### 5e. What removing components costs

| Component removed | Effect |
|------------------|--------|
| End-effector position history | Complete training failure |
| End-effector velocity | Robot balances but never walks |
| Recent EE at 4ms/8ms | Robot gets stuck on tile edges |
| Action space bounds | Policy fails to converge |
| Gravity direction + pitch | Base exhibits unnatural movements |

---

## SECTION 6 — WHEN TO USE THIS METHOD

### 6a. Ideal conditions
- Task requires locomotion on elastically compliant terrain
- Terrain compliance is unknown or varies
- Phase-based control (slow gait decisions) is acceptable
- Simulation-only project (no sim-to-real transfer attempted)

### 6b. Practical sweet spot
Any project where soft/sinking terrain is the primary challenge and the robot needs to adapt its gait pattern to the terrain's compliance level. The end-effector history observation is generalizable beyond the specific Bezier curve framework.

### 6c. Scale considerations
- Laikago: 25kg, ~55cm base length, 5cm sinkage = ~9% of leg length
- Scales to larger robots with proportionally deeper sinkage
- Tile size should be ~foot contact area or smaller for accurate sinkage modeling

### 6d. Computational profile
- Training: moderate (not reported, but PPO with PyBullet is CPU-feasible)
- No GPU required for training (CPU-based PyBullet)
- Inference: fast MLP, phase-based (only 4 queries per gait cycle)

---

## SECTION 7 — WHEN NOT TO USE THIS METHOD

### 7a. Failure modes
- Terrain stiffness changes faster than 2m: policy learns conservative worst-case behavior
- Very deep sinkage (>5cm): training becomes unstable, robot cannot overcome tile edges
- Need for rapid reactive control: phase-based control (0.75s intervals) is too slow

### 7b. Known limitations (from authors)
- Non-interactive to ground: motion planned once per 0.75s, cannot adjust mid-phase
- Flat terrain only: sloped compliant terrain not tested
- Elastic only: plastic deformation (permanent) not modeled
- Fixed phase order: cannot adapt gait sequence

### 7c. Hidden assumptions
- Terrain deforms elastically (springs back when foot lifts)
- Tile dimensions ~foot size (20cm tiles for ~10cm feet)
- Sinkage depth is bounded and predictable enough to navigate
- Robot can physically step over 5cm tile edges at height transitions

---

## SECTION 8 — EXPECTED RESULTS

### 8a. What good output looks like
- Robot walks forward on all terrain types with ~33cm base height
- Smoother gait on soft terrain (more compliant contact = better damping)
- Visibly higher foot clearance on softer terrain
- Can transition between rigid and soft without falling

### 8b. Characteristic artifacts
- Asymmetric gait (left side favored due to fixed phase order)
- Conservative landing heights on mixed-stiffness terrain
- Slight base height oscillation (~3cm std) at phase transitions

### 8c. What failure looks like
- Robot balances but doesn't walk: end-effector velocity missing from observation
- Training doesn't converge: action bounds too wide or too narrow
- Robot gets stuck on tile edges: missing recent EE history (4ms, 8ms terms)

---

## SECTION 9 — CONNECTIONS TO OTHER WORK

### 9a. Key predecessors
- Tan et al. 2018 (your paper 5) — contact parameter randomization for quadruped
- Peng et al. 2018 (your paper 4) — dynamics randomization
- Lee et al. 2020 (your paper 1) — teacher-student quadruped locomotion

### 9b. Key successors
This paper's terrain model (spring-loaded tiles) is directly extended in subsequent work on deformable terrain locomotion. The observation design insight (EE history + velocity critical for non-rigid terrain) is validated by later papers.

### 9c. Position in field
**Targeted contribution paper.** This is not a landmark systems paper like Lee et al. 2020 or RMA. It's a focused paper that addresses one specific gap: no prior DRL work had demonstrated locomotion on physically deformable terrain in simulation. Its contribution is the terrain model and the observation design insight. For your project, it's the most directly relevant paper — it builds exactly the thing you want to build, at Level 2 of your terrain hierarchy.

---

## SECTION 10 — PROJECT APPLICATION MAPPING

**Your project:** MuJoCo quadruped, Level 1 (contact param randomization) and Level 2 (soft patches). Compare Policy A (rigid) vs. Policy B (randomized).

### 10a. How this paper applies
This is your **closest prior work** and your **primary Level 2 terrain reference**. Everything in Section IV is a direct template for your Level 2 implementation. The spring-loaded tile model IS your Level 2 target — translated from PyBullet to MuJoCo.

### 10b. What is directly usable vs. needs adaptation

| Component | Usable | Notes |
|-----------|--------|-------|
| Spring-loaded tile terrain model | ✅ Direct | Translate from PyBullet prismatic+spring to MuJoCo flex bodies or compliant contact geoms |
| End-effector position history (48D) | ✅ Critical | Must include in Policy B observation — this is the key finding |
| End-effector velocity (12D) | ✅ Critical | Required for forward locomotion on compliant terrain |
| Fine-grained EE history (4ms, 8ms) | ✅ Recommended | Add for tile edge robustness |
| 5-term reward function | ✅ Adapt | Replace goal-based reward with velocity tracking; keep torque + height + roll terms |
| Terrain stiffness curriculum | ✅ Direct | 2cm → 3cm → 4cm → 5cm matches your Level 2 progression |
| Bezier curve action space | ❌ Replace | Use standard joint position targets; Bezier adds complexity without benefit for your comparison study |
| Phase-based control (0.75s) | ❌ Replace | Use continuous per-timestep control for more reactive adaptation |
| PPO from Stable Baselines | ✅ Direct | Use Stable Baselines3 (updated version) |

### 10c. The most important finding for your project

**The end-effector history observation is not optional — it's a hard requirement for non-rigid terrain locomotion.** Section V-B states explicitly: "The learning failed with the memory of even one less phase."

This has a direct implication for how you design Policy B's observation space: Policy B must include a history of end-effector positions over recent timesteps, not just current foot positions. If you give Policy A and Policy B identical observations (current state only), Policy B may fail to generalize to Level 2 terrain the same way this paper's policy fails without history.

**Design recommendation:** Give Policy B a 10-20 step history of foot positions and velocities concatenated to the standard observation. This is the simplified version of the full 48D EE history — much simpler to implement but captures the same physical insight. Policy A can have current-state-only observations (it doesn't need history to walk on rigid ground).

### 10d. MuJoCo Level 2 terrain implementation

MuJoCo 3.x has three approaches to simulating compliant sinking terrain:

**Option 1 (Recommended — closest to this paper):** Flex bodies
```xml
<flex name="terrain_patch" dim="2" radius="0.01"
      stiffness="1000" damping="10">
  <vertex pos="0 0 0" ... />
</flex>
```
Use MuJoCo's flex body system to create a deformable ground patch. Set stiffness to produce ~5cm sinkage under foot contact.

**Option 2 (Your Level 1+2 bridge):** Compliant contact parameters
```xml
<geom name="soft_patch" type="box" pos="0 0 0" size="0.2 0.2 0.02"
      solimp="0.99 0.99 0.001" solref="0.02 1"/>
```
Use `solimp` and `solref` to create springy contact. Simpler than flex bodies, doesn't geometrically deform but produces compliant force response.

**Option 3 (Closest physics):** Multiple rigid tiles on springs
```xml
<body name="tile_00" pos="0.0 0.0 0.0">
  <joint name="tile_00_slide" type="slide" axis="0 0 1" 
         limited="true" range="-0.05 0"/>
  <geom type="box" size="0.1 0.1 0.01"/>
</body>
```
Plus a tendon or equality constraint acting as a spring:
```xml
<tendon>
  <fixed name="spring_00">
    <joint joint="tile_00_slide" coef="1000"/>
  </fixed>
</tendon>
```

For your Level 2, **Option 3** most directly matches this paper's model and gives you the most physical realism for the sinking behavior. **Option 2** (solimp/solref) is simpler and bridges Level 1 and Level 2 — you're already using it in Level 1, just making it more compliant.

### 10e. What your Policy B observation should include

Based on this paper's findings, Policy B's observation should add:

```python
# Standard observation (both Policy A and B)
standard_obs = [joint_positions(12), joint_velocities(12), 
                base_orientation(4), base_angular_vel(3),
                base_linear_vel(3), commands(3)]  # ~37D

# Policy B additional: EE history
# Simplified version of paper's 48D EE history
ee_history = []
for t in [0, -5, -10, -20]:  # current + 3 recent timesteps
    ee_history.append(foot_positions_base_frame[t])  # 4 feet × 3D = 12D each
ee_history_flat = np.concatenate(ee_history)  # 48D

# EE velocities
ee_velocities = foot_velocities_base_frame.flatten()  # 4 × 3 = 12D

policy_b_obs = np.concatenate([standard_obs, ee_history_flat, ee_velocities])
# Total: 37 + 48 + 12 = 97D
```

Policy A keeps the 37D standard observation. The comparison is then: does adding EE history help Policy B generalize to Level 2 terrain? This is the most direct test of this paper's finding in your context.

### 10f. The conservative policy finding and your experiment

Table I shows that when terrain stiffness changes every 2m, the policy learns a conservative strategy matching the hardest terrain (T5c) — it "plays it safe" for all terrain as if everything is maximally compliant.

This is directly relevant to your experiment: **Policy B trained on the full compliance spectrum may learn a conservatively unified gait rather than a dynamically adaptive one.** Your evaluation should check not just whether Policy B succeeds but whether it shows different behavior on different terrain types (like Tv8 does) or whether it converges to one conservative strategy for all terrains.

To enable terrain-adaptive behavior (like Tv8), your terrain in Policy B's training should persist long enough per terrain type for the policy to identify it. Changing terrain parameters every episode (standard RL) is equivalent to the paper's Tv2 setting — the policy sees all terrain as equally uncertain. Changing less frequently (closer to Tv8) would allow the policy to learn to discriminate. **This is a design choice worth documenting explicitly in your project.**

---

## IMPLEMENTATION CHEAT SHEET

### 3 most important things to understand

1. **End-effector position history is the hardest dependency to implement but non-negotiable for non-rigid terrain.** The paper reports "learning failed with the memory of even one less phase" — this is not a soft improvement, it's a hard requirement. For your Level 2 terrain, Policy B must have foot position history in its observation. The simplified version (last 3-4 timestep foot positions concatenated) is sufficient to capture the same physical signal.

2. **End-effector velocity is what enables locomotion vs. just balancing.** Without it, the policy learns to stand still safely. For your project: include foot velocities in Policy B's observation as a matter of course. The difference between "robot that can balance on soft terrain" and "robot that can walk on soft terrain" comes down to this single signal.

3. **Terrain that changes too frequently forces conservative universal behavior.** When the paper changes stiffness every 2m, the policy defaults to treating all terrain as maximally soft. When it changes every 8m, the policy learns to adapt. Your experiment design determines whether Policy B learns genuine terrain adaptation or just conservative robustness — both are valid findings to document, but you need to know which you're testing.

### 5-step implementation recipe

1. **Build your Level 2 terrain** in MuJoCo using Option 3 (spring-loaded rigid tiles) or Option 2 (solimp/solref compliance). Calibrate stiffness to produce 2-5cm sinkage under robot's standing weight.
2. **Add EE history to Policy B's observation** — store foot positions in base frame at the last 3-4 timesteps and concatenate. Add foot velocities. This is your critical differentiator from Policy A.
3. **Use the paper's 5-term reward** as a starting template, replacing the goal-distance reward with velocity tracking (your project doesn't use waypoint goals).
4. **Implement terrain curriculum** — start Policy B on 2cm sinking only, then add 3cm, 4cm, 5cm progressively as training converges per level.
5. **Document which observation Policy A and B receive** in your README — this is the most important design choice and determines what your ablation measures.

### 3 most common implementation mistakes

1. **Not including EE history in Policy B's observation** — this is the most common failure mode for non-rigid terrain RL. If Policy B has the same observation as Policy A, it cannot detect compliance from foot feedback. Always include foot position history.

2. **Setting tile sizes too small** — tiles smaller than the foot contact area produce unstable contact dynamics (foot partially spans two tiles simultaneously). Use tiles at least 1.5× the foot contact diameter. For a standard quadruped with 3-4cm feet, tiles should be ≥10cm.

3. **Skipping the terrain curriculum** — trying to train directly on 5cm sinking terrain fails. Always start with 2cm and progress. This matches the paper's finding that T⁵c requires curriculum learning via Tv2 first.

### Most important hyperparameter to tune first
**End-effector history length** — how many timesteps of foot position history to include. The paper uses 4 phase starts (3 seconds at 0.75s/phase). For your per-timestep control (50Hz), equivalent is ~150 timesteps, which is expensive. Start with 20-step history (0.4s at 50Hz) — this is RMA's adaptation window and likely sufficient to detect compliance changes from gait perturbation. If Policy B still fails to adapt on Level 2, increase to 50 steps before changing anything else.

### Should I use this method for my project?
**Yes — this paper IS your Level 2 terrain implementation.** Use the spring-loaded tile model in MuJoCo (Option 3), include EE position history and velocity in Policy B's observation, and use the terrain curriculum (2cm → 5cm). The Bezier curve action space can be replaced with standard joint position targets. The key insight — EE history enables compliance identification, EE velocity enables locomotion — directly informs how Policy B should differ from Policy A in observation design, not just in training terrain.
