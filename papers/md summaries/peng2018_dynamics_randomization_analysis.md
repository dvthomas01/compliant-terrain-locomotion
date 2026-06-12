# Paper Analysis: Peng et al. 2018
## Sim-to-Real Transfer of Robotic Control with Dynamics Randomization
**Authors:** Xue Bin Peng, Marcin Andrychowicz, Wojciech Zaremba, Pieter Abbeel
**Venue:** ICRA 2018
**arXiv:** 1710.06537

---

## SECTION 1 — PROBLEM STATEMENT

**One-sentence summary:** Policies trained in simulation overfit to the specific physics of the simulator and fail on real robots — this paper shows that randomizing the physics parameters during training forces policies to generalize across a wide range of dynamics, enabling zero-shot transfer to the real world without any real-world data collection.

The failure mode is fundamental: any simulator is an approximation of reality. Friction, mass, damping, actuator latency — all are imprecisely modeled. A policy that learns to exploit the exact contact dynamics of the simulated puck will fail catastrophically when the real puck behaves even slightly differently. Prior approaches either tried to make the simulator more accurate (expensive, never perfect) or collected real-world data to fine-tune the policy (dangerous, costly). Memoryless feedforward policies trained with randomization improved robustness but still had a ceiling — they couldn't adapt within an episode because they had no memory of what the current episode's dynamics feel like.

The gap this paper fills: how to train a policy that can identify the current episode's dynamics from recent experience and adapt its behavior accordingly — entirely in simulation, without ever touching a real robot. The key insight is that the LSTM hidden state learns to implicitly perform system identification from the history of state-action pairs.

---

## SECTION 2 — CORE IDEA & INTUITION

The central insight is: **randomize the physics so widely that reality is just another sample from your training distribution, and give the policy memory so it can figure out which sample it's currently in.**

Two complementary ideas:

1. **Dynamics randomization:** Sample new physics parameters (mass, friction, damping, latency, noise) at the start of every training episode. The policy sees such a wide variety of physics during training that the real world's physics falls within the range it has already learned to handle.

2. **Recurrent policy (LSTM):** A memoryless policy is given the same current state regardless of whether it's in a high-friction or low-friction world. It can't adapt. An LSTM policy accumulates a history of states and actions, and its hidden state becomes an implicit estimate of the current dynamics. The policy learns to read its own recent behavior to figure out what world it's in, then act accordingly.

Why simpler approaches fail: a feedforward network with randomization (FF) achieves 67% on the real robot vs. 89% for LSTM (Table II). A feedforward network augmented with explicit history (FF+Hist) achieves 70%. The LSTM outperforms explicit history because it learns what aspects of history matter for dynamics identification, rather than treating all past observations equally.

---

## SECTION 3 — METHODOLOGY

### 3a. Pipeline / Architecture

```
TRAINING PIPELINE
==================
For each update iteration:
    1. Sample goal g ~ ρ_g           (target puck location)
    2. Sample dynamics μ ~ ρ_μ       (friction, mass, damping, latency, noise)
    3. Run episode with policy π(a_t | s_t, z_t, g):
       - z_t = LSTM hidden state, updated each step from (s_t, a_{t-1})
       - Dynamics μ held fixed for entire episode
       - Timestep Δt varies every step: Δt ~ Δt_0 + Exp(λ)
       - Observation noise applied: σ = 5% of running std per feature
    4. Store episode (τ, rewards, g, μ) in replay buffer M
    5. With probability k=0.8: relabel with HER goal
    6. Update policy π (RDPG) and value function Q (omniscient critic)

DEPLOYMENT
===========
- Same policy, no fine-tuning
- LSTM hidden state z_t carries implicit dynamics estimate
- Policy adapts within episode as LSTM accumulates experience
```

**Two-branch network architecture:**

```
Recurrent branch (dynamics inference):
  Input: (s_t, a_{t-1})   [state + previous action — contains dynamics info]
  → Embedding layer: 128 FC units → ReLU
  → LSTM: 128 units
  → Output: z_t (hidden state, implicitly encodes dynamics)

Feedforward branch (current state + goal):
  Input: (s_t, g)          [current state + goal — not dynamics-specific]
  → Processed independently

Combined:
  → Concatenate [z_t, FF features]
  → 128 FC → ReLU → 128 FC → ReLU → action a_t (tanh output)

Value network (omniscient critic, training only):
  Same structure, additionally receives μ as input to FF branch
  → Outputs Q(s_t, a_t, z_t, g, μ)
  Note: μ given to critic but NOT to policy at training or deployment
```

**Why separate branches?**
- The goal g has no information about dynamics → feedforward branch
- The state s_t and previous action a_{t-1} contain dynamics information (how the arm moved given the commanded action reveals friction, mass, latency) → recurrent branch
- Giving s_t to both branches gives the current step more direct influence on actions without requiring dynamics inference for simple current-state responses

### 3b. Key Algorithms & Formulas

**Modified RL objective with dynamics distribution:**
```
π* = argmax_π E_{μ~ρ_μ} [E_{τ~p(τ|π,μ)} [Σ_{t=0}^{T-1} r(s_t, a_t)]]
```
- `μ`: dynamics parameter vector (95 parameters total)
- `ρ_μ`: distribution over dynamics parameters (uniform or log-uniform over ranges in Table I)
- Standard RL objective, but expectation taken over dynamics distribution, not just trajectories

**Sparse binary reward:**
```
r(s_t, g) = 0   if puck within 0.07m of target g
           = -1  otherwise
```

**Hindsight Experience Replay (HER) goal relabeling:**
```
For episode ending in state s_T:
    g' = m(s_T)     # goal satisfied by final state
    r_t' = r(s_t, g') for all t    # recompute rewards under new goal
```
- With probability k=0.8: relabel with HER
- Converts failed episodes into successful ones under different goals
- Critical for learning from sparse rewards

**LSTM update (recurrent branch):**
```
z_t = LSTM(z_{t-1}, embed(s_t, a_{t-1}))
```
- `z_{t-1}`: previous hidden state (carries dynamics estimate)
- `embed(·)`: 128-unit FC embedding layer
- `z_t`: updated hidden state — implicit dynamics encoding

**RDPG policy gradient:**
```
∇_θ J ≈ (1/T) Σ_t [∂Q(s_t, â_t, z_t, g, μ)/∂a · ∂â_t/∂θ]
```
where `â_t = π_θ(s_t, z_t, g)` is the deterministic policy output.

**Omniscient critic:**
```
Q(s_t, a_t, z_t, g, μ) → scalar
```
- μ is given to critic during training (it knows the true dynamics)
- μ is NOT given to policy — policy must infer from history
- This reduces variance of critic estimates: knowing the true physics helps the critic evaluate whether an action was good given those physics

**Action timestep randomization:**
```
Δt ~ Δt_0 + Exp(λ)
Δt_0 = 0.04s (default control timestep)
λ sampled from [125, 1000] s^{-1} per episode
```
- Models controller latency variability
- Most impactful single parameter to randomize (see ablation Table III: removing it drops real-world success from 89% to 29%)

**Observation noise:**
```
noise_i ~ N(0, 0.05 * running_std_i)
```
- 5% of running standard deviation per feature
- Applied every timestep (varies timestep to timestep)
- Second most impactful parameter (removing drops success to 25%)

### 3c. Design Decisions & Tradeoffs

**LSTM vs. FF + explicit history:**
The LSTM outperforms FF+Hist (8-step explicit history) — 89% vs 70% real-world success. Why? FF+Hist treats all 8 past (state, action) pairs equally via concatenation. The LSTM learns which aspects of history are informative for dynamics identification. It selectively attends to informative events (unexpected arm movement, contact force anomalies) rather than weighting all history uniformly.

**Logarithmic vs. uniform sampling of physics parameters:**
Mass, damping, friction, and controller gains are logarithmically sampled. This makes sense because physical properties have multiplicative effects — doubling friction is qualitatively different from adding a fixed amount. Log-sampling ensures the policy sees proportionally similar changes across the range, not just extreme values dominating.

**Action timestep as latency model:**
Rather than explicitly modeling the communication delay stack of a real robot controller, the paper uses a random action timestep as a simple proxy. This captures the effect of latency (actions applied for varying durations) without requiring accurate modeling. The exponential distribution Exp(λ) models memoryless latency, which is a reasonable approximation for communication jitter.

**Omniscient critic (μ given to Q, not π):**
Standard in multi-task and privileged learning. Giving the critic ground-truth dynamics reduces value function variance, producing cleaner gradients for policy improvement. The policy doesn't receive μ because it won't be available at deployment — this asymmetry is intentional and critical.

**HER probability k=0.8:**
80% of training updates use a relabeled goal. This is high but appropriate for sparse rewards — without HER, the agent almost never receives a positive reward and cannot learn. HER converts the learning problem from extremely sparse to reasonably dense.

**95 randomized parameters:**
This is the total count: 7 links × mass + damping (per joint) + puck mass, friction, damping + table height + controller gains + timestep + noise. The comprehensiveness matters — leaving out any important parameter (especially timestep and noise) significantly degrades transfer.

### 3d. Reward Function (Loss Function)

This paper uses a sparse binary reward, not a shaped reward:
```
r(s_t, g) = 0   if ||puck_pos - g|| < 0.07m
           = -1  otherwise
```

No intermediate shaping. The HER augmentation is what makes this tractable — without it, the policy would rarely see reward=0 (success) and learning would stall.

For the value function (critic), the TD target with HER:
```
q_t = r_t + γ * Q(s_{t+1}, â_{t+1}, z_{t+1}, g, μ)
ΔQ = q_t - Q(s_t, a_t, z_t, g, μ)
```

---

## SECTION 4 — IMPLEMENTATION GUIDE

### 4a. Step-by-step pseudocode

```
# ============================================================
# DYNAMICS RANDOMIZATION TRAINING
# ============================================================

initialize:
    policy_LSTM = LSTMPolicy(state_dim=52, goal_dim=3, action_dim=7)
    critic_LSTM = LSTMCritic(state_dim=52, goal_dim=3, action_dim=7, mu_dim=95)
    replay_buffer M = []
    optimizer = Adam(lr=5e-4)
    k_HER = 0.8  # HER probability

for iteration in range(8000):
    # Sample goal and dynamics for this episode
    g = sample_goal()              # target puck position
    μ = sample_dynamics()          # 95 physics parameters from Table I
    
    # Run episode with current dynamics
    trajectory = []
    z_t = zeros(128)  # initial LSTM hidden state
    s_t = env.reset(mu=μ)
    
    for t in range(100):  # 100 control timesteps
        # Sample action timestep latency for this step
        delta_t = delta_t0 + exponential(λ)  # λ fixed for episode
        
        # Apply observation noise
        s_t_noisy = s_t + N(0, 0.05 * running_std)
        
        # Policy forward pass
        a_t, z_{t+1} = policy_LSTM(s_t_noisy, z_t, g)
        
        # Step environment for delta_t seconds
        s_{t+1}, r_t = env.step(a_t, delta_t=delta_t, mu=μ)
        r_t = reward(s_{t+1}, g)  # binary reward
        
        trajectory.append((s_t, a_t, r_t, s_{t+1}))
        z_t = z_{t+1}
        s_t = s_{t+1}
    
    # Store trajectory with goal and dynamics
    M.append((trajectory, g, μ))
    
    # Sample training batch
    episode, g_ep, μ_ep = sample(M)
    
    # HER: relabel goal with probability k
    if random() < k_HER:
        g_ep = extract_goal(episode[-1].s)  # goal from final state
        # Recompute rewards under new goal
        for t in episode:
            t.r = reward(t.s_next, g_ep)
    
    # Compute LSTM states for full episode
    z_states = compute_lstm_states(episode, policy_LSTM)
    y_states = compute_lstm_states(episode, critic_LSTM)
    
    # RDPG update
    for t in range(len(episode)):
        s_t, a_t, r_t, s_next = episode[t]
        
        # TD target (omniscient critic sees μ)
        a_next = policy_LSTM(s_next, z_states[t+1], g_ep)
        q_target = r_t + γ * critic_LSTM(s_next, a_next, y_states[t+1], g_ep, μ_ep)
        
        # Critic loss
        q_pred = critic_LSTM(s_t, a_t, y_states[t], g_ep, μ_ep)
        critic_loss = (q_target.detach() - q_pred).pow(2)
        
        # Policy gradient (through critic)
        a_pred = policy_LSTM(s_t, z_states[t], g_ep)
        policy_loss = -critic_LSTM(s_t, a_pred, y_states[t], g_ep, μ_ep)
    
    update(critic_loss + policy_loss)

# ============================================================
# DEPLOYMENT (no changes to policy)
# ============================================================

z_t = zeros(128)  # reset LSTM state at episode start
for t in range(episode_length):
    s_t = read_sensors()
    a_t, z_{t+1} = policy_LSTM(s_t, z_t, g)
    execute_action(a_t)
    z_t = z_{t+1}
    # LSTM hidden state z_t implicitly tracks dynamics estimate
    # No explicit system identification required
```

### 4b. Python implementation skeleton

```python
import torch
import torch.nn as nn
import numpy as np
from collections import deque

# ============================================================
# NETWORK ARCHITECTURES
# ============================================================

class LSTMPolicy(nn.Module):
    """Recurrent policy with two branches:
    - Recurrent branch: infers dynamics from (state, prev_action) history
    - Feedforward branch: processes goal (no dynamics info)
    Combined output: action"""
    
    def __init__(self, state_dim=52, goal_dim=3, action_dim=7,
                 embed_dim=128, lstm_dim=128, hidden_dim=128):
        super().__init__()
        
        # Recurrent branch: dynamics inference from history
        self.state_embed = nn.Sequential(
            nn.Linear(state_dim + action_dim, embed_dim),  # (s_t, a_{t-1})
            nn.ReLU()
        )
        self.lstm = nn.LSTMCell(embed_dim, lstm_dim)
        
        # Feedforward branch: goal processing
        self.goal_embed = nn.Sequential(
            nn.Linear(goal_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Combined head
        self.combined = nn.Sequential(
            nn.Linear(lstm_dim + state_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # actions bounded
        )
        
        # Action scaling (to joint angle bounds)
        # TODO: set action_scale to span joint angle ranges
        self.action_scale = nn.Parameter(torch.ones(action_dim), requires_grad=False)
    
    def forward(self, s_t, a_prev, z_prev, g):
        """
        Args:
            s_t: current state [batch, state_dim]
            a_prev: previous action [batch, action_dim]
            z_prev: previous LSTM hidden/cell state tuple
            g: goal [batch, goal_dim]
        Returns:
            action: [batch, action_dim]
            z_next: updated LSTM state tuple
        """
        # Recurrent branch: embed (state, prev_action) and update LSTM
        sa = torch.cat([s_t, a_prev], dim=-1)
        embedded = self.state_embed(sa)
        h_next, c_next = self.lstm(embedded, z_prev)  # update hidden state
        
        # Feedforward branch: process goal
        goal_features = self.goal_embed(g)
        
        # Combine: LSTM hidden state + current state (direct access) + goal
        combined_input = torch.cat([h_next, s_t, goal_features], dim=-1)
        action = self.combined(combined_input) * self.action_scale
        
        return action, (h_next, c_next)
    
    def init_hidden(self, batch_size=1):
        """Initialize LSTM hidden state to zeros."""
        return (torch.zeros(batch_size, 128),
                torch.zeros(batch_size, 128))


class LSTMCritic(nn.Module):
    """Omniscient critic: knows true dynamics μ during training.
    Same structure as policy but also receives action a_t and μ."""
    
    def __init__(self, state_dim=52, action_dim=7, goal_dim=3,
                 mu_dim=95, embed_dim=128, lstm_dim=128, hidden_dim=128):
        super().__init__()
        
        # Recurrent branch (same as policy — infers dynamics from history)
        self.state_embed = nn.Sequential(
            nn.Linear(state_dim + action_dim, embed_dim), nn.ReLU()
        )
        self.lstm = nn.LSTMCell(embed_dim, lstm_dim)
        
        # Feedforward branch: goal + current action + true dynamics μ
        self.ff_branch = nn.Sequential(
            nn.Linear(goal_dim + action_dim + mu_dim, hidden_dim), nn.ReLU()
        )
        
        # Combined head → scalar Q value
        self.combined = nn.Sequential(
            nn.Linear(lstm_dim + state_dim + hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # linear output for Q
        )
    
    def forward(self, s_t, a_t, a_prev, y_prev, g, mu):
        """
        Args:
            s_t: current state
            a_t: QUERY action (action being evaluated)
            a_prev: previous action (for LSTM dynamics inference)
            y_prev: LSTM hidden state of critic
            g: goal
            mu: true dynamics parameters [batch, 95]
        Returns:
            Q value scalar
        """
        sa = torch.cat([s_t, a_prev], dim=-1)
        embedded = self.state_embed(sa)
        h_next, c_next = self.lstm(embedded, y_prev)
        
        # FF branch: goal + query action + true dynamics
        ff_input = torch.cat([g, a_t, mu], dim=-1)
        ff_features = self.ff_branch(ff_input)
        
        combined_input = torch.cat([h_next, s_t, ff_features], dim=-1)
        return self.combined(combined_input).squeeze(-1), (h_next, c_next)


# ============================================================
# DYNAMICS RANDOMIZATION
# ============================================================

class DynamicsRandomizer:
    """Samples physics parameters for each episode.
    Based on Table I from the paper."""
    
    # Parameter ranges from Table I
    RANGES = {
        'link_mass_multiplier': (0.25, 4.0),        # log-uniform
        'joint_damping_multiplier': (0.2, 20.0),     # log-uniform
        'puck_mass': (0.1, 0.4),                     # uniform (kg)
        'puck_friction': (0.1, 5.0),                 # log-uniform
        'puck_damping': (0.01, 0.2),                 # uniform (Ns/m)
        'table_height': (0.73, 0.77),                # uniform (m)
        'controller_gains_multiplier': (0.5, 2.0),   # log-uniform
        'action_timestep_lambda': (125.0, 1000.0),   # uniform (s^{-1})
    }
    
    def sample(self):
        """Sample a complete set of dynamics parameters for one episode."""
        params = {}
        
        # Log-uniform sampling for multiplicative parameters
        for key in ['link_mass_multiplier', 'joint_damping_multiplier',
                    'puck_friction', 'controller_gains_multiplier']:
            lo, hi = self.RANGES[key]
            params[key] = np.exp(np.random.uniform(np.log(lo), np.log(hi)))
        
        # Uniform sampling for additive parameters
        for key in ['puck_mass', 'puck_damping', 'table_height', 'action_timestep_lambda']:
            lo, hi = self.RANGES[key]
            params[key] = np.random.uniform(lo, hi)
        
        return params
    
    def apply_to_sim(self, sim, params):
        """Apply sampled parameters to MuJoCo simulation."""
        # TODO: use sim.model.body_mass, sim.model.dof_damping etc.
        for i, link_mass in enumerate(sim.model.body_mass):
            sim.model.body_mass[i] = link_mass * params['link_mass_multiplier']
        
        for i, damping in enumerate(sim.model.dof_damping):
            sim.model.dof_damping[i] = damping * params['joint_damping_multiplier']
        
        # Apply puck-specific parameters
        puck_id = sim.model.body_name2id('puck')  # TODO: correct body name
        sim.model.body_mass[puck_id] = params['puck_mass']
        # Set friction: sim.model.geom_friction[puck_geom_id]
        
        sim.model.forward()  # recompute derived quantities
        return params


# ============================================================
# OBSERVATION NOISE
# ============================================================

class ObservationNoiseModel:
    """Running statistics for per-feature noise scaling.
    Noise std = 5% of running std per feature."""
    
    def __init__(self, state_dim):
        self.running_mean = np.zeros(state_dim)
        self.running_var = np.ones(state_dim)
        self.count = 0
        self.noise_frac = 0.05  # 5% of running std
    
    def update(self, observation):
        """Update running statistics."""
        self.count += 1
        delta = observation - self.running_mean
        self.running_mean += delta / self.count
        self.running_var += (observation - self.running_mean) * delta
    
    def add_noise(self, observation):
        """Apply per-feature Gaussian noise scaled to 5% of running std."""
        if self.count < 2:
            return observation
        running_std = np.sqrt(self.running_var / (self.count - 1))
        noise = np.random.normal(0, self.noise_frac * running_std)
        return observation + noise


# ============================================================
# HINDSIGHT EXPERIENCE REPLAY
# ============================================================

class HERBuffer:
    """Replay buffer with Hindsight Experience Replay."""
    
    def __init__(self, max_size=10000, her_prob=0.8):
        self.buffer = []
        self.max_size = max_size
        self.her_prob = her_prob
    
    def push(self, trajectory, goal, mu):
        """Store episode with original goal and dynamics."""
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append({'trajectory': trajectory, 'goal': goal, 'mu': mu})
    
    def sample(self, batch_size=128):
        """Sample episodes, applying HER relabeling with probability her_prob."""
        episodes = np.random.choice(self.buffer, batch_size)
        processed = []
        
        for ep in episodes:
            traj = ep['trajectory']
            goal = ep['goal']
            mu = ep['mu']
            
            if np.random.random() < self.her_prob:
                # HER: relabel goal with final state's achieved goal
                final_state = traj[-1]['s_next']
                goal = self.state_to_goal(final_state)  # extract puck position
                
                # Recompute rewards under new goal
                traj = [{'s': t['s'], 'a': t['a'],
                         'r': self.compute_reward(t['s_next'], goal),
                         's_next': t['s_next']} for t in traj]
            
            processed.append({'trajectory': traj, 'goal': goal, 'mu': mu})
        
        return processed
    
    def state_to_goal(self, state):
        """Extract puck position from state as goal."""
        # TODO: define which state dimensions are puck position
        return state[puck_pos_indices]
    
    def compute_reward(self, state, goal):
        """Sparse binary reward."""
        puck_pos = state[puck_pos_indices]
        dist = np.linalg.norm(puck_pos - goal)
        return 0.0 if dist < 0.07 else -1.0


# ============================================================
# COMPLETE TRAINING LOOP
# ============================================================

def train(env, policy, critic, buffer, n_iterations=8000):
    """Main training loop with dynamics randomization and HER."""
    randomizer = DynamicsRandomizer()
    noise_model = ObservationNoiseModel(state_dim=52)
    optimizer_policy = torch.optim.Adam(policy.parameters(), lr=5e-4)
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=5e-4)
    gamma = 0.98  # discount
    
    for iteration in range(n_iterations):
        # ---- EPISODE COLLECTION ----
        g = env.sample_goal()
        mu = randomizer.sample()
        randomizer.apply_to_sim(env.sim, mu)
        
        s_t = env.reset()
        a_prev = torch.zeros(7)
        z_t = policy.init_hidden()
        trajectory = []
        
        # Sample episode-level latency parameter
        lambda_ep = np.random.uniform(125, 1000)
        
        for t in range(100):
            # Add observation noise
            noise_model.update(s_t.numpy())
            s_noisy = torch.tensor(noise_model.add_noise(s_t.numpy()), dtype=torch.float32)
            
            # Sample per-step action timestep (latency model)
            delta_t = 0.04 + np.random.exponential(1.0 / lambda_ep)
            
            # Policy forward pass
            with torch.no_grad():
                a_t, z_next = policy(s_noisy.unsqueeze(0), a_prev.unsqueeze(0),
                                      z_t, g.unsqueeze(0))
            
            # Add action exploration noise
            a_t_noisy = a_t + torch.randn_like(a_t) * 0.01
            
            # Step environment
            s_next, _ = env.step(a_t_noisy.squeeze(0).numpy(), delta_t=delta_t)
            r_t = buffer.compute_reward(s_next, g.numpy())
            
            trajectory.append({
                's': s_noisy, 'a': a_t.squeeze(0), 'r': r_t,
                's_next': torch.tensor(s_next)
            })
            
            z_t = z_next
            a_prev = a_t.squeeze(0)
            s_t = torch.tensor(s_next)
        
        buffer.push(trajectory, g, mu)
        
        # ---- POLICY UPDATE ----
        if len(buffer.buffer) < 128:
            continue
        
        episodes = buffer.sample(batch_size=128)
        
        total_policy_loss = 0
        total_critic_loss = 0
        
        for ep in episodes:
            traj = ep['trajectory']
            g_ep = ep['goal']
            mu_ep = torch.tensor([v for v in ep['mu'].values()], dtype=torch.float32)
            
            # Recompute LSTM states for policy and critic
            z = policy.init_hidden()
            y = critic.init_hidden()
            
            for t in range(len(traj)):
                s_t = traj[t]['s'].unsqueeze(0)
                a_t = traj[t]['a'].unsqueeze(0)
                r_t = traj[t]['r']
                
                a_prev_t = traj[t-1]['a'].unsqueeze(0) if t > 0 else torch.zeros(1, 7)
                
                # Policy action for critic evaluation
                a_policy, z_next = policy(s_t, a_prev_t, z, g_ep.unsqueeze(0))
                
                # Critic evaluation (omniscient: sees μ)
                q_pred, y_next = critic(s_t, a_t, a_prev_t, y, g_ep.unsqueeze(0),
                                        mu_ep.unsqueeze(0))
                
                # TD target
                if t < len(traj) - 1:
                    s_next = traj[t+1]['s'].unsqueeze(0)
                    a_next_policy, _ = policy(s_next, a_t, z_next, g_ep.unsqueeze(0))
                    q_next, _ = critic(s_next, a_next_policy, a_t, y_next,
                                       g_ep.unsqueeze(0), mu_ep.unsqueeze(0))
                    q_target = r_t + gamma * q_next.detach()
                else:
                    q_target = torch.tensor(r_t)
                
                # Losses
                critic_loss = (q_target - q_pred).pow(2)
                policy_loss = -critic(s_t, a_policy, a_prev_t, y, g_ep.unsqueeze(0),
                                       mu_ep.unsqueeze(0))[0]
                
                total_critic_loss += critic_loss
                total_policy_loss += policy_loss
                
                z = z_next
                y = y_next
        
        (total_critic_loss / (128 * 100)).backward()
        optimizer_critic.step()
        optimizer_critic.zero_grad()
        
        (total_policy_loss / (128 * 100)).backward()
        optimizer_policy.step()
        optimizer_policy.zero_grad()
```

### 4c. Key Hyperparameters

| Parameter | Value | Controls | Sensitivity |
|-----------|-------|---------|------------|
| LSTM hidden dim | 128 | Dynamics encoding capacity | Medium — larger captures more complex dynamics patterns |
| Embedding dim | 128 | Per-step feature compression | Low |
| FC hidden dims | 128, 128 | Policy expressiveness | Low |
| γ (discount) | 0.98 | Horizon (100 steps → ~20 step effective horizon) | Medium |
| HER probability k | 0.8 | Fraction of updates using relabeled goals | High — sparse rewards require high HER ratio |
| Learning rate (Adam) | 5e-4 | Both policy and critic | Low — standard |
| Batch size | 128 episodes × 100 steps | Sample efficiency | Low |
| n_iterations | 8000 | Total training | Low |
| Action exploration noise | σ=0.01 rad | Exploration | Low |
| Observation noise | 5% running std | Sim-to-real transfer quality | High — removing drops performance to 25% |
| Link mass range | [0.25, 4]× default | Body dynamics variation | Medium |
| Puck friction range | [0.1, 5] | Contact dynamics variation | High — critical for manipulation |
| Action timestep λ | [125, 1000] s⁻¹ | Latency variation | High — removing drops performance to 29% |

**The two most critical parameters to randomize:**
1. Action timestep (latency model) — removing drops real-world success from 89% to 29%
2. Observation noise — removing drops success from 89% to 25%

Both capture real-world imperfections that aren't in the nominal simulator model.

### 4d. Data Requirements

**State space (52D):**
- Robot arm: joint positions (7) + joint velocities (7) + gripper position (3)
- Puck: position (3) + orientation (4) + linear velocity (3) + angular velocity (6)
- Previous action: relative joint offsets (7)
- Goal: target puck position (3)
- Note: NOT 52D if goal is separate. This is the paper's full state encoding.

**Action space (7D):** Relative joint position offsets for 7-DOF arm.

**Dynamics parameter vector μ (95D):**
- 7 link masses (multipliers)
- 7 joint damping values (multipliers)
- Puck mass (1), friction (1), damping (1)
- Table height (1)
- Controller gains (7 multipliers — one per joint)
- Action timestep lambda (1)
- Observation noise (implicitly parameterized)
Total: approximately 95 independently randomized values.

### 4e. Dependencies

| Tool | Purpose |
|------|---------|
| MuJoCo | Physics simulation (paper's simulator) |
| PyTorch | LSTM and MLP implementation |
| RDPG (custom) | Off-policy recurrent policy gradient |
| HER implementation | Hindsight experience relay — implement from scratch or use stable-baselines3-contrib |
| PhaseSpace / motion capture | Puck tracking on real robot (deployment only) |

---

## SECTION 5 — BENCHMARK RESULTS

### 5a. Evaluation
- Real Fetch Robotics arm (7-DOF)
- Puck pushing task: random start and goal positions in 0.3m × 0.3m bound
- Goal satisfied if puck within 0.07m of target at end of 200-step episode

### 5b. Metrics
- **Success rate:** Fraction of episodes where goal is achieved at end
- 4 random seeds per architecture in simulation
- Limited real-world trials (10-28 per condition) due to hardware safety

### 5c. Quantitative results (Tables II and III)

**Architecture comparison:**

| Model | Success (Sim) | Success (Real) |
|-------|--------------|---------------|
| FF no Rand | 0.51 ± 0.05 | 0.00 ± 0.00 |
| FF (with Rand) | 0.83 ± 0.04 | 0.67 ± 0.14 |
| FF + Hist (8 steps) | 0.87 ± 0.03 | 0.70 ± 0.10 |
| **LSTM** | **0.91 ± 0.03** | **0.89 ± 0.06** |

Key finding: sim-to-real gap is largest without randomization (FF no Rand: 51% sim → 0% real). Randomization helps (FF: 83% → 67%). LSTM closes the remaining gap nearly completely (91% sim → 89% real).

**Parameter ablation (Table III):**

| Configuration | Real Success |
|--------------|-------------|
| All parameters randomized | 0.89 ± 0.06 |
| Fixed action timestep | 0.29 ± 0.11 |
| No observation noise | 0.25 ± 0.12 |
| Fixed link mass | 0.64 ± 0.10 |
| Fixed puck friction | 0.48 ± 0.10 |

**Robustness test:** Modified puck contact dynamics (chips packet attached) → LSTM still achieves 0.91 ± 0.04 (vs. 0.89 baseline).

### 5d. Qualitative findings
- Emergent manipulation strategies from sparse reward + HER: pressing puck from the side to partially upend it, correcting overshoot
- LSTM hidden state implicitly encodes dynamics — policy adapts within episode
- Sim-to-real joint trajectories show significant mismatch (Figure 5), yet LSTM still transfers

### 5e. Comparison to baselines
- FF no Rand: complete failure on real robot (overfit to nominal simulator)
- FF with Rand: moderate improvement but still 22% worse than LSTM on real robot
- FF + Hist: better than FF alone but still 19% worse than LSTM — explicit history less efficient than learned recurrent encoding

---

## SECTION 6 — WHEN TO USE THIS METHOD

### 6a. Ideal conditions
- Tasks where physics properties (friction, mass, latency) vary significantly between simulation and reality
- Contact-rich manipulation where exact friction modeling is impossible
- Any scenario where you cannot afford real-world training data
- Policies that need to adapt within an episode to changing or unknown dynamics

### 6b. Practical sweet spot
Any sim-to-real task where the primary challenge is contact dynamics uncertainty. The method is particularly powerful when latency and sensor noise are significant sources of sim-to-real gap (which they almost always are in real robotic systems).

### 6c. Scale considerations
- Demonstrated on 7-DOF arm manipulation
- The principle scales to any contact-rich task
- Number of randomized parameters (95 here) should cover all significant sources of sim-to-real mismatch

### 6d. Computational profile
- 8 hours on 100-core cluster
- Heavy compute requirement from HER + RDPG (off-policy recurrent)
- Inference: LSTM at control frequency — fast, CPU-viable

---

## SECTION 7 — WHEN NOT TO USE THIS METHOD

### 7a. Failure modes
- If real-world dynamics fall completely outside the training range
- Tasks requiring very precise control where conservative LSTM behavior is insufficient
- Very long-horizon tasks where LSTM truncation during BPTT causes credit assignment issues

### 7b. Known limitations (from authors)
- Demonstrated on manipulation (pushing) not locomotion
- Compute-intensive: 100M samples, 8 hours on 100-core cluster
- LSTM training with RDPG is more complex than PPO — harder to debug

### 7c. Hidden assumptions
- The real-world dynamics fall within the training distribution of μ
- Task dynamics are identifiable from recent (state, action) history
- Latency and sensor noise are the primary sim-to-real bottlenecks (this is usually true)

### 7d. Cases where simpler approaches suffice
- If real-world and simulator match well: direct sim-to-real without randomization
- For locomotion on well-characterized terrain: domain randomization with MLP + wider ranges may suffice (Lee et al., RMA)

---

## SECTION 8 — EXPECTED RESULTS

### 8a. What good output looks like
- LSTM policy achieves comparable simulation and real-world success rates
- Policy demonstrates in-episode adaptation (different behavior at start vs. after a few steps)
- Emergent strategies from sparse reward

### 8b. Characteristic artifacts
- Early in episode: policy is conservative (LSTM hasn't identified dynamics yet)
- Post-adaptation: more confident, dynamics-appropriate behavior
- Occasionally uses unexpected strategies discovered from HER relabeling

### 8c. What failure looks like
- All architectures fail similarly on real robot: randomization ranges too narrow
- LSTM performs same as FF: LSTM not learning to use history (check LSTM gradient flow)
- HER not helping: check HER implementation for goal relabeling correctness

### 8d. Realistic numbers
- LSTM with full randomization: ~89% real-world success on pushing task
- Without action timestep randomization: ~29%
- Without observation noise: ~25%
- FF without randomization: 0% on real robot

---

## SECTION 9 — CONNECTIONS TO OTHER WORK

### 9a. Key predecessors
- Tobin et al. 2017 — visual domain randomization (complementary to physics randomization)
- Antonova et al. 2017 — memoryless version of dynamics randomization for manipulation
- DDPG (Lillicrap et al. 2015) — base algorithm extended to RDPG
- HER (Andrychowicz et al. 2017) — sparse reward learning technique

### 9b. Key successors
- RMA (Kumar et al. 2021) — extends to locomotion, replaces LSTM with explicit adaptation module
- Lee et al. 2020 — applies dynamics randomization to rough terrain locomotion
- Rudin et al. 2022 — scales dynamics randomization to massively parallel training
- This paper is the foundational reference cited by all three of your other papers

### 9c. Position in field
**Foundational paper.** This is the paper that established dynamics randomization as the standard approach for sim-to-real transfer in robotic manipulation. Every subsequent paper on locomotion sim-to-real (Lee et al., RMA, Rudin) cites this paper as the conceptual foundation for their physics parameter randomization. It's the "proof of concept" that the idea works on real hardware.

---

## SECTION 10 — PROJECT APPLICATION MAPPING

**Your project:** Train PPO quadruped in MuJoCo with Level 1 contact parameter randomization and Level 2 soft patches. Compare Policy A (rigid) vs. Policy B (randomized).

### 10a. How this paper applies
This paper is your **conceptual and theoretical foundation** for why physics parameter randomization works. It is not your implementation reference (the algorithm is RDPG+HER for manipulation, not PPO for locomotion), but it provides the core theoretical justification for your entire experiment.

Specifically, this paper establishes three claims your project builds on:
1. Randomizing dynamics parameters produces policies that generalize to unseen dynamics
2. Recurrent/history-based policies outperform memoryless policies under dynamics variation
3. The most impactful parameters to randomize are latency (timestep) and sensor noise — not just mass and friction

### 10b. What is directly usable

| Component | Usable | Notes |
|-----------|--------|-------|
| Core DR principle (randomize μ per episode) | ✅ Direct | This is exactly what your Level 1 terrain does |
| Parameter ranges and sampling distributions | ✅ Adapt | Their Table I gives you a template for your contact parameter ranges |
| Log-uniform sampling for multiplicative params | ✅ Direct | Use log-uniform for friction, damping multipliers |
| Observation noise during training | ✅ Direct | 5% of running std — apply this to your quadruped policy |
| Action timestep randomization | ✅ Adapt | Randomize your simulation timestep slightly for robustness |
| LSTM architecture | ⚠️ Optional | You can use MLP with history instead; simpler for your project |
| RDPG algorithm | ❌ Replace | Use PPO — RDPG is complex and designed for off-policy learning |
| HER | ❌ Not applicable | Your reward function is not sparse; shaped reward + PPO is better |

### 10c. Direct mapping to your Level 1 terrain

Your Level 1 contact parameters map directly to Peng et al.'s randomized dynamics:

| Your parameter | Paper equivalent | Sampling |
|---------------|-----------------|---------|
| MuJoCo friction | Puck friction [0.1, 5] | Log-uniform |
| MuJoCo damping | Joint damping [0.2, 20]× | Log-uniform |
| MuJoCo solimp stiffness | Puck damping [0.01, 0.2] | Uniform |
| Observation noise | Observation noise (5% running std) | Normal per feature |
| Simulation timestep variation | Action timestep λ | Small variation around dt=0.002 |

Use their sampling distributions and roughly their multiplier ranges as a starting point for your parameters.

### 10d. The critical insight for your robustness-efficiency tradeoff

Table III in this paper is the most directly relevant finding for your project's core question. It shows that:
- Removing friction randomization: 89% → 48% (−41%)
- Removing action timestep: 89% → 29% (−60%)
- Removing observation noise: 89% → 25% (−64%)

This tells you that **not all physics parameters are equal in importance**. For your compliance terrain experiment, the equivalent insight is: **some of your Level 1 parameters will matter more than others for Level 2 generalization**. Your ablation should test which contact parameters (friction vs. damping vs. solimp) contribute most to the robustness improvement you observe.

This gives you a richer analysis structure: not just "Policy B is more robust than Policy A" but "which randomized parameters drove the improvement, and which didn't matter?"

### 10e. Why this is the right foundational citation

When you describe your experiment in a README or interview, the logical chain is:
1. Peng et al. 2018 established that randomizing physics parameters enables sim-to-real transfer
2. Lee et al. 2020 and RMA applied this to locomotion over challenging terrain
3. Your project applies it specifically to the compliance spectrum — soft patches and varied contact properties — and measures the robustness-efficiency tradeoff

This paper is position 1 in that chain. Cite it when you first introduce the phrase "dynamics randomization" in your README.

### 10f. Minimal viable implementation

```
What to extract from Peng et al. for your project:
1. Randomize per-episode physics: sample new friction, damping, solimp at episode start
2. Use log-uniform sampling for multiplicative parameters (friction, damping)
3. Add observation noise: N(0, 0.05 * running_std) per feature
4. Add small action timestep variation: dt = nominal_dt + Exp(λ)
5. Run ablation: which parameters matter most for compliance generalization?
   → Train separate Policy B variants with each parameter fixed
   → Report which single parameter gives the most generalization to Level 2
```

This turns your two-policy comparison into a richer ablation study matching Table III.

---

## IMPLEMENTATION CHEAT SHEET

### 3 most important things to understand

1. **Randomizing physics parameters is necessary but not sufficient — memory is what closes the sim-to-real gap.** A memoryless policy with randomization achieves 67% real-world success. An LSTM achieves 89%. The LSTM implicitly identifies the current episode's dynamics from its own movement history. For your project: adding a 10-20 step history to Policy B's observation is the simplest version of this — the policy can learn to detect which terrain it's on from how its gait felt.

2. **Action timestep and observation noise are the two most important parameters to randomize.** This is counterintuitive — most people focus on mass and friction. But latency (how long an action is actually applied) and sensor noise (how accurately you read your own joints) are the dominant sources of sim-to-real mismatch, and they're usually not randomized by default. Add both to your Level 1 randomization.

3. **Log-uniform sampling for multiplicative parameters.** Friction and damping have multiplicative effects — a robot that's seen friction ranging uniformly from 0.1 to 5.0 has mostly experienced friction near 5.0 (high values dominate a uniform distribution). Log-uniform sampling gives equal "coverage" at each order of magnitude. Use `exp(uniform(log(lo), log(hi)))` for all multiplier-type parameters.

### 5-step implementation recipe

1. **Define your parameter ranges** based on Table I — map friction [0.1, 5], damping [0.2, 20]×default to your MuJoCo contact parameters.
2. **Sample at episode start** using log-uniform for multiplicative params, uniform for additive params.
3. **Add observation noise** at each step: N(0, 0.05 × running_std_per_feature).
4. **Add timestep jitter:** dt_step = nominal_dt + exponential(λ) where λ is sampled per episode.
5. **Run your ablation** (mirror Table III): train Policy B variants with each parameter fixed to measure which ones drive generalization to soft patches.

### 3 most common implementation mistakes

1. **Using uniform sampling for all parameters** — friction and damping multipliers should be log-uniform. Uniform sampling makes the policy mostly see extreme high values and almost never low values proportionally.

2. **Not randomizing action timestep and observation noise** — these are the two most impactful parameters per Table III, and they're easy to overlook because they're not physics properties of the robot itself. Add them from the start.

3. **Randomizing without checking that the nominal case is inside the range** — your nominal rigid-ground baseline must fall within the center of your randomization range, not at an edge. If nominal friction is 1.0, your range should be something like [0.3, 3.0], not [1.0, 5.0].

### Most important hyperparameter to tune first
**Friction randomization range lower bound** — this is the single parameter with the most impact on whether Policy B generalizes to soft/compliant terrain. Set it low enough (0.1-0.2) to simulate near-frictionless conditions, but not so low that training becomes unstable. Start with [0.2, 3.0] and monitor training convergence.

### Should I use this method for my project?
**Yes — use the dynamics randomization framework and parameter sampling approach, but replace RDPG+HER with PPO.** The core contribution is the principle, not the algorithm. Your Level 1 terrain IS dynamics randomization applied to contact parameters. Use Table I as your parameter range template, add observation noise and timestep jitter from the ablation findings, and use log-uniform sampling for multiplicative parameters. Reference this paper when explaining why compliance randomization should generalize — it's your theoretical foundation.
