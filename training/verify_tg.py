"""
Kinematic verification of the residual trot trajectory generator (v14).

The single biggest implementation risk for the PMTG prior is the joint-sign
convention: the calf (KFE) offset must LIFT the swing foot off the ground, not
press it down. This script verifies that purely kinematically (no dynamics): for
each phase it sets the joints to NOMINAL + TG(phase), runs mj_forward, and reads
the foot site heights. It checks:

  1. The baked-in _TG_LIFT_AMP sign makes the swing foot rise above its stance height.
  2. The trot phasing is correct: FR+RL move together, FL+RR move together,
     and the two diagonals are 180° out of phase.
  3. The thigh (HFE) swing produces real fore/aft foot travel (propulsion).

Run:  python training/verify_tg.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mujoco

from environments.rigid_env import RigidTerrainEnv
from environments import base_env as B

N_PHASE = 48  # samples over one full gait cycle


def foot_xz_over_cycle(env, lift_sign: float):
    """Return (phases, z[N,4], x[N,4]) foot world positions over one cycle for a
    given calf-lift sign, holding the base at the keyframe pose (pure kinematics)."""
    phases = np.linspace(0.0, 2.0 * np.pi, N_PHASE, endpoint=False)
    z = np.zeros((N_PHASE, 4))
    x = np.zeros((N_PHASE, 4))
    for i, phi in enumerate(phases):
        mujoco.mj_resetDataKeyframe(env._model, env._data, 0)  # base + nominal joints
        theta = phi + B._TG_PHASE_OFFSET
        off = np.zeros((4, 3))
        off[:, 1] = B._TG_STRIDE_AMP * np.cos(theta)                       # thigh
        off[:, 2] = abs(B._TG_LIFT_AMP) * lift_sign * np.maximum(0.0, np.sin(theta))  # calf
        env._data.qpos[7:] = B.NOMINAL_JOINT_POS + off.reshape(12)
        mujoco.mj_forward(env._model, env._data)
        sites = env._data.site_xpos[env._foot_site_ids]
        z[i] = sites[:, 2]
        x[i] = sites[:, 0]
    return phases, z, x


def main():
    env = RigidTerrainEnv(target_lin_vel=(0.2, 0.0), use_tg=True)
    env.reset(seed=0)

    print(f"TG config: freq={B._TG_FREQ_HZ} Hz, stride={B._TG_STRIDE_AMP}, "
          f"lift={B._TG_LIFT_AMP}, ref_speed={B._TG_REF_SPEED}")
    print(f"Δφ per policy step = {env._tg_dphase:.4f} rad "
          f"({2*np.pi/env._tg_dphase:.1f} steps/cycle)\n")

    # --- 1. which sign lifts? test both ---
    print("=== foot-lift sign test (swing-phase mean z minus stance-phase mean z) ===")
    correct_sign = None
    for sign in (+1.0, -1.0):
        phases, z, _ = foot_xz_over_cycle(env, sign)
        # FR leg (idx 0): swing = sin(phase)>0, stance = sin(phase)<0
        swing = np.sin(phases) > 0
        dz = z[swing, 0].mean() - z[~swing, 0].mean()
        verdict = "LIFTS ✓" if dz > 0.01 else "presses down ✗"
        print(f"  calf sign {sign:+.0f}: FR swing-z − stance-z = {dz:+.4f} m  → {verdict}")
        if dz > 0.01:
            correct_sign = sign

    baked_sign = np.sign(B._TG_LIFT_AMP)
    print(f"\n  → correct lift sign = {correct_sign:+.0f};  baked _TG_LIFT_AMP sign = {baked_sign:+.0f}")
    if correct_sign is None:
        print("  ✗✗ NEITHER sign lifts the foot — TG parametrisation is wrong, do NOT train.")
        env.close(); return
    if baked_sign != correct_sign:
        print(f"  ✗ MISMATCH: set _TG_LIFT_AMP to {correct_sign*abs(B._TG_LIFT_AMP):+.2f} in base_env.py.")
        env.close(); return
    print("  ✓ baked sign is correct — feet lift during swing.")

    # --- 2. trot phasing + 3. fore/aft travel, using the baked sign ---
    phases, z, x = foot_xz_over_cycle(env, baked_sign)
    print("\n=== per-foot vertical lift & fore/aft travel (baked sign) ===")
    for fi, name in enumerate(B.FOOT_NAMES):
        print(f"  {name}: z range {z[:,fi].max()-z[:,fi].min():.3f} m  "
              f"(min {z[:,fi].min():.3f}, max {z[:,fi].max():.3f})   "
              f"x travel {x[:,fi].max()-x[:,fi].min():.3f} m")

    # diagonal pairs FR(0)+RL(3) vs FL(1)+RR(2): same-diagonal corr ≈ +1, cross ≈ −1
    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1])
    print("\n=== trot phasing (foot-z correlations) ===")
    print(f"  FR·RL (same diagonal)   = {corr(z[:,0], z[:,3]):+.2f}  (want ≈ +1)")
    print(f"  FL·RR (same diagonal)   = {corr(z[:,1], z[:,2]):+.2f}  (want ≈ +1)")
    print(f"  FR·FL (opposite diag)   = {corr(z[:,0], z[:,1]):+.2f}  (want ≈ −1)")

    # --- propulsion direction: a FORWARD gait plants the foot ahead of the hip at
    #     touchdown (θ=π) and sweeps it back through stance to liftoff (θ=0/2π), i.e.
    #     foot_x (body frame) ∝ −cos(θ). If foot_x ∝ +cos(θ) the gait is retrograde
    #     (drives the robot backward) → flip the sign of _TG_STRIDE_AMP. ---
    print("\n=== propulsion direction (foot fore/aft vs gait phase) ===")
    c = corr(x[:, 0], np.cos(phases))   # FR foot-x vs cos(phase)
    fwd = c < -0.5
    print(f"  corr(FR foot-x, cos φ) = {c:+.2f}   "
          f"→ {'FORWARD-propulsive ✓ (foot ∝ −cos φ)' if fwd else 'RETROGRADE ✗ — flip _TG_STRIDE_AMP sign'}")

    ok = (corr(z[:,0], z[:,3]) > 0.9 and corr(z[:,1], z[:,2]) > 0.9
          and corr(z[:,0], z[:,1]) < -0.5
          and z[:,0].max() - z[:,0].min() > 0.02
          and fwd)
    print(f"\n{'✓ TG VERIFIED — feet lift, trot phasing correct, forward-propulsive. Ready to train.' if ok else '✗ TG check FAILED — inspect/flip signs before training.'}")
    env.close()


if __name__ == "__main__":
    main()
