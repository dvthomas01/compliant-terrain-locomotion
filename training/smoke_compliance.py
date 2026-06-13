"""Phase-1 validation for CompliantTerrainEnv (Policy B).

Checks: (1) obs dims (B=89, A=49, finite); (2) foot-history actually populates and
t-10/t-20 differ from t after motion; (3) the priority fix makes compliance PHYSICAL
— a soft floor produces measurable foot sinkage vs a rigid floor; (4) curriculum
advances on a long episode. Run: python training/smoke_compliance.py
"""
import numpy as np
import mujoco
from environments.compliance_env import CompliantTerrainEnv
from environments.rigid_env import RigidTerrainEnv


def test_dims():
    b = CompliantTerrainEnv(target_lin_vel=(0.2, 0.0), use_tg=True)
    a = RigidTerrainEnv(target_lin_vel=(0.2, 0.0), use_tg=True)
    ob, _ = b.reset(seed=0); oa, _ = a.reset(seed=0)
    assert ob.shape == (89,), ob.shape
    assert oa.shape == (49,), oa.shape
    assert np.all(np.isfinite(ob)) and np.all(np.isfinite(oa))
    print(f"[1] dims OK  — Policy B 89D, Policy A 49D, both finite")


def test_history_populates():
    b = CompliantTerrainEnv(target_lin_vel=(0.2, 0.0), use_tg=True)
    b._level = 3
    b.reset(seed=1)
    for _ in range(40):
        ob, *_ = b.step(b.action_space.sample() * 0.0)  # zero action: TG+gravity move feet
    # obs layout: [49 base][24 hist=(t,t-10,t-20)×4feet×xy][12 vel][4 contact]
    hist = ob[49:73].reshape(3, 4, 2)
    d_t10 = np.abs(hist[0] - hist[1]).max()
    d_t20 = np.abs(hist[0] - hist[2]).max()
    assert d_t10 > 1e-4 and d_t20 > 1e-4, (d_t10, d_t20)
    print(f"[2] foot history OK — |t−(t-10)|={d_t10:.4f}, |t−(t-20)|={d_t20:.4f} (feet moving, history distinct)")


def _settle_height(env, soft):
    """Force a fixed rigid OR soft floor, settle 300 steps zero-action, return trunk z."""
    env.reset(seed=7)
    gid = env._floor_geom_id
    if soft:
        env._model.geom_priority[gid] = 2          # > foot(1): floor governs
        env._model.geom_solref[gid]   = [0.20, 1.0]  # very soft (long time const)
        soft = env._floor_solimp0.copy(); soft[0] = 0.6; soft[1] = 0.6
        env._model.geom_solimp[gid]   = soft
    else:
        env._model.geom_priority[gid] = env._floor_priority0
        env._model.geom_solref[gid]   = env._floor_solref0
        env._model.geom_solimp[gid]   = env._floor_solimp0
    mujoco.mj_forward(env._model, env._data)
    for _ in range(300):
        env.step(np.zeros(12, dtype=np.float32))
    return float(env._data.qpos[2])


def test_compliance_is_physical():
    env = CompliantTerrainEnv(target_lin_vel=(0.0, 0.0), use_tg=True)  # cmd 0 → TG off, just stand
    h_rigid = _settle_height(env, soft=False)
    h_soft  = _settle_height(env, soft=True)
    sink = h_rigid - h_soft
    assert sink > 0.01, f"soft floor should sink >1cm vs rigid; got rigid={h_rigid:.4f} soft={h_soft:.4f} sink={sink:.4f}"
    print(f"[3] compliance PHYSICAL — rigid z={h_rigid:.4f}, soft z={h_soft:.4f}, sinkage={sink*100:.1f} cm "
          f"(priority fix works: floor governs contact)")


def test_level0_matches_rigid_contact():
    """Level 0 must leave the floor foot-governed (priority unchanged) == Policy A."""
    env = CompliantTerrainEnv(target_lin_vel=(0.2, 0.0), use_tg=True)
    env._level = 0
    env.reset(seed=3)
    gid = env._floor_geom_id
    assert env._model.geom_priority[gid] == env._floor_priority0, "level 0 must not raise floor priority"
    assert np.allclose(env._model.geom_solref[gid], env._floor_solref0)
    print(f"[4] level-0 contact == Policy A (floor priority {env._floor_priority0}, solref unchanged)")


def test_level2_patch_placement():
    """Level >=4 activates patches in x∈[1,3.5]; levels <4 bury them all."""
    env = CompliantTerrainEnv(target_lin_vel=(0.2, 0.0), use_tg=True)
    env._level = 6; env.reset(seed=11)
    z = np.array([env._model.geom_pos[pid][2] for pid in env._patch_geom_ids])
    x = np.array([env._model.geom_pos[pid][0] for pid in env._patch_geom_ids])
    active = z > -1.0
    assert active.any(), "level 6 must activate >=1 patch"
    from environments.compliance_env import _PATCH_RAISE
    for k in np.where(active)[0]:
        assert 1.0 <= x[k] <= 3.5, f"active patch x out of path range: {x[k]}"
        top = z[k] + env._patch_half_z
        assert abs(top - _PATCH_RAISE) < 1e-9, f"active patch top must be at +{_PATCH_RAISE}, got {top}"
    env._level = 2; env.reset(seed=12)
    z2 = np.array([env._model.geom_pos[pid][2] for pid in env._patch_geom_ids])
    assert np.all(z2 < -1.0), "levels <4 must bury all patches"
    print(f"[6] Level-2 patches OK — level 6: {int(active.sum())} active in-path & flush; level 2: all buried")


def _settle_on_pad(env, solref, solimp01):
    """Place a raised pad (top at +_PATCH_RAISE) under the robot with given hardness,
    settle 300 steps zero-action, return trunk z. Isolates softness from the step height."""
    from environments.compliance_env import _PATCH_RAISE
    env._level = 0; env.reset(seed=20)
    pid = env._patch_geom_ids[0]
    env._model.geom_pos[pid]    = [0.0, 0.0, _PATCH_RAISE - env._patch_half_z]
    env._model.geom_solref[pid] = solref
    sm = env._model.geom_solimp[pid].copy(); sm[0], sm[1] = solimp01
    env._model.geom_solimp[pid] = sm
    mujoco.mj_forward(env._model, env._data)
    for _ in range(300): env.step(np.zeros(12, dtype=np.float32))
    return float(env._data.qpos[2])


def test_level2_patch_is_physical():
    """At equal pad HEIGHT, a soft pad lets the foot penetrate deeper than a rigid pad
    (isolates compliance from the 3cm step)."""
    env = CompliantTerrainEnv(target_lin_vel=(0.0, 0.0), use_tg=True)
    h_hard = _settle_on_pad(env, [0.005, 1.0], (0.99, 0.99))   # stiff pad
    h_soft = _settle_on_pad(env, [0.15, 1.0],  (0.70, 0.70))   # soft pad, same height
    sink = h_hard - h_soft
    assert sink > 0.005, f"soft pad should penetrate >5mm deeper than rigid pad; hard={h_hard:.4f} soft={h_soft:.4f}"
    print(f"[7] Level-2 patch PHYSICAL — hard-pad z={h_hard:.4f}, soft-pad z={h_soft:.4f}, extra sinkage={sink*100:.1f} cm")


def test_curriculum_advances():
    env = CompliantTerrainEnv(target_lin_vel=(0.2, 0.0), use_tg=True)
    env.reset(seed=5)
    env._step_count = 1
    env._data.qpos[0] = 0.6 * _target()        # pretend it travelled >50% target
    env.reset(seed=6)
    assert env._level == 1, env._level
    env._step_count = 1
    env._data.qpos[0] = 0.0                     # travelled nothing → retreat
    env.reset(seed=7)
    assert env._level == 0, env._level
    print(f"[5] curriculum OK — advances on success, retreats on failure")


def _target():
    from environments.compliance_env import _TARGET_DISTANCE
    return _TARGET_DISTANCE


if __name__ == "__main__":
    test_dims()
    test_history_populates()
    test_compliance_is_physical()
    test_level0_matches_rigid_contact()
    test_level2_patch_placement()
    test_level2_patch_is_physical()
    test_curriculum_advances()
    print("\nALL SMOKE CHECKS PASSED (Phase 1 + Phase 2)")
