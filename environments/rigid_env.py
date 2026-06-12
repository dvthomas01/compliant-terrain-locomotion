"""Policy A environment: flat rigid terrain with fixed physics."""

from environments.base_env import Go1BaseEnv


class RigidTerrainEnv(Go1BaseEnv):
    """
    Policy A: rigid flat ground, no domain randomization.

    Inherits all observation (47D), reward (9 terms), and termination logic
    from Go1BaseEnv. _reset_terrain() does nothing — no curriculum, no compliance
    variation. This gives Policy A the maximum possible rigid-ground performance.
    """
    pass
