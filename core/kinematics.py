import numpy as np
from collections import deque


def compute_world_transforms_multi(modules, topology, root_id):
    """
    Compute world transforms for multiple modules - placeholder: set each to identity with q offset.
    Ignores topology for kinematics; topology used only for connections.
    """
    world_Ts = {}
    for mid, m in modules.items():
        m.world_T = np.eye(4)
        m.world_T[:3, 3] = [m.q[0], m.q[1], 0.0]
        world_Ts[mid] = m.world_T
    return world_Ts
