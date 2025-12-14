import numpy as np


def get_joint_vector(modules):
    """Concatenate all module q values into a single numpy array."""
    return np.concatenate([m.q for m in modules.values()])


def set_joint_vector(modules, joint_vector):
    """Update each module's q from the concatenated joint_vector."""
    idx = 0
    for m in modules.values():
        m.q = joint_vector[idx:idx + 2].copy()
        idx += 2
