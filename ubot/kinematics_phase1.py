import numpy as np
from typing import Dict, Union
from .spec import JointSpec


def rodrigues_rot(axis, theta):
    """
    Compute rotation matrix using Rodrigues formula for arbitrary axis.
    """
    axis = axis / np.linalg.norm(axis)
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])


def forward_kinematics(q: np.ndarray, T_root: np.ndarray, joint1: JointSpec, joint2: JointSpec) -> Dict[str, np.ndarray]:
    """
    Minimal FK for a single module: q = [q1, q2], T_root = root pose (4x4).
    ax = root, ma = ax * joint1 TF, mb = ax * joint2 TF.
    Assumptions: joint1 affects ma relative to ax, joint2 affects mb relative to ax.
    """
    T_ax = T_root.copy()
    T_ma = T_ax @ transform_from_joint(q[0], joint1)
    T_mb = T_ax @ transform_from_joint(q[1], joint2)
    return {"ax": T_ax, "ma": T_ma, "mb": T_mb}


def transform_from_joint(q, joint: JointSpec):
    """Displacement by joint.pos, rotation by q around joint.axis."""
    R = rodrigues_rot(joint.axis, q)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = joint.pos
    return T
