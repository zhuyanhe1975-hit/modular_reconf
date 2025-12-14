import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class SitePose:
    name: str
    position: np.ndarray  # (3,)
    quat_wxyz: np.ndarray  # (4,)


@dataclass
class ConnectParams:
    eps_pos: float = 0.003
    eps_normal_deg: float = 3.0
    eps_yaw_deg: float = 5.0
    eps_parallel: float = 1e-8


@dataclass
class ConnectResult:
    feasible: bool
    reason: str
    pos_err: float
    normal_err_deg: float
    yaw_deg: float
    yaw_snap_deg: Optional[int]


def quat_to_rot(wxyz):
    """Convert MuJoCo wxyz quaternion to rotation matrix."""
    w, x, y, z = wxyz
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])


def unit(v):
    return v / np.linalg.norm(v)


def project_to_plane(v, n):
    return v - np.dot(v, n) * n


def angle_deg(u, v):
    cos_angle = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    cos_angle = np.clip(cos_angle, -1, 1)
    return np.degrees(np.arccos(cos_angle))


def wrap_deg_360(a):
    return a % 360


def can_connect(a: SitePose, b: SitePose, params=ConnectParams()) -> ConnectResult:
    """
    Check if two sites can connect: positions coincide, normals oppose, yaw snaps.
    """
    pos_err = np.linalg.norm(a.position - b.position)
    if pos_err > params.eps_pos:
        return ConnectResult(feasible=False, reason="pos_mismatch",
                           pos_err=pos_err, normal_err_deg=0, yaw_deg=0, yaw_snap_deg=None)

    R_a = quat_to_rot(a.quat_wxyz)
    R_b = quat_to_rot(b.quat_wxyz)
    z_a = R_a[:, 2]
    z_b = R_b[:, 2]
    y_a = R_a[:, 1]
    y_b = R_b[:, 1]

    # Normal opposition
    normal_err_deg = angle_deg(z_a, -z_b)
    if normal_err_deg > params.eps_normal_deg:
        return ConnectResult(feasible=False, reason="normal_not_opposing",
                           pos_err=pos_err, normal_err_deg=normal_err_deg, yaw_deg=0, yaw_snap_deg=None)

    # Yaw alignment
    y_a_proj = unit(project_to_plane(y_a, z_a))
    y_b_proj = unit(project_to_plane(y_b, z_a))

    if np.linalg.norm(y_a_proj) < params.eps_parallel or np.linalg.norm(y_b_proj) < params.eps_parallel:
        return ConnectResult(feasible=False, reason="degenerate_tangent",
                           pos_err=pos_err, normal_err_deg=normal_err_deg, yaw_deg=0, yaw_snap_deg=None)

    # Signed angle
    dot_ab = np.dot(y_a_proj, y_b_proj)
    cross_ab = np.dot(z_a, np.cross(y_a_proj, y_b_proj))
    yaw_rad = np.arctan2(cross_ab, dot_ab)
    yaw_deg = wrap_deg_360(np.degrees(yaw_rad))

    # Snap to nearest
    snaps = [0, 90, 180, 270]
    errors = [min((yaw_deg - s) % 360, (s - yaw_deg) % 360) for s in snaps]
    min_err = min(errors)
    yaw_snap = snaps[np.argmin(errors)] if min_err <= params.eps_yaw_deg else None

    if yaw_snap is None:
        return ConnectResult(feasible=False, reason="yaw_not_snapped",
                           pos_err=pos_err, normal_err_deg=normal_err_deg, yaw_deg=yaw_deg, yaw_snap_deg=yaw_snap)

    return ConnectResult(feasible=True, reason="ok",
                        pos_err=pos_err, normal_err_deg=normal_err_deg,
                        yaw_deg=yaw_deg, yaw_snap_deg=yaw_snap)
