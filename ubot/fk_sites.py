#!/usr/bin/env python3
"""
UBot Real Site-Frame FK (ax-centered model).
Phase-2.4: Compute internal FK and site frames.
"""

import numpy as np
from .mjcf_parser import load_ubot_mjcf
from .spec import UbotModuleSpec


def rotz(theta_deg: float) -> np.ndarray:
    """Rotation matrix around Z by theta_deg degrees."""
    theta_rad = np.deg2rad(theta_deg)
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]], dtype=np.float64)


def rotx(theta_deg: float) -> np.ndarray:
    """Rotation matrix around X by theta_deg degrees."""
    theta_rad = np.deg2rad(theta_deg)
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s,  c]], dtype=np.float64)


def roty(theta_deg: float) -> np.ndarray:
    """Rotation matrix around Y by theta_deg degrees."""
    theta_rad = np.deg2rad(theta_deg)
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]], dtype=np.float64)


def quat_to_rot(wxyz):
    """Convert MuJoCo wxyz quaternion to rotation matrix."""
    w, x, y, z = wxyz
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])


def hinge_T(axis_local: np.ndarray, theta_rad: float, pos_local: np.ndarray = np.zeros(3)) -> np.ndarray:
    """Hinge joint transform: displacement pos, rotation theta around axis."""
    from .kinematics_phase1 import rodrigues_rot, transform_from_joint
    # Reuse existing robustness
    R = rodrigues_rot(axis_local, theta_rad)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = pos_local
    return T


def compute_site_world_Ts_for_module(spec: UbotModuleSpec, q: np.ndarray, Tw_module: np.ndarray, mjcf_path: str) -> dict[str, np.ndarray]:
    """Compute world T for each site, given q and Tw_module and path for sites."""
    kin = UBotKinematics(mjcf_path)
    kin.spec = spec  # override spec
    site_names = [name for name in kin.site_T_half if "connector" in name]
    T_sites = {}
    for name in site_names:
        T_sites[name] = Tw_module @ kin.T_ax_site(q, name)
    return T_sites


class UBotKinematics:
    """Real UBot kinematics for ax-centered model."""

    def __init__(self, mjcf_path, verbose: bool = False):
        if isinstance(mjcf_path, str):
            from .mjcf_parser import load_ubot_mjcf
            self.spec = load_ubot_mjcf(mjcf_path, verbose=verbose)
        else:
            self.spec = mjcf_path  # Passed spec for compute helper
        # Precompute site transforms in half frames (from XML pos+quat)
        self.site_T_half: dict[str, np.ndarray] = {}  # site_name -> T_half^site

        if isinstance(mjcf_path, str):
            # Proper XML parsing for sites
            import xml.etree.ElementTree as ET
            root = ET.parse(mjcf_path).getroot()
            for site in root.findall(".//site"):
                name = site.get("name")
                if name and "connector" in name:  # e.g. ma_connector_right
                    pos_str = site.get("pos", "0 0 0")
                    pos = np.array(list(map(float, pos_str.split())))
                    quat_str = site.get("quat", "1 0 0 0")
                    quat = np.array(list(map(float, quat_str.split())))  # MuJoCo w x y z
                    # Local quat_to_rot defined above
                    R = quat_to_rot(quat)
                    T_half_site = np.eye(4, dtype=np.float64)
                    T_half_site[:3, :3] = R
                    T_half_site[:3, 3] = pos
                    self.site_T_half[name] = T_half_site

            if verbose:
                print(f"DEBUG: Loaded sites: {list(self.site_T_half.keys())}")

        if not hasattr(self.spec, 'joints'):
            # If passed a path as spec, try to load spec
            from .mjcf_parser import load_ubot_mjcf
            self.spec = load_ubot_mjcf(str(mjcf_path), verbose=verbose)

        # Assume joints[0] is ma, joints[1] is mb (parser order)
        self.j_ma = self.spec.joints[0]  # j1 for ma
        self.j_mb = self.spec.joints[1]  # j2 for mb

    def T_ax_ma(self, q_rad: float) -> np.ndarray:
        """T_ax^ma for q_rad (hinge around j_ma.axis at j_ma.pos)."""
        pos_local = self.j_ma.pos
        axis_local = self.j_ma.axis
        return hinge_T(axis_local, q_rad, pos_local)

    def T_ax_mb(self, q_rad: float) -> np.ndarray:
        """T_ax^mb for q_rad (hinge around j_mb.axis at j_mb.pos)."""
        pos_local = self.j_mb.pos
        axis_local = self.j_mb.axis
        return hinge_T(axis_local, q_rad, pos_local)

    def T_ax_site(self, q: np.ndarray, site_name: str) -> np.ndarray:
        """T_ax^site for q=[q_ma, q_mb]."""
        half_name, _ = site_name.split('_', 1)  # e.g. "ma_connector_right" -> "ma"
        if half_name == "ma":
            T_ax_half = self.T_ax_ma(q[0])
        elif half_name == "mb":
            T_ax_half = self.T_ax_mb(q[1])
        else:
            raise ValueError(f"Unknown half {half_name}")
        T_half_site = self.site_T_half.get(site_name)
        if T_half_site is None:
            raise ValueError(f"Site {site_name} not found")
        T_ax_site = T_ax_half @ T_half_site
        return T_ax_site
