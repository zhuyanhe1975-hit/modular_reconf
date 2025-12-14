#!/usr/bin/env python3
"""
Load site poses from MJCF XML for UBot connection sites.
"""

import xml.etree.ElementTree as ET
import numpy as np
from reconfiguration.connection import SitePose


def load_site_poses(xml_path):
    """Load the 4 connection sites from MJCF."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    poses = {}
    for site in root.findall(".//site"):
        name = site.get("name")
        if name in ["ma_connector_right", "ma_connector_bottom", "mb_connector_left", "mb_connector_top"]:
            pos_str = site.get("pos", "0 0 0")
            quat_str = site.get("quat", "1 0 0 0")  # MuJoCo wxyz
            pos = np.array(list(map(float, pos_str.split())))
            quat = np.array(list(map(float, quat_str.split())))
            poses[name] = SitePose(name=name, position=pos, quat_wxyz=quat)
    return poses


if __name__ == "__main__":
    # Assume xml path relative to repo root
    import os
    repo_root = os.path.dirname(os.path.dirname(__file__))
    xml_path = os.path.join(repo_root, "assets", "ubot_ax_centered.xml")
    poses = load_site_poses(xml_path)

    from reconfiguration.connection import quat_to_rot
    print("=== Site Poses and Axes ===")
    for name, pose in poses.items():
        print(f"{name}: pos={pose.position}, quat={pose.quat_wxyz}")
        R = quat_to_rot(pose.quat_wxyz)
        z_axis = R[:, 2]  # outward normal
        y_axis = R[:, 1]
        print(f"  z-axis (outward): {z_axis}")
        print(f"  y-axis: {y_axis}")
    print("\nNote: z should align with pos direction for outward; y near [0,1,0] for q=identity.")
