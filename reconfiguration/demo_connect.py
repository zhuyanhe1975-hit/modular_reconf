#!/usr/bin/env python3
"""
Demo connection checking for UBot sites.
Loads poses, sets positions equal, calls can_connect.
"""

import os
import sys
from pathlib import Path

def main():

    sys.path.insert(0, Path(__file__).parent.parent)

    from ubot.site_pose_loader import load_site_poses
    from reconfiguration.connection import can_connect
    repo_root = os.path.dirname(os.path.dirname(__file__))
    xml_path = os.path.join(repo_root, "assets", "ubot_ax_centered.xml")
    poses = load_site_poses(xml_path)

    print("=== Connection Demo ===")

    # ma_right and mb_left
    ma_right = poses["ma_connector_right"]
    mb_left = poses["mb_connector_left"]
    contact_pos = (ma_right.position + mb_left.position) / 2  # midpoint
    ma_right.position = contact_pos
    mb_left.position = contact_pos
    result1 = can_connect(ma_right, mb_left)
    print(f"ma_right - mb_left: feasible={result1.feasible}, reason={result1.reason}, yaw_snap={result1.yaw_snap_deg}")

    # ma_bottom and mb_top
    ma_bottom = poses["ma_connector_bottom"]
    mb_top = poses["mb_connector_top"]
    contact_pos2 = (ma_bottom.position + mb_top.position) / 2
    ma_bottom.position = contact_pos2
    mb_top.position = contact_pos2
    result2 = can_connect(ma_bottom, mb_top)
    print(f"ma_bottom - mb_top: feasible={result2.feasible}, reason={result2.reason}, yaw_snap={result2.yaw_snap_deg}")

if __name__ == "__main__":
    main()
