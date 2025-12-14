#!/usr/bin/env python3
"""
Demo Connection Graph: attach and detach events.
"""

import sys
from pathlib import Path

sys.path.insert(0, Path(__file__).parent.parent)

from ubot.site_pose_loader import load_site_poses
from reconfiguration.connection_graph import ConnectionGraph, make_attach_event, ConnectionEvent

def main():
    repo_root = Path(__file__).parent.parent
    xml_path = repo_root / "assets" / "ubot_ax_centered.xml"
    poses = load_site_poses(str(xml_path))

    print("=== Connection Graph Demo ===")

    g = ConnectionGraph()

    # Attach ma_right (module 1) to mb_left (module 2)
    pose_right = poses["ma_connector_right"]
    pose_left = poses["mb_connector_left"]
    pose_left.position = pose_right.position.copy()  # Force contact

    try:
        event_attach1 = make_attach_event(1, "ma_connector_right", pose_right, 2, "mb_connector_left", pose_left)
        g.apply(event_attach1)
        print(f"Attach 1: {event_attach1.a} <-> {event_attach1.b}, yaw_snap={event_attach1.yaw_snap_deg}°")
        print(f"Active edges: {len(g.active_edges())}")
    except ValueError as e:
        print(f"Attach 1 failed: {e}")

    # Try to attach ma_bottom (module 1) to mb_top (module 3), but use a different mb_top site
    pose_bottom = poses["ma_connector_bottom"]
    pose_top = poses["mb_connector_top"]
    pose_top.position = pose_bottom.position.copy()

    try:
        event_attach2 = make_attach_event(1, "ma_connector_bottom", pose_bottom, 3, "mb_connector_top", pose_top)
        g.apply(event_attach2)
        print(f"Attach 2: {event_attach2.a} <-> {event_attach2.b}, yaw_snap={event_attach2.yaw_snap_deg}°")
        print(f"Active edges after attach2: {len(g.active_edges())}")
    except ValueError as e:
        print(f"Attach 2 failed: {e}")

    # Try illegal attach reusing ma site from module 1
    pose_right_dup = poses["ma_connector_right"]
    pose_right_dup.position = pose_right.position.copy()
    try:
        event_attach_illegal = make_attach_event(1, "ma_connector_right", pose_right_dup, 4, "mb_connector_left", pose_left)  # Different mb_left but same ma
        g.apply(event_attach_illegal)
        print("Illegal attach succeeded unexpectedly")
    except ValueError as e:
        print(f"Illegal attach blocked: {e}")

    # Detach first edge
    event_detach1 = ConnectionEvent(kind="detach", a=event_attach1.a, b=event_attach1.b)
    g.apply(event_detach1)
    print(f"Detached {event_detach1.a} <-> {event_detach1.b}")
    print(f"Active edges after detach: {len(g.active_edges())}")

if __name__ == "__main__":
    main()
