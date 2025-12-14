#!/usr/bin/env python3
"""
Demo Kinematic Executor V2 Phase-2.4: Propagation with real site constraints.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, Path(__file__).parent.parent)

from reconfiguration.connection_graph import SiteRef, EdgeKey, ConnectionEdge, ConnectionGraph
from reconfiguration.kinematic_compiler import compile_kinematic_tree
from reconfiguration.kinematic_executor_v2 import propagate_world_poses_with_sites
from reconfiguration.site_naming import site_full_name
from reconfiguration.site_alignment import compute_constraint_metrics, compute_rel_yaw_deg
import ubot


def main():
    ubot_kin = ubot.UBotKinematics(Path(__file__).parent.parent / "assets" / "ubot_ax_centered.xml")

    graph = ConnectionGraph()

    # Edge 1 ma_right <-> 2 mb_left, yaw=0
    site_a1 = SiteRef(1, "ma", "right")
    site_b1 = SiteRef(2, "mb", "left")
    edge1 = ConnectionEdge(key=EdgeKey.normalized(site_a1, site_b1), yaw_snap_deg=0, T_a_b=np.eye(4, dtype=np.float64))
    graph.edges[edge1.key] = edge1

    # Edge 2 ma_bottom <-> 3 mb_top, yaw=90
    site_a2 = SiteRef(2, "ma", "bottom")
    site_b2 = SiteRef(3, "mb", "top")
    T_edge2 = np.eye(4, dtype=np.float64)
    T_edge2[0, 3] = 2.0  # dummy
    edge2 = ConnectionEdge(key=EdgeKey.normalized(site_a2, site_b2), yaw_snap_deg=90, T_a_b=T_edge2)
    graph.edges[edge2.key] = edge2

    print("=== Kinematic Executor V2 Demo ===")

    tree = compile_kinematic_tree(1, graph)

    # q with small angles
    q_by_module = {
        1: np.array([0.1, -0.05]),  # q_ma, q_mb for module 1
        2: np.array([0.05, 0.0]),   # for module 2
        3: np.array([0.0, 0.1])    # for module 3
    }

    T_root = np.eye(4, dtype=np.float64)
    result = propagate_world_poses_with_sites(tree, T_root, q_by_module, ubot_kin)

    print("Module poses:")
    for mid in sorted(result.T_world.keys()):
        pos = result.T_world[mid][:3, 3]
        print(f"  Module {mid}: x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}")

    print("\nConstraint checks:")
    for child, att in tree.attachments.items():
        parent_site_name = site_full_name(att.parent_site)
        child_site_name = site_full_name(att.child_site)

        T_world_site_p = result.T_world[att.parent] @ ubot_kin.T_ax_site(q_by_module[att.parent], parent_site_name)
        T_world_site_c = result.T_world[att.child] @ ubot_kin.T_ax_site(q_by_module[att.child], child_site_name)

        metrics = compute_constraint_metrics(T_world_site_p, T_world_site_c, att.yaw_snap_deg)

        # Raw yaw after flip (before snap compensation)
        R_p = T_world_site_p[:3, :3]
        R_c = T_world_site_c[:3, :3]
        R_flip = ubot.fk_sites.roty(180.0)
        raw_yaw_after_flip = compute_rel_yaw_deg(R_p, R_c @ R_flip)

        print(f"  {parent_site_name} <-> {child_site_name}: pos_err={metrics['pos_err']:.6f}, z_dot={metrics['z_dot']:.6f}, yaw_snap={att.yaw_snap_deg}°, raw_yaw_after_flip={raw_yaw_after_flip:.1f}°, residual_yaw={metrics['rel_yaw_deg']:.1f}°")


if __name__ == "__main__":
    main()
