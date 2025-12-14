#!/usr/bin/env python3
"""
Demo Connection Feasibility Phase-2.5
Test pre-attach checks.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, Path(__file__).parent.parent)

from reconfiguration.connection_feasibility import FeasibilityParams, check_attach_feasible, auto_attach
from reconfiguration.connection_graph import SiteRef, ConnectionGraph, ConnectionEdge, EdgeKey
from reconfiguration.kinematic_executor_v2 import propagate_world_poses_with_sites
from reconfiguration.kinematic_compiler import compile_kinematic_tree
import ubot


def main():
    path = Path(__file__).parent.parent / "assets" / "ubot_ax_centered.xml"
    ubot_kin = ubot.UBotKinematics(str(path))

    # Setup modules like executor demo
    graph = ConnectionGraph()

    # Edges for chain
    edge1 = ConnectionEdge(key=EdgeKey.normalized(
        SiteRef(1, "ma", "right"), SiteRef(2, "mb", "left")
    ), yaw_snap_deg=0, T_a_b=np.eye(4, dtype=np.float64))
    edge2 = ConnectionEdge(key=EdgeKey.normalized(
        SiteRef(2, "ma", "bottom"), SiteRef(3, "mb", "top")
    ), yaw_snap_deg=90, T_a_b=np.eye(4, dtype=np.float64))
    graph.edges[edge1.key] = edge1
    graph.edges[edge2.key] = edge2

    q_by_module = {
        1: np.array([0.1, -0.05]),
        2: np.array([0.05, 0.0]),
        3: np.array([0.0, 0.1])
    }
    T_root = np.eye(4, dtype=np.float64)

    tree = compile_kinematic_tree(1, graph)
    result_exe = propagate_world_poses_with_sites(tree, T_root, q_by_module, ubot_kin)

    print("=== Connection Feasibility Demo ===")

    # Test attach pair 1 right - 2 left (already attached, but test feasibility)
    Tw_a = result_exe.T_world[1]  # module 1
    Tw_b = result_exe.T_world[2]  # module 2
    T_sites = ubot.compute_site_world_Ts_for_module(ubot_kin.spec, q_by_module[1], Tw_a, str(path))
    T_sites_m2 = ubot.compute_site_world_Ts_for_module(ubot_kin.spec, q_by_module[2], Tw_b, str(path))

    Tw_site1 = T_sites["ma_connector_right"]
    Tw_site2 = T_sites_m2["mb_connector_left"]

    params = FeasibilityParams()
    result = check_attach_feasible(Tw_site1, Tw_site2, params)

    print(f"Pair ma_connector_right <-> mb_connector_left:")
    print(f"  Feasible: {result.feasible}, Reason: '{result.reason}', Best yaw: {result.best_yaw_deg}°")
    print(f"  Pos err: {result.pos_err:.6f}, Z dot: {result.z_dot:.6f}")
    print(f"  Candidates: {[c['yaw_deg'] for c in result.candidate_table]} with residuals {[c['residual_yaw_deg'] for c in result.candidate_table]}")

    # Test pair 3 top with something misaligned for fail example
    # Offset pair 3 top vs nothing (simulate fail)
    Tw_wrong = np.eye(4, dtype=np.float64)
    Tw_wrong[2, 3] = 0.1  # Offset in Z
    T_sites_m3_wrong = ubot.compute_site_world_Ts_for_module(ubot_kin.spec, q_by_module[3], Tw_wrong, str(path))
    Tw_site3_top_wrong = T_sites_m3_wrong["mb_connector_top"]

    # Test with site ma_bottom module 2 (same as connected)
    Tw_site2_bottom = T_sites_m2["ma_connector_bottom"]

    result_wrong = check_attach_feasible(Tw_site2_bottom, Tw_site3_top_wrong, params)

    print(f"\nPair ma_connector_bottom <-> mb_connector_top (misaligned):")
    print(f"  Feasible: {result_wrong.feasible}, Reason: '{result_wrong.reason}'")


if __name__ == "__main__":
    main()
