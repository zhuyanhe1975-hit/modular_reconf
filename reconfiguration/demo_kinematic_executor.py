#!/usr/bin/env python3
"""
Demo Kinematic Executor Phase-2.3
Propagate poses along tree, test detach.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, Path(__file__).parent.parent)

from reconfiguration.connection_graph import SiteRef, EdgeKey, ConnectionEdge, ConnectionGraph, ConnectionEvent
from reconfiguration.kinematic_compiler import compile_kinematic_tree
from reconfiguration.kinematic_executor import propagate_world_poses


def main():
    graph = ConnectionGraph()

    # Edge 1-2 with translate x=1
    T12 = np.eye(4, dtype=np.float64)
    T12[0, 3] = 1.0
    site_a1 = SiteRef(1, "ma", "right")
    site_b1 = SiteRef(2, "mb", "left")
    edge12 = ConnectionEdge(key=EdgeKey.normalized(site_a1, site_b1), yaw_snap_deg=0, T_a_b=T12)
    graph.edges[edge12.key] = edge12

    # Edge 2-3 with translate x=2
    T23 = np.eye(4, dtype=np.float64)
    T23[0, 3] = 2.0
    site_a2 = SiteRef(2, "ma", "right")
    site_b2 = SiteRef(3, "mb", "left")
    edge23 = ConnectionEdge(key=EdgeKey.normalized(site_a2, site_b2), yaw_snap_deg=0, T_a_b=T23)
    graph.edges[edge23.key] = edge23

    print("=== Kinematic Executor Demo ===")

    # Initial propagation
    tree = compile_kinematic_tree(1, graph)
    T_root = np.eye(4, dtype=np.float64)
    result = propagate_world_poses(tree, T_root)

    print("Initial poses (all connected):")
    for mid in sorted(result.T_world.keys()):
        pos = result.T_world[mid][:3, 3]
        print(f"  Module {mid}: x={pos[0]:.1f}, y={pos[1]:.1f}, z={pos[2]:.1f}")

    # Detach 2-3
    print("\nDetaching edge 2-3...")
    ev_detach = ConnectionEvent(
        kind="detach",
        a=site_a2, b=site_b2
    )
    graph.apply(ev_detach)

    # Recompile and propagate
    tree2 = compile_kinematic_tree(1, graph)
    result2 = propagate_world_poses(tree2, T_root)

    print("After detach (module 3 unreachable):")
    for mid in sorted(result2.T_world.keys()):
        pos = result2.T_world[mid][:3, 3]
        print(f"  Module {mid}: x={pos[0]:.1f}, y={pos[1]:.1f}, z={pos[2]:.1f}")
    print(f"Reachable modules: {result2.reachable}")


if __name__ == "__main__":
    main()
