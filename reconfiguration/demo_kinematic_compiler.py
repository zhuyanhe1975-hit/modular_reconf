#!/usr/bin/env python3
"""
Demo Kinematic Compiler: build tree from chain.
"""

import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, Path(__file__).parent.parent)

from reconfiguration.connection_graph import SiteRef, EdgeKey, ConnectionEdge, ConnectionGraph
from reconfiguration.kinematic_compiler import compile_kinematic_tree


def main():
    graph = ConnectionGraph()

    # Edge 1-2
    T12 = np.eye(4)
    T12[0, 3] = 1.0  # translate x=1
    site_a1 = SiteRef(1, "ma", "right")
    site_b1 = SiteRef(2, "mb", "left")
    edge12 = ConnectionEdge(key=EdgeKey.normalized(site_a1, site_b1), yaw_snap_deg=0, T_a_b=T12)

    # Edge 2-3
    T23 = np.eye(4)
    T23[0, 3] = 2.0  # translate x=2
    site_a2 = SiteRef(2, "ma", "right")
    site_b2 = SiteRef(3, "mb", "left")
    edge23 = ConnectionEdge(key=EdgeKey.normalized(site_a2, site_b2), yaw_snap_deg=0, T_a_b=T23)

    # Add manually (since only active are used)
    graph.edges[edge12.key] = edge12
    graph.edges[edge23.key] = edge23

    print("=== Kinematic Compiler Demo ===")
    tree = compile_kinematic_tree(1, graph, verbose=True)

    print("\nParent map:")
    for child, parent in tree.parent_of.items():
        print(f"  {child} -> {parent}")

    print(f"BFS order: {tree.order}")

    print("\nAttachments:")
    for child, att in tree.attachments.items():
        print(f"  Child {child}: parent={att.parent}, sites {att.parent_site} <-> {att.child_site}, T shape={att.T_parent_child.shape}")


if __name__ == "__main__":
    main()
