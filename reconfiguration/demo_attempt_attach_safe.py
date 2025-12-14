#!/usr/bin/env python3
"""
Demo Safe Attempt Attach (post-check + rollback).
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, Path(__file__).parent.parent)

from reconfiguration.connection_feasibility import FeasibilityParams
from reconfiguration.connection_graph import SiteRef, ConnectionGraph
from reconfiguration.connection_api import ExecutorWrapper, attempt_attach
import ubot


def main():
    path = Path(__file__).parent.parent / "assets" / "ubot_ax_centered.xml"
    ubot_kin = ubot.UBotKinematics(str(path))

    # Setup 2 modules for valid attach, then perturb to demo rollback
    T_world = {
        1: np.eye(4, dtype=np.float64),
        2: np.eye(4, dtype=np.float64) + np.array([[0,0,0,0.1],[0,0,0,0],[0,0,0,0],[0,0,0,0]])  # x=0.1
    }
    q_by_module = {1: np.array([0., 0.]), 2: np.array([0., 0.])}
    executor = ExecutorWrapper(T_world, q_by_module, ubot_kin)

    graph = ConnectionGraph()
    params = FeasibilityParams()

    print("=== Safe Attempt Attach Demo ===")

    # Case A: Valid attach, should succeed
    a_ref = SiteRef(1, "ma", "right")
    b_ref = SiteRef(2, "mb", "left")
    result = attempt_attach(graph, executor, a_ref, b_ref, params)
    print(f"Valid attach: feasible={result.feasible}, reason='{result.reason}', yaw={result.best_yaw_deg}°")
    print(f"Active edges: {len(graph.active_edges())}")

    # Case B: Perturb executor to force postcheck failure (e.g., over-offset position)
    # Backup original T_world[2]
    T_orig = executor.T_world[2].copy()
    # Perturb slightly to break pos or yaw
    executor.T_world[2][1, 3] += 0.05  # y offset to break pos

    result_bad = attempt_attach(graph, executor, SiteRef(1, "ma", "bottom"), SiteRef(2, "mb", "top"), params)
    print(f"Postcheck fail (perturbed y): feasible={result_bad.feasible}, reason='{result_bad.reason}'")
    print(f"Active edges after rollback: {len(graph.active_edges())} (should be 1, rollback happened)")

    # Restore
    executor.T_world[2] = T_orig


if __name__ == "__main__":
    main()
