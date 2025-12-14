#!/usr/bin/env python3
"""
Demo Attempt Attach: minimal API flow.
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

    # Setup 2 modules: module1 at origin, module2 translated by 0.1 in x to align right<->left
    T_world = {
        1: np.eye(4, dtype=np.float64),
        2: np.eye(4, dtype=np.float64) + np.array([[0,0,0,0.1],[0,0,0,0],[0,0,0,0],[0,0,0,0]])  # T_world[2] has x=0.1
    }
    q_by_module = {
        1: np.array([0., 0.]),
        2: np.array([0., 0.])
    }
    executor = ExecutorWrapper(T_world, q_by_module, ubot_kin)

    graph = ConnectionGraph()
    params = FeasibilityParams()

    print("=== Attempt Attach Demo ===")

    # Attempt attach ma_right (1) <-> mb_left (2)
    a_ref = SiteRef(1, "ma", "right")
    b_ref = SiteRef(2, "mb", "left")
    result = attempt_attach(graph, executor, a_ref, b_ref, params)

    print(f"Attach ma_right <-> mb_left: feasible={result.feasible}, reason='{result.reason}', best_yaw={result.best_yaw_deg}°")
    print(f"Active edges: {len(graph.active_edges())}")

    # Attempt again with same a: should fail occupied
    result2 = attempt_attach(graph, executor, a_ref, SiteRef(1, "ma", "bottom"), params)  # Different but same module
    print(f"Re-attempt with occupied site: feasible={result2.feasible}, reason='{result2.reason}'")

    # For misaligned, offset module2 in y for pos fail
    executor.T_world[2] = T_world[2].copy()
    executor.T_world[2][1, 3] += 0.01  # y offset 1cm
    a2_ref = SiteRef(2, "ma", "bottom")
    b2_ref = SiteRef(1, "mb", "top")  # Dummy, as no module 3, but for pos fail, any
    result3 = attempt_attach(graph, executor, a2_ref, b2_ref, params)
    print(f"Attach with pos mismatch (y offset): feasible={result3.feasible}, reason='{result3.reason}'")


if __name__ == "__main__":
    main()
