#!/usr/bin/env python3
"""
Demo Local Attach Solver Phase-2.6
Micro-correction using joint adjustments.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, Path(__file__).parent.parent)

from reconfiguration.local_attach_solver import LocalSolveParams, solve_local_attach
from reconfiguration.connection_feasibility import FeasibilityParams
from reconfiguration.connection_api import ExecutorWrapper, attempt_attach
from reconfiguration.connection_graph import SiteRef, ConnectionGraph
import ubot


def main():
    path = Path(__file__).parent.parent / "assets" / "ubot_ax_centered.xml"
    ubot_kin = ubot.UBotKinematics(str(path))

    # Setup modules for near miss: misalign by 2mm in x (pos fails)
    T_world = {
        1: np.eye(4, dtype=np.float64),
        2: np.eye(4, dtype=np.float64) + np.array([[0,0,0,0.102],[0,0,0,0],[0,0,0,0],[0,0,0,0]])  # x=0.1 + 2mm
    }
    q_by_module = {
        1: np.array([0., 0.]),
        2: np.array([0., 0.])
    }
    executor = ExecutorWrapper(T_world, q_by_module, ubot_kin)

    graph = ConnectionGraph()
    params_local = LocalSolveParams(pos_tol=5e-3)  # Set target to 5mm
    params_attach = FeasibilityParams(enable_local_solve=True, pos_tol=5e-3, yaw_tol_deg=2.0)

    print("=== Local Attach Solver Demo ===")

    a_ref = SiteRef(1, "ma", "right")
    b_ref = SiteRef(2, "mb", "left")

    # Run local solve directly
    result_solve = solve_local_attach(executor, a_ref, b_ref, params_local)

    print("Local solve:")
    if result_solve.success:
        print(f"  Success, iters={result_solve.iters}")
        print(f"  Before: pos_err={result_solve.metrics_before['pos_err']:.4f}, z_dot={result_solve.metrics_before['z_dot']:.4f}")
        print(f"  After: pos_err={result_solve.metrics_after['pos_err']:.4f}, z_dot={result_solve.metrics_after['z_dot']:.4f}")
    else:
        print(f"  Failed: {result_solve.reason}")

    # Now attempt attach with local solve enabled
    result_attach = attempt_attach(graph, executor, a_ref, b_ref, params_attach)

    print(f"\nAttach with local solve enabled: feasible={result_attach.feasible}, reason='{result_attach.reason}'")
    if result_attach.feasible:
        print(f"  Active edges: {len(graph.active_edges())}")


if __name__ == "__main__":
    main()
