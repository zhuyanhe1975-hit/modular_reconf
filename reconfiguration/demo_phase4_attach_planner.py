#!/usr/bin/env python3
"""
Phase-4 Demo: Feasibility-driven Attach Planner.

Shows planner finding best attach candidate.
"""

import sys
sys.path.insert(0, '/home/yhzhu/myWorks/UBot/modular_reconf')
import numpy as np
from reconfiguration.attach_planner import find_attach_candidates, plan_one_attach
from reconfiguration.connection_graph import ConnectionGraph
from reconfiguration.connection_api import ExecutorWrapper
from reconfiguration.connection_feasibility import FeasibilityParams
import ubot


def main():
    print("=== Phase-4: Feasibility-driven Attach Planner Demo ===\n")

    # Setup similar to simulation initial state
    graph = ConnectionGraph(edges={})
    ubot_kin = ubot.UBotKinematics("assets/ubot_ax_centered.xml")
    T_world = {1: np.eye(4), 2: np.eye(4), 3: np.eye(4)}
    T_world[1][:3,3] = [0, 0, 0]
    T_world[2][:3,3] = [0.3, 0, 0]
    T_world[3][:3,3] = [0.6, 0, 0]
    q_by_module = {1: np.array([0.0, 0.0]), 2: np.array([0.0, 0.0]), 3: np.array([0.0, 0.0])}
    executor = ExecutorWrapper(T_world, q_by_module, ubot_kin)
    params = FeasibilityParams(pos_tol=1.0, yaw_tol_deg=10, z_dot_max=-0.99, enable_local_solve=False)

    modules = [1, 2, 3]

    # Find candidates
    candidates = find_attach_candidates(graph, modules)
    print(f"Found {len(candidates)} candidate pairs:")
    for i, (a, b) in enumerate(candidates[:5]):  # Show first 5
        print(f"  {i+1}. {a} -> {b}")
    if len(candidates) > 5:
        print(f"  ... and {len(candidates)-5} more")

    # Plan attach
    print("\nPlanning best attach (enable_local_solve=False):")
    event, reason = plan_one_attach(graph, executor, params, enable_local_solve=False)
    if event:
        print(f"Selected: {event.a} -> {event.b} with yaw={event.yaw_snap_deg}°")
    else:
        print(f"No feasible attach: {reason}")

    # Apply and show graph
    if event:
        from reconfiguration.event_applier import apply_event
        result = apply_event(graph, executor, event, params)
        if result.ok:
            print(f"Applied successfully. Active edges: {len(graph.active_edges())}")
        else:
            print(f"Failed to apply: {result.reason}")

    print("\nDemo complete.")


if __name__ == "__main__":
    main()
