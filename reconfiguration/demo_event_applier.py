#!/usr/bin/env python3
"""
Demo EventApplier Phase-3: Atomic event application and result reporting.
"""

from reconfiguration.event_applier import apply_event, EventResult
from reconfiguration.connection_graph import ConnectionGraph, ConnectionEvent, SiteRef
from reconfiguration.connection_api import ExecutorWrapper
from reconfiguration.connection_feasibility import FeasibilityParams
import numpy as np
import ubot

# Setup
ubot_kin = ubot.UBotKinematics("assets/ubot_ax_centered.xml")
T_world = {
    1: np.eye(4),
    2: np.eye(4) + np.array([[0,0,0,0.3],[0,0,0,0],[0,0,0,0],[0,0,0,0]]),
    3: np.eye(4) + np.array([[0,0,0,0.6],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
}
executor = ExecutorWrapper(T_world, {1: np.zeros(2), 2: np.zeros(2), 3: np.zeros(2)}, ubot_kin)
graph = ConnectionGraph()
params = FeasibilityParams(enable_local_solve=False, pos_tol=1e-3, yaw_tol_deg=10, z_dot_max=-0.999)

print("=== EventApplier Demo ===")

# 1. Successful attach
event1 = ConnectionEvent(kind="attach", a=SiteRef(1, "ma", "right"), b=SiteRef(2, "mb", "left"))
result1 = apply_event(graph, executor, event1, params)
print(f"Attach Success: {result1}")

# 2. Occupied site reject
event2 = ConnectionEvent(kind="attach", a=SiteRef(1, "ma", "right"), b=SiteRef(3, "mb", "left"))
result2 = apply_event(graph, executor, event2, params)
print(f"Occupied Reject: {result2}")

# 3. Detach success
event3 = ConnectionEvent(kind="detach", a=SiteRef(1, "ma", "right"), b=SiteRef(2, "mb", "left"))
result3 = apply_event(graph, executor, event3, params)
print(f"Detach Success: {result3}")

# 4. Detach missing
event4 = ConnectionEvent(kind="detach", a=SiteRef(1, "ma", "right"), b=SiteRef(2, "mb", "left"))
result4 = apply_event(graph, executor, event4, params)
print(f"Detach Missing: {result4}")
