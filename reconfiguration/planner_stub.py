#!/usr/bin/env python3
"""
Phase-3.1: Minimal PlannerStub for generating event schedules.
Returns (path, event_schedule) for reconfiguration simulation.
"""

import numpy as np
from typing import Dict, List, Tuple
from .connection_graph import ConnectionEvent, SiteRef
from .connection_feasibility import FeasibilityParams


def generate_dummy_path(num_steps=8):
    """
    Generate a dummy linear path.
    """
    start = np.array([0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.6, 0.0, 0.0])
    end = np.array([0.2, 0.0, 0.0, 0.5, 0.0, 0.0, 0.8, 0.0, 0.0])
    path = []
    for t in np.linspace(0, 1, num_steps + 1):
        config = start + t * (end - start)
        path.append(config.tolist())
    return path


def plan_modular_reconfig(num_steps: int = 8, detach_step: int = 4, attach_success_step: int = 6, detach_then_attach: bool = False) -> Tuple[List, Dict[int, List[ConnectionEvent]]]:
    """
    Minimal planner stub: generates path and event schedule.
    - detaches edge at detach_step
    - attempts successful attach back at attach_success_step (if set)
    - if detach_then_attach: adjust steps to detach early (step 1), attach later (step 7)
    """
    path = generate_dummy_path(num_steps)

    schedule = {}
    if detach_step is not None:
        detach_at = 1 if detach_then_attach else detach_step
        # Detach full chain if attach_success_step set to enable attach
        if attach_success_step is not None:
            schedule[detach_at] = [
                ConnectionEvent(kind="detach", a=SiteRef(module_id=1, half="ma", site="right"), b=SiteRef(module_id=2, half="mb", site="left")),
                ConnectionEvent(kind="detach", a=SiteRef(module_id=2, half="ma", site="right"), b=SiteRef(module_id=3, half="mb", site="left"))
            ]
        elif detach_then_attach:
            schedule[detach_at] = [
                ConnectionEvent(kind="detach", a=SiteRef(module_id=1, half="ma", site="right"), b=SiteRef(module_id=2, half="mb", site="left")),
                ConnectionEvent(kind="detach", a=SiteRef(module_id=2, half="ma", site="right"), b=SiteRef(module_id=3, half="mb", site="left"))
            ]
        else:
            schedule[detach_at] = [
                ConnectionEvent(kind="detach", a=SiteRef(module_id=2, half="ma", site="right"), b=SiteRef(module_id=3, half="mb", site="left"))
            ]
    if attach_success_step is not None:
        if detach_then_attach:
            # Phase-3.2: use attach_planner
            detach_at = 1
            attach_at = 7
            # Simulate graph and executor as initial chain detached, at initial positions
            import ubot
            ubot_kin = ubot.UBotKinematics("assets/ubot_ax_centered.xml")
            # Use initial positions
            init_config = path[0]  # config at the start
            T_world = {mid: np.eye(4) for mid in [1, 2, 3]}
            T_world[1][:3,3] = init_config[:3]
            T_world[2][:3,3] = init_config[3:6]
            T_world[3][:3,3] = init_config[6:]
            q_by_module = {mid: np.array([0.0, 0.0]) for mid in [1, 2, 3]}
            from .connection_api import ExecutorWrapper
            from .connection_graph import ConnectionGraph
            graph_sim = ConnectionGraph()
            executor_sim = ExecutorWrapper(T_world, q_by_module, ubot_kin)
            params = FeasibilityParams(pos_tol=10.0, yaw_tol_deg=5, z_dot_max=-0.99, enable_local_solve=True)
            from .attach_planner import plan_one_attach
            event, reason = plan_one_attach(graph_sim, executor_sim, params, enable_local_solve=True)
            if event:
                schedule[attach_at] = [event]
            # else no attach
        else:
            # Phase-3.1: keep hard coded attach event for compatibility
            detach_at = detach_step
            attach_at = attach_success_step
            schedule[detach_at] = [
                ConnectionEvent(kind="detach", a=SiteRef(module_id=2, half="ma", site="right"), b=SiteRef(module_id=3, half="mb", site="left"))
            ] if detach_step is not None else []
            schedule[attach_at] = [
                ConnectionEvent(kind="attach", a=SiteRef(module_id=1, half="ma", site="right"), b=SiteRef(module_id=3, half="mb", site="left"), yaw_snap_deg=0, T_a_b=np.eye(4))
            ]

    return path, schedule
