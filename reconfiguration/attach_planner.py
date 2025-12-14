#!/usr/bin/env python3
"""
Phase-4: Feasibility-driven Attach Planner.

Finds candidate free site pairs and plans best feasible attach.
"""

import numpy as np
from typing import List, Tuple, Optional
from .connection_graph import ConnectionGraph, SiteRef, ConnectionEvent
from .connection_feasibility import FeasibilityParams
from .connection_api import ExecutorWrapper, get_site_Tw


def find_attach_candidates(graph: ConnectionGraph, modules: List[int]) -> List[Tuple[SiteRef, SiteRef]]:
    """
    Generate all free/unoccupied ma-mb site pairs amenable to attach.
    ma halves connect to mb halves only.
    """
    candidates = []
    for m1 in modules:
        for m2 in modules:
            if m1 == m2:  # don't attach same module
                continue
            # Connect ma.right to mb.left (as in demo)
            a = SiteRef(module_id=m1, half="ma", site="right")
            if not graph.site_is_free(a):
                continue
            b = SiteRef(module_id=m2, half="mb", site="left")
            if graph.site_is_free(b):
                candidates.append((a, b))

    return candidates


def plan_one_attach(graph: ConnectionGraph, executor: ExecutorWrapper, params: FeasibilityParams, *, enable_local_solve: bool = False) -> Tuple[Optional[ConnectionEvent], str]:
    """
    Find best feasible attach among candidates.

    Iterates candidates, checks feasibility, picks min pos_err / min residual yaw.

    If enable_local_solve: try solving for infeasible but pos_err close.

    Returns (event, reason_or_empty)
    """
    modules = list(executor.q_by_module.keys())
    candidates = find_attach_candidates(graph, modules)
    if not candidates:
        return None, "no_candidates"

    best_pos_err = float('inf')
    best_yaw_deg = None
    best_T_a_b = None
    best_a = None
    best_b = None

    for a, b in candidates:
        from .site_naming import site_full_name
        Tw_a = get_site_Tw(executor, a.module_id, site_full_name(a))
        Tw_b = get_site_Tw(executor, b.module_id, site_full_name(b))
        from .connection_feasibility import FeasibilityParams
        params_local = FeasibilityParams(
            enable_local_solve=False,
            pos_tol=params.pos_tol,
            yaw_candidates_deg=params.yaw_candidates_deg,
            yaw_tol_deg=params.yaw_tol_deg,
            z_dot_max=params.z_dot_max
        )
        from .connection_feasibility import check_attach_feasible
        feas_result = check_attach_feasible(Tw_a, Tw_b, params_local)

        if feas_result.feasible:
            if (feas_result.pos_err < best_pos_err or
                (feas_result.pos_err == best_pos_err and
                 abs(feas_result.residual_yaw_deg) < abs(best_yaw_deg or 0))):
                best_pos_err = feas_result.pos_err
                best_yaw_deg = feas_result.best_yaw_deg
                # Compute T_a_b per conventions: A-site frame to B-site frame
                T_a_b = np.linalg.inv(Tw_a) @ Tw_b
                best_T_a_b = T_a_b
                best_a, best_b = a, b

    if best_a is None:
        return None, "no_feasible_pair"

    # If enable_local_solve and best_pos_err close to tol, try solving
    if enable_local_solve and best_pos_err > params.pos_tol * 0.5:  # some threshold
        from .local_attach_solver import LocalSolveParams, solve_local_attach
        solve_params = LocalSolveParams(pos_tol=params.pos_tol, yaw_tol_deg=params.yaw_tol_deg, z_dot_tol_above=params.z_dot_max)
        solve_result = solve_local_attach(executor, best_a, best_b, solve_params)
        if solve_result.success:
            # Update executor.q_by_module
            executor.q_by_module[best_a.module_id] = solve_result.q_a
            executor.q_by_module[best_b.module_id] = solve_result.q_b
            # Re-check
            Tw_a = get_site_Tw(executor, best_a.module_id, site_full_name(best_a))
            Tw_b = get_site_Tw(executor, best_b.module_id, site_full_name(best_b))
            feas_result = check_attach_feasible(Tw_a, Tw_b, params)
            if feas_result.feasible:
                best_yaw_deg = feas_result.best_yaw_deg

    # Create event
    event = ConnectionEvent(
        kind="attach",
        a=best_a,
        b=best_b,
        yaw_snap_deg=best_yaw_deg,
        T_a_b=best_T_a_b
    )
    return event, ""
