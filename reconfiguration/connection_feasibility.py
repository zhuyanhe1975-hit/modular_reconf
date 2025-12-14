#!/usr/bin/env python3
"""
Connection Feasibility for UBot Phase-2.5
Pre-attach check with yaw snapping.
"""

from dataclasses import dataclass, field
import numpy as np
from typing import List, Dict

from .site_alignment import compute_constraint_metrics


@dataclass
class FeasibilityParams:
    pos_tol: float = 1e-3
    z_dot_max: float = -0.999  # Must have dot <= this for opposition
    yaw_candidates_deg: tuple[int, ...] = (0, 90, 180, 270)
    yaw_tol_deg: float = 2.0  # Max |residual_yaw_deg| for feasible
    enable_local_solve: bool = False  # If true, try local joint correction on near failures


@dataclass
class FeasibilityResult:
    feasible: bool = False
    best_yaw_deg: int | None = None
    pos_err: float = 0.0
    z_dot: float = 0.0
    raw_yaw_after_flip_deg: float = 0.0
    residual_yaw_deg: float = 0.0
    candidate_table: List[Dict] = field(default_factory=list)
    reason: str = ""


def check_attach_feasible(Tw_site_a: np.ndarray, Tw_site_b: np.ndarray, params: FeasibilityParams) -> FeasibilityResult:
    """
    Check if two sites can be attached.
    """
    # Compute metrics with test yaw (0), but for pos, z_dot use any
    metrics_base = compute_constraint_metrics(Tw_site_a, Tw_site_b, 0)  # Use yaw 0 for base

    pos_err = metrics_base['pos_err']
    z_dot = metrics_base['z_dot']

    # Check pos feasibility
    if pos_err > params.pos_tol:
        return FeasibilityResult(
            feasible=False, best_yaw_deg=None,
            pos_err=pos_err, z_dot=z_dot,
            raw_yaw_after_flip_deg=0.0, residual_yaw_deg=0.0,  # Placeholder
            candidate_table=[],
            reason="pos"
        )

    # Check normal feasibility
    if z_dot > params.z_dot_max:
        return FeasibilityResult(
            feasible=False, best_yaw_deg=None,
            pos_err=pos_err, z_dot=z_dot,
            raw_yaw_after_flip_deg=0.0, residual_yaw_deg=0.0,
            candidate_table=[],
            reason="normal"
        )

    # Compute after flip (equivalent to compute_constraint_metrics with yaw=0 but z_dot flipped)
    metrics_0 = compute_constraint_metrics(Tw_site_a, Tw_site_b, 0)
    raw_yaw_after_flip_deg = metrics_0['rel_yaw_deg']  # Since rel_yaw after flip 0 is the after flip yaw

    # Check yaw candidates
    candidates = []
    best_abs_residual = float('inf')
    best_yaw = None

    for yaw in params.yaw_candidates_deg:
        metrics = compute_constraint_metrics(Tw_site_a, Tw_site_b, yaw)
        residual = metrics['rel_yaw_deg']
        abs_res = abs(residual)
        candidates.append({'yaw_deg': yaw, 'residual_yaw_deg': residual})

        if abs_res < best_abs_residual:
            best_abs_residual = abs_res
            best_yaw = yaw

    residual_yaw_deg = best_abs_residual

    feasible = best_abs_residual <= params.yaw_tol_deg

    reason = "" if feasible else "yaw"

    return FeasibilityResult(
        feasible=feasible,
        best_yaw_deg=best_yaw,
        pos_err=pos_err, z_dot=z_dot,
        raw_yaw_after_flip_deg=raw_yaw_after_flip_deg,
        residual_yaw_deg=residual_yaw_deg,
        candidate_table=candidates,
        reason=reason
    )


def auto_attach(graph, a_ref, b_ref, Tw_a, Tw_b, params: FeasibilityParams) -> FeasibilityResult:
    """
    Check feasibility and attach if possible.
    """
    from .connection_graph import EdgeKey, ConnectionEvent

    # Check sites free
    if not graph.site_is_free(a_ref):
        return FeasibilityResult(feasible=False, reason="occupied_a", best_yaw_deg=None, pos_err=0, z_dot=0, raw_yaw_after_flip_deg=0, residual_yaw_deg=0)
    if not graph.site_is_free(b_ref):
        return FeasibilityResult(feasible=False, reason="occupied_b", best_yaw_deg=None, pos_err=0, z_dot=0, raw_yaw_after_flip_deg=0, residual_yaw_deg=0)

    result = check_attach_feasible(Tw_a, Tw_b, params)
    if result.feasible:
        # Attach
        ev = ConnectionEvent(kind="attach", a=a_ref, b=b_ref, yaw_snap_deg=result.best_yaw_deg, T_a_b=np.eye(4, dtype=np.float64))
        graph.apply(ev)

    return result
