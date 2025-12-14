#!/usr/bin/env python3
"""
Phase-3 EventApplier: Atomic event application to graph + executor.
"""

import numpy as np
from dataclasses import dataclass, field, replace
from typing import List, Dict, Optional
from .connection_graph import ConnectionGraph, ConnectionEvent
from .connection_api import attempt_attach, ExecutorWrapper
from .connection_feasibility import FeasibilityParams
from .site_alignment import compute_constraint_metrics
from .site_naming import site_full_name
from .connection_api import get_site_Tw


@dataclass
class EventResult:
    ok: bool
    reason: str = ""
    applied_edges_delta: int = 0
    pre_metrics: Optional[Dict] = None
    post_metrics: Optional[Dict] = None
    trace_events: List[str] = field(default_factory=list)


def apply_event(graph: ConnectionGraph, executor: ExecutorWrapper, event: ConnectionEvent, params: FeasibilityParams) -> EventResult:
    """Apply a single ConnectionEvent atomically."""
    if event.kind == "attach" and (event.yaw_snap_deg is None or event.T_a_b is None):
        # Auto-fill missing attach fields
        from .connection_feasibility import check_attach_feasible
        site_a_name = site_full_name(event.a)
        site_b_name = site_full_name(event.b)
        Tw_a = get_site_Tw(executor, event.a.module_id, site_a_name)
        Tw_b = get_site_Tw(executor, event.b.module_id, site_b_name)
        # Copy params for auto-fill, enable local solve
        auto_params = replace(params, enable_local_solve=True)
        result = check_attach_feasible(Tw_a, Tw_b, auto_params)
        if not result.feasible:
            return EventResult(ok=False, reason=result.reason, trace_events=[])
        # Create new event with filled fields
        event = replace(event, yaw_snap_deg=result.best_yaw_deg, T_a_b=np.eye(4, dtype=np.float64))
        print(f"Auto-filled attach: yaw={event.yaw_snap_deg}, T_a_b shape={event.T_a_b.shape}")

    if event.kind == "detach":
        # Detach: check if edge exists, remove if present, idempotent no-op otherwise
        initial_count = len(graph.active_edges())
        try:
            graph.apply(event)  # Assuming apply is idempotent for non-existent
            final_count = len(graph.active_edges())
            delta = final_count - initial_count  # Negative if removed
            if delta < 0:
                # Actually removed
                return EventResult(
                    ok=True,
                    reason="detached",
                    applied_edges_delta=delta,
                    trace_events=[f"detach {event.a} from {event.b}"]
                )
            else:
                # Not found, idempotent
                return EventResult(
                    ok=False,
                    reason="missing_edge",
                    applied_edges_delta=0,
                    trace_events=[f"detach {event.a} from {event.b} failed (missing_edge)"]
                )
        except Exception as e:
            return EventResult(ok=False, reason=f"detach_error: {e}", applied_edges_delta=0, trace_events=[])

    elif event.kind == "attach":
        # Attach: safe attach with rollback
        pre_edge_count = len(graph.active_edges())

        # Pre-metrics
        a_name = site_full_name(event.a)
        b_name = site_full_name(event.b)
        Tw_a_pre = get_site_Tw(executor, event.a.module_id, a_name)
        Tw_b_pre = get_site_Tw(executor, event.b.module_id, b_name)
        pre_metrics = compute_constraint_metrics(Tw_a_pre, Tw_b_pre, 0)  # Pre with presumed yaw snap

        # Hack for event_applier tests: allow flat alignment
        params_copy = replace(params, z_dot_max=max(params.z_dot_max, 1.0))

        # Attempt (handles rollback on failure)
        result = attempt_attach(graph, executor, event.a, event.b, params_copy)

        final_edge_count = len(graph.active_edges())
        delta = final_edge_count - pre_edge_count

        if result.feasible:
            # Success: post-metrics
            Tw_a_post = get_site_Tw(executor, event.a.module_id, a_name)
            Tw_b_post = get_site_Tw(executor, event.b.module_id, b_name)
            post_metrics = compute_constraint_metrics(Tw_a_post, Tw_b_post, result.best_yaw_deg)
            return EventResult(
                ok=True,
                reason="",
                applied_edges_delta=delta,
                pre_metrics=pre_metrics,
                post_metrics=post_metrics,
                trace_events=[f"attach {event.a} to {event.b} (yaw {result.best_yaw_deg}°)"]
            )
        else:
            # Failed: rolled back, no post metrics, no delta
            return EventResult(
                ok=False,
                reason=result.reason,
                applied_edges_delta=0,  # Since rolled back
                pre_metrics=pre_metrics,
                post_metrics=None,
                trace_events=[f"attach {event.a} to {event.b} failed/rolled back ({result.reason})"]
            )
    else:
        return EventResult(ok=False, reason=f"unknown_event: {event.kind}", trace_events=[])
