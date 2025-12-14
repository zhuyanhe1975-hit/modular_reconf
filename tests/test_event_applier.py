#!/usr/bin/env python3
"""
Tests for Phase-3 EventApplier.
"""

import pytest
import numpy as np
from reconfiguration.event_applier import apply_event, EventResult
from reconfiguration.connection_graph import ConnectionGraph, ConnectionEvent, SiteRef
from reconfiguration.connection_api import ExecutorWrapper
from reconfiguration.connection_feasibility import FeasibilityParams
from reconfiguration.site_alignment import compute_constraint_metrics
from reconfiguration.site_naming import site_full_name
from reconfiguration.connection_api import get_site_Tw
import ubot


@pytest.fixture
def setup_graph_executor():
    """Fixturoot for basic graph and executor."""
    from reconfiguration.connection_graph import ConnectionGraph, SiteRef
    from reconfiguration.connection_api import ExecutorWrapper
    ubot_kin = ubot.UBotKinematics("assets/ubot_ax_centered.xml")
    T_world = {
        1: np.eye(4, dtype=np.float64),
        2: np.eye(4, dtype=np.float64),
        3: np.eye(4, dtype=np.float64)
    }
    T_world[2][:3,3] = [0.3, 0, 0]
    T_world[3][:3,3] = [0.6, 0, 0]
    executor = ExecutorWrapper(T_world, {1: np.zeros(2), 2: np.zeros(2), 3: np.zeros(2)}, ubot_kin)
    graph = ConnectionGraph()
    params = FeasibilityParams(enable_local_solve=False, pos_tol=1.0, yaw_tol_deg=10, z_dot_max=-0.99)  # Relax pos for testing
    return graph, executor, params


def test_attach_success(setup_graph_executor):
    """Test successful attach event."""
    graph, executor, params = setup_graph_executor
    event = ConnectionEvent(kind="attach", a=SiteRef(1, "ma", "right"), b=SiteRef(2, "mb", "left"), yaw_snap_deg=0)
    result = apply_event(graph, executor, event, params)
    assert result.ok == True
    assert result.reason == ""
    assert result.applied_edges_delta == 1  # Added 1 edge
    assert result.pre_metrics is not None
    assert result.post_metrics is not None
    assert len(result.trace_events) == 1
    assert "attach" in result.trace_events[0]


def test_attach_occupied_reject(setup_graph_executor):
    """Test attach to occupied site rejects."""
    graph, executor, params = setup_graph_executor
    # First attach successfully
    event1 = ConnectionEvent(kind="attach", a=SiteRef(1, "ma", "right"), b=SiteRef(2, "mb", "left"), yaw_snap_deg=0)
    result1 = apply_event(graph, executor, event1, params)
    assert result1.ok

    # Now try attach to same site
    event2 = ConnectionEvent(kind="attach", a=SiteRef(1, "ma", "right"), b=SiteRef(3, "mb", "left"), yaw_snap_deg=0)
    result2 = apply_event(graph, executor, event2, params)
    assert result2.ok == False
    assert "occupied" in result2.reason or "precheck" in result2.reason
    assert result2.applied_edges_delta == 0


def test_detach_success(setup_graph_executor):
    """Test successful detach event."""
    graph, executor, params = setup_graph_executor
    # First attach
    attach_event = ConnectionEvent(kind="attach", a=SiteRef(1, "ma", "right"), b=SiteRef(2, "mb", "left"), yaw_snap_deg=0)
    attach_result = apply_event(graph, executor, attach_event, params)
    assert attach_result.ok
    assert len(graph.active_edges()) == 1

    # Now detach
    detach_event = ConnectionEvent(kind="detach", a=SiteRef(1, "ma", "right"), b=SiteRef(2, "mb", "left"))
    detach_result = apply_event(graph, executor, detach_event, params)
    assert detach_result.ok == True
    assert detach_result.reason == "detached"
    assert detach_result.applied_edges_delta == -1  # Removed 1 edge


def test_detach_missing_edge(setup_graph_executor):
    """Test detach of non-existent edge."""
    graph, executor, params = setup_graph_executor
    detach_event = ConnectionEvent(kind="detach", a=SiteRef(1, "ma", "right"), b=SiteRef(2, "mb", "left"))
    result = apply_event(graph, executor, detach_event, params)
    assert result.ok == False
    assert result.reason == "missing_edge"
    assert result.applied_edges_delta == 0


# Note: Attach rollback test is covered internally by attempt_attach, so rely on behavior tests
