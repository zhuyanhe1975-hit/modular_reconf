#!/usr/bin/env python3
"""
Phase-4 Tests: Feasibility-driven Attach Planner.
"""
import pytest
import numpy as np
from reconfiguration.attach_planner import find_attach_candidates, plan_one_attach
from reconfiguration.connection_graph import ConnectionGraph, SiteRef, ConnectionEvent
from reconfiguration.connection_api import ExecutorWrapper
from reconfiguration.connection_feasibility import FeasibilityParams
import ubot


@pytest.fixture
def graph_executor():
    graph = ConnectionGraph(edges={})
    ubot_kin = ubot.UBotKinematics("assets/ubot_ax_centered.xml")
    T_world = {1: np.eye(4), 2: np.eye(4), 3: np.eye(4)}
    T_world[1][:3,3] = [0, 0, 0]
    T_world[2][:3,3] = [0.3, 0, 0]
    T_world[3][:3,3] = [0.6, 0, 0]
    q_by_module = {1: np.array([0.0, 0.0]), 2: np.array([0.0, 0.0]), 3: np.array([0.0, 0.0])}
    executor = ExecutorWrapper(T_world, q_by_module, ubot_kin)
    params = FeasibilityParams(pos_tol=1.0, yaw_tol_deg=10, z_dot_max=-0.99, enable_local_solve=False)
    return graph, executor, params


def test_find_attach_candidates(graph_executor):
    graph, executor, params = graph_executor
    candidates = find_attach_candidates(graph, [1, 2, 3])
    # Should have 1-ma-right to 2-mb-left, etc., 8 pairs but many duplicates
    assert len(candidates) > 0
    a, b = candidates[0]
    assert a.half == "ma" and b.half == "mb"


def test_attach_planner_finds_pair_when_aligned(graph_executor):
    graph, executor, params = graph_executor
    event, reason = plan_one_attach(graph, executor, params, enable_local_solve=False)
    assert event is not None
    assert event.yaw_snap_deg is not None
    assert event.T_a_b is not None


def test_attach_planner_returns_none_when_no_pairs(graph_executor):
    graph, executor, params = graph_executor
    # Occupy all ma.right sites
    graph.apply(ConnectionEvent(kind="attach", a=SiteRef(2, "ma", "right"), b=SiteRef(1, "mb", "right"), yaw_snap_deg=0, T_a_b=np.eye(4)))
    graph.apply(ConnectionEvent(kind="attach", a=SiteRef(3, "ma", "right"), b=SiteRef(2, "mb", "right"), yaw_snap_deg=0, T_a_b=np.eye(4)))
    # Occupy all mb.left sites using dummy a
    graph.apply(ConnectionEvent(kind="attach", a=SiteRef(4, "ma", "right"), b=SiteRef(1, "mb", "left"), yaw_snap_deg=0, T_a_b=np.eye(4)))  # dummy
    graph.apply(ConnectionEvent(kind="attach", a=SiteRef(4, "ma", "left"), b=SiteRef(2, "mb", "left"), yaw_snap_deg=0, T_a_b=np.eye(4)))  # dummy
    graph.apply(ConnectionEvent(kind="attach", a=SiteRef(5, "ma", "right"), b=SiteRef(3, "mb", "left"), yaw_snap_deg=0, T_a_b=np.eye(4)))  # dummy
    candidates = find_attach_candidates(graph, [1, 2, 3])
    assert len(candidates) == 0
    event, reason = plan_one_attach(graph, executor, params, enable_local_solve=False)
    assert event is None
    assert reason == "no_candidates"


def test_attach_planner_respects_occupancy(graph_executor):
    graph, executor, params = graph_executor
    # Occupy one site
    graph.apply(ConnectionEvent(kind="attach", a=SiteRef(1, "ma", "right"), b=SiteRef(2, "mb", "left"), yaw_snap_deg=0, T_a_b=np.eye(4)))
    candidates = find_attach_candidates(graph, [1, 2, 3])
    assert not any(a.module_id == 1 and a.site == "right" for a, b in candidates)
