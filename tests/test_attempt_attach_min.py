import pytest
import numpy as np
from reconfiguration.connection_feasibility import FeasibilityParams
from reconfiguration.connection_graph import SiteRef, ConnectionGraph
from reconfiguration.connection_api import ExecutorWrapper, attempt_attach
import ubot


@pytest.fixture
def setup_2_modules():
    """Setup 2 modules aligned for right<->left attach."""
    ubot_kin = ubot.UBotKinematics("assets/ubot_ax_centered.xml")
    T_world = {
        1: np.eye(4, dtype=np.float64),
        2: np.eye(4, dtype=np.float64)
    }
    T_world[2][0, 3] += 0.1  # Align x
    q_by_module = {1: np.array([0., 0.]), 2: np.array([0., 0.])}
    executor = ExecutorWrapper(T_world, q_by_module, ubot_kin)
    graph = ConnectionGraph()
    params = FeasibilityParams()
    return graph, executor, params


def test_success_attach_adds_edge(setup_2_modules):
    """Attach adds active edge to graph."""
    graph, executor, params = setup_2_modules
    a_ref = SiteRef(1, "ma", "right")
    b_ref = SiteRef(2, "mb", "left")
    result = attempt_attach(graph, executor, a_ref, b_ref, params)
    assert result.feasible
    assert len(graph.active_edges()) == 1


def test_occupied_site_rejected(setup_2_modules):
    """Occupied site rejects attach."""
    graph, executor, params = setup_2_modules
    a_ref = SiteRef(1, "ma", "right")
    b_ref = SiteRef(2, "mb", "left")
    attempt_attach(graph, executor, a_ref, b_ref, params)  # Attach first

    # Try attach using same a_ref
    result2 = attempt_attach(graph, executor, a_ref, SiteRef(1, "ma", "bottom"), params)
    assert not result2.feasible
    assert result2.reason == "occupied_a"


def test_pos_misalignment_rejected(setup_2_modules):
    """Pos mismatch rejects attach."""
    graph, executor, params = setup_2_modules
    # Offset executor.T_world[2] in y
    executor.T_world[2][1, 3] += 0.01
    a_ref = SiteRef(2, "ma", "bottom")
    b_ref = SiteRef(1, "mb", "top")
    result = attempt_attach(graph, executor, a_ref, b_ref, params)
    assert not result.feasible
    assert result.reason == "pos"
