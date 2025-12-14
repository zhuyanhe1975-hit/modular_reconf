import pytest
import numpy as np
from reconfiguration.connection_feasibility import FeasibilityParams, FeasibilityResult, check_attach_feasible, auto_attach
from reconfiguration.connection_graph import SiteRef, ConnectionGraph
import ubot





def test_pos_fail():
    """Test pos fail by offsetting."""
    Tw_a = np.eye(4, dtype=np.float64)
    Tw_b = np.eye(4, dtype=np.float64)
    Tw_b[0, 3] = 0.01  # 1cm offset

    params = FeasibilityParams(pos_tol=1e-4)  # Strict
    result = check_attach_feasible(Tw_a, Tw_b, params)

    assert not result.feasible
    assert result.reason == "pos"
    assert result.pos_err == 0.01


def test_normal_fail():
    """Test normal fail by removing opposition."""
    Tw_a = np.eye(4, dtype=np.float64)
    Tw_b = np.eye(4, dtype=np.float64)  # Same orientation, z parallel not opposite

    params = FeasibilityParams()
    result = check_attach_feasible(Tw_a, Tw_b, params)

    assert not result.feasible
    assert result.reason == "normal"


def test_yaw_fail():
    """Test yaw fail by large misalign."""
    Tw_a = np.eye(4, dtype=np.float64)
    Tw_b = np.eye(4, dtype=np.float64)
    # Flip b z to oppose
    from ubot.fk_sites import roty, rotz
    Tw_b[:3, :3] = roty(180.0)
    # Rotate b by 45 deg around z (after flip)
    Rz45 = rotz(45.0)
    Tw_b[:3, :3] = Tw_b[:3, :3] @ Rz45

    params = FeasibilityParams(yaw_tol_deg=5.0)  # Can't find snap within 5 deg
    result = check_attach_feasible(Tw_a, Tw_b, params)

    assert not result.feasible
    assert result.reason == "yaw"


def test_auto_attach(ubot_kin: ubot.UBotKinematics):
    """Test auto attach with good pair."""
    graph = ConnectionGraph()
    a_ref = SiteRef(1, "ma", "right")
    b_ref = SiteRef(2, "mb", "left")

    Tw_a = np.eye(4, dtype=np.float64)
    Tw_b = np.eye(4, dtype=np.float64)
    from ubot.fk_sites import roty
    Tw_b[:3, :3] = roty(180.0)  # Flip z_b

    params = FeasibilityParams()
    result = auto_attach(graph, a_ref, b_ref, Tw_a, Tw_b, params)

    assert result.feasible
    assert len(graph.active_edges()) == 1


@pytest.fixture
def ubot_kin():
    return ubot.UBotKinematics("assets/ubot_ax_centered.xml")
