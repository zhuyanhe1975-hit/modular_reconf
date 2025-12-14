import pytest
import numpy as np

from reconfiguration.connection import ConnectParams
from reconfiguration.connection_graph import (
    SiteRef, ConnectionEvent, EdgeKey, ConnectionEdge, ConnectionGraph,
    parse_site_name, make_attach_event
)
from ubot.site_pose_loader import load_site_poses


@pytest.fixture
def sample_poses():
    xml_path = "assets/ubot_ax_centered.xml"
    return load_site_poses(xml_path)


def test_edgekey_normalized():
    """Test EdgeKey normalization."""
    a = SiteRef(module_id=1, half="ma", site="connector_right")
    b = SiteRef(module_id=2, half="mb", site="connector_left")
    key_ab = EdgeKey.normalized(a, b)
    key_ba = EdgeKey.normalized(b, a)
    assert key_ab == key_ba


def test_parse_site_name():
    """Test parsing site names."""
    assert parse_site_name("ma_connector_right") == ("ma", "connector_right")
    assert parse_site_name("mb_connector_top") == ("mb", "connector_top")


def test_graph_apply_attach(sample_poses):
    """Test applying attach event creates active edge."""
    g = ConnectionGraph()
    a_ref = SiteRef(1, "ma", "connector_right")
    b_ref = SiteRef(2, "mb", "connector_left")
    # Override positions for contact
    pose_a = sample_poses["ma_connector_right"]
    pose_b = sample_poses["mb_connector_left"]
    pose_b.position = pose_a.position.copy()

    event = make_attach_event(1, "ma_connector_right", pose_a, 2, "mb_connector_left", pose_b)
    g.apply(event)
    assert len(g.active_edges()) == 1
    assert g.is_connected(a_ref, b_ref)


def test_graph_apply_detach(sample_poses):
    """Test applying detach deactivates edge."""
    g = ConnectionGraph()
    pose_a = sample_poses["ma_connector_right"]
    pose_b = sample_poses["mb_connector_left"]
    pose_b.position = pose_a.position.copy()

    event_attach = make_attach_event(1, "ma_connector_right", pose_a, 2, "mb_connector_left", pose_b)
    g.apply(event_attach)
    assert len(g.active_edges()) == 1

    event_detach = ConnectionEvent(kind="detach", a=event_attach.a, b=event_attach.b)
    g.apply(event_detach)
    assert len(g.active_edges()) == 0


def test_double_attach_raises(sample_poses):
    """Test attaching to a used site raises."""
    g = ConnectionGraph()
    pose_a = sample_poses["ma_connector_right"]
    pose_b = sample_poses["mb_connector_left"]
    pose_b.position = pose_a.position.copy()

    event1 = make_attach_event(1, "ma_connector_right", pose_a, 2, "mb_connector_left", pose_b)
    g.apply(event1)

    # Now try to attach to the same site (ma_connector_right)
    pose_d = sample_poses["mb_connector_top"]
    pose_d.position = pose_a.position.copy()  # Force contact
    pose_d.quat_wxyz = np.array([0.707, 0, 0.707, 0])  # Override for test
    params_large_eps = ConnectParams(eps_normal_deg=200)  # Allow any to test graph logic
    event2 = make_attach_event(1, "ma_connector_right", pose_a, 3, "mb_connector_top", pose_d, params=params_large_eps)
    with pytest.raises(ValueError, match="already connected"):
        g.apply(event2)


def test_make_attach_with_real_poses(sample_poses):
    """Test make_attach_event with positions set equal (feasible)."""
    pose_a = sample_poses["ma_connector_right"]
    pose_b = sample_poses["mb_connector_left"]
    pose_b.position = pose_a.position.copy()

    event = make_attach_event(1, "ma_connector_right", pose_a, 2, "mb_connector_left", pose_b)
    assert event.yaw_snap_deg in [0, 90, 180, 270]
    assert event.T_a_b.shape == (4, 4)


def test_make_attach_negative(sample_poses):
    """Test make_attach_event with positions not equal (not feasible)."""
    pose_a = sample_poses["ma_connector_right"]
    pose_b = sample_poses["mb_connector_left"]
    # Keep positions different

    with pytest.raises(ValueError, match="not feasible"):
        make_attach_event(1, "ma_connector_right", pose_a, 2, "mb_connector_left", pose_b)


def test_site_is_free(sample_poses):
    """Test site_is_free when disconnected vs connected."""
    g = ConnectionGraph()
    a_ref = SiteRef(1, "ma", "connector_right")
    b_ref = SiteRef(2, "mb", "connector_left")
    pose_a = sample_poses["ma_connector_right"]
    pose_b = sample_poses["mb_connector_left"]
    pose_b.position = pose_a.position.copy()

    assert g.site_is_free(a_ref)
    assert g.site_is_free(b_ref)

    event = make_attach_event(1, "ma_connector_right", pose_a, 2, "mb_connector_left", pose_b)
    g.apply(event)

    assert not g.site_is_free(a_ref)
    assert not g.site_is_free(b_ref)
