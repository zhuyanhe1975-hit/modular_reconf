import pytest
import numpy as np

from reconfiguration.connection_graph import SiteRef, EdgeKey, ConnectionEdge, ConnectionGraph
from reconfiguration.kinematic_compiler import compile_kinematic_tree
from reconfiguration.kinematic_executor_v2 import propagate_world_poses_with_sites
from reconfiguration.site_naming import site_full_name
from reconfiguration.site_alignment import compute_constraint_metrics
import ubot


def create_test_connection_graph() -> ConnectionGraph:
    """Simple graph with one connection."""
    graph = ConnectionGraph()
    # Edge 1 ma_right <-> 2 mb_left, yaw=0
    site_a = SiteRef(1, "ma", "right")
    site_b = SiteRef(2, "mb", "left")
    edge = ConnectionEdge(key=EdgeKey.normalized(site_a, site_b), yaw_snap_deg=0, T_a_b=np.eye(4, dtype=np.float64))
    graph.edges[edge.key] = edge
    return graph


@pytest.fixture
def ubot_kin():
    """UBotKinematics instance from XML."""
    return ubot.UBotKinematics("assets/ubot_ax_centered.xml")


def test_site_coincidence_constraint(ubot_kin: ubot.UBotKinematics):
    """Test that attached sites origins coincide and Z axes oppose."""
    graph = create_test_connection_graph()

    tree = compile_kinematic_tree(1, graph)
    assert len(tree.attachments) == 1

    # q all zero
    q_by_module = {1: np.array([0., 0.]), 2: np.array([0., 0.])}
    T_world_root = np.eye(4, dtype=np.float64)
    result = propagate_world_poses_with_sites(tree, T_world_root, q_by_module, ubot_kin)

    # World site frames for the attached sites
    att = tree.attachments[2]
    parent_site_name = site_full_name(att.parent_site)  # "ma_connector_right"
    child_site_name = site_full_name(att.child_site)    # "mb_connector_left"

    T_world_site1 = result.T_world[1] @ ubot_kin.T_ax_site(q_by_module[1], parent_site_name)
    T_world_site2 = result.T_world[2] @ ubot_kin.T_ax_site(q_by_module[2], child_site_name)

    # Origins coincide
    pos1 = T_world_site1[:3, 3]
    pos2 = T_world_site2[:3, 3]
    dist = np.linalg.norm(pos1 - pos2)
    assert dist < 1e-6, f"Origins not coincident: {pos1} vs {pos2}"

    # Z axes oppose (dot < -0.999)
    z1 = T_world_site1[:3, 2]
    z2 = T_world_site2[:3, 2]
    dot_z = np.dot(z1, z2)
    assert dot_z < -0.999, f"Z axes not opposing: dot={dot_z}"


def test_yaw_snap(ubot_kin: ubot.UBotKinematics):
    """Test yaw snapping with yaw=90, rel_yaw close to 0 after compensation."""
    graph = ConnectionGraph()
    # Edge 1 ma_right <-> 2 mb_left, yaw=90
    site_a = SiteRef(1, "ma", "right")
    site_b = SiteRef(2, "mb", "left")
    edge = ConnectionEdge(key=EdgeKey.normalized(site_a, site_b), yaw_snap_deg=90, T_a_b=np.eye(4, dtype=np.float64))
    graph.edges[edge.key] = edge

    tree = compile_kinematic_tree(1, graph)

    q_by_module = {1: np.array([0., 0.]), 2: np.array([0., 0.])}
    T_world_root = np.eye(4, dtype=np.float64)
    result = propagate_world_poses_with_sites(tree, T_world_root, q_by_module, ubot_kin)

    att = tree.attachments[2]
    parent_site_name = site_full_name(att.parent_site)
    child_site_name = site_full_name(att.child_site)

    T_world_site_p = result.T_world[att.parent] @ ubot_kin.T_ax_site(q_by_module[att.parent], parent_site_name)
    T_world_site_c = result.T_world[att.child] @ ubot_kin.T_ax_site(q_by_module[att.child], child_site_name)

    metrics = compute_constraint_metrics(T_world_site_p, T_world_site_c, att.yaw_snap_deg)

    # After compensation, rel_yaw should be small
    assert abs(metrics['rel_yaw_deg']) < 1.0
