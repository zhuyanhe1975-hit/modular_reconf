import pytest
import numpy as np
from reconfiguration.connection_graph import SiteRef, EdgeKey, ConnectionEdge
from reconfiguration.connection_graph import ConnectionGraph, ConnectionEvent
from reconfiguration.kinematic_compiler import compile_kinematic_tree, KinematicTree


def _make_edge(module_a: int, site_a: SiteRef | str, module_b: int, site_b: SiteRef | str, T_a_b: np.ndarray) -> ConnectionEdge:
    """Helper to create edge for testing."""
    if isinstance(site_a, str):
        site_a = SiteRef(module_a, "ma", site_a)  # dummy
    if isinstance(site_b, str):
        site_b = SiteRef(module_b, "mb", site_b)
    key = EdgeKey.normalized(site_a, site_b)
    return ConnectionEdge(key=key, yaw_snap_deg=0, T_a_b=T_a_b)


def _add_edges_to_graph(graph: ConnectionGraph, edges: list[ConnectionEdge]):
    """Directly add active edges for testing."""
    for edge in edges:
        graph.edges[edge.key] = edge


def test_compile_chain():
    """Test chain 1->2->3, root=1."""
    graph = ConnectionGraph()

    # Edge 1-2
    T12 = np.eye(4)
    T12[0, 3] = 1.0  # translate x=1
    edge12 = _make_edge(1, "right", 2, "left", T12)

    # Edge 2-3
    T23 = np.eye(4)
    T23[0, 3] = 2.0  # translate x=2
    edge23 = _make_edge(2, "right", 3, "left", T23)

    _add_edges_to_graph(graph, [edge12, edge23])

    tree = compile_kinematic_tree(1, graph)

    assert tree.root == 1
    assert tree.order == [1, 2, 3]
    assert tree.parent_of == {1: None, 2: 1, 3: 2}
    assert 2 in tree.attachments
    assert 3 in tree.attachments

    # Check transforms
    assert np.allclose(tree.attachments[2].T_parent_child, T12)
    assert np.allclose(tree.attachments[3].T_parent_child, T23)


def test_compile_reverse_root():
    """Test same chain but root=3, should invert transforms."""
    graph = ConnectionGraph()

    T12 = np.eye(4)
    T12[0, 3] = 1.0
    edge12 = _make_edge(1, "right", 2, "left", T12)

    T23 = np.eye(4)
    T23[0, 3] = 2.0
    edge23 = _make_edge(2, "right", 3, "left", T23)

    _add_edges_to_graph(graph, [edge12, edge23])

    tree = compile_kinematic_tree(3, graph)

    assert tree.root == 3
    assert tree.order == [3, 2, 1]
    assert tree.parent_of[3] is None
    assert tree.parent_of[2] == 3
    assert tree.parent_of[1] == 2

    # From root=3 to 2, going against T23? Wait, when parent=3 child=2, but edge23 has a=2 b=3? Wait
    # Edge23 key normalized, but T23 is T(2<-3), so inv for T(3<-2)
    # So T_parent_child for parent=3 child=2 should be inv(T23)
    expected_3_to_2 = np.linalg.inv(T23)
    assert np.allclose(tree.attachments[2].T_parent_child, expected_3_to_2)


def test_compile_cycle():
    """Test cycle 1-2,2-3,1-3; root=1 should build spanning tree."""
    graph = ConnectionGraph()

    T12 = np.eye(4)
    T12[0, 3] = 1.0
    edge12 = _make_edge(1, "right", 2, "left", T12)

    T23 = np.eye(4)
    T23[0, 3] = 2.0
    edge23 = _make_edge(2, "right", 3, "left", T23)

    T13 = np.eye(4)
    T13[0, 3] = 3.0  # direct 1-3
    edge13 = _make_edge(1, "right", 3, "left", T13)

    _add_edges_to_graph(graph, [edge12, edge23, edge13])

    tree = compile_kinematic_tree(1, graph)

    # Should have 1,2,3 with 2 attachments (spanning tree)
    assert tree.root == 1
    assert set(tree.order) == {1, 2, 3}
    assert len(tree.attachments) == 2  # 1->2, 2->3 or 1->3 and 2->3
    assert 1 not in tree.attachments  # root
    assert tree.parent_of[1] is None
