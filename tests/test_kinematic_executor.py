import pytest
import numpy as np
from reconfiguration.connection_graph import SiteRef, EdgeKey, ConnectionEdge, ConnectionGraph, ConnectionEvent
from reconfiguration.kinematic_compiler import compile_kinematic_tree
from reconfiguration.kinematic_executor import propagate_world_poses, assert_T, make_T, relative_T


def _make_edge(module_a: int, site_a: str, module_b: int, site_b: str, T_a_b: np.ndarray) -> ConnectionEdge:
    s_a = SiteRef(module_a, "ma", site_a)
    s_b = SiteRef(module_b, "mb", site_b)
    return ConnectionEdge(key=EdgeKey.normalized(s_a, s_b), yaw_snap_deg=0, T_a_b=T_a_b)


def _add_edge(graph, edge):
    graph.edges[edge.key] = edge


def test_propagate_chain():
    """Test propagation in chain 1-2-3."""
    graph = ConnectionGraph()

    # T1 (1<-2): translate x=1
    T12 = np.eye(4, dtype=np.float64)
    T12[0, 3] = 1.0
    edge12 = _make_edge(1, "right", 2, "left", T12)
    _add_edge(graph, edge12)

    # T2 (2<-3): translate x=2
    T23 = np.eye(4, dtype=np.float64)
    T23[0, 3] = 2.0
    edge23 = _make_edge(2, "right", 3, "left", T23)
    _add_edge(graph, edge23)

    tree = compile_kinematic_tree(1, graph)

    # Propagate root at origin
    T_root = np.eye(4, dtype=np.float64)
    result = propagate_world_poses(tree, T_root)

    assert 1 in result.T_world
    assert 2 in result.T_world
    assert 3 in result.T_world
    assert result.reachable == {1, 2, 3}

    # Check positions
    assert np.allclose(result.T_world[1][:3, 3], [0, 0, 0])  # Root
    assert np.allclose(result.T_world[2][:3, 3], [1, 0, 0])  # Parent1 + T1
    assert np.allclose(result.T_world[3][:3, 3], [3, 0, 0])  # Parent2 + T2 = 1 + 2


def test_propagate_non_identity_root():
    """Test with root pose shifted."""
    graph = ConnectionGraph()

    # Only 1-2, translate x=5
    T12 = np.eye(4, dtype=np.float64)
    T12[0, 3] = 5.0
    edge12 = _make_edge(1, "right", 2, "left", T12)
    _add_edge(graph, edge12)

    tree = compile_kinematic_tree(1, graph)

    # Root at x=10
    T_root = np.eye(4, dtype=np.float64)
    T_root[0, 3] = 10.0
    result = propagate_world_poses(tree, T_root)

    assert np.allclose(result.T_world[1][:3, 3], [10, 0, 0])
    assert np.allclose(result.T_world[2][:3, 3], [15, 0, 0])  # 10 + 5


def test_detach_changes_tree():
    """Test detach removes unreachable modules."""
    graph = ConnectionGraph()

    T12 = np.eye(4, dtype=np.float64)
    T12[0, 3] = 1.0
    edge12 = _make_edge(1, "right", 2, "left", T12)
    _add_edge(graph, edge12)

    T23 = np.eye(4, dtype=np.float64)
    T23[0, 3] = 2.0
    edge23 = _make_edge(2, "right", 3, "left", T23)
    _add_edge(graph, edge23)

    # Initial tree has 3 modules
    tree1 = compile_kinematic_tree(1, graph)
    result1 = propagate_world_poses(tree1, np.eye(4, dtype=np.float64))
    assert 3 in result1.T_world

    # Detach edge 2-3
    ev_detach = ConnectionEvent(
        kind="detach",
        a=SiteRef(2, "ma", "right"),
        b=SiteRef(3, "mb", "left")
    )
    graph.apply(ev_detach)

    # Recompile tree
    tree2 = compile_kinematic_tree(1, graph)
    result2 = propagate_world_poses(tree2, np.eye(4, dtype=np.float64))

    # Module 3 should be unreachable
    assert 3 not in result2.T_world
    assert result2.reachable == {1, 2}
    assert len(result2.T_world) == 2


def test_assert_T():
    """Test assert_T validates matrix."""
    T_good = np.eye(4, dtype=np.float64)
    assert_T(T_good)

    # Wrong shape
    with pytest.raises(AssertionError):
        assert_T(np.eye(3))

    # Wrong dtype
    with pytest.raises(AssertionError):
        assert_T(np.eye(4, dtype=np.float32))

    # Wrong last row
    bad = np.eye(4, dtype=np.float64)
    bad[3, 1] = 1.0
    with pytest.raises(AssertionError):
        assert_T(bad)


def test_make_T():
    """Test compose T from R and p."""
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    p = np.array([1, 2, 3], dtype=np.float64)
    T = make_T(R, p)
    assert np.allclose(T[:3, :3], R)
    assert np.allclose(T[:3, 3], p)
    assert np.allclose(T[3, :], [0, 0, 0, 1])


def test_relative_T():
    """Test compute relative between world poses."""
    T_a = np.eye(4, dtype=np.float64)
    T_b = np.eye(4, dtype=np.float64)
    T_b[0, 3] = 2.0  # x=2

    rel = relative_T(T_a, T_b)
    assert np.allclose(rel[:3, 3], [2, 0, 0])  # T_b in T_a frame

    # Inv direction
    rel_inv = relative_T(T_b, T_a)
    assert np.allclose(rel_inv[:3, 3], [-2, 0, 0])
