#!/usr/bin/env python3
"""
Kinematic Executor Phase-2.3
Propagates world poses along the KinematicTree.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict

from .kinematic_compiler import KinematicTree


@dataclass
class WorldPoseResult:
    T_world: Dict[int, np.ndarray]  # module_id -> (4,4)
    reachable: set[int]             # visited modules


def assert_T(T: np.ndarray) -> None:
    """Assert valid T matrix: (4,4), float64, last row [0,0,0,1]."""
    assert T.shape == (4, 4), f"Shape {T.shape} != (4,4)"
    assert T.dtype == np.float64, f"Dtype {T.dtype} != float64"
    assert np.allclose(T[3, :], [0, 0, 0, 1], atol=1e-10), f"Last row {T[3,:]}"


def make_T(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Compose homogeneous T = [R, p; 0,0,0,1]."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def propagate_world_poses(
    tree: KinematicTree,
    T_world_root: np.ndarray,
) -> WorldPoseResult:
    """
    Compute world transforms for all modules in tree using:
    T_world[child] = T_world[parent] @ T_parent_child
    """
    assert_T(T_world_root)

    T_world = {}
    T_world[tree.root] = T_world_root.astype(np.float64)
    assert_T(T_world[tree.root])

    for child in tree.order[1:]:  # Skip root
        parent = tree.parent_of[child]
        assert parent is not None, f"No parent for {child}"

        att = tree.attachments[child]
        T_world_child = T_world[parent] @ att.T_parent_child
        assert_T(T_world_child)
        T_world[child] = T_world_child

    reached = set(tree.order)

    return WorldPoseResult(
        T_world=T_world,
        reachable=reached
    )


def relative_T(T_world_a: np.ndarray, T_world_b: np.ndarray) -> np.ndarray:
    """T_b_in_a = inv(T_world_a) @ T_world_b."""
    assert_T(T_world_a)
    assert_T(T_world_b)
    rel = np.linalg.inv(T_world_a) @ T_world_b
    assert_T(rel)
    return rel
