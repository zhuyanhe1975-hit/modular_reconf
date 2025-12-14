#!/usr/bin/env python3
"""
Kinematic Executor V2 Phase-2.4: Propagation using dynamic site-frame constraint transforms.
"""

import numpy as np
from typing import Dict

from .kinematic_executor import WorldPoseResult, assert_T
from .kinematic_compiler import KinematicTree, KinematicAttachment
from .connection_graph import SiteRef
from .kinematic_executor import make_T
import ubot


def propagate_world_poses_with_sites(
    tree: KinematicTree,
    T_world_root: np.ndarray,
    q_by_module: Dict[int, np.ndarray],  # module_id -> q=np.array([q_ma_rad, q_mb_rad])
    ubot_kin: ubot.UBotKinematics,
) -> WorldPoseResult:
    """
    Propagate world poses with real constraint T_computed from site frames.
    """
    assert_T(T_world_root)

    T_world = {}
    T_world[tree.root] = T_world_root.astype(np.float64)
    assert_T(T_world[tree.root])

    # Constraint construction helpers
    R_flip = ubot.fk_sites.roty(180)  # Flip Z to -Z
    # make_T(R, p) combines

    for child in tree.order[1:]:  # Skip root
        parent = tree.parent_of[child]
        assert parent is not None, f"No parent for {child}"

        att: KinematicAttachment = tree.attachments[child]

        # Compute T_parent_Psite and T_child_Csite
        from .site_naming import site_full_name
        parent_site_full = site_full_name(att.parent_site)
        child_site_full = site_full_name(att.child_site)

        q_parent = q_by_module[parent]
        q_child = q_by_module[child]

        T_parent_Psite = ubot_kin.T_ax_site(q_parent, parent_site_full)
        T_child_Csite = ubot_kin.T_ax_site(q_child, child_site_full)

        # T_Psite_Csite: zero translation, R_yaw @ R_flip
        R_yaw = ubot.fk_sites.rotz(att.yaw_snap_deg)
        R_constraint = R_yaw @ R_flip
        T_Psite_Csite = make_T(R_constraint, np.zeros(3, dtype=np.float64))

        # T_parent_child = T_parent_Psite @ T_Psite_Csite @ inv(T_child_Csite)
        T_parent_child = T_parent_Psite @ T_Psite_Csite @ np.linalg.inv(T_child_Csite)
        assert_T(T_parent_child)

        # Propagate
        T_world[child] = T_world[parent] @ T_parent_child
        assert_T(T_world[child])

    reached = set(tree.order)

    return WorldPoseResult(
        T_world=T_world,
        reachable=reached
    )
