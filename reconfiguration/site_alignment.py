#!/usr/bin/env python3
"""
Site Alignment Utilities for Phase-2.4
Computes yaw and constraint metrics.
"""

import numpy as np
from ubot.fk_sites import roty, rotz


def compute_rel_yaw_deg(R_parent: np.ndarray, R_child: np.ndarray) -> float:
    """
    Compute relative yaw of child site frame relative to parent site frame, about parent z-axis.

    Positive yaw: child y-axis rotated clockwise about parent z when viewed from above.

    R_parent, R_child: 3x3 matrices in the SAME world frame.
    """
    z_p = R_parent[:, 2]
    x_p = R_parent[:, 0]
    y_p = R_parent[:, 1]

    # Child y-axis projected to plane orthogonal to z_p
    y_c = R_child[:, 1]
    y_c_proj = y_c - np.dot(y_c, z_p) * z_p
    norm_y_c_proj = np.linalg.norm(y_c_proj)
    if norm_y_c_proj < 1e-8:
        return 0.0  # Degenerate, assume 0

    y_c_proj /= norm_y_c_proj

    # Signed angle in radians: atan2(dot(z_p, cross(y_p, y_c_proj)), dot(y_p, y_c_proj))
    # But since y_c_proj is in plane, cross(y_p, y_c_proj) is along z_p, so atan2(dot(z_p, normal), dot)
    dot_yy = np.dot(y_p, y_c_proj)
    cross_yy = np.cross(y_p, y_c_proj)
    sin_val = np.dot(z_p, cross_yy)

    yaw_rad = np.arctan2(sin_val, dot_yy)
    yaw_deg = np.rad2deg(yaw_rad)

    return yaw_deg


def compute_constraint_metrics(Tw_parent_site: np.ndarray, Tw_child_site: np.ndarray, yaw_snap_deg: int) -> dict:
    """
    Compute site constraint metrics: pos_err, z_dot, rel_yaw_deg.

    rel_yaw_deg: residual yaw after applying flip + yaw_snap compensation to child site frame
    """
    R_p = Tw_parent_site[:3, :3]
    R_c = Tw_child_site[:3, :3]
    p_p = Tw_parent_site[:3, 3]
    p_c = Tw_child_site[:3, 3]

    pos_err = np.linalg.norm(p_p - p_c)

    z_p = R_p[:, 2]
    z_c = R_c[:, 2]
    z_dot = np.dot(z_p, z_c)

    # Apply flip to child site frame: z becomes -z
    R_flip = roty(180.0)  # RotY 180 flips z to -z
    R_c_flip = R_c @ R_flip

    # Apply yaw snap compensation: rotate child about its (now flipped) z by -yaw_snap_deg
    # To align, we rotate the child frame by -yaw_snap_deg about parent z
    # But since frames are in world, better to apply to R_c_flip about the axis, but known axis is parent's z in world? Wait, it's tricky.

    # Since the compensation in executor is Rz(yaw_snap) @ R_flip, which applies to the parent->child tf.
    # For measurement, the 'correct' child frame after compensation should have R_correct_child = R_p @ Rz(yaw_snap) @ R_flip @ R_c_inv or something.

    # Simpler: since the propagation enforces z_p = -z_c_after, to measure residual yaw, compute yaw between R_p and R_c_corrected.

    # The 'corrected' child rotation is R_p @ Rz(yaw_snap) @ R_flip @ R_p.T @ R_p, but that's not right.

    # Let's think: the constraint is T_Psite_Csite includes Rz(yaw_snap) @ R_flip, so the 'desired' R_child is R_p @ Rz(yaw_snap) @ R_flip.

    # To find residual, rel_yaw = compute_rel_yaw_deg(R_p, R_c_desired - but it's not.

    # The executor computes T_parent_child such that T_world_child = T_world_parent @ T_parent_child, and it includes the constraint T_Psite_Csite = Rz(yaw_snap) @ R_flip, so in the propagation, the actual T_world_site_c = T_world_site_p @ T_Psite_Csite @ inv(T_child_site_local), but wait.

    # In the executor, the site frames after propagation should satisfy:
    # T_world_site_p @ T_Psite_Csite == T_world_site_c

    # Since T_Psite_Csite = translation 0 * rotation Rz(yaw_snap) @ R_flip

    # So T_world_site_c ~ T_world_site_p @ Rz(yaw_snap) @ R_flip

    # So the z_c = R_world_site_c [:,2] = R_world_site_p @ Rz(yaw_snap) @ roty(180) @ [0,0,1]^T

    # To get 'aligned' R_c, we need R_position_c_married @ rotz(-yaw_snap_deg) @ roty(-180) , but it's complicated.

    # For simplicity, since z are adjusted, the rel_yaw is to compute the yaw after assuming z are opposed.

    # The task suggests by applying flip and snap to child and computing yaw against parent.

    # R_c_flip = R_c @ R_flip

    # Then to apply yaw_snap, since the snap is about the axis, but to compensate, apply Rz(-yaw_snap_deg) to R_c_flip关于 the axis, but since axis is not known, assume it's about the current z.

    # Let's assume the compensation is applied to the child frame around its local z.

    # So R_c_aligned = R_c @ R_flip @ rotz(-yaw_snap_deg)

    # Then rel_yaw = compute_rel_yaw_deg(R_p, R_c_aligned")

    # This should give 0 if the compensation matches the required.

    # Yes, try that.

    R_c_flip = R_c @ roty(180.0)
    R_c_aligned = R_c_flip @ rotz(yaw_snap_deg)   # Note: since executor has Rz(yaw_snap) @ R_flip, to match, here R_flip @ Rz(yaw_snap), but R_flip is roty(180), so order.

    # If executor uses R = rub rotz(yaw_snap) @ roty(180)

    # Then for child, the effective rotation is R_c_needed = R_p @ R, so the 'applied' to child is R_c_aligned = inv(R) @ R_p.T @ R_p @ R_c_original

    # To make it simple, let's assume the compensation is to apply the inverse to measure residual.

    # The task says: "R_c_aligned = R_c_flip @ Rz(-yaw_snap_deg)"

    # Where R_flip is 180deg about local y for z flip.

    R_c_aligned = R_c_flip @ rotz(-yaw_snap_deg)

    rel_yaw_deg = compute_rel_yaw_deg(R_p, R_c_aligned)

    return {
        "pos_err": pos_err,
        "z_dot": z_dot,
        "rel_yaw_deg": rel_yaw_deg
    }
