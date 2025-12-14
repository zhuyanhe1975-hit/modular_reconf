#!/usr/bin/env python3
"""
Phase-2.6: Local solver for near-feasible attaches.
Tweaks q_ma, q_mb of involved modules to satisfy pos/yaw constraints.
"""

from dataclasses import dataclass
import numpy as np
from typing import List, Dict
from .connection_graph import SiteRef
from .connection_feasibility import FeasibilityResult


@dataclass
class LocalSolveParams:
    pos_tol: float = 1e-3  # Target pos err
    z_dot_tol_above: float = -0.999  # z_dot <= this
    yaw_tol_deg: float = 2.0
    max_iters: int = 10
    step: float = 1e-4  # Finite diff step
    damping: float = 0.1  # For LS


@dataclass
class LocalSolveResult:
    success: bool
    reason: str
    q_new_a: np.ndarray  # [q_ma, q_mb]
    q_new_b: np.ndarray
    metrics_before: dict
    metrics_after: dict
    iters: int


def compute_error_vector(Tw_a, Tw_b, snap_yaw_deg, params):
    """
    Compute 5D error vector for local solver.
    (1) Face-normal alignment (z_dot +1 for opposed)
    (2) Planar yaw in canonical face-to-face frame (rel_yaw to 0)
    (3) Position coincidence (diff minimized)
    """
    from .site_alignment import compute_constraint_metrics, compute_rel_yaw_deg
    from ubot.fk_sites import rotz, roty

    # Compute metrics
    metrics = compute_constraint_metrics(Tw_a, Tw_b, snap_yaw_deg)

    # Position error: p_a - p_b (3D)
    pos_diff = Tw_a[:3, 3] - Tw_b[:3, 3]
    err_pos = pos_diff

    # Normal alignment: |z_dot +1| minimized (for opposed z)
    err_normal = metrics['z_dot'] + 1.0

    # Planar yaw: relative yaw after face-to-face alignment
    # Apply flip and compensation to child for canonical comparison
    R_a = Tw_a[:3, :3]
    R_b = Tw_b[:3, :3]
    R_b_aligned = R_b @ roty(180.0) @ rotz(-snap_yaw_deg)  # Canonical child frame
    err_yaw = compute_rel_yaw_deg(R_a, R_b_aligned)

    return np.concatenate([err_pos, [err_normal, err_yaw]])


def solve_local_attach(executor, a: SiteRef, b: SiteRef, params: LocalSolveParams) -> LocalSolveResult:
    """
    Solve for q_a and q_b to satisfy constraints.
    Modifies executor.q_by_module in place if succeeds.
    """
    from .connection_graph import SiteRef
    from .site_naming import site_full_name
    from .site_alignment import compute_constraint_metrics

    site_a_name = site_full_name(a)
    site_b_name = site_full_name(b)

    # Current q
    q_a_old = executor.q_by_module[a.module_id].copy()
    q_b_old = executor.q_by_module[b.module_id].copy()

    # Best yaw from precheck, assume 0 for simplicity
    snap_yaw_deg = 0  # TODO: could be from feasibility

    # Compute before metrics
    Tw_a_old = executor.T_world[a.module_id] @ executor.ubot_kin.T_ax_site(q_a_old, site_a_name)
    Tw_b_old = executor.T_world[b.module_id] @ executor.ubot_kin.T_ax_site(q_b_old, site_b_name)
    metrics_before = compute_constraint_metrics(Tw_a_old, Tw_b_old, snap_yaw_deg)

    # If already satisfied, return success
    if metrics_before['pos_err'] <= params.pos_tol and metrics_before['z_dot'] <= params.z_dot_tol_above and abs(metrics_before['rel_yaw_deg']) <= params.yaw_tol_deg:
        return LocalSolveResult(True, "", q_a_old, q_b_old, metrics_before, metrics_before, 0)

    # Vars: dqa0, dqa1, dqb0, dqb1
    q_init = np.concatenate([q_a_old, q_b_old])

    def get_error(q):
        """ q = [qa0, qa1, qb0, qb1] """
        qa = q[:2]
        qb = q[2:4]
        Twa = executor.T_world[a.module_id] @ executor.ubot_kin.T_ax_site(qa, site_a_name)
        Twb = executor.T_world[b.module_id] @ executor.ubot_kin.T_ax_site(qb, site_b_name)
        return compute_error_vector(Twa, Twb, snap_yaw_deg, params)

    # Levenberg-Marquardt style
    alpha = params.damping
    q = q_init.copy()
    for it in range(params.max_iters):
        err = get_error(q)
        err_norm = np.linalg.norm(err)
        if err_norm < 1e-3:  # Converged
            break
        # Jacobian via finite diff
        J = np.zeros((5, 4))
        for j in range(4):
            q_eps = q.copy()
            q_eps[j] += params.step
            err_eps = get_error(q_eps)
            J[:, j] = (err_eps - err) / params.step
        # Solve (J^T J + alpha I) d = - J^T err
        JTJ = J.T @ J
        JTJ_reg = JTJ + alpha * np.eye(4)
        d = np.linalg.solve(JTJ_reg, -J.T @ err)
        # Update
        q_new = q + d
        # Clamp to joint limits, assume +/- pi
        q_new = np.clip(q_new, -np.pi, np.pi)
        # Recompute
        q = q_new

    # Final check
    # Update temp q for metrics after
    executor.q_by_module[a.module_id] = q[:2]
    executor.q_by_module[b.module_id] = q[2:4]
    Tw_a_new = executor.T_world[a.module_id] @ executor.ubot_kin.T_ax_site(q[:2], site_a_name)
    Tw_b_new = executor.T_world[b.module_id] @ executor.ubot_kin.T_ax_site(q[2:4], site_b_name)
    metrics_after_temp = compute_constraint_metrics(Tw_a_new, Tw_b_new, snap_yaw_deg)

    success = (metrics_after_temp['pos_err'] <= params.pos_tol and
               abs(metrics_after_temp['rel_yaw_deg']) <= params.yaw_tol_deg)
    if success:
        # Update executor
        executor.q_by_module[a.module_id] = q[:2]
        executor.q_by_module[b.module_id] = q[2:4]
        # Recompute metrics after update
        Tw_a_new = executor.T_world[a.module_id] @ executor.ubot_kin.T_ax_site(q[:2], site_a_name)
        Tw_b_new = executor.T_world[b.module_id] @ executor.ubot_kin.T_ax_site(q[2:4], site_b_name)
        metrics_after = compute_constraint_metrics(Tw_a_new, Tw_b_new, snap_yaw_deg)
        return LocalSolveResult(True, "", q[:2], q[2:4], metrics_before, metrics_after, it)
    else:
        return LocalSolveResult(False, f"Did not converge, final err {err_norm:.3e}", q_a_old, q_b_old, metrics_before, {}, it)


from reconfiguration.connection_feasibility import FeasibilityParams
from reconfiguration.connection_api import attempt_attach

def attempt_attach_with_local(graph, executor, a: SiteRef, b: SiteRef, feas_params: FeasibilityParams, enable_local_solve: bool = False) -> FeasibilityResult:
    """
    attempt_attach with optional local solve.
    """
    # First try normal attach
    result = attempt_attach(graph, executor, a, b, feas_params)
    if result.feasible or not enable_local_solve:
        return result

    # If failed on pos or yaw, but z_dot ok, try local solve
    if result.reason in ('pos', 'yaw') and result.z_dot <= feas_params.z_dot_max:
        # Get site names
        from .site_naming import site_full_name
        site_a_name = site_full_name(a)
        site_b_name = site_full_name(b)

        # Run local solve
        solve_params = LocalSolveParams(
            pos_tol=feas_params.pos_tol,
            yaw_tol_deg=feas_params.yaw_tol_deg,
            z_dot_tol_above=feas_params.z_dot_max
        )
        solve_result = solve_local_attach(executor, a, b, solve_params)
        if solve_result.success:
            # Retry attach with new q
            result = attempt_attach(graph, executor, a, b, feas_params)
            return result  # Will have updated q
    return result
