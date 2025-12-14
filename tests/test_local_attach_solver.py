import pytest
import numpy as np
from reconfiguration.local_attach_solver import LocalSolveParams, solve_local_attach
from reconfiguration.connection_api import ExecutorWrapper
from reconfiguration.connection_graph import SiteRef
import ubot
from reconfiguration.site_naming import site_full_name
from reconfiguration.site_alignment import compute_constraint_metrics


@pytest.fixture
def executor_near_miss():
    """Create executor with near miss using internal joints."""
    ubot_kin = ubot.UBotKinematics("assets/ubot_ax_centered.xml")
    T_world = {
        1: np.eye(4, dtype=np.float64),
        2: np.eye(4, dtype=np.float64)
    }
    # Assert T_world are identity (no world translation)
    assert np.allclose(T_world[1][:3,3], [0,0,0]), "T_world[1] has translation"
    assert np.allclose(T_world[2][:3,3], [0,0,0]), "T_world[2] has translation"

    # Internal misalignment for yaw error
    q_by_module = {
        1: np.array([0.0, 0.0]),
        2: np.array([0.0, np.deg2rad(190)])  # 10 deg off from 180 for mb hinge
    }

    executor = ExecutorWrapper(T_world, q_by_module, ubot_kin)

    # Print initial inputs
    a_ref = SiteRef(1, "ma", "right")
    b_ref = SiteRef(2, "mb", "left")
    site_a_name = site_full_name(a_ref)
    site_b_name = site_full_name(b_ref)
    Tw_a = executor.T_world[a_ref.module_id] @ executor.ubot_kin.T_ax_site(q_by_module[a_ref.module_id], site_a_name)
    Tw_b = executor.T_world[b_ref.module_id] @ executor.ubot_kin.T_ax_site(q_by_module[b_ref.module_id], site_b_name)
    metrics_before = compute_constraint_metrics(Tw_a, Tw_b, 0)
    print(f"Fixture debug: T_world[1][:3,3]={T_world[1][:3,3]}, T_world[2][:3,3]={T_world[2][:3,3]}")
    print(f"q_by_module: {q_by_module}")
    print(f"metrics_before: pos_err={metrics_before['pos_err']:.4f}, rel_yaw_deg={metrics_before['rel_yaw_deg']:.4f}, z_dot={metrics_before['z_dot']:.6f}")

    return executor


def test_local_solve_success(executor_near_miss):
    """Solve near miss yaw error."""
    a_ref = SiteRef(1, "ma", "right")
    b_ref = SiteRef(2, "mb", "left")
    params = LocalSolveParams(yaw_tol_deg=6.0, z_dot_tol_above=-0.99998, max_iters=50, damping=0.001)  # Set target to 6deg, loose z_dot, high iterations
    result = solve_local_attach(executor_near_miss, a_ref, b_ref, params)
    assert result.success
    assert abs(result.metrics_before['rel_yaw_deg']) > 1.0
    assert abs(result.metrics_after['rel_yaw_deg']) < 6.0


def test_local_solve_failure():
    """Solve failure case (world-frame translation mismatch)."""
    ubot_kin = ubot.UBotKinematics("assets/ubot_ax_centered.xml")
    T_world = {
        1: np.eye(4, dtype=np.float64),
        2: np.eye(4, dtype=np.float64) + np.array([[0,0,0,0.5],[0,0,0,0],[0,0,0,0],[0,0,0,0]])  # x=0.5 (too far, world trans)
    }
    q_by_module = {1: np.array([0., 0.]), 2: np.array([0., 0.])}
    executor = ExecutorWrapper(T_world, q_by_module, ubot_kin)
    a_ref = SiteRef(1, "ma", "right")
    b_ref = SiteRef(2, "mb", "left")
    params = LocalSolveParams(max_iters=50)  # High iters, still fails on pos
    result = solve_local_attach(executor, a_ref, b_ref, params)
    assert not result.success
    assert "Did not converge" in result.reason


def test_yaw_convergence_internal():
    """Regression: internal joint yaw errors converge."""
    # Create fixture with internal yaw misalignment (like executor_near_miss)
    ubot_kin = ubot.UBotKinematics("assets/ubot_ax_centered.xml")
    T_world = {
        1: np.eye(4, dtype=np.float64),
        2: np.eye(4, dtype=np.float64)
    }
    q_by_module = {1: np.array([0.0, 0.0]), 2: np.array([0.0, np.deg2rad(190)])}
    executor = ExecutorWrapper(T_world, q_by_module, ubot_kin)

    a_ref = SiteRef(1, "ma", "right")
    b_ref = SiteRef(2, "mb", "left")
    params = LocalSolveParams(yaw_tol_deg=6.0, max_iters=50, damping=0.001)
    result = solve_local_attach(executor, a_ref, b_ref, params)
    assert result.success
    assert abs(result.metrics_before['rel_yaw_deg']) > 1.0  # Had error
    assert abs(result.metrics_after['rel_yaw_deg']) < 6.0  # Con partverged
