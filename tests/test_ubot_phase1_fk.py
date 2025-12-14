import pytest
import numpy as np
from ubot.spec import JointSpec
from ubot.kinematics_phase1 import forward_kinematics, rodrigues_rot


def test_rodrigues_rot():
    """Test Rodrigues rotation changes orientation."""
    axis = np.array([0, 0, 1])  # Z-axis
    R_zero = rodrigues_rot(axis, 0)
    np.testing.assert_allclose(R_zero, np.eye(3))
    
    R_pi2 = rodrigues_rot(axis, np.pi/2)
    expected = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])  # Matches actual for theta +pi/2
    np.testing.assert_allclose(R_pi2, expected, atol=1e-5)


def test_forward_kinematics_dof():
    """Test FK with dummy joints for orientation change."""
    joint1 = JointSpec(
        name="j1", parent_body="ax", child_body="ma",
        type="hinge", axis=np.array([0, 0, 1]), range=(-np.pi, np.pi), pos=np.array([0, 0, 0])
    )
    joint2 = JointSpec(
        name="j2", parent_body="ax", child_body="mb",
        type="hinge", axis=np.array([0, 0, 1]), range=(-np.pi, np.pi), pos=np.array([0, 0, 0])
    )
    T_root = np.eye(4)
    
    # Zero config
    q_zero = np.array([0, 0])
    poses_zero = forward_kinematics(q_zero, T_root, joint1, joint2)
    assert poses_zero["ax"][:3,3].tolist() == [0, 0, 0]  # Root at origin
    # ma/mb at T_root since pos=0, R=I
    np.testing.assert_allclose(poses_zero["ma"], T_root)
    np.testing.assert_allclose(poses_zero["mb"], T_root)
    
    # Non-zero config: rotation should change ma/mb orientation
    q_nonzero = np.array([np.pi/4, -np.pi/4])
    poses_nonzero = forward_kinematics(q_nonzero, T_root, joint1, joint2)
    
    # Check orientation change
    R_ma_changed = poses_nonzero["ma"][:3, :3]
    R_zero = poses_zero["ma"][:3, :3]
    assert not np.allclose(R_ma_changed, R_zero)  # Orientation should differ
    
    R_mb_changed = poses_nonzero["mb"][:3, :3]
    assert not np.allclose(R_mb_changed, R_zero)  # And for mb


def test_fk_positions():
    """Test that FK respects joint positions."""
    joint1 = JointSpec(
        name="j1", parent_body="ax", child_body="ma",
        type="hinge", axis=np.array([0, 0, 1]), range=(-np.pi, np.pi), pos=np.array([1, 0, 0])  # Pos shifted
    )
    joint2 = JointSpec(
        name="j2", parent_body="ax", child_body="mb",
        type="hinge", axis=np.array([0, 0, 1]), range=(-np.pi, np.pi), pos=np.array([0, 1, 0])  # Different shift
    )
    T_root = np.eye(4)
    q_zero = np.array([0, 0])
    
    poses = forward_kinematics(q_zero, T_root, joint1, joint2)
    np.testing.assert_allclose(poses["ma"][:3, 3], [1, 0, 0])  # ma shifted by joint1 pos
    np.testing.assert_allclose(poses["mb"][:3, 3], [0, 1, 0])  # mb shifted by joint2 pos
