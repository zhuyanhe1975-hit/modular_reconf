import pytest
from reconfiguration.modular_reconfig import ExecutionStep, ExecutionTrace
from reconfiguration.verifier import verify_execution


def test_collision_consistency():
    """Test that collision_free is computed correctly and consistently."""
    trace = ExecutionTrace()
    # Step 0, no collision
    step0 = ExecutionStep(
        step_index=0,
        positions={1: [0, 0, 0]},
        states={1: 'attached'},
        collision=False,
        events=[]
    )
    trace.append(step0)

    # Step 1, collision
    step1 = ExecutionStep(
        step_index=1,
        positions={1: [0, 0, 0]},
        states={1: 'attached'},
        collision=True,  # Collision at step 1
        events=[]
    )
    trace.append(step1)

    # Step 2, no collision
    step2 = ExecutionStep(
        step_index=2,
        positions={1: [0, 0, 0]},
        states={1: 'attached'},
        collision=False,
        events=[]
    )
    trace.append(step2)

    # Verify
    report = verify_execution(trace, expected_detach_steps=[], expected_collision_free=True)

    assert report.collision_free == False
    assert report.collision_failures == [1]
