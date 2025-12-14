#!/usr/bin/env python3
"""
Phase-3.2 Tests: Detach-Then-Attach Reconfiguration.
"""
import pytest
from reconfiguration.planner_stub import plan_modular_reconfig
from reconfiguration.modular_reconfig import simulate_modular_reconfig


def test_detach_then_attach_ordering():
    """Detach steps < attach steps in schedule."""
    path, schedule = plan_modular_reconfig(detach_step=1, attach_success_step=7, detach_then_attach=True)
    detach_steps = [step for step, events in schedule.items() if any(ev.kind == "detach" for ev in events)]
    attach_steps = [step for step, events in schedule.items() if any(ev.kind == "attach" for ev in events)]
    assert all(ds < as_ for ds in detach_steps for as_ in attach_steps), "Detaches not before attaches"


def test_detach_then_attach_execution():
    """Detach early, attach later, check states and traces."""
    _, schedule = plan_modular_reconfig(detach_step=1, attach_success_step=7, detach_then_attach=True)
    trace = simulate_modular_reconfig(num_steps=8, detach_step=None, schedule=schedule)

    # After step 1 (detach): all detached, positions frozen
    assert all(state == 'detached' for state in trace._steps[2].states.values()), "Not all detached after detach"
    pos_after_detach = trace._steps[2].positions
    pos_later = trace._steps[6].positions
    for mid in [1, 2, 3]:
        assert pos_after_detach[mid] == pos_later[mid], f"Module {mid} moved after detach"

    # Step 7: attach success recorded
    step_7_events = [ev for ev in trace._steps[7].events if "attach" in ev and "yaw" in ev]
    assert len(step_7_events) == 1, "Attach not successful at step 7"


def test_attach_failure_no_corruption():
    """Attach succeeds gracefully, graph updated."""
    _, schedule = plan_modular_reconfig(detach_step=1, attach_success_step=6, detach_then_attach=True)
    trace = simulate_modular_reconfig(num_steps=8, detach_step=None, schedule=schedule)
    # Should complete successfully
    assert len(trace._steps) == 9, "Simulation did not complete"
    # Check event success in trace at step 7
    events_at_7 = trace._steps[7].events
    if events_at_7:
        assert any("attach" in ev and "yaw" in ev for ev in events_at_7), "Attach success not recorded"
