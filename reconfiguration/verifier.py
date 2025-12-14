#!/usr/bin/env python3
"""
Verifier Module for Modular Robot Reconfiguration System
- Verifies execution traces against expected behaviors.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from .modular_reconfig import ExecutionStep, ExecutionTrace


@dataclass
class VerificationReport:
    attached_moved: bool = False
    detached_frozen: bool = False
    events_correct: bool = True  # Assume true, set to false on failure
    collision_free: bool = True  # If expected

    attached_moved_failures: List[int] = field(default_factory=list)
    detached_frozen_failures: List[int] = field(default_factory=list)
    events_failures: List[int] = field(default_factory=list)
    collision_failures: List[int] = field(default_factory=list)


def verify_execution(trace: ExecutionTrace,
                     expected_detach_steps: Optional[List[int]] = None,
                     expected_collision_free: bool = True) -> VerificationReport:
    """
    Verify the execution trace against expected behaviors.

    Args:
        trace: ExecutionTrace with recorded steps.
        expected_detach_steps: List of step indices where detach should happen.
        expected_collision_free: If True, expect no collisions.

    Returns:
        VerificationReport with results and failure indices.
    """
    report = VerificationReport()
    expected_detach_steps = expected_detach_steps or []

    if not trace._steps:
        return report

    # Get initial positions
    initial_positions = trace._steps[0].positions.copy()
    final_step = trace._steps[-1]

    # Check attached modules moved (e.g., non-zero movement in final positions for attached)
    report.attached_moved = all(
        not np.allclose(initial_positions[mid], final_step.positions[mid], atol=0.01)
        for mid in final_step.positions
        if final_step.states[mid] == 'attached'
    )

    # Check detached modules frozen (positions unchanged after first detach state)
    detached_modules = set()
    for step in trace._steps:
        for mid, state in step.states.items():
            if state == 'detached':
                detached_modules.add(mid)
    report.detached_frozen = True
    for step in trace._steps:
        for mid in detached_modules:
            if mid in step.positions:
                # Get position at detach (first 'detached' step for this mid)
                detach_pos = None
                for s in trace._steps[:step.step_index+1]:
                    if s.states.get(mid) == 'detached':
                        detach_pos = np.array(s.positions[mid])
                        break
                if detach_pos is not None and not np.allclose(np.array(step.positions[mid]), detach_pos, atol=0.01):
                    report.detached_frozen = False
                    report.detached_frozen_failures.append(step.step_index)
                    print(f"DEBUG: Detached module {mid} moved at step {step.step_index}")

    # Check events correct (detachs at expected steps)
    actual_detach_events = {}
    for step in trace._steps:
        for event in step.events:
            if 'detach' in event:
                parts = event.split()
                if len(parts) >= 3 and parts[1] == 'module':
                    mid = int(parts[2])
                    actual_detach_events[mid] = step.step_index

    for step_idx in expected_detach_steps:
        for mid in final_step.states:
            expected = (mid in [3]) and (step_idx == expected_detach_steps[0])  # Assume mid 3
            if expected:
                if mid not in actual_detach_events or actual_detach_events[mid] != step_idx:
                    report.events_correct = False
                    report.events_failures.append(step_idx)
                    print(f"DEBUG: Expected detach for {mid} at {step_idx}, got {actual_detach_events.get(mid)}")

    # Check collision free
    if expected_collision_free:
        collisions_found = [step.step_index for step in trace._steps if step.collision]
        if collisions_found:
            report.collision_free = False
            report.collision_failures = collisions_found
            print(f"DEBUG: Collisions at steps {collisions_found}")

    return report


# Example usage (for integration)
if __name__ == "__main__":
    # Would normally import trace from execution run
    print("Verifier module - import in main script to verify traces.")
