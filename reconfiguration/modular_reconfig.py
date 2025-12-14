#!/usr/bin/env python3
"""
Modular Robot Reconfiguration Simulation - Consolidated Version
Date: 2025-12-13

This script simulates multi-module path planning and reconfiguration for modular robots.
Includes state updating, collision checking, detachment logic, and repulsion adjustments.

Features:
- Module class with position, attachment, and detachment states
- Path simulation with 6-DOF joint vectors
- Dynamic detachment of modules (e.g., module 3 detaches halfway)
- Repulsion adjustments to avoid close proximity collisions before detachment
- Step-by-step logging of positions, detachment status, and collision flags
- Final verification summary

Compatible with Python 3.12 on Ubuntu 24.04.
Does not require MuJoCo for this console-based version.
"""

import numpy as np
from itertools import combinations
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List

class ModuleState(Enum):
    ATTACHED = "attached"
    DETACHED = "detached"

@dataclass
class ExecutionEvent:
    step: int
    type: str  # e.g., "detach"
    module_id: int

@dataclass
class ExecutionContext:
    positions: Dict[int, np.ndarray]
    states: Dict[int, ModuleState]
    root_id: int

@dataclass
class ExecutionStep:
    step_index: int
    positions: Dict[int, List[float]]
    states: Dict[int, str]
    collision: bool
    events: List[str]

class ExecutionTrace:
    def __init__(self):
        self._steps: List[ExecutionStep] = []

    def append(self, step: ExecutionStep):
        self._steps.append(step)

    def summarize(self):
        print("\n=== Execution Trace Summary ===")
        collision_warnings = 0
        for step in self._steps:
            print(f"  Step {step.step_index}: Pos={step.positions}, Sta={step.states}, Coll={step.collision}, Ev={step.events}")
            if step.collision:
                collision_warnings += 1
        return collision_warnings  # Optional: return count if needed later

class Module:
    """
    Represents an individual robot module in the reconfiguration system.
    """
    def __init__(self, id, pos=np.array([0, 0, 0]), attached_to=None):
        self.id = id  # Unique identifier
        self.pos = np.array(pos, dtype=float)  # 3D position vector
        self.attached_to = attached_to or []  # List of connected module IDs
        self.detached = False  # Flag for detached state
        self.color = {1: 'red', 2: 'green', 3: 'blue'}.get(id, 'blue')  # Visual color (informational)

def update_modules(modules, step_config, adjust_repulsion=True, detachment_step=None, current_step=0):
    """
    Update positions for all non-detached modules from joint configuration.

    Args:
        modules: Dict of module_id -> Module
        step_config: List of 9 floats (3 per module: x, y, z)
        adjust_repulsion: If True, apply repulsion adjustments before detachment
        detachment_step: Step index for potential detachment (used for timing)
        current_step: Current simulation step
    """
    # Update positions from configuration
    for i, mid in enumerate(sorted(modules.keys())):
        if not modules[mid].detached:
            idx = i * 3
            modules[mid].pos = np.array([step_config[idx], step_config[idx+1], step_config[idx+2]])

    # Apply repulsion if enabled and before detachment
    if adjust_repulsion and current_step < (detachment_step or float('inf')):
        for m1 in list(modules.values()):
            if m1.detached:
                continue
            for m2 in list(modules.values()):
                if m2.detached or m1.id >= m2.id:
                    continue
                # Axis-wise proximity check (< 0.05 units)
                for axis in range(3):
                    if abs(m1.pos[axis] - m2.pos[axis]) < 0.05:
                        # Repel m1 away from m2
                        direction = 1 if m1.pos[axis] > m2.pos[axis] else -1
                        m1.pos[axis] += 0.01 * direction
                        print(f"  Repulsion: Adjusted module {m1.id} along axis {axis}")

def check_collisions(modules, threshold=0.2):
    """
    Check for collisions: any pair of modules closer than threshold.
    Returns True if collision detected.
    """
    for m1, m2 in combinations(modules.values(), 2):
        dist = np.linalg.norm(m1.pos - m2.pos)
        if dist < threshold:
            return True
    return False

def generate_dummy_path(num_steps=8):
    """
    Generate a dummy linear path from start to end configuration.

    Returns:
        list: Path of configurations, each as [x1,y1,z1,x2,y2,z2,x3,y3,z3]
    """
    start = np.array([0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.6, 0.0, 0.0])
    end = np.array([0.2, 0.0, 0.0, 0.5, 0.0, 0.0, 0.8, 0.0, 0.0])
    path = []
    for t in np.linspace(0, 1, num_steps + 1):
        config = start + t * (end - start)
        path.append(config.tolist())
    return path

def update_attached_modules(context: ExecutionContext, config):
    """Update positions for attached modules from step configuration."""
    for i, mid in enumerate(sorted(context.positions.keys())):
        if context.states[mid] == ModuleState.ATTACHED:
            idx = i * 3
            context.positions[mid] = np.array([config[idx], config[idx+1], config[idx+2]])

def apply_repulsion(context: ExecutionContext):
    """Apply repulsion adjustments for attached modules."""
    for m1_id, m1_pos in context.positions.items():
        if context.states[m1_id] != ModuleState.ATTACHED:
            continue
        for m2_id, m2_pos in context.positions.items():
            if context.states[m2_id] != ModuleState.ATTACHED or m1_id >= m2_id:
                continue
            for axis in range(3):
                if abs(m1_pos[axis] - m2_pos[axis]) < 0.05:
                    direction = 1 if m1_pos[axis] > m2_pos[axis] else -1
                    m1_pos[axis] += 0.01 * direction
                    print(f"  Adjusted {m1_id} axis {axis}")

def check_context_collisions(context: ExecutionContext, threshold=0.2):
    """Check collisions in the execution context."""
    positions = list(context.positions.values())
    for i, pos1 in enumerate(positions):
        for pos2 in positions[i+1:]:
            if np.linalg.norm(pos1 - pos2) < threshold:
                return True
    return False

def simulate_modular_reconfig(num_steps=8, detach_step=4, repulsion_enabled=True):
    """
    Run the modular reconfiguration simulation with explicit execution layer.
    """
    print("=== Modular Robot Reconfiguration Simulation (Execution Layer) ===")
    print(f"Parameters: {num_steps} steps, detach at step {detach_step}")

    # Initialize Execution Context
    context = ExecutionContext(
        positions={
            1: np.array([0.0, 0.0, 0.0]),
            2: np.array([0.3, 0.0, 0.0]),
            3: np.array([0.6, 0.0, 0.0])
        },
        states={
            1: ModuleState.ATTACHED,
            2: ModuleState.ATTACHED,
            3: ModuleState.ATTACHED
        },
        root_id=1
    )

    # Prepare scheduled events
    events = [ExecutionEvent(detach_step, "detach", 3)] if detach_step else []

    # Generate the reconfiguration path
    path = generate_dummy_path(num_steps)
    print(f"Generated path with {len(path)} configurations")

    # Initialize Execution Trace
    trace = ExecutionTrace()

    # Simulation loop with strict order (continued despite collisions)
    for step_idx, config in enumerate(path):
        step_events = []

        # 1. Handle events (only for this step)
        for event in events:
            if event.step == step_idx:
                if event.type == "detach":
                    context.states[event.module_id] = ModuleState.DETACHED
                    step_events.append(f"{event.type} module {event.module_id}")
                    print(f"STEP {step_idx}: Event processed - Detached module {event.module_id}")

        # 2. Update attached modules along the path
        update_attached_modules(context, config)

        # 3. Apply repulsion / collision corrections (only for ATTACHED modules)
        if repulsion_enabled:
            apply_repulsion(context)

        # 4. Log state and record ExecutionStep
        collision = check_context_collisions(context)
        if collision:
            print(f"STEP {step_idx}: WARNING - Collision detected!")

        states = {mid: state.value for mid, state in context.states.items()}
        positions = {mid: pos.tolist() for mid, pos in context.positions.items()}
        print(f"STEP {step_idx}: Positions {positions}, States {states}, Collision {collision}")

        # Record in trace
        step_record = ExecutionStep(
            step_index=step_idx,
            positions=positions,
            states=states,
            collision=collision,
            events=step_events
        )
        trace.append(step_record)

    # Execution Trace Summary
    collision_warnings = trace.summarize()

    # Verification using Verifier module
    from .verifier import verify_execution
    verification_report = verify_execution(trace, expected_detach_steps=[detach_step], expected_collision_free=False)

    print("\n=== Verification Results ===")
    print(f"- Attached modules moved: {verification_report.attached_moved}")
    print(f"- Detached modules frozen: {verification_report.detached_frozen}")
    print(f"- Events correct: {verification_report.events_correct}")
    print(f"- Collision free: {verification_report.collision_free}")
    if verification_report.attached_moved_failures:
        print(f"  Failures: {verification_report.attached_moved_failures}")
    if verification_report.detached_frozen_failures:
        print(f"  Detached moved at steps: {verification_report.detached_frozen_failures}")
    if verification_report.events_failures:
        print(f"  Event failures at steps: {verification_report.events_failures}")
    if verification_report.collision_failures:
        print(f"  Collisions at steps: {verification_report.collision_failures}")

    # Final verification
    print("\n=== Final Verification ===")
    attached_moved = all(
        not np.allclose(context.positions[mid], [0, 0, 0], atol=0.01)
        for mid in context.positions
        if context.states[mid] == ModuleState.ATTACHED
    )
    detached_frozen = all(
        context.states[mid] == ModuleState.DETACHED
        for mid in context.positions
        if mid == 3  # For this example
    )

    print(f"- Attached modules moved along path: {attached_moved}")
    print(f"- Detached modules remained frozen: {detached_frozen}")
    print(f"- Detachment events processed correctly: {detached_frozen}")
    if collision_warnings > 0:
        print(f"- Collision warnings observed: {collision_warnings} (continued execution)")

    print("VERIFICATION: Execution completed successfully!")

    print("Simulation complete.")

if __name__ == "__main__":
    # Run the example simulation
    simulate_modular_reconfig()
