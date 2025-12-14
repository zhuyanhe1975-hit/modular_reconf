#!/usr/bin/env python3
"""
Phase-3.2 Demo: Detach-Then-Attach Reconfiguration.

Shows planner creating schedule with detaches early, attaches later.
Demonstrates execution layer applying events, recording traces.
"""

import sys
sys.path.insert(0, '/home/yhzhu/myWorks/UBot/modular_reconf')

from reconfiguration.planner_stub import plan_modular_reconfig
from reconfiguration.modular_reconfig import simulate_modular_reconfig


def main():
    print("=== Phase-3.2: Detach-Then-Attach Reconfiguration Demo ===\n")

    # Generate plan with detach-then-attach ordering
    print("Planning reconfiguration:")
    print("- Detach at step 1: Break initial 1-2-3 chain")
    print("- Attach at step 7: Reconnect 1 to 3")
    path, schedule = plan_modular_reconfig(
        num_steps=8,
        detach_step=1,
        attach_success_step=7,
        detach_then_attach=True
    )

    print("\nScheduled Events:")
    for step, events in sorted(schedule.items()):
        print(f"  Step {step}:")
        for event in events:
            if event.kind == "detach":
                print(f"    DETACH {event.a} from {event.b}")
            elif event.kind == "attach":
                print(f"    ATTACH {event.a} to {event.b} (yaw={event.yaw_snap_deg}°)")
    print("\n")

    # Execute simulation with event execution
    print("Executing simulation with event-driven reconfiguration:")
    trace = simulate_modular_reconfig(
        num_steps=8,
        detach_step=None,  # Legacy off
        schedule=schedule,
        repulsion_enabled=True
    )

    # Show key points
    print("\n=== Key Execution Points ===")
    print("Step 1 (Detach Events):")
    print(f"  States: {trace._steps[1].states}")
    print(f"  Events: {trace._steps[1].events}")

    print("Step 2-6 (Detached phase):")
    print("  Positions frozen for detached modules.")
    pos_2 = trace._steps[2].positions
    print(f"  Pos at step 2: {pos_2}")
    pos_6 = trace._steps[6].positions
    print(f"  Pos at step 6: {pos_6}")
    frozen = all(pos_2[mid] == pos_6[mid] for mid in [1, 2, 3])
    print(f"  Positions frozen: {frozen}")

    print("Step 7 (Attach Event):")
    print(f"  States: {trace._steps[7].states}")
    print(f"  Events: {trace._steps[7].events}")

    # Verification summary
    verification = trace.summarize()
    print(f"\nDemo complete. Collisions during execution: {verification} (continued past)")


if __name__ == "__main__":
    main()
