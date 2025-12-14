#!/usr/bin/env python3
"""
Demo Phase-3.1: Planner + Scheduler in Execution Layer.
Run simulation with planned detach/attach events, print compact trace summary.
"""

from reconfiguration.planner_stub import plan_modular_reconfig
from reconfiguration.modular_reconfig import simulate_modular_reconfig

def main():
    # Plan: detach at 4, attach success at 6
    path, schedule = plan_modular_reconfig(num_steps=8, detach_step=4, attach_success_step=6)
    print("Planned schedule:", {k: [f"{e.kind} {e.a} to {e.b}" for e in v] for k,v in schedule.items()})

    # Simulate with schedule
    simulate_modular_reconfig(num_steps=8, detach_step=None, schedule=schedule)

if __name__ == "__main__":
    main()
