#!/usr/bin/env python3
"""
Demo Phase-3: Execution Layer with ConnectionGraph Events.
Run the 8-step simulation with detach at step 4, and attach attempts at step 6.
"""

from reconfiguration.modular_reconfig import simulate_modular_reconfig
from reconfiguration.connection_graph import ConnectionEvent, SiteRef

def main():
    # Events: step 4 detach edge, step 6 attempt attach (one success, one fail to show rollback)
    events_by_step = {
        4: [ConnectionEvent(kind="detach", a=SiteRef(module_id=2, half="ma", site="right"), b=SiteRef(module_id=3, half="mb", site="left"))],
        6: [
            ConnectionEvent(kind="attach", a=SiteRef(module_id=1, half="mb", site="left"), b=SiteRef(module_id=4, half="ma", site="right")),  # Fail, invalid module 4
            ConnectionEvent(kind="attach", a=SiteRef(module_id=3, half="ma", site="right"), b=SiteRef(module_id=3, half="mb", site="left")),  # Fail, detached module
        ]
    }

    simulate_modular_reconfig(num_steps=8, detach_step=None, events_by_step=events_by_step)

if __name__ == "__main__":
    main()
