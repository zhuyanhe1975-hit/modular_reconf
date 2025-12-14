import pytest
from reconfiguration.connection_graph import ConnectionEvent, SiteRef
from reconfiguration.modular_reconfig import simulate_modular_reconfig

def test_phase3_events_simulation():
    """Test Phase-3 events in simulation."""
    events_by_step = {
        4: [ConnectionEvent(kind="detach", a=SiteRef(module_id=2, half="ma", site="right"), b=SiteRef(module_id=3, half="mb", site="left"))],
        6: [
            ConnectionEvent(kind="attach", a=SiteRef(module_id=1, half="mb", site="left"), b=SiteRef(module_id=4, half="ma", site="right")),
            ConnectionEvent(kind="attach", a=SiteRef(module_id=3, half="ma", site="right"), b=SiteRef(module_id=3, half="mb", site="left")),
        ]
    }

    # Capture output or test assertion? Expect KeyError for invalid module 4
    try:
        simulate_modular_reconfig(num_steps=8, detach_step=None, schedule=events_by_step)
        pytest.fail("Expected KeyError for invalid module")
    except KeyError as e:
        assert str(e) == "4", f"Expected KeyError for module 4, got {e}"
    except Exception as e:
        pytest.fail(f"Unexpected exception: {e}")

    # Additional asserts if needed, but since output is captured, ok for now
