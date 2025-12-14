"""Reconfiguration package initialization.

Provides high-level reconfiguration planning and execution APIs.
"""

# Import key modules for top-level access
from .modular_reconfig import simulate_modular_reconfig
from .connection_api import attempt_attach, ExecutorWrapper
from .connection_feasibility import FeasibilityParams
from .connection_graph import ConnectionGraph, ConnectionEvent, SiteRef
from .verifier import verify_execution

# Optional: export demos for quick access
__all__ = [
    "simulate_modular_reconfig",
    "attempt_attach",
    "ExecutorWrapper",
    "FeasibilityParams",
    "ConnectionGraph",
    "ConnectionEvent",
    "SiteRef",
    "verify_execution",
]
