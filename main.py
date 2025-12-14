"""
Modular Reconfiguration Main Entry Point
Calls the execution layer simulation.
"""

print("Modular reconfiguration planner ready")

# Call the unified execution layer simulation
from reconfiguration.modular_reconfig import simulate_modular_reconfig
simulate_modular_reconfig()
