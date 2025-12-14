"""
Core Simulation Script for Modular Robot Reconfiguration
- Implements core mechanics of module states, topology, kinematics, and motion execution.
- Focuses on detachment logic, validation, and extensibility.
- Compatible with the modular robot reconfiguration project.
"""

import numpy as np

# Import existing project components
try:
    from core.module import Module
    from core.topology_graph import TopologyGraph
    from core.multi_state_space import get_joint_vector, set_joint_vector
    from core.kinematics import compute_world_transforms_multi
except ImportError as e:
    print(f"Error importing components: {e}")
    exit(1)

# Placeholder for future collision checking
def check_collisions(modules, topology):
    """Placeholder for collision detection. Returns True if collisions detected."""
    # TODO: Integrate with geometry.collision for actual checking
    return False

class ModularRobotSimulation:
    """Core simulation class for modular robot reconfiguration."""

    def __init__(self, modules, topology):
        self.modules = modules  # dict {id: Module}
        self.topology = topology  # TopologyGraph
        self.detached = {}  # dict {id: fixed_world_T} for detached modules

    def compute_transforms(self, root_id):
        """Compute world transforms for connected modules."""
        return compute_world_transforms_multi(self.modules, self.topology, root_id)

    def detach_module(self, module_id, root_id):
        """Detach a module: freeze its position and update topology."""
        if module_id in self.modules:
            # Compute current transform before detaching
            world_Ts = self.compute_transforms(root_id)
            self.detached[module_id] = world_Ts.get(module_id, np.eye(4)).copy()
            # No topology detach here, do in execute_path

    def get_positions(self):
        """Get positions of all modules, using detached or computed."""
        world_Ts = self.compute_transforms(list(self.modules)[0])
        positions = {}
        for mid in self.modules:
            if mid in self.detached:
                positions[mid] = self.detached[mid][:3, 3]
            elif mid in world_Ts:
                positions[mid] = world_Ts[mid][:3, 3]
        return positions

    def execute_path(self, path, detachment_step=None, detach_id=None):
        """Execute a path of joint vectors, with optional detachment."""
        root_id = list(self.modules)[0]  # Assume first as root
        print(f"Starting simulation with {len(path)} steps. Root: {root_id}")

        for step_idx, joint_vec in enumerate(path):
            # Update joints
            set_joint_vector(self.modules, joint_vec)

            # Detach if at step
            if detachment_step is not None and step_idx == detachment_step and detach_id:
                self.detach_module(detach_id, root_id)
                # Update topology for connections
                if detach_id == 3:
                    self.topology.detach(2, 3)
                print(f"Detached module {detach_id} at step {step_idx}")

            # Compute transforms for connected
            self.compute_transforms(root_id)

            # Get positions
            positions = self.get_positions()
            print(f"Step {step_idx}: {positions}")

            # Post-step validation
            expected_positions = self.expected_positions_from_vec(joint_vec)
            actual_positions = self.get_positions()
            self.validate_step(expected_positions, actual_positions, step_idx)

        print("Simulation complete.")

    def expected_positions_from_vec(self, joint_vec):
        """Placeholder: Compute expected positions from joint vec (integrate kinematics later)."""
        idx = 0
        expected = {}
        for mid in sorted(self.modules):
            if mid not in self.detached:  # Only connected have expected from vec
                expected[mid] = np.array([joint_vec[idx], joint_vec[idx+1], 0.0])  # Assuming simple mapping
                idx += 2
        return expected

    def validate_step(self, expected, actual, step_idx):
        """Validate positions against expected."""
        for mid in expected:
            if mid in actual:
                exp_pos = expected[mid][:2]  # xy only
                act_pos = actual[mid][:2]
                if not np.allclose(exp_pos, act_pos, atol=0.01):
                    print(f"WARNING at step {step_idx}: Module {mid} position mismatch - Expected {exp_pos}, Actual {act_pos}. "
                          f"Check kinematics or detachment logic.")
                    return False
        return True

# Demo execution
if __name__ == "__main__":
    # Create 3 modules
    modules = {
        1: Module(id=1, q=np.array([0.0, 0.0]), world_T=np.eye(4)),
        2: Module(id=2, q=np.array([0.0, 0.0]), world_T=np.eye(4)),
        3: Module(id=3, q=np.array([0.0, 0.0]), world_T=np.eye(4))
    }

    # Topology
    topology = TopologyGraph()
    topology.add_module(1)
    topology.add_module(2)
    topology.add_module(3)
    topology.attach(1, 2)
    topology.attach(2, 3)

    # Generate dummy path (6 DOF)
    num_steps = 10
    start_vec = np.array([0.0, 0.0, 0.4, 0.0, 0.8, 0.0])
    end_vec = np.array([0.3, 0.0, 0.6, 0.0, 1.0, 0.0])
    path = [start_vec + t * (end_vec - start_vec) / num_steps for t in range(num_steps + 1)]

    # Simulation
    sim = ModularRobotSimulation(modules, topology)
    sim.execute_path(path, detachment_step=num_steps // 2, detach_id=3)
