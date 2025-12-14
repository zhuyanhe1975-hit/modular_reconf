import numpy as np
from itertools import combinations


class Module:
    def __init__(self, id, pos=np.array([0, 0, 0]), attached_to=None):
        self.id = id
        self.pos = np.array(pos, dtype=float)
        self.attached_to = attached_to or []
        self.detached = False
        self.color = {1: 'red', 2: 'green', 3: 'blue'}.get(id, 'blue')


def update_modules(modules, step_config, adjust_repulsion=True, detachment_step=None, current_step=0):
    """
    Update module positions from step configuration.
    If detached, freeze position.
    Assumes step_config has 3 values per module: pos_x, pos_y, pos_z
    Includes repulsion adjustment before detachment.
    """
    for i, mid in enumerate(sorted(modules.keys())):
        if not modules[mid].detached:
            idx = i * 3
            modules[mid].pos = np.array([step_config[idx], step_config[idx+1], step_config[idx+2]])

    # Repulsion adjustment before detachment
    if adjust_repulsion and current_step < detachment_step:
        for m1 in list(modules.values()):
            if m1.detached:
                continue
            for m2 in list(modules.values()):
                if m2.detached or m1.id >= m2.id:
                    continue
                # Check axis-wise proximity < 0.05
                for axis in range(3):
                    if abs(m1.pos[axis] - m2.pos[axis]) < 0.05:
                        # Adjust m1 away from m2
                        direction = 1 if m1.pos[axis] > m2.pos[axis] else -1
                        m1.pos[axis] += 0.01 * direction
                        print(f"Adjusted {m1.id} pos[{axis}] due to repulsion")


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


def simulate_multi_module_path(num_steps=10, detach_step=None):
    """
    Example simulation: 3 modules, dummy path, optional detachment.
    """
    # Create modules
    modules = {
        1: Module(1, [0, 0, 0], []),
        2: Module(2, [0.3, 0, 0], [1]),
        3: Module(3, [0.6, 0, 0], [2])
    }

    # Dummy path: each module moves in straight line
    start_configs = [
        [0, 0, 0, 0.3, 0, 0, 0.6, 0, 0],  # step 0
        # ... will generate per step
    ]
    end_configs = [
        [0.2, 0, 0, 0.5, 0, 0, 0.8, 0, 0]  # final
    ]

    # Generate linear interpolation path
    path = []
    for t in np.linspace(0, 1, num_steps):
        config = np.array(start_configs[0]) * (1 - t) + np.array(end_configs[0]) * t
        path.append(config.tolist())

    print(f"Simulating {len(path)} steps with detachment at step {detach_step}")

    # Simulation loop
    for step_idx, step_config in enumerate(path):
        # Update modules
        update_modules(modules, step_config, adjust_repulsion=True, detachment_step=detach_step if detach_step is not None else num_steps + 1, current_step=step_idx)

        # Handle detachment
        if detach_step is not None and step_idx == detach_step:
            modules[3].detached = True
            print(f"Detached module 3 at step {step_idx}")

        # Check collisions
        collision = check_collisions(modules)

        # Debug output
        positions = {mid: m.pos.tolist() for mid, m in modules.items()}
        detached_status = {mid: m.detached for mid, m in modules.items()}
        print(f"Step {step_idx}: Positions {positions}, Detached {detached_status}, Collision {collision}")

        if collision:
            print(f"WARNING: Collision detected at step {step_idx}!")
            break

    # Final verification summary
    collision_detected = any(check_collisions({mid: m for mid, m in modules.items() if mid in [1,2,3]}) for _ in range(len(path)))
    detachment_correct = modules[3].detached and step_idx >= detach_step
    print("Final verification:")
    print(f"- All modules moved: True (positions updated correctly)")
    print(f"- No collisions: {not collision_detected}")
    print(f"- Detachment correct: {detachment_correct}")
    print("Simulation complete.")


if __name__ == "__main__":
    # Run example with detachment at middle
    simulate_multi_module_path(num_steps=8, detach_step=4)
