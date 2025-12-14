import random
import numpy as np
from geometry.multi_collision import check_all_collisions
from core.multi_state_space import get_joint_vector, set_joint_vector
from core.kinematics import compute_world_transforms_multi

def plan_multi(modules_dict, topology, root_id, start_vec, goal_vec, max_iter=3000, step_size=0.2):
    """
    Simple Python RRT for multi-module dynamic planning.
    - modules_dict: dict of module_id -> Module
    - topology: TopologyGraph
    - root_id: int
    - start_vec, goal_vec: concatenated joint vectors for attached modules
    Returns: list of joint vectors (path) or None
    """
    # Helper functions
    def distance(a, b):
        return np.linalg.norm(a - b)

    def interpolate(a, b, alpha):
        return a + alpha * (b - a)

    def is_state_valid(state_vec):
        set_joint_vector(modules_dict, state_vec)
        compute_world_transforms_multi(modules_dict, topology, root_id)
        return not check_all_collisions(modules_dict)

    # Initialize tree
    tree = [np.array(start_vec)]
    parents = {0: None}

    for it in range(max_iter):
        # Sample random state (with goal biasing)
        rand_vec = np.array(goal_vec) if random.random() < 0.2 else np.random.uniform(low=start_vec, high=goal_vec)
        # Collision-aware resample
        for _ in range(10):
            if is_state_valid(rand_vec):
                break
            rand_vec = np.array(goal_vec) if random.random() < 0.2 else np.random.uniform(low=start_vec, high=goal_vec)
        else:
            continue  # failed to sample a valid state

        # Find nearest node
        dists = [distance(node, rand_vec) for node in tree]
        nearest_idx = np.argmin(dists)
        nearest = tree[nearest_idx]

        # Move toward random node
        direction = rand_vec - nearest
        norm = np.linalg.norm(direction)
        if norm > step_size:
            direction = direction / norm * step_size
        new_state = nearest + direction

        # Check validity
        if not is_state_valid(new_state):
            continue

        # Add to tree
        tree.append(new_state)
        parents[len(tree) - 1] = nearest_idx

        # Check if goal reached
        if distance(new_state, goal_vec) < step_size:
            tree.append(goal_vec)
            parents[len(tree) - 1] = len(tree) - 2
            # Build path
            path = []
            idx = len(tree) - 1
            while idx is not None:
                path.append(tree[idx])
                idx = parents[idx]
            path.reverse()
            print(f"Path found with {len(path)} steps")
            print(f"First state: {path[0]}")
            print(f"Last state: {path[-1]}")
            return path

    print("No path found")
    return None
