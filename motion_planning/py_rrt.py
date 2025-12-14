import math
import random
import numpy as np


def plan_2dof_py(start, goal, is_state_valid, max_iter=1000, step_size=0.1):
    """Plan motion for 2-DOF using pure Python RRT implementation."""
    start = list(start)
    goal = list(goal)

    nodes = [start.copy()]
    parent = {tuple(start): None}

    for _ in range(max_iter):
        # Sample random state within [-pi, pi]
        rand = [random.uniform(-math.pi, math.pi) for _ in range(2)]

        # Find nearest node
        nearest = min(nodes, key=lambda node: distance(node, rand))

        # Steer towards the random sample
        d = distance(nearest, rand)
        if d <= step_size:
            new_node = rand
        else:
            ratio = step_size / d
            new_node = [n + ratio * (r - n) for n, r in zip(nearest, rand)]

        # Check if new node is valid
        if is_state_valid(new_node):
            nodes.append(new_node)
            parent[tuple(new_node)] = tuple(nearest)

            # Check if close to goal
            if distance(new_node, goal) < step_size:
                # Reconstruct path to goal
                path = []
                current = tuple(new_node)
                while current is not None:
                    path.append(list(current))
                    current = parent[current]
                path.reverse()
                return path

    return None


def distance(state1, state2):
    """Euclidean distance between two states."""
    return math.sqrt(sum((a - b)**2 for a, b in zip(state1, state2)))
