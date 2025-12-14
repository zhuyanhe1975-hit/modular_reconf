import numpy as np
from core.module import Module
from core.topology_graph import TopologyGraph
from core.multi_state_space import get_joint_vector, set_joint_vector
from core.kinematics import compute_world_transforms_multi
from geometry.multi_collision import check_all_collisions
from motion_planning.py_rrt_multi import plan_multi

print("Modular reconfiguration planner ready")

# --- Step 1: Create modules ---
modules = {
    1: Module(id=1, q=np.zeros(2), world_T=np.eye(4)),
    2: Module(id=2, q=np.zeros(2), world_T=np.eye(4)),
    3: Module(id=3, q=np.zeros(2), world_T=np.eye(4))
}

# Set world_T offsets to avoid collision (placeholder for joint kinematics)
modules[1].world_T[0, 3] = 0.0
modules[2].world_T[0, 3] = 0.3
modules[3].world_T[0, 3] = 0.6

# --- Step 2: Build topology ---
topology = TopologyGraph()
for m in modules.values():
    topology.add_module(m.id)

# Attach modules in a chain: 1->2->3
topology.attach(1, 2)
topology.attach(2, 3)

# --- Step 3: Compute initial FK ---
compute_world_transforms_multi(modules, topology, root_id=1)
print("Computed world_Ts for modules:", list(modules.keys()))

# --- Step 4: Check initial collision ---
collision_initial = check_all_collisions(modules)
print("Initial collision check:", collision_initial)

# --- Step 5: Prepare start and goal vectors ---
start_vec = get_joint_vector(modules)
goal_vec = start_vec + 0.5  # simple offset
print("Start vector:", start_vec)
print("Goal vector:", goal_vec)

# --- Step 6: Try to plan for all modules ---
path_all = plan_multi(modules, topology, root_id=1, start_vec=start_vec, goal_vec=goal_vec)
if path_all:
    print("Python RRT path found with", len(path_all), "states")
else:
    print("No path found for all modules")

# --- Step 7: Detach module 3 dynamically ---
topology.detach(2, 3)
del modules[3]  # simulate dynamic detach
compute_world_transforms_multi(modules, topology, root_id=1)
print("Modules after detach:", list(modules.keys()))

# --- Step 8: Check collision after detach ---
collision_after_detach = check_all_collisions(modules)
print("Collision check after detach:", collision_after_detach)

# --- Step 9: Plan for remaining modules ---
start_vec_detach = get_joint_vector(modules)
goal_vec_detach = start_vec_detach + 0.5
path_detach = plan_multi(modules, topology, root_id=1, start_vec=start_vec_detach, goal_vec=goal_vec_detach)
if path_detach:
    print("Path found after detach with", len(path_detach), "steps")
    print("First state:", path_detach[0])
    print("Last state:", path_detach[-1])
else:
    print("No path found after detach")
