import time
import numpy as np
import open3d as o3d
from core.module import Module
from core.topology_graph import TopologyGraph
from core.multi_state_space import get_joint_vector, set_joint_vector
from core.kinematics import compute_world_transforms_multi
from motion_planning.py_rrt_multi import plan_multi

# --- Setup modules and topology ---
modules = {
    1: Module(id=1, q=np.zeros(2), world_T=np.eye(4)),
    2: Module(id=2, q=np.zeros(2), world_T=np.eye(4)),
    3: Module(id=3, q=np.zeros(2), world_T=np.eye(4))
}

# Initial offsets
modules[1].world_T[0, 3] = 0.0
modules[2].world_T[0, 3] = 0.5
modules[3].world_T[0, 3] = 1.0

topology = TopologyGraph()
for m in modules.values():
    topology.add_module(m.id)
topology.attach(1, 2)
topology.attach(2, 3)

# --- Plan multi-module path ---
start_vec = get_joint_vector(modules)
goal_vec = start_vec + np.ones_like(start_vec) * 2.0  # amplify motion for visualization
path = plan_multi(modules, topology, 1, start_vec, goal_vec)

if path is None:
    print("No path found to visualize")
    exit()

# --- Visualization setup ---
vis = o3d.visualization.Visualizer()
vis.create_window()
ctr = vis.get_view_control()
ctr.set_lookat([1.0, 0, 0])
ctr.set_front([-1, -1, -1])
ctr.set_up([0, 0, 1])
ctr.set_zoom(0.6)

# Save initial X offsets for scaling
initial_x = [m.world_T[0,3] for m in modules.values()]
goal_x = [initial + 2.0 for initial in initial_x]  # scaled goal

# Animation loop
half_len = len(path) // 2
traces = []  # trajectory markers
colors_module = [(1,0,0), (0,1,0), (0,0,1)]  # RGB for attached modules
color_detached = (0.7,0.7,0.7)  # gray for detached

for j, step in enumerate(path):
    # Update joint vectors
    set_joint_vector(modules, step)
    compute_world_transforms_multi(modules, topology, 1)

    # Apply scaled linear motion for visualization
    for idx, m in enumerate(modules.values()):
        fraction = j / len(path)
        m.world_T[0,3] = initial_x[idx] + (goal_x[idx]-initial_x[idx]) * fraction

    # Print module positions
    positions = {m.id: m.world_T[:3, 3] for m in modules.values()}
    print(f"Step {j} module positions: {positions}")

    # Detach at halfway
    if j == half_len and 3 in modules:
        topology.detach(2,3)
        detached_module = modules.pop(3)
        previous_detached_world_T = detached_module.world_T.copy()
        print(f"Detachment at step {j} - module 3 removed from scene")

    # Clear geometries
    vis.clear_geometries()
    for trace in traces:
        vis.add_geometry(trace)

    # Draw attached modules
    for idx, m in enumerate(modules.values()):
        box = o3d.geometry.TriangleMesh.create_box(0.2,0.2,0.2)
        box.translate([-0.1,-0.1,-0.1])
        box.transform(m.world_T)
        box.paint_uniform_color(colors_module[idx % len(colors_module)])
        vis.add_geometry(box)

        # ID sphere
        id_sphere = o3d.geometry.TriangleMesh.create_sphere(0.03)
        id_sphere.translate(m.world_T[:3,3])
        id_sphere.paint_uniform_color([1,1,1])
        vis.add_geometry(id_sphere)

        # Trajectory marker
        trace_sphere = o3d.geometry.TriangleMesh.create_sphere(0.04)
        trace_sphere.translate(m.world_T[:3,3])
        trace_sphere.paint_uniform_color(colors_module[idx % len(colors_module)])
        traces.append(trace_sphere)

    # Detached module
    if j >= half_len and 'detached_module' in locals():
        gray_box = o3d.geometry.TriangleMesh.create_box(0.2,0.2,0.2)
        gray_box.translate([-0.1,-0.1,-0.1])
        gray_box.transform(previous_detached_world_T)
        gray_box.paint_uniform_color(color_detached)
        vis.add_geometry(gray_box)

    # Render
    vis.poll_events()
    vis.update_renderer()
    time.sleep(0.2)

print("Visualization complete")
vis.destroy_window()
