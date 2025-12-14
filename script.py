#!/usr/bin/env python3
"""
MuJoCo Visualization Script for Modular Robot Reconfiguration
- Visualizes three modules (red, green, blue cubes) moving along a 6-DOF joint vector path.
- Module 3 detaches halfway, leaving a gray ghost at its last position.
- Uses the latest MuJoCo API (Python 3.12 on Ubuntu 24.04).
"""

import time
import numpy as np
import mujoco
import mujoco.viewer

# MJCF model string with free joints for 6-DOF motion and gray ghost
mjcf = """
<mujoco>
  <worldbody>
    <body name="module_1" pos="0 0 0">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1" rgba="1 0 0 1"/>
    </body>
    <body name="module_2" pos="0.4 0 0">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1" rgba="0 1 0 1"/>
    </body>
    <body name="module_3" pos="0.8 0 0">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1" rgba="0 0 1 1"/>
    </body>
    <body name="ghost_3" pos="0.8 0 0">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1" rgba="0.5 0.5 0.5 1"/>
    </body>
  </worldbody>
</mujoco>
"""

# Simulated precomputed path (6 DOF: joint vectors for modules 1,2,3)
# In practice, integrate with your planner; here we generate a simple path
def generate_path(num_steps=10):
    """Generate a dummy 6-DOF path for 3 modules."""
    start_config = np.array([0, 0, 0.4, 0, 0.8, 0])  # q for modules 1,2,3 (x positions)
    end_config = np.array([0.3, 0, 0.6, 0, 1.0, 0])  # goal q
    path = []
    for i in range(num_steps + 1):
        t = i / num_steps
        config = start_config + t * (end_config - start_config)
        path.append(config)
    return path

# Main visualization function
def main():
    # Load model and data
    model = mujoco.MjModel.from_xml_string(mjcf)
    data = mujoco.MjData(model)

    # Generate path
    path = generate_path(num_steps=20)  # 21 steps
    half_len = len(path) // 2

    # Track expected positions for each module (xpos from q)
    expected_positions = {1: [], 2: [], 3: []}

    # Launch viewer
    viewer = mujoco.viewer.launch_passive(model, data)

    # Animation loop
    for j, config in enumerate(path):
        # Update data.qpos for free joints (7 DOF each: 3 pos, 4 quat)
        # Set pos part, keep quat as identity (no rotation)
        for i, mid in enumerate([1, 2, 3]):
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"module_{mid}")
            body_start = model.jnt_qposadr[model.body_jntadr[body_id]]
            data.qpos[body_start:body_start + 3] = [config[i*2], config[i*2+1], 0.0]  # x, y, z=0

            # Store expected position
            expected_positions[mid].append([config[i*2], config[i*2+1]])

        # Simulate detachment: at halfway, freeze ghost and move module_3 away
        if j == half_len:
            ghost_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ghost_3")
            ghost_start = model.jnt_qposadr[model.body_jntadr[ghost_id]]
            data.qpos[ghost_start:ghost_start + 3] = data.qpos[body_start:body_start + 3]  # Ghost at current position
            data.qpos[body_start:body_start + 3] = [10, 0, 0]  # Move away
            expected_positions[3][-1] = [10, 0]  # Update expected
            print(f"Detachment at step {j}: Module 3 moved to {data.qpos[body_start:body_start + 3]}")

        # Step simulation
        mujoco.mj_step(model, data)

        # Print debug positions
        positions = {}
        for mid in [1, 2, 3]:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"module_{mid}")
            pos = data.body(body_id).xpos[:2]  # xy position
            positions[mid] = pos
        print(f"Step {j}: Positions - Module1: {positions[1]}, Module2: {positions[2]}, Module3: {positions[3]}")

        # Sync viewer
        viewer.sync()
        time.sleep(0.2)  # Pause for visibility

    # Post-animation verification
    print("\nPost-animation verification:")
    motion_correct = True
    for mid, exp_chain in expected_positions.items():
        last_exp = exp_chain[-1]
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"module_{mid}")
        act_pos = data.body(body_id).xpos[:2]
        if not np.allclose(act_pos, last_exp, atol=0.01):
            motion_correct = False
            print(f"WARNING: Module {mid} final position {act_pos} != expected {last_exp}. "
                  f"Suggest checking qpos update logic or joint configuration.")
    if motion_correct:
        print("Validation passed: All modules moved correctly along the planned path!")

    # Close viewer after user interaction
    input("Press Enter to close viewer...")
    viewer.close()

if __name__ == "__main__":
    main()
