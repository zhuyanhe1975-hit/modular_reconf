#!/usr/bin/env python3
"""
MuJoCo Visualization Script for Modular Robot Reconfiguration
- Red, Green, Blue cubes move along a 6-DOF path
- Blue cube detaches halfway, leaving a gray ghost
- Python 3.12, Ubuntu 24.04, latest MuJoCo
"""

import time
import numpy as np
import mujoco
import mujoco.viewer

# MJCF model: free joints for motion + ghost for detached module
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

def generate_path(num_steps=10):
    """Generate a dummy 6-DOF path for modules 1,2,3 (x,y positions)"""
    start = np.array([0,0, 0.4,0, 0.8,0])  # x,y for modules 1,2,3
    end   = np.array([0.3,0, 0.6,0, 1.0,0])
    path = []
    for i in range(num_steps + 1):
        t = i / num_steps
        path.append(start + t*(end - start))
    return path

def main():
    model = mujoco.MjModel.from_xml_string(mjcf)
    data = mujoco.MjData(model)

    path = generate_path(num_steps=20)
    half_len = len(path)//2

    detached_pos = None

    viewer = mujoco.viewer.launch_passive(model, data)

    for step_idx, config in enumerate(path):
        # Update positions for red and green cubes always
        for i, mid in enumerate([1,2]):
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"module_{mid}")
            body_start = model.jnt_qposadr[model.body_jntadr[body_id]]
            data.qpos[body_start:body_start+3] = [config[i*2], config[i*2+1], 0.0]

        # Blue cube / ghost
        if step_idx < half_len:
            # Blue still moving
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "module_3")
            body_start = model.jnt_qposadr[model.body_jntadr[body_id]]
            data.qpos[body_start:body_start+3] = [config[4], config[5], 0.0]
        elif step_idx == half_len:
            # Detach blue cube
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "module_3")
            body_start = model.jnt_qposadr[model.body_jntadr[body_id]]
            detached_pos = data.qpos[body_start:body_start+3].copy()
            # Move blue far away
            data.qpos[body_start:body_start+3] = [10,0,0]
            # Place ghost
            ghost_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ghost_3")
            ghost_start = model.jnt_qposadr[model.body_jntadr[ghost_id]]
            data.qpos[ghost_start:ghost_start+3] = detached_pos
            print(f"Step {step_idx}: Module 3 detached at {detached_pos}")
        else:
            # Keep ghost at detached_pos
            ghost_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ghost_3")
            ghost_start = model.jnt_qposadr[model.body_jntadr[ghost_id]]
            data.qpos[ghost_start:ghost_start+3] = detached_pos

        mujoco.mj_step(model, data)

        # Print positions for verification
        positions = {}
        for mid in [1,2]:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"module_{mid}")
            positions[mid] = data.body(body_id).xpos[:2]
        positions[3] = detached_pos[:2] if detached_pos is not None else data.body(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "module_3")).xpos[:2]

        print(f"Step {step_idx}: Positions - Module1: {positions[1]}, Module2: {positions[2]}, Module3: {positions[3]}")

        viewer.sync()
        time.sleep(0.2)

    print("\nAnimation complete. Verify visually:")
    print("- Red and green cubes moved")
    print("- Blue cube detached and moved away")
    print("- Gray ghost stays at detach position")

    input("Press Enter to close viewer...")
    viewer.close()

if __name__ == "__main__":
    main()
