# UBot Physical Model Integration - Phase-1

This phase introduces integration with UBot's physical model from MJCF files, focusing on static geometry, connection faces, internal joints, and minimal kinematics. **No dynamics or MuJoCo simulation**.

## Conventions
- **Coordinate Frame**: 
  - +X = right
  - +Y = forward (unused for connections)
  - +Z = up
- **Module Halves**: Red = ma, Blue = mb, ax = central internal piece (not connectable externally).
- **External Connection Faces** (only on ma/mb, not ax - no Y faces):
  - ma: RIGHT (+X), DOWN (-Z)
  - mb: LEFT (-X), UP (+Z)
- **Internal Joints**: 2-DOF serial chain: ma <-> ax <-> mb.

## Components
- `spec.py`: Dataclasses for FaceID, FaceSpec, JointSpec, UbotModuleSpec, etc.
- `mjcf_parser.py`: Loads MJCF XML into UbotModuleSpec without depending on MuJoCo runtime.
- `kinematics_phase1.py`: Minimal FK using Rodrigues rotation for internal joint poses.
- `README_phase1.md`: This documentation.
- `tools/ubot_phase1_check.py`: Integration checker for summary and FK sanity.

## How to Run the Checker
1. Ensure MJCF path is accessible.
2. Run: `python3 tools/ubot_phase1_check.py <path_to_ubot_mjcf.xml>`.

This will print body names, joints with axes/ranges, faces, and FK results for q=[0,0] and q=[0.1,-0.1].

## How to Run Tests
1. Ensure the real MJCF path, e.g.:
   `export UBOT_MJCF_PATH="/home/yhzhu/ubot-materials/3d model/mjcf/ubot.xml"`
2. Run: `UBOT_MJCF_PATH="/home/yhzhu/ubot-materials/3d model/mjcf/ubot.xml" python3 -m pytest -q tests/test_ubot_phase1_*`

Tests skip if env var not set, fail if parsing incorrect, pass if asserts match (e.g., 2 joints, 4 faces).

## Next Phases
Future phases will add dynamics, face-based connections, and planner hooks.
