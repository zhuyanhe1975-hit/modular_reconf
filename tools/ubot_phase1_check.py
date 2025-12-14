#!/usr/bin/env python3
"""
UBot Phase-1 Integration Checker
Loads MJCF, prints summary, runs FK sanity check.
"""

import sys
import argparse
from pathlib import Path
import numpy as np

# Add ubot to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ubot.mjcf_parser import load_ubot_mjcf
from ubot.kinematics_phase1 import forward_kinematics

def main(mjcf_path):
    spec = load_ubot_mjcf(mjcf_path)

    print("\n=== UBot Phase-1 Summary ===")
    print(f"Module: {spec.module_name}")
    print(f"Bodies: ax={spec.ax_body_name}, ma={spec.ma_body_name}, mb={spec.mb_body_name}")
    print("Joints:")
    for j in spec.joints:
        print(f"  {j.name}: axis={j.axis}, range={j.range}, pos={j.pos}")
    print("Faces:")
    for half_name, half_spec in spec.halves.items():
        faces_names = [face.id.value for face in half_spec.faces.values()]
        print(f"  {half_name}: {faces_names}")

    # FK sanity check
    print("\n=== FK Sanity Checks ===")
    T_root = np.eye(4)
    q0 = np.array([0, 0])
    q1 = np.array([0.1, -0.1])
    
    poses0 = forward_kinematics(q0, T_root, spec.joints[0], spec.joints[1])
    poses1 = forward_kinematics(q1, T_root, spec.joints[0], spec.joints[1])
    
    print(f"q=[0,0]: ma pos={poses0['ma'][:3,3]}, mb pos={poses0['mb'][:3,3]}")
    print(f"q=[0.1,-0.1]: ma pos={poses1['ma'][:3,3]}, mb pos={poses1['mb'][:3,3]}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 tools/ubot_phase1_check.py <mjcf_path>")
        sys.exit(1)
    main(sys.argv[1])
