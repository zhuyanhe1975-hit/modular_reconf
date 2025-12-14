import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional
import numpy as np
from .spec import UbotModuleSpec, JointSpec, HalfSpec, FaceID, default_faces, default_faces_for_half

# Manual override for ambiguous names (key: regex token in XML, value: role)
UBOT_NAME_MAP = {
    "ma": "ma",
    "mb": "mb",
    "ax": "ax"
}

def load_ubot_mjcf(path: str, verbose: bool = False) -> UbotModuleSpec:
    tree = ET.parse(path)
    root = tree.getroot()
    
    # Extract body names (simple heuristic: find bodies containing tokens)
    bodies = {}
    for body in root.findall(".//body"):
        name = body.get("name", "")
        if "ma" in name.lower():
            bodies["ma"] = name
        elif "mb" in name.lower():
            bodies["mb"] = name
        elif "ax" in name.lower() or "center" in name.lower():
            bodies["ax"] = name

    # Override with UBOT_NAME_MAP if set
    for key, role in UBOT_NAME_MAP.items():
        if key in bodies:
            bodies[role] = bodies[key]

    if not all(k in bodies for k in ["ax", "ma", "mb"]):
        raise ValueError("Could not identify ax/ma/mb bodies from MJCF")


    # Extract joints (assume 2 hinge joints that connect ma<->ax and ax<->mb; read axis, range, pos)
    all_joints = [j for j in root.findall(".//joint")]
    if verbose:
        print(f"DEBUG: Found {len(all_joints)} joints total")
        for j in all_joints:
            print(f"  Joint: name={j.get('name')}, parent={j.get('parent')}, body={j.get('body')}, type={j.get('type')}")
    joints = []
    hinge_joints = [j for j in all_joints if j.get('type') == 'hinge']
    if verbose:
        print(f"DEBUG: Found {len(hinge_joints)} hinge joints")

    for i, joint in enumerate(hinge_joints):
        if i >= 2:
            break
        parent = joint.get("parent", "")
        name = joint.get("name", "")
        axis = joint.get("axis", "0 0 1")
        range_min, range_max = map(float, joint.get("range", "-pi pi").split())
        pos_el = joint.get("pos", "0 0 0")
        joints.append(JointSpec(
            name=name,
            parent_body=parent,
            child_body=joint.get("body", ""),
            type="hinge",
            axis=np.array(list(map(float, axis.split()))),
            range=(range_min, range_max),
            pos=np.array(list(map(float, pos_el.split())))
        ))

    if len(joints) < 2:
        if verbose:
            print(f"DEBUG: Only {len(joints)} joints found, adding dummy joints.")
        while len(joints) < 2:
            joints.append(JointSpec(
                name=f"dummy{len(joints)+1}", parent_body=bodies.get('ax', 'dummy'),
                child_body=bodies.get('ma' if len(joints) <1 else 'mb', 'dummy'), type="hinge",
                axis=np.array([0,0,1]), range=(-3.14,3.14), pos=np.array([0,0,0])
            ))

    if len(joints) != 2:
        raise ValueError("Expected exactly 2 internal joints")

    # Extract geoms for ma/mb if present (box/capsule/cylinder); store references
    halves = {"ma": HalfSpec("ma"), "mb": HalfSpec("mb")}
    for geom in root.findall(".//geom"):
        attrs = geom.attrib
        geom_ref = attrs.get("name", "")
        body_ref = attrs.get("body", "")
        if body_ref == bodies["ma"]:
            halves["ma"].collision_geoms.append(geom_ref)
        elif body_ref == bodies["mb"]:
            halves["mb"].collision_geoms.append(geom_ref)
    # Add default faces
    halves["ma"].faces = default_faces_for_half("ma")
    halves["mb"].faces = default_faces_for_half("mb")

    return UbotModuleSpec(
        module_name="ubot_module",
        ax_body_name=bodies["ax"],
        ma_body_name=bodies["ma"],
        mb_body_name=bodies["mb"],
        joints=joints,
        halves=halves
    )


def default_faces_for_half(half_name):
    df = default_faces()
    if half_name == "ma":
        return {k: v for k, v in df.items() if "ma" in k.value}
    elif half_name == "mb":
        return {k: v for k, v in df.items() if "mb" in k.value}
    return {}
