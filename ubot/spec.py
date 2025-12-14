from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
import numpy as np


class FaceID(Enum):
    MA_RIGHT = "ma_right"
    MA_DOWN = "ma_down"
    MB_LEFT = "mb_left"
    MB_UP = "mb_up"


@dataclass
class FaceSpec:
    id: FaceID
    local_normal: np.ndarray  # Shape (3,)
    local_origin: np.ndarray  # Shape (3,)
    plane_offset: Optional[float] = None
    size: Optional[np.ndarray] = None  # Shape (3,) if known (e.g., width, height, depth)


@dataclass
class HalfSpec:
    name: str  # "ma" or "mb"
    collision_geoms: List[str] = field(default_factory=list)  # XML geom references
    visual_geoms: List[str] = field(default_factory=list)
    faces: Dict[FaceID, FaceSpec] = field(default_factory=dict)


@dataclass
class JointSpec:
    name: str
    parent_body: str
    child_body: str
    type: str  # e.g., "hinge"
    axis: np.ndarray  # Shape (3,)
    range: tuple[float, float]  # (min, max)
    pos: np.ndarray  # Shape (3,)


@dataclass
class UbotModuleSpec:
    module_name: str
    ax_body_name: str
    ma_body_name: str
    mb_body_name: str
    joints: List[JointSpec]
    halves: Dict[str, HalfSpec]  # "ma": HalfSpec, "mb": HalfSpec


def default_faces() -> Dict[FaceID, FaceSpec]:
    """
    Returns default FaceSpec for 4 faces using conventions.
    Origins = [0,0,0] (TODO: set exact); normals in local half-frame.
    """
    return {
        FaceID.MA_RIGHT: FaceSpec(
            id=FaceID.MA_RIGHT,
            local_normal=np.array([1, 0, 0]),  # +X right
            local_origin=np.array([0, 0, 0]),  # TODO: Set to actual
            size=None  # TODO: Add size [width, height, depth]
        ),
        FaceID.MA_DOWN: FaceSpec(
            id=FaceID.MA_DOWN,
            local_normal=np.array([0, 0, -1]),  # -Z down
            local_origin=np.array([0, 0, 0])  # TODO: Set to actual
        ),
        FaceID.MB_LEFT: FaceSpec(
            id=FaceID.MB_LEFT,
            local_normal=np.array([-1, 0, 0]),  # -X left
            local_origin=np.array([0, 0, 0])  # TODO: Set to actual
        ),
        FaceID.MB_UP: FaceSpec(
            id=FaceID.MB_UP,
            local_normal=np.array([0, 0, 1]),  # +Z up
            local_origin=np.array([0, 0, 0])  # TODO: Set to actual
        )
    }


def default_faces_for_half(half_name: str) -> Dict[FaceID, FaceSpec]:
    """Returns default faces for a specific half (ma or mb)."""
    df = default_faces()
    if half_name == "ma":
        return {k: v for k, v in df.items() if "ma" in k.value}
    elif half_name == "mb":
        return {k: v for k, v in df.items() if "mb" in k.value}
