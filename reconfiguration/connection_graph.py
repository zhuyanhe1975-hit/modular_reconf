#!/usr/bin/env python3
"""
Connection Graph for UBot modular reconfiguration.
Manages active connections between modules.
"""

from dataclasses import dataclass, field
import numpy as np
from typing import Literal, Optional, Dict, List
from .connection import can_connect, SitePose


YawSnap = Literal[0, 90, 180, 270]


@dataclass(frozen=True)
class SiteRef:
    module_id: int
    half: Literal["ma", "mb"]
    site: str


@dataclass(frozen=True)
class ConnectionEvent:
    kind: Literal["attach", "detach"]
    a: SiteRef
    b: SiteRef
    yaw_snap_deg: Optional[YawSnap] = None
    T_a_b: Optional[np.ndarray] = None  # (4,4)


@dataclass(frozen=True)
class EdgeKey:
    a: SiteRef
    b: SiteRef

    def __eq__(self, other):
        if not isinstance(other, EdgeKey):
            return NotImplemented
        return ((self.a == other.a and self.b == other.b) or
                (self.a == other.b and self.b == other.a))

    def __hash__(self):
        # Sort tuples lexicographically for hashing
        key1 = (self.a.module_id, self.a.half, self.a.site)
        key2 = (self.b.module_id, self.b.half, self.b.site)
        if key1 < key2:
            return hash((key1, key2))
        else:
            return hash((key2, key1))

    @classmethod
    def normalized(cls, a_start: SiteRef, b_end: SiteRef) -> 'EdgeKey':
        key_a = (a_start.module_id, a_start.half, a_start.site)
        key_b = (b_end.module_id, b_end.half, b_end.site)
        if key_a < key_b:
            return cls(a=a_start, b=b_end)
        else:
            return cls(a=b_end, b=a_start)


@dataclass
class ConnectionEdge:
    key: EdgeKey
    yaw_snap_deg: YawSnap
    T_a_b: np.ndarray  # (4,4)
    active: bool = True


@dataclass
class ConnectionGraph:
    edges: Dict[EdgeKey, ConnectionEdge] = field(default_factory=dict)

    def apply(self, ev: ConnectionEvent) -> None:
        if ev.kind == "attach":
            if ev.yaw_snap_deg is None or ev.T_a_b is None:
                raise ValueError(f"Attach event must have yaw_snap_deg and T_a_b")
            if not self.site_is_free(ev.a):
                raise ValueError(f"Site {ev.a} is already connected")
            if not self.site_is_free(ev.b):
                raise ValueError(f"Site {ev.b} is already connected")
            edge = ConnectionEdge(
                key=EdgeKey.normalized(ev.a, ev.b),
                yaw_snap_deg=ev.yaw_snap_deg,
                T_a_b=ev.T_a_b
            )
            self.edges[edge.key] = edge
        elif ev.kind == "detach":
            key = EdgeKey.normalized(ev.a, ev.b)
            if key in self.edges and self.edges[key].active:
                self.edges[key].active = False

    def is_connected(self, a: SiteRef, b: SiteRef) -> bool:
        key = EdgeKey.normalized(a, b)
        return key in self.edges and self.edges[key].active

    def active_edges(self) -> List[ConnectionEdge]:
        return [e for e in self.edges.values() if e.active]

    def module_states(self, all_module_ids: List[int]) -> Dict[int, 'ModuleState']:
        """Return {module_id: ModuleState} based on active connections."""
        from .modular_reconfig import ModuleState
        connected = set()
        for edge in self.active_edges():
            connected.add(edge.key.a.module_id)
            connected.add(edge.key.b.module_id)
        return {mid: ModuleState.ATTACHED if mid in connected else ModuleState.DETACHED for mid in all_module_ids}

    def site_is_free(self, site: SiteRef) -> bool:
        for edge in self.active_edges():
            if edge.key.a == site or edge.key.b == site:
                return False
        return True


def parse_site_name(xml_site_name: str) -> tuple[str, str]:
    """Parse 'ma_connector_right' -> ('ma', 'connector_right')"""
    parts = xml_site_name.split('_', 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid site name: {xml_site_name}")
    return parts[0], parts[1]


def make_attach_event(moduleA: int, siteA_name: str, poseA: SitePose,
                      moduleB: int, siteB_name: str, poseB: SitePose,
                      params=None) -> ConnectionEvent:
    """Create attach event after verifying connection."""
    if params is None:
        from .connection import ConnectParams
        params = ConnectParams()

    result = can_connect(poseA, poseB, params)
    if not result.feasible:
        raise ValueError(f"Connection not feasible: {result.reason}")

    # Parse site names
    halfA, siteA = parse_site_name(siteA_name)
    halfB, siteB = parse_site_name(siteB_name)

    a = SiteRef(module_id=moduleA, half=halfA, site=siteA)
    b = SiteRef(module_id=moduleB, half=halfB, site=siteB)

    # Compute T_a_b = inv(T_world_a) @ T_world_b
    T_world_a = np.eye(4)
    T_world_a[:3, 3] = poseA.position
    from .connection import quat_to_rot
    T_world_a[:3, :3] = quat_to_rot(poseA.quat_wxyz)

    T_world_b = np.eye(4)
    T_world_b[:3, 3] = poseB.position
    T_world_b[:3, :3] = quat_to_rot(poseB.quat_wxyz)

    T_a_b = np.linalg.inv(T_world_a) @ T_world_b

    return ConnectionEvent(
        kind="attach",
        a=a, b=b,
        yaw_snap_deg=result.yaw_snap_deg,
        T_a_b=T_a_b
    )
