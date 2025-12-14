#!/usr/bin/env python3
"""
Kinematic Compiler Phase-2.2
Compiles ConnectionGraph active edges into a deterministic kinematic tree.
"""

from collections import deque
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List

from .connection_graph import ConnectionGraph, EdgeKey, SiteRef, ConnectionEdge


@dataclass(frozen=True)
class KinematicAttachment:
    parent: int
    child: int
    parent_site: SiteRef
    child_site: SiteRef
    T_parent_child: np.ndarray  # (4,4) from child module frame to parent module frame
    yaw_snap_deg: int  # yaw snap angle for constraint


@dataclass
class KinematicTree:
    root: int
    parent_of: Dict[int, Optional[int]]  # child -> parent, root -> None
    attachments: Dict[int, KinematicAttachment]  # keyed by child id
    order: List[int]  # BFS order


def compile_kinematic_tree(root: int, graph: ConnectionGraph, verbose: bool = False) -> KinematicTree:
    """
    Compile active connections into kinematic tree.

    active edges connect modules via SiteRef(module_id, ...)

    Semantics:
    - BFS from root, deterministic order (module_ids sorted in neighbors).
    - For each edge traversal, determine parent/child direction and compute T_parent_child from stored T_a_b.
    - Assumption: T_parent_child ≈ T_parentSite_childSite (module frames at site origins, Phase-2.2 approx).
    """
    # Build adjacency: module_id -> set of connected module_ids (from active edges)
    adj: dict[int, set[int]] = {}
    edge_map: dict[tuple[int, int], ConnectionEdge] = {}  # sorted(tuple) -> edge
    for edge in graph.active_edges():
        m1 = edge.key.a.module_id
        m2 = edge.key.b.module_id
        if m1 not in adj:
            adj[m1] = set()
        adj[m1].add(m2)
        if m2 not in adj:
            adj[m2] = set()
        adj[m2].add(m1)
        key = tuple(sorted([m1, m2]))
        edge_map[key] = edge

    # BFS
    visited = set()
    parent_of = {root: None}
    attachments = {}
    order = [root]
    queue = deque([root])

    visited.add(root)

    while queue:
        u = queue.popleft()

        # Sort neighbors for deterministic order
        neighbors = sorted(adj.get(u, set()))

        for v in neighbors:
            if v in visited:
                if verbose:
                    print(f"Ignoring back edge {u}-{v} (cycle)")
                continue

            visited.add(v)
            order.append(v)
            parent_of[v] = u

            # Get edge
            edge_key = tuple(sorted([u, v]))
            edge = edge_map[edge_key]
            assert edge is not None

            # Determine parent/child sites
            site_a = edge.key.a
            site_b = edge.key.b

            # Assumption: edge.T_a_b is T(a <- b), coords in b expressed in a
            if u == site_a.module_id:
                parent_site = site_a
                child_site = site_b
                T_parent_child = edge.T_a_b.copy()  # already T(parent <- child) since parent=A, child=B
            else:
                parent_site = site_b
                child_site = site_a
                T_parent_child = np.linalg.inv(edge.T_a_b)

            T_parent_child = T_parent_child.astype(np.float64)
            assert T_parent_child.shape == (4, 4)

            attachments[v] = KinematicAttachment(
                parent=u, child=v,
                parent_site=parent_site, child_site=child_site,
                T_parent_child=T_parent_child,
                yaw_snap_deg=edge.yaw_snap_deg
            )

            queue.append(v)

    # Modules without attachments (not reachable) are not included

    if verbose:
        print("BFS order:", order)
        for child, att in attachments.items():
            print(f"Attachment {att.parent} <-> {att.child} via sites {att.parent_site} <-> {att.child_site}, T_shape={att.T_parent_child.shape}")

    return KinematicTree(
        root=root,
        parent_of=parent_of,
        attachments=attachments,
        order=order
    )
