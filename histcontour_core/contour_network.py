"""Assemble candidate fragments at existing nodes without inventing geometry.

Only degree-two endpoint nodes are merged. Branches, source tiles, and every
original vertex remain distinct. A network containing an approved connector
does NOT thereby become a human-approved contour.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
import math

from .assisted_review import _line
from .human_feedback import geometry_digest, map_to_pixel


def merge_parts(parts, tiles, *, endpoint_tolerance_pixels=1e-5):
    import numpy as np
    from scipy.spatial import cKDTree
    if not 0 < endpoint_tolerance_pixels <= 1e-4:
        raise ValueError("only numerical round-off, not spatial snapping, may join endpoints")
    by_tile = defaultdict(list)
    part_ids = [f["properties"]["part_id"] for f in parts]
    if len(part_ids) != len(set(part_ids)):
        raise ValueError("duplicate part identities")
    for feature in parts:
        _line(feature["geometry"])
        tid = feature["properties"]["tile_id"]
        if tid not in tiles:
            raise ValueError("part has no source tile")
        by_tile[tid].append(feature)
    features, memberships, tile_audits = [], [], []
    for tid in sorted(by_tile):
        edges = sorted(by_tile[tid], key=lambda f: f["properties"]["part_id"])
        endpoints = []
        for feature in edges:
            p = feature["geometry"]["coordinates"]
            endpoints.extend(map_to_pixel(tiles[tid], [p[0],p[-1]]))
        points = np.asarray(endpoints); parent = list(range(len(points)))
        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]; i = parent[i]
            return i
        for a,b in sorted(cKDTree(points).query_pairs(endpoint_tolerance_pixels)):
            a,b = find(a),find(b)
            if a != b:
                parent[max(a,b)] = min(a,b)
        nodes = [find(i) for i in range(len(points))]
        adjacency = defaultdict(list)
        for i in range(len(edges)):
            adjacency[nodes[2*i]].append((i,0)); adjacency[nodes[2*i+1]].append((i,1))
        # Refuse transitive chains of near points whose full cluster is wider
        # than the explicit round-off tolerance.
        clusters = defaultdict(list)
        for i,node in enumerate(nodes):
            clusters[node].append(points[i])
        for values in clusters.values():
            for i,p in enumerate(values):
                if any(np.linalg.norm(p-q) > endpoint_tolerance_pixels for q in values[i+1:]):
                    raise ValueError("endpoint round-off cluster is spatially ambiguous")
        visited = set(); tile_lines = 0; maximum_join_error = 0.
        def walk(edge_index, entry_side):
            nonlocal tile_lines, maximum_join_error
            coordinates, members, kinds = [], [], Counter()
            while edge_index not in visited:
                visited.add(edge_index)
                feature = edges[edge_index]; props = feature["properties"]
                ordered = feature["geometry"]["coordinates"]
                if entry_side:
                    ordered = list(reversed(ordered))
                if coordinates:
                    seam = map_to_pixel(tiles[tid],[coordinates[-1],ordered[0]])
                    error = math.dist(*seam)
                    if error > endpoint_tolerance_pixels:
                        raise ValueError("joined path exceeds round-off tolerance")
                    maximum_join_error = max(maximum_join_error,error)
                    # Keep BOTH coordinates when different. This preserves
                    # exact approved vertices; no hidden coordinate snapping.
                    if coordinates[-1] == ordered[0]:
                        ordered = ordered[1:]
                coordinates.extend(copy.deepcopy(ordered))
                members.append({"part_id":props["part_id"],"reversed":bool(entry_side)})
                kinds[props["part_kind"]] += 1
                node = nodes[2*edge_index+(1-entry_side)]
                if len(adjacency[node]) != 2:
                    break
                available = [(index,side) for index,side in adjacency[node] if index not in visited]
                if not available:
                    break
                edge_index,entry_side = available[0]
            line_id = f"C{len(features)+1:06}"
            geometry = {"type":"LineString","coordinates":coordinates}
            pixels = map_to_pixel(tiles[tid],coordinates)
            properties = {"line_id":line_id,"tile_id":tid,"part_count":len(members),
                "source_part_count":kinds["retained_source"],"approved_connection_count":kinds["approved_connection"],
                "automatic_connection_count":kinds["automatic_connection"],
                "human_approved":False,"whole_line_semantics_approved":False,"training_eligible":False,
                "dataset_role":"review_only_not_training","review_status":"unreviewed_network",
                "elevation_status":"not_assigned","closed":math.dist(pixels[0],pixels[-1]) <= endpoint_tolerance_pixels,
                "length_pixels":sum(math.dist(a,b) for a,b in zip(pixels,pixels[1:])),
                "geometry_sha256":geometry_digest(geometry)}
            features.append({"type":"Feature","properties":properties,"geometry":geometry})
            memberships.append({"line_id":line_id,"tile_id":tid,"members":members})
            tile_lines += 1
        # Start at boundaries/branches; remaining components are cycles.
        for node in sorted(adjacency):
            if len(adjacency[node]) != 2:
                for index,side in adjacency[node]:
                    if index not in visited:
                        walk(index,side)
        for index in range(len(edges)):
            if index not in visited:
                walk(index,0)
        assert len(visited) == len(edges)
        tile_audits.append({"tile_id":tid,"parts":len(edges),"merged_lines":tile_lines,
                           "endpoint_nodes":len(adjacency),"open_endpoint_nodes":sum(len(v)==1 for v in adjacency.values()),
                           "branch_nodes":sum(len(v)>2 for v in adjacency.values()),
                           "maximum_roundoff_join_error_pixels":maximum_join_error})
    return {"features":features,"memberships":memberships,"tiles":tile_audits,
            "part_count":len(parts),"line_count":len(features),
            "open_endpoint_nodes":sum(t["open_endpoint_nodes"] for t in tile_audits),
            "endpoint_tolerance_pixels":endpoint_tolerance_pixels}
