"""Machine-selected trace prompts, never human-confirmed endpoints or truth."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import copy
import hashlib
import math


def native_feature_collection(features, tiles, *, source_crs=None):
    """Keep native coordinates explicit at every intermediate export stage."""
    crs_ids = {tile.get("crs_authid") for tile in tiles}
    if len(crs_ids) != 1 or not all(isinstance(value, str) and value.strip() for value in crs_ids):
        raise ValueError("native GeoJSON export needs one explicit source CRS")
    crs = {"type": "name", "properties": {"name": next(iter(crs_ids))}}
    if source_crs is not None and source_crs != crs:
        raise ValueError("source GeoJSON CRS conflicts with tile metadata")
    return {"type": "FeatureCollection", "crs": crs, "features": copy.deepcopy(features)}


def validate_machine_drawing_report(report):
    """Automated screens must never process human-approved or held-out packets."""
    if (report.get("schema") != "jap-map-assisted-contour-drawing/1"
            or report.get("holdout_used") is not False or report.get("human_approvals") != 0):
        raise ValueError("automated screening requires an unapproved development report")
    if not report.get("tiles") or any(t.get("split") != "development" or t.get("sheet_id") == "178-gongju" for t in report["tiles"]):
        raise ValueError("automated screening cannot use held-out sources")
    if any(p.get("human_approved") is not False or p.get("dataset_role") != "review_only_not_training" for p in report["proposals"]):
        raise ValueError("automated screening cannot reinterpret human approvals or training labels")


@dataclass(frozen=True)
class DrawingConfig:
    minimum_source_length: float = 24.0
    tangent_lookback: float = 12.0
    minimum_gap: float = 3.0
    maximum_gap: float = 96.0
    maximum_angle_degrees: float = 30.0
    alternatives_per_endpoint: int = 2

    def validate(self):
        values = (self.minimum_source_length, self.tangent_lookback, self.minimum_gap,
                  self.maximum_gap, self.maximum_angle_degrees)
        if any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError("drawing bounds must be finite and positive")
        if self.minimum_gap > self.maximum_gap or self.maximum_gap > 128:
            raise ValueError("gap bounds must stay inside the upstream 128px limit")
        if self.maximum_angle_degrees > 45 or not 1 <= self.alternatives_per_endpoint <= 3:
            raise ValueError("ambiguous endpoint search must stay bounded")
        return self


def line_length(points):
    return sum(math.dist(a, b) for a, b in zip(points, points[1:]))


def endpoint(points, side, lookback=12.0):
    ordered = points if side == 0 else list(reversed(points))
    start = tuple(ordered[0])
    remaining = lookback
    previous = start
    inner = None
    for current in ordered[1:]:
        length = math.dist(previous, current)
        if length >= remaining and length > 0:
            ratio = remaining / length
            inner = tuple(a + ratio * (b-a) for a, b in zip(previous, current))
            break
        remaining -= length
        previous = current
    if inner is None:
        inner = tuple(ordered[-1])
    distance = math.dist(start, inner)
    if distance < 3:
        return None
    outward = tuple((a-b)/distance for a, b in zip(start, inner))
    return start, inner, outward


def endpoint_pairs(lines, config=DrawingConfig()):
    """Enumerate plausible continuations from retained lines, including conflicts.

    All raw line endpoints count toward junction degree. A rejected neighbor is
    not silently erased to turn an existing junction into a free endpoint.
    """
    from scipy.spatial import cKDTree

    config.validate()
    if len({line["uid"] for line in lines}) != len(lines):
        raise ValueError("duplicate source line identities")
    degree = Counter()
    for line in lines:
        points = line["points"]
        if len(points) < 2 or any(len(p) != 2 or not all(math.isfinite(v) for v in p) for p in points):
            raise ValueError("source line must have finite 2-D geometry")
        degree.update(tuple(round(v, 5) for v in p) for p in (points[0], points[-1]))
    anchors = []
    for line in lines:
        if not line["retained"] or line_length(line["points"]) < config.minimum_source_length:
            continue
        for side in (0, 1):
            value = endpoint(line["points"], side, config.tangent_lookback)
            if value is None:
                continue
            point, previous, direction = value
            if degree[tuple(round(v, 5) for v in point)] != 1:
                continue
            anchors.append({"id": f"{line['uid']}:{side}", "uid": line["uid"],
                            "point": point, "previous": previous, "direction": direction,
                            "length": line_length(line["points"]), "score": line["score"]})
    if not anchors:
        return [], {"eligible_endpoints": 0, "plausible_pairs": 0}
    tree = cKDTree([row["point"] for row in anchors])
    minimum_dot = math.cos(math.radians(config.maximum_angle_degrees))
    pairs = []
    for first_index, second_index in sorted(tree.query_pairs(config.maximum_gap)):
        first, second = anchors[first_index], anchors[second_index]
        if first["uid"] == second["uid"]:
            continue
        gap = math.dist(first["point"], second["point"])
        if gap < config.minimum_gap or gap > min(first["length"], second["length"]) * 1.5:
            continue
        chord = tuple((b-a)/gap for a, b in zip(first["point"], second["point"]))
        align_a = sum(a*b for a, b in zip(first["direction"], chord))
        align_b = -sum(a*b for a, b in zip(second["direction"], chord))
        if min(align_a, align_b) < minimum_dot:
            continue
        identity = "|".join(sorted((first["id"], second["id"])))
        priority = min(align_a, align_b) * math.sqrt(min(first["length"], second["length"])) / (gap+12)
        priority *= .5 + .5 * min(first["score"], second["score"])
        pairs.append({"pair_id": "pair-"+hashlib.sha256(identity.encode()).hexdigest()[:16],
                      "first": first, "second": second, "gap_pixels": gap,
                      "maximum_angle_degrees": math.degrees(math.acos(min(1., min(align_a, align_b)))),
                      "priority_score_not_probability": priority})
    pairs.sort(key=lambda row: (-row["priority_score_not_probability"], row["pair_id"]))
    usage = Counter()
    selected = []
    for row in pairs:
        ends = (row["first"]["id"], row["second"]["id"])
        if any(usage[end] >= config.alternatives_per_endpoint for end in ends):
            continue
        usage.update(ends)
        selected.append(row)
    for row in selected:
        row["competing_endpoint_pair"] = any(usage[row[key]["id"]] > 1 for key in ("first", "second"))
    return selected, {"eligible_endpoints": len(anchors), "plausible_pairs": len(pairs),
                      "bounded_pair_count": len(selected)}


def assess_path(points, start, end, score, source_mask):
    """Measure proposal evidence without asserting contour semantics."""
    import numpy as np
    from scipy.ndimage import distance_transform_edt, map_coordinates

    if len(points) < 2 or any(len(p) != 2 or not all(math.isfinite(v) for v in p) for p in points):
        raise ValueError("path must have finite 2-D points")
    height, width = score.shape
    if any(not 0 <= x < width or not 0 <= y < height for x, y in points):
        raise ValueError("path leaves its source tile")
    sampled = [tuple(points[0])]
    for a, b in zip(points, points[1:]):
        count = max(1, math.ceil(math.dist(a, b)))
        sampled.extend(tuple(x+(y-x)*n/count for x, y in zip(a, b)) for n in range(1, count+1))
    xy = np.asarray(sampled, dtype=float).T
    values = map_coordinates(score, (xy[1], xy[0]), order=1, mode="nearest")
    distance = distance_transform_edt(~np.asarray(source_mask, dtype=bool))
    new = map_coordinates(distance, (xy[1], xy[0]), order=1, mode="nearest") > 1.5
    length = line_length(points)
    return {"path_length_pixels": length, "detour_ratio": length/max(math.dist(start, end), 1e-9),
            "endpoint_error_pixels": max(math.dist(points[0], start), math.dist(points[-1], end)),
            "mean_ink_support": float(values.mean()), "support_q10": float(np.quantile(values, .1)),
            "supported_fraction": float((values >= .28).mean()),
            "new_fraction_outside_original_1_5px": float(new.mean())}
