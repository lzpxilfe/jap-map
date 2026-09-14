"""Bounded, order-preserving hypotheses between two caller-supplied banks.

Only endpoint geometry is compared. No pixels, contour semantics, elevations,
text classes, routes, or approvals are inferred. Costs describe hypothetical
gap connections; even a unique optimum requires routing and human review.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Real


@dataclass(frozen=True)
class RegionalGapMatchingConfig:
    maximum_endpoints_per_bank: int = 64
    maximum_dp_cells: int = 4225
    minimum_pair_distance_px: float = .5
    maximum_pair_distance_px: float = 96.
    maximum_lateral_displacement_px: float = 32.
    maximum_tangent_error_degrees: float = 65.
    distance_weight: float = 1.
    tangent_weight: float = .75
    lateral_weight: float = 1.
    support_weight: float = .15
    skip_cost: float = .65
    ambiguity_cost_margin: float = .12
    ordering_tolerance_px: float = .5
    boundary_fraction: float = .05

    def validate(self):
        for key, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{key} must be finite and numeric")
            if value < 0 or (value == 0 and key not in {
                    "support_weight", "ambiguity_cost_margin", "ordering_tolerance_px"}):
                raise ValueError(f"{key} is outside its allowed range")
        if type(self.maximum_endpoints_per_bank) is not int or not 1 <= self.maximum_endpoints_per_bank <= 128:
            raise ValueError("maximum_endpoints_per_bank must be an integer in 1..128")
        if type(self.maximum_dp_cells) is not int or not 1 <= self.maximum_dp_cells <= 16641:
            raise ValueError("maximum_dp_cells must be an integer in 1..16641")
        if not self.minimum_pair_distance_px < self.maximum_pair_distance_px <= 512:
            raise ValueError("pair distances must be ordered and bounded by 512 pixels")
        if self.maximum_lateral_displacement_px > self.maximum_pair_distance_px:
            raise ValueError("lateral displacement cannot exceed the pair distance cap")
        if any(getattr(self, key) > 1e6 for key in ("distance_weight", "tangent_weight", "lateral_weight",
                                                  "support_weight", "skip_cost", "ambiguity_cost_margin")):
            raise ValueError("cost settings must remain bounded by 1e6")
        if not 0 < self.maximum_tangent_error_degrees < 90 or self.boundary_fraction > .25:
            raise ValueError("tangent and boundary guards must remain local")
        return self


def _vector(value, name, *, unit=False):
    if isinstance(value, (str, bytes)) or not hasattr(value, "__len__") or len(value) != 2:
        raise ValueError(f"{name} must contain two finite coordinates")
    if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) or abs(v) > 1e9 for v in value):
        raise ValueError(f"{name} must contain finite coordinates bounded by 1e9")
    result = tuple(float(v) for v in value)
    if unit:
        length = math.hypot(*result)
        if length <= 1e-12:
            raise ValueError(f"{name} must have a nonzero direction")
        result = tuple(v / length for v in result)
    return result


def _bank(values, name, axis, cap, identities):
    if isinstance(values, (str, bytes, dict)) or not hasattr(values, "__len__"):
        raise ValueError("endpoint banks must be bounded sequences")
    if len(values) > cap:
        raise ValueError("endpoint bank exceeds the configured work cap")
    bank = []
    for index, value in enumerate(values):
        if not isinstance(value, dict) or not {"point", "outward_tangent", "source_path_id"} <= value.keys():
            raise ValueError("each endpoint needs point, outward_tangent, and source_path_id")
        source = value["source_path_id"]
        if not isinstance(source, str) or not source.strip():
            raise ValueError("source_path_id must be a nonempty string")
        point = _vector(value["point"], "point")
        tangent = _vector(value["outward_tangent"], "outward_tangent", unit=True)
        support = value.get("support")
        if support is not None and (isinstance(support, bool) or not isinstance(support, Real)
                                    or not math.isfinite(support) or not 0 <= support <= 1):
            raise ValueError("optional endpoint support must be finite in 0..1")
        identity = (source, point)
        if identity in identities:
            raise ValueError("the same source endpoint cannot occur twice, including across banks")
        identities.add(identity)
        bank.append({"input_index": index, "endpoint_id": f"{name}:{index}", "point": point,
                     "outward_tangent": tangent, "source_path_id": source,
                     "support": None if support is None else float(support),
                     "order": sum(a*b for a, b in zip(point, axis))})
    return sorted(bank, key=lambda e: (e["order"], e["input_index"]))


def _pair(a, b, axis, config):
    if a["source_path_id"] == b["source_path_id"]:
        return None, "same_source_path"
    delta = tuple(y-x for x, y in zip(a["point"], b["point"]))
    distance = math.hypot(*delta)
    if not config.minimum_pair_distance_px <= distance <= config.maximum_pair_distance_px:
        return None, "distance_bound"
    direction = tuple(v/distance for v in delta)
    lateral = abs(sum(v*w for v, w in zip(delta, axis)))
    if lateral > config.maximum_lateral_displacement_px:
        return None, "lateral_bound"
    cosines = (sum(v*w for v, w in zip(a["outward_tangent"], direction)),
               -sum(v*w for v, w in zip(b["outward_tangent"], direction)))
    errors = tuple(math.degrees(math.acos(max(-1., min(1., v)))) for v in cosines)
    if max(errors) > config.maximum_tangent_error_degrees:
        return None, "tangent_bound"
    supports = [e["support"] for e in (a, b) if e["support"] is not None]
    terms = {"distance": config.distance_weight*distance/config.maximum_pair_distance_px,
             "tangent": config.tangent_weight*sum(errors)/(2*config.maximum_tangent_error_degrees),
             "lateral": config.lateral_weight*lateral/config.maximum_lateral_displacement_px,
             "endpoint_support": config.support_weight*(1-sum(supports)/len(supports)) if supports else 0.}
    return {"a_index": a["input_index"], "b_index": b["input_index"],
            "endpoint_ids": [a["endpoint_id"], b["endpoint_id"]],
            "source_path_ids": [a["source_path_id"], b["source_path_id"]],
            "start": list(a["point"]), "end": list(b["point"]),
            "distance_px": distance, "lateral_displacement_px": lateral,
            "tangent_errors_degrees": list(errors), "cost_terms": terms, "cost": sum(terms.values()),
            "inferred_gap": True, "human_approved": False, "contour_semantics_assigned": False,
            "routing_performed": False}, None


def _intersect(a, b, c, d):
    """Inclusive straight-chord intersection, including touches and overlap."""
    def cross(p, q, r):
        return (q[0]-p[0])*(r[1]-p[1])-(q[1]-p[1])*(r[0]-p[0])
    values = cross(a, b, c), cross(a, b, d), cross(c, d, a), cross(c, d, b)
    tolerance = 1e-9 * max(1., math.dist(a, b), math.dist(c, d))
    def on(p, q, r):
        return all(min(p[k], q[k])-tolerance <= r[k] <= max(p[k], q[k])+tolerance for k in (0, 1))
    if any(abs(v) <= tolerance and on(p, q, r) for v, p, q, r in (
            (values[0], a, b, c), (values[1], a, b, d),
            (values[2], c, d, a), (values[3], c, d, b))):
        return True
    return ((values[0] > tolerance and values[1] < -tolerance or values[0] < -tolerance and values[1] > tolerance)
            and (values[2] > tolerance and values[3] < -tolerance or values[2] < -tolerance and values[3] > tolerance))


def _best_two(candidates):
    unique = {}
    for cost, pairs in candidates:
        unique[pairs] = min(cost, unique.get(pairs, math.inf))
    return sorted(((cost, pairs) for pairs, cost in unique.items()), key=lambda item: (item[0], item[1]))[:2]


def match_endpoint_banks(bank_a, bank_b, *, ordering_axis, config=RegionalGapMatchingConfig()):
    """Return the exact best and runner-up distinct monotone match sets.

    Each bank is a sequence of dictionaries with ``point``, outward-facing
    ``outward_tangent``, ``source_path_id``, and optional ``support`` in 0..1.
    Tangents and the explicit 2-D ``ordering_axis`` are normalized internally.
    Coordinates remain exact; no input is modified. An endpoint identity is
    (source_path_id, point), and duplicate identities are rejected. Pairing two
    ends of the same source path is prohibited; this module does not infer loops.

    Both banks sort in the same axis direction. Skipping costs ``skip_cost``
    per endpoint. Dynamic programming retains two DISTINCT match sets, so
    changing the order of skip operations is not a runner-up hypothesis.
    Optional support is endpoint evidence only, never evidence inside the gap.

    Optimality concerns the order constraint and pair gates, not routed geometry.
    Irregular banks can produce crossing straight chords despite monotone order;
    such hypotheses are explicitly ineligible for routing without review. All
    hypotheses remain inferred, unapproved, and unsuitable as observed-ink truth.
    """
    if not isinstance(config, RegionalGapMatchingConfig):
        raise ValueError("config must be RegionalGapMatchingConfig")
    config.validate()
    axis = _vector(ordering_axis, "ordering_axis", unit=True)
    identities = set()
    a = _bank(bank_a, "a", axis, config.maximum_endpoints_per_bank, identities)
    b = _bank(bank_b, "b", axis, config.maximum_endpoints_per_bank, identities)
    m, n = len(a), len(b)
    cells = (m+1)*(n+1)
    if cells > config.maximum_dp_cells:
        raise ValueError("alignment exceeds the configured DP work cap")
    pairs, rejections = {}, {}
    for i, first in enumerate(a):
        for j, second in enumerate(b):
            pair, reason = _pair(first, second, axis, config)
            if pair is not None:
                pairs[i, j] = pair
            else:
                rejections[reason] = rejections.get(reason, 0)+1
    previous = None
    for i in range(m+1):
        current = []
        for j in range(n+1):
            options = [(0., ())] if i == j == 0 else []
            if i:
                options.extend((cost+config.skip_cost, match) for cost, match in previous[j])
            if j:
                options.extend((cost+config.skip_cost, match) for cost, match in current[j-1])
            if i and j and (i-1, j-1) in pairs:
                pair_cost = pairs[i-1, j-1]["cost"]
                options.extend((cost+pair_cost, match+((i-1, j-1),)) for cost, match in previous[j-1])
            current.append(_best_two(options))
        previous = current
    ranked = previous[-1]
    common = []
    if not a or not b:
        common.append("empty_endpoint_bank")
    if any(second["order"]-first["order"] <= config.ordering_tolerance_px
           for bank in (a, b) for first, second in zip(bank, bank[1:])):
        common.append("bank_order_tie")
    all_points = [e["point"] for e in (*a, *b)]
    if len(set(all_points)) < len(all_points):
        common.append("coincident_source_endpoints")
    normal = (-axis[1], axis[0])
    if a and b:
        across_a = [sum(x*y for x, y in zip(e["point"], normal)) for e in a]
        across_b = [sum(x*y for x, y in zip(e["point"], normal)) for e in b]
        if not (max(across_a) < min(across_b) or max(across_b) < min(across_a)):
            common.append("bank_separation_not_established")
    margin = ranked[1][0]-ranked[0][0] if len(ranked) > 1 else None
    if margin is not None and margin <= config.ambiguity_cost_margin+1e-12:
        common.append("near_tied_hypotheses")

    def hypothesis(item):
        cost, selected = item
        # Copy each public nested record so best/runner-up do not alias.
        import copy
        matches = [copy.deepcopy(pairs[key]) for key in selected]
        reasons = list(common)
        if not matches:
            reasons.append("no_pairs_selected")
        for i, first in enumerate(matches):
            if any(_intersect(first["start"], first["end"], second["start"], second["end"])
                   for second in matches[i+1:]):
                reasons.append("straight_chords_cross_or_touch")
        for key, match in zip(selected, matches):
            if any(other != key and (other[0] == key[0] or other[1] == key[1])
                   and abs(value["cost"]-match["cost"]) <= config.ambiguity_cost_margin
                   for other, value in pairs.items()):
                reasons.append("competing_endpoint_pair")
            near = 1-config.boundary_fraction
            if (match["distance_px"] >= near*config.maximum_pair_distance_px
                    or match["distance_px"] <= (1+config.boundary_fraction)*config.minimum_pair_distance_px):
                reasons.append("pair_near_distance_boundary")
            if match["lateral_displacement_px"] >= near*config.maximum_lateral_displacement_px:
                reasons.append("pair_near_lateral_boundary")
            if max(match["tangent_errors_degrees"]) >= near*config.maximum_tangent_error_degrees:
                reasons.append("pair_near_tangent_boundary")
        reasons = sorted(set(reasons))
        used_a, used_b = {i for i, _ in selected}, {j for _, j in selected}
        return {"cost": cost, "matches": matches,
                "skipped_a_indices": [e["input_index"] for i, e in enumerate(a) if i not in used_a],
                "skipped_b_indices": [e["input_index"] for j, e in enumerate(b) if j not in used_b],
                "ambiguous": bool(reasons), "ambiguity_reasons": reasons,
                "eligible_for_routing": not reasons, "requires_human_review": True,
                "inferred_gap": True, "human_approved": False, "routing_performed": False}

    best = hypothesis(ranked[0])
    runner = hypothesis(ranked[1]) if len(ranked) > 1 else None
    return {"schema": "jap-map-regional-gap-matching/1", "status": "inferred_hypotheses_not_applied",
            "best": best, "runner_up": runner, "cost_margin": margin,
            "ambiguous": best["ambiguous"], "ambiguity_reasons": list(best["ambiguity_reasons"]),
            "ordered_a_indices": [e["input_index"] for e in a],
            "ordered_b_indices": [e["input_index"] for e in b], "ordering_axis": list(axis),
            "diagnostics": {"dp_cells": cells, "feasible_pair_count": len(pairs), "rejected_pairs": rejections},
            "config": asdict(config), "inferred_gap": True, "human_approved": False,
            "contour_semantics_assigned": False, "training_eligible": False, "routing_performed": False,
            "limitations": ["Endpoint costs do not establish contour identity or observed ink inside a gap.",
                            "Monotone alignment optimality does not establish a noncrossing routed geometry.",
                            "Synthetic checks do not establish performance on historical maps."]}
