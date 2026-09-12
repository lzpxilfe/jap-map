"""Label-free context from already touching, unambiguous linework arms.

Chains are descriptors, never replacement geometry. Only exactly shared pixel
endpoints are considered; no gap, four-way junction, or ambiguous branch is
connected. Classification still returns one score per original proposal.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .segment_review import geometry_features
from .contour_context import CONTEXT_FEATURE_NAMES, NEIGHBOR_FEATURE_NAMES, _fit_descriptor_model

CHAIN_SCHEMA = "ink-contour-touching-chain/1"
CHAIN_GEOMETRY_NAMES = ("log_length", "straightness", "mean_turn", "max_turn", "bbox_fill", "bbox_aspect_log")
CHAIN_CONTEXT_NAMES = (
    "coherence_r36", "normal_alignment_r36", "ink_fraction_r36",
    "coherence_r84", "normal_alignment_r84", "ink_fraction_r84",
    "profile_peaks", "profile_far_peaks", "profile_bilateral", "profile_peak_spacing_cv",
    "smooth_turn_mean", "smooth_turn_max",
)
CHAIN_EXTRA_NAMES = tuple("chain_delta_"+name for name in CHAIN_GEOMETRY_NAMES) + ("chain_other_member_fraction", "chain_member_count_log")
CHAIN_SHAPE_FEATURE_NAMES = CONTEXT_FEATURE_NAMES + CHAIN_EXTRA_NAMES
CHAIN_FEATURE_NAMES = CHAIN_SHAPE_FEATURE_NAMES + tuple("chain_delta_"+name for name in CHAIN_CONTEXT_NAMES)
PRIOR_CONTEXT_FEATURE_NAMES = ("legacy_log_odds",) + NEIGHBOR_FEATURE_NAMES
PRIOR_CHAIN_FEATURE_NAMES = PRIOR_CONTEXT_FEATURE_NAMES + CHAIN_EXTRA_NAMES + tuple("chain_delta_"+name for name in CHAIN_CONTEXT_NAMES)
SUPPORTED_FEATURE_SETS = (CHAIN_SHAPE_FEATURE_NAMES, CHAIN_FEATURE_NAMES, PRIOR_CONTEXT_FEATURE_NAMES, PRIOR_CHAIN_FEATURE_NAMES)


@dataclass(frozen=True)
class VisibleChain:
    member_indices: tuple[int, ...]
    points: tuple[tuple[float, float], ...]


def visible_chains(proposals, *, tangent_distance=8.0, maximum_deviation=20.0, ambiguity_margin=25.0):
    """Partition every original line into one unmodified context chain."""
    if (not all(math.isfinite(v) for v in (tangent_distance, maximum_deviation, ambiguity_margin))
            or tangent_distance <= 0 or not 0 <= maximum_deviation < 90 or not 0 <= ambiguity_margin <= 180):
        raise ValueError("invalid chain direction settings")
    lines = [tuple((float(x), float(y)) for x, y in proposal.points) for proposal in proposals]
    if any(len(line) < 2 or not all(math.isfinite(v) for point in line for v in point) for line in lines):
        raise ValueError("chain input requires finite polylines")
    if any(sum(math.dist(a, b) for a, b in zip(line, line[1:])) <= 0 for line in lines):
        raise ValueError("chain input lines need positive length")
    ends = {}
    for index, line in enumerate(lines):
        if line[0] != line[-1]:
            for side in (0, 1):
                ends.setdefault(line[0 if side == 0 else -1], []).append((index, side))

    def outward(key):
        index, side = key
        line = lines[index] if side == 0 else tuple(reversed(lines[index]))
        travelled, point = 0.0, line[-1]
        for a, b in zip(line, line[1:]):
            length = math.dist(a, b)
            if length and travelled+length >= tangent_distance:
                fraction = (tangent_distance-travelled)/length
                point = (a[0]+fraction*(b[0]-a[0]), a[1]+fraction*(b[1]-a[1]))
                travelled = tangent_distance
                break
            travelled += length
        vector = point[0]-line[0][0], point[1]-line[0][1]
        norm = math.hypot(*vector)
        return (vector[0]/norm, vector[1]/norm, travelled) if norm else (0.0, 0.0, 0.0)

    pairs = {}
    observed_degrees, ambiguous = {}, 0
    for attached in ends.values():
        degree = len(attached)
        observed_degrees[degree] = observed_degrees.get(degree, 0)+1
        if degree not in (2, 3) or len({index for index, _ in attached}) != degree:
            continue
        directions = {key: outward(key) for key in attached}
        if any(direction[2] < 2 for direction in directions.values()):
            continue
        ranked = []
        for offset, first in enumerate(attached):
            for second in attached[offset+1:]:
                a, b = directions[first], directions[second]
                cosine = -a[0]*b[0]-a[1]*b[1]
                deviation = math.degrees(math.acos(max(-1., min(1., cosine))))
                ranked.append((deviation, first, second))
        ranked.sort()
        best = ranked[0]
        if best[0] > maximum_deviation or (degree == 3 and ranked[1][0]-best[0] < ambiguity_margin):
            ambiguous += 1
            continue
        if min(directions[best[1]][2], directions[best[2]][2]) < tangent_distance:
            continue
        pairs[best[1]], pairs[best[2]] = best[2], best[1]

    consumed, chains = set(), []

    def consume(start):
        index, side = start
        members, points = [], []
        while index not in consumed:
            consumed.add(index)
            members.append(index)
            ordered = lines[index] if side == 0 else tuple(reversed(lines[index]))
            if points and points[-1] != ordered[0]:
                raise ValueError("context grouping attempted to cross a physical gap")
            points.extend(ordered if not points else ordered[1:])
            following = pairs.get((index, 1-side))
            if following is None:
                break
            index, side = following
        return VisibleChain(tuple(members), tuple(points))

    for index in range(len(lines)):
        if index in consumed:
            continue
        for side in (0, 1):
            if (index, side) not in pairs:
                chains.append(consume((index, side)))
                break
    for index in range(len(lines)):
        if index not in consumed:
            chains.append(consume((index, 0)))
    if sorted(member for chain in chains for member in chain.member_indices) != list(range(len(lines))):
        raise ValueError("chain membership is not a complete one-to-one partition")
    return chains, {"chains": len(chains), "members": len(lines), "paired_junctions": len(pairs)//2,
                    "ambiguous_junctions_not_paired": ambiguous, "observed_endpoint_degrees": observed_degrees,
                    "multi_member_chains": sum(len(chain.member_indices) > 1 for chain in chains),
                    "members_in_multi_member_chains": sum(len(chain.member_indices) for chain in chains if len(chain.member_indices) > 1),
                    "gap_connections_created": 0, "geometry_replacements": 0}


def chain_features(proposals, image_cache, local_contexts=None):
    chains, audit = visible_chains(proposals)
    result = [None]*len(proposals)
    for chain in chains:
        shape = geometry_features(chain.points)
        context = image_cache.describe(chain.points)
        total_length = sum(math.dist(a, b) for a, b in zip(chain.points, chain.points[1:]))
        for index in chain.member_indices:
            length = sum(math.dist(a, b) for a, b in zip(proposals[index].points, proposals[index].points[1:]))
            own_shape = geometry_features(proposals[index].points)
            own_context = local_contexts[index] if local_contexts is not None else image_cache.describe(proposals[index].points)
            # Deltas are exactly zero for an isolated line: duplicated local
            # features must not masquerade as a benefit from chain context.
            result[index] = {**{"chain_delta_"+name: shape[name]-own_shape[name] for name in CHAIN_GEOMETRY_NAMES},
                             **{"chain_delta_"+name: context[name]-own_context[name] for name in CHAIN_CONTEXT_NAMES},
                             "chain_member_count_log": math.log(len(chain.member_indices)),
                             "chain_other_member_fraction": max(0.0, 1.0-length/total_length)}
    return result, chains, audit


def fit_chain_classifier(records, *, feature_names=CHAIN_FEATURE_NAMES, l2=0.1):
    names = tuple(feature_names)
    if names not in SUPPORTED_FEATURE_SETS:
        raise ValueError("unsupported chain descriptor order")
    return _fit_descriptor_model(records, names, feature_schema=CHAIN_SCHEMA, l2=l2)


def validate_chain_classifier(model):
    if model.get("feature_schema") != CHAIN_SCHEMA or tuple(model.get("feature_names", ())) not in SUPPORTED_FEATURE_SETS:
        raise ValueError("unsupported chain model schema or feature order")
    count = len(model["feature_names"])
    for name in ("means", "scales", "coefficients"):
        if len(model.get(name, [])) != count or any(type(v) not in (int, float) or not math.isfinite(v) for v in model[name]):
            raise ValueError("chain model needs matching finite numeric parameters")
    if any(v <= 0 for v in model["scales"]) or type(model.get("intercept")) not in (int, float) or not math.isfinite(model["intercept"]):
        raise ValueError("chain model scales/intercept are invalid")
    return model


def legacy_log_odds(score):
    if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("legacy score must be finite and in [0, 1]")
    bounded = min(1.0-1e-9, max(1e-9, score))
    return max(-6.0, min(6.0, math.log(bounded/(1.0-bounded))))
