"""Conservative completion proposals between existing linework endpoints.

The completion backend does not classify black pixels as contours.  It starts
from pre-existing line proposals (or, later, human-approved contours), pairs
only mutually facing endpoints, and returns review-only bridge candidates.

Two completion modes are intentionally distinct:

``ink_path``
    A short route supported by a junction-free Ink centreline inside a narrow
    cubic-Hermite corridor.

``hermite_occlusion`` / ``hermite_gap``
    A smooth amodal completion used when text, a symbol, or a genuinely blank
    gap interrupts the visible centreline.  These candidates require longer
    anchors and tighter tangent agreement and receive a lower score.

No candidate is an asserted contour.  Semantic contour filtering still needs
human labels or a trained segmentation model; this module only prevents the
Ink backend from vectorising every black mark in the source image.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import heapq
import math
from typing import Iterable, Sequence

from .contours import Point
from .vectorization import simplify_polyline


COMPLETION_BACKEND_ID = "ridge_endpoints_ink_completion_v1"
_NEIGHBOURS = (
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
)


class CompletionBackendUnavailable(RuntimeError):
    """Raised when the optional numerical dependency is unavailable."""


@dataclass(frozen=True)
class ContourEndpointAnchor:
    """One outward-facing end of an existing review line."""

    anchor_id: str
    proposal_id: str
    endpoint_role: str
    point: Point
    outward_tangent: Point
    source_length_px: float
    source_confidence: float = 1.0

    def __post_init__(self) -> None:
        if not self.anchor_id or not self.proposal_id:
            raise ValueError("anchor and proposal identifiers must not be empty")
        if self.endpoint_role not in {"start", "end"}:
            raise ValueError("endpoint_role must be start or end")
        values = (*self.point, *self.outward_tangent, self.source_length_px, self.source_confidence)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("anchor values must be finite")
        norm = math.hypot(*self.outward_tangent)
        if abs(norm - 1.0) > 1e-6:
            raise ValueError("outward_tangent must be a unit vector")
        if self.source_length_px <= 0 or not 0 <= self.source_confidence <= 1:
            raise ValueError("source length and confidence are out of range")


@dataclass(frozen=True)
class ContourCompletionSettings:
    """Conservative defaults for 1,024-pixel historical-map review tiles."""

    minimum_anchor_length_px: float = 40.0
    minimum_interpolation_anchor_length_px: float = 60.0
    tangent_lookback_px: float = 10.0
    minimum_gap_px: float = 4.0
    maximum_ink_gap_px: float = 40.0
    maximum_interpolation_gap_px: float = 72.0
    maximum_ink_tangent_error_degrees: float = 35.0
    maximum_interpolation_tangent_error_degrees: float = 18.0
    endpoint_snap_radius_px: int = 4
    ink_corridor_radius_px: int = 9
    maximum_ink_path_stretch: float = 1.65
    maximum_internal_junction_pixels: int = 0
    existing_linework_buffer_px: int = 1
    minimum_missing_run_px: int = 3
    minimum_missing_fraction: float = 0.2
    border_margin_px: float = 12.0
    hermite_control_fraction: float = 0.36
    hermite_sample_spacing_px: float = 0.75
    ambiguity_score_margin: float = 0.06
    simplify_tolerance_px: float = 0.5

    def __post_init__(self) -> None:
        positive = (
            self.minimum_anchor_length_px,
            self.minimum_interpolation_anchor_length_px,
            self.tangent_lookback_px,
            self.minimum_gap_px,
            self.maximum_ink_gap_px,
            self.maximum_interpolation_gap_px,
            self.endpoint_snap_radius_px,
            self.ink_corridor_radius_px,
            self.maximum_ink_path_stretch,
            self.existing_linework_buffer_px,
            self.minimum_missing_run_px,
            self.border_margin_px,
            self.hermite_control_fraction,
            self.hermite_sample_spacing_px,
        )
        if any(float(value) <= 0 for value in positive):
            raise ValueError("completion distances and scales must be positive")
        if self.minimum_anchor_length_px > self.minimum_interpolation_anchor_length_px:
            raise ValueError("interpolation anchors cannot be shorter than Ink anchors")
        if not self.minimum_gap_px < self.maximum_ink_gap_px <= self.maximum_interpolation_gap_px:
            raise ValueError("completion gap limits must be increasing")
        if not 0 < self.maximum_interpolation_tangent_error_degrees <= self.maximum_ink_tangent_error_degrees < 90:
            raise ValueError("tangent error limits must be increasing and below 90 degrees")
        if self.maximum_internal_junction_pixels < 0:
            raise ValueError("junction limit must be non-negative")
        if not 0 <= self.minimum_missing_fraction <= 1 or not 0 <= self.ambiguity_score_margin <= 1:
            raise ValueError("completion fractions must be in [0, 1]")
        if self.simplify_tolerance_px < 0:
            raise ValueError("simplify tolerance must be non-negative")


@dataclass(frozen=True)
class ContourCompletionCandidate:
    """One review-only bridge between two source proposal endpoints."""

    completion_id: str
    first_anchor_id: str
    second_anchor_id: str
    first_proposal_id: str
    second_proposal_id: str
    mode: str
    points: tuple[Point, ...]
    direct_gap_px: float
    path_length_px: float
    maximum_tangent_error_degrees: float
    missing_fraction: float
    maximum_missing_run_px: int
    ink_support_fraction: float
    junction_pixels: int
    score: float
    backend: str = COMPLETION_BACKEND_ID

    def __post_init__(self) -> None:
        if self.mode not in {"ink_path", "hermite_occlusion", "hermite_gap"}:
            raise ValueError("unknown completion mode")
        if len(self.points) < 2 or self.direct_gap_px <= 0 or self.path_length_px <= 0:
            raise ValueError("completion geometry must be a non-empty line")
        if not 0 <= self.score <= 1 or not 0 <= self.ink_support_fraction <= 1:
            raise ValueError("completion scores must be in [0, 1]")


@dataclass(frozen=True)
class ContourCompletionResult:
    """Accepted proposals plus auditable rejection counts."""

    candidates: tuple[ContourCompletionCandidate, ...]
    rejection_counts: dict[str, int]
    anchor_count: int
    eligible_anchor_count: int
    evaluated_pair_count: int
    backend: str = COMPLETION_BACKEND_ID


def _numpy():
    try:
        import numpy as np
    except ImportError as error:
        raise CompletionBackendUnavailable("contour completion requires NumPy") from error
    return np


def _unit(vector: Point) -> Point | None:
    norm = math.hypot(*vector)
    if norm <= 1e-9:
        return None
    return vector[0] / norm, vector[1] / norm


def _distance(first: Point, second: Point) -> float:
    return math.hypot(second[0] - first[0], second[1] - first[1])


def _polyline_length(points: Sequence[Point]) -> float:
    return sum(_distance(first, second) for first, second in zip(points, points[1:]))


def _endpoint_tangent(points: Sequence[Point], *, start: bool, lookback_px: float) -> Point | None:
    ordered = list(points if start else reversed(points))
    endpoint = ordered[0]
    travelled = 0.0
    interior = ordered[-1]
    for first, second in zip(ordered, ordered[1:]):
        segment = _distance(first, second)
        if segment <= 1e-9:
            continue
        if travelled + segment >= lookback_px:
            fraction = (lookback_px - travelled) / segment
            interior = (
                first[0] + (second[0] - first[0]) * fraction,
                first[1] + (second[1] - first[1]) * fraction,
            )
            break
        travelled += segment
        interior = second
    return _unit((endpoint[0] - interior[0], endpoint[1] - interior[1]))


def anchors_from_polyline(
    proposal_id: str,
    points: Sequence[Point],
    *,
    source_length_px: float | None = None,
    source_confidence: float = 1.0,
    tangent_lookback_px: float = 10.0,
) -> tuple[ContourEndpointAnchor, ...]:
    """Build start/end anchors while ignoring closed or degenerate lines."""

    values = tuple((float(x), float(y)) for x, y in points)
    if len(values) < 2 or not proposal_id:
        return ()
    length = float(source_length_px) if source_length_px is not None else _polyline_length(values)
    if length <= 0 or _distance(values[0], values[-1]) <= 1e-6:
        return ()
    anchors = []
    for role, start in (("start", True), ("end", False)):
        tangent = _endpoint_tangent(values, start=start, lookback_px=tangent_lookback_px)
        if tangent is None:
            continue
        anchors.append(
            ContourEndpointAnchor(
                anchor_id=f"{proposal_id}:{role}",
                proposal_id=proposal_id,
                endpoint_role=role,
                point=values[0] if start else values[-1],
                outward_tangent=tangent,
                source_length_px=length,
                source_confidence=float(source_confidence),
            )
        )
    return tuple(anchors)


def _hermite_curve(first: ContourEndpointAnchor, second: ContourEndpointAnchor, settings, gap: float) -> tuple[Point, ...]:
    control = gap * settings.hermite_control_fraction
    p0, p3 = first.point, second.point
    p1 = (p0[0] + first.outward_tangent[0] * control, p0[1] + first.outward_tangent[1] * control)
    p2 = (p3[0] + second.outward_tangent[0] * control, p3[1] + second.outward_tangent[1] * control)
    sample_count = max(3, int(math.ceil(gap / settings.hermite_sample_spacing_px)) + 1)
    points = []
    for index in range(sample_count):
        t = index / (sample_count - 1)
        inverse = 1.0 - t
        points.append(
            (
                inverse**3 * p0[0] + 3 * inverse**2 * t * p1[0] + 3 * inverse * t**2 * p2[0] + t**3 * p3[0],
                inverse**3 * p0[1] + 3 * inverse**2 * t * p1[1] + 3 * inverse * t**2 * p2[1] + t**3 * p3[1],
            )
        )
    return tuple(points)


def _line_pixels(points: Sequence[Point]) -> tuple[tuple[int, int], ...]:
    pixels = []
    for first, second in zip(points, points[1:]):
        span = max(abs(second[0] - first[0]), abs(second[1] - first[1]))
        steps = max(1, int(math.ceil(span * 2.0)))
        for index in range(steps + 1):
            fraction = index / steps
            pixel = (
                int(round(first[0] + (second[0] - first[0]) * fraction)),
                int(round(first[1] + (second[1] - first[1]) * fraction)),
            )
            if not pixels or pixels[-1] != pixel:
                pixels.append(pixel)
    return tuple(pixels)


def rasterize_polylines(shape: tuple[int, int], polylines: Iterable[Sequence[Point]]):
    """Rasterize review vectors on their source-pixel grid without GIS deps."""

    np = _numpy()
    mask = np.zeros(shape, dtype=bool)
    height, width = shape
    for points in polylines:
        for x, y in _line_pixels(points):
            if 0 <= x < width and 0 <= y < height:
                mask[y, x] = True
    return mask


def _dilate(np, mask, radius: int):
    active = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return active.copy()
    height, width = active.shape
    output = np.zeros(active.shape, dtype=bool)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            source_y = slice(max(0, -dy), min(height, height - dy))
            target_y = slice(max(0, dy), min(height, height + dy))
            source_x = slice(max(0, -dx), min(width, width - dx))
            target_x = slice(max(0, dx), min(width, width + dx))
            output[target_y, target_x] |= active[source_y, source_x]
    return output


def _corridor_mask(np, shape, guide: Sequence[Point], radius: int):
    return _dilate(np, rasterize_polylines(shape, (guide,)), radius)


def _crossing_number(np, centerline):
    active = np.asarray(centerline, dtype=bool)
    padded = np.pad(active, 1, mode="constant", constant_values=False)
    neighbours = (
        padded[:-2, 1:-1], padded[:-2, 2:], padded[1:-1, 2:], padded[2:, 2:],
        padded[2:, 1:-1], padded[2:, :-2], padded[1:-1, :-2], padded[:-2, :-2],
    )
    crossings = np.zeros(active.shape, dtype=np.uint8)
    for current, following in zip(neighbours, neighbours[1:] + neighbours[:1]):
        crossings += (~current & following).astype(np.uint8)
    crossings[~active] = 0
    return crossings


def _snap_to_centerline(centerline, point: Point, radius: int, corridor) -> tuple[int, int] | None:
    height, width = centerline.shape
    center_x, center_y = point
    candidates = []
    for y in range(max(0, int(round(center_y)) - radius), min(height, int(round(center_y)) + radius + 1)):
        for x in range(max(0, int(round(center_x)) - radius), min(width, int(round(center_x)) + radius + 1)):
            distance = math.hypot(x - center_x, y - center_y)
            if distance <= radius and centerline[y, x] and corridor[y, x]:
                candidates.append((distance, y, x))
    if not candidates:
        return None
    _distance_value, y, x = min(candidates)
    return x, y


def _shortest_centerline_path(centerline, corridor, start, goal, maximum_length: float):
    distances = {start: 0.0}
    previous = {}
    queue = [(0.0, start)]
    height, width = centerline.shape
    while queue:
        cost, current = heapq.heappop(queue)
        if cost != distances[current]:
            continue
        if current == goal:
            path = [current]
            while current in previous:
                current = previous[current]
                path.append(current)
            return tuple(reversed(path)), cost
        if cost > maximum_length:
            continue
        x, y = current
        for dx, dy in _NEIGHBOURS:
            next_point = x + dx, y + dy
            next_x, next_y = next_point
            if not 0 <= next_x < width or not 0 <= next_y < height:
                continue
            if not centerline[next_y, next_x] or not corridor[next_y, next_x]:
                continue
            next_cost = cost + (math.sqrt(2.0) if dx and dy else 1.0)
            if next_cost < distances.get(next_point, math.inf):
                distances[next_point] = next_cost
                previous[next_point] = current
                heapq.heappush(queue, (next_cost, next_point))
    return None, None


def _direction_error_degrees(tangent: Point, direction: Point) -> float:
    direction_unit = _unit(direction)
    if direction_unit is None:
        return 180.0
    dot = max(-1.0, min(1.0, tangent[0] * direction_unit[0] + tangent[1] * direction_unit[1]))
    return math.degrees(math.acos(dot))


def _path_endpoint_error(path: Sequence[Point], first, second, lookahead_px: float = 5.0) -> float:
    def direction(values):
        origin = values[0]
        for point in values[1:]:
            if _distance(origin, point) >= lookahead_px:
                return point[0] - origin[0], point[1] - origin[1]
        last = values[-1]
        return last[0] - origin[0], last[1] - origin[1]

    first_error = _direction_error_degrees(first.outward_tangent, direction(path))
    second_error = _direction_error_degrees(second.outward_tangent, direction(tuple(reversed(path))))
    return max(first_error, second_error)


def _maximum_true_run(values: Sequence[bool]) -> int:
    maximum = current = 0
    for value in values:
        current = current + 1 if value else 0
        maximum = max(maximum, current)
    return maximum


def _path_metrics(np, pixels, ink_score, buffered_existing, junction_mask):
    height, width = ink_score.shape
    in_bounds = [(x, y) for x, y in pixels if 0 <= x < width and 0 <= y < height]
    if not in_bounds:
        return 0.0, 0, 0.0, 0
    missing = [not buffered_existing[y, x] for x, y in in_bounds]
    support = [float(ink_score[y, x]) >= 0.04 for x, y in in_bounds]
    junctions = sum(bool(junction_mask[y, x]) for x, y in in_bounds[2:-2])
    return (
        float(sum(missing) / len(missing)),
        _maximum_true_run(missing),
        float(sum(support) / len(support)),
        int(junctions),
    )


def _candidate_score(first, second, gap, tangent_error, support, mode, settings):
    anchor_score = (first.source_confidence + second.source_confidence) / 2.0
    tangent_limit = (
        settings.maximum_ink_tangent_error_degrees
        if mode == "ink_path"
        else settings.maximum_interpolation_tangent_error_degrees
    )
    tangent_score = max(0.0, 1.0 - tangent_error / tangent_limit)
    gap_limit = settings.maximum_ink_gap_px if mode == "ink_path" else settings.maximum_interpolation_gap_px
    gap_score = max(0.0, 1.0 - gap / gap_limit)
    mode_base = 0.58 if mode == "ink_path" else (0.42 if mode == "hermite_occlusion" else 0.34)
    value = mode_base + 0.16 * anchor_score + 0.12 * tangent_score + 0.08 * gap_score + 0.06 * support
    return max(0.0, min(1.0, value))


def _inside_border(anchor, shape, margin) -> bool:
    height, width = shape
    x, y = anchor.point
    return margin <= x <= width - 1 - margin and margin <= y <= height - 1 - margin


def propose_contour_completions(
    anchors: Iterable[ContourEndpointAnchor],
    ink_centerline,
    ink_score,
    *,
    existing_linework=None,
    settings: ContourCompletionSettings = ContourCompletionSettings(),
) -> ContourCompletionResult:
    """Return one-to-one, review-only completions between compatible anchors."""

    if not isinstance(settings, ContourCompletionSettings):
        raise TypeError("settings must be ContourCompletionSettings")
    np = _numpy()
    centerline = np.asarray(ink_centerline, dtype=bool)
    scores = np.asarray(ink_score, dtype=np.float32)
    if centerline.ndim != 2 or scores.shape != centerline.shape:
        raise ValueError("Ink centerline and score must share one 2-D shape")
    if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("Ink score must contain finite values in [0, 1]")
    existing = np.zeros(centerline.shape, dtype=bool) if existing_linework is None else np.asarray(existing_linework, dtype=bool)
    if existing.shape != centerline.shape:
        raise ValueError("existing linework must match the Ink arrays")
    buffered_existing = _dilate(np, existing, settings.existing_linework_buffer_px)
    topology = _crossing_number(np, centerline)
    junction_mask = _dilate(np, topology >= 3, 2)
    rejection_counts = Counter()
    all_anchors = tuple(sorted(anchors, key=lambda item: item.anchor_id))
    eligible = []
    for anchor in all_anchors:
        if anchor.source_length_px < settings.minimum_anchor_length_px:
            rejection_counts["short_anchor"] += 1
        elif not _inside_border(anchor, centerline.shape, settings.border_margin_px):
            rejection_counts["border_anchor"] += 1
        else:
            eligible.append(anchor)

    raw_candidates = []
    evaluated_pairs = 0
    for first_index, first in enumerate(eligible):
        for second in eligible[first_index + 1:]:
            if first.proposal_id == second.proposal_id:
                continue
            gap = _distance(first.point, second.point)
            if gap > settings.maximum_interpolation_gap_px:
                continue
            evaluated_pairs += 1
            if gap < settings.minimum_gap_px:
                rejection_counts["gap_too_short"] += 1
                continue
            gap_vector = (second.point[0] - first.point[0], second.point[1] - first.point[1])
            reverse_gap = (-gap_vector[0], -gap_vector[1])
            tangent_error = max(
                _direction_error_degrees(first.outward_tangent, gap_vector),
                _direction_error_degrees(second.outward_tangent, reverse_gap),
            )
            if tangent_error > settings.maximum_ink_tangent_error_degrees:
                rejection_counts["tangent_mismatch"] += 1
                continue
            guide = _hermite_curve(first, second, settings, gap)
            guide_pixels = _line_pixels(guide)
            mode = None
            output_points = None
            output_pixels = None
            path_length = None
            path_junctions = 0
            if gap <= settings.maximum_ink_gap_px:
                corridor = _corridor_mask(np, centerline.shape, guide, settings.ink_corridor_radius_px)
                start = _snap_to_centerline(centerline, first.point, settings.endpoint_snap_radius_px, corridor)
                goal = _snap_to_centerline(centerline, second.point, settings.endpoint_snap_radius_px, corridor)
                if start is not None and goal is not None and start != goal:
                    path, length = _shortest_centerline_path(
                        centerline,
                        corridor,
                        start,
                        goal,
                        gap * settings.maximum_ink_path_stretch + 2 * settings.endpoint_snap_radius_px,
                    )
                    if path is not None and length / gap <= settings.maximum_ink_path_stretch:
                        path_points = (
                            first.point,
                            *(tuple((float(x), float(y)) for x, y in path)),
                            second.point,
                        )
                        path_error = _path_endpoint_error(path_points, first, second)
                        _missing, _run, _support, path_junctions = _path_metrics(
                            np, path, scores, buffered_existing, junction_mask
                        )
                        if (
                            path_error <= settings.maximum_ink_tangent_error_degrees
                            and path_junctions <= settings.maximum_internal_junction_pixels
                        ):
                            mode = "ink_path"
                            output_points = path_points
                            output_pixels = _line_pixels(path_points)
                            path_length = _polyline_length(path_points)
                            tangent_error = max(tangent_error, path_error)

            if mode is None:
                if (
                    first.source_length_px < settings.minimum_interpolation_anchor_length_px
                    or second.source_length_px < settings.minimum_interpolation_anchor_length_px
                ):
                    rejection_counts["interpolation_anchor_too_short"] += 1
                    continue
                if tangent_error > settings.maximum_interpolation_tangent_error_degrees:
                    rejection_counts["interpolation_tangent_mismatch"] += 1
                    continue
                output_points = guide
                output_pixels = guide_pixels
                path_length = _polyline_length(guide)
                nearby_junctions = sum(
                    bool(junction_mask[y, x])
                    for x, y in guide_pixels
                    if 0 <= x < centerline.shape[1] and 0 <= y < centerline.shape[0]
                )
                guide_support = sum(
                    float(scores[y, x]) >= 0.04
                    for x, y in guide_pixels
                    if 0 <= x < centerline.shape[1] and 0 <= y < centerline.shape[0]
                ) / max(1, len(guide_pixels))
                mode = "hermite_occlusion" if nearby_junctions or guide_support >= 0.15 else "hermite_gap"
                path_junctions = int(nearby_junctions)

            missing_fraction, missing_run, support, measured_junctions = _path_metrics(
                np, output_pixels, scores, buffered_existing, junction_mask
            )
            if missing_run < settings.minimum_missing_run_px or missing_fraction < settings.minimum_missing_fraction:
                rejection_counts["already_covered"] += 1
                continue
            junction_count = max(path_junctions, measured_junctions)
            score = _candidate_score(first, second, gap, tangent_error, support, mode, settings)
            simplified = simplify_polyline(tuple(output_points), settings.simplify_tolerance_px)
            raw_candidates.append(
                ContourCompletionCandidate(
                    completion_id="pending",
                    first_anchor_id=first.anchor_id,
                    second_anchor_id=second.anchor_id,
                    first_proposal_id=first.proposal_id,
                    second_proposal_id=second.proposal_id,
                    mode=mode,
                    points=simplified,
                    direct_gap_px=gap,
                    path_length_px=float(path_length),
                    maximum_tangent_error_degrees=tangent_error,
                    missing_fraction=missing_fraction,
                    maximum_missing_run_px=missing_run,
                    ink_support_fraction=support,
                    junction_pixels=junction_count,
                    score=score,
                )
            )

    by_anchor = defaultdict(list)
    for candidate in raw_candidates:
        by_anchor[candidate.first_anchor_id].append(candidate)
        by_anchor[candidate.second_anchor_id].append(candidate)
    ambiguous = set()
    for anchor_id, values in by_anchor.items():
        ranked = sorted(values, key=lambda item: (-item.score, item.direct_gap_px, item.first_anchor_id, item.second_anchor_id))
        if len(ranked) > 1 and ranked[0].score - ranked[1].score <= settings.ambiguity_score_margin:
            ambiguous.add(anchor_id)

    accepted = []
    used = set()
    ordered = sorted(
        raw_candidates,
        key=lambda item: (-item.score, item.direct_gap_px, item.first_anchor_id, item.second_anchor_id),
    )
    for candidate in ordered:
        anchor_ids = {candidate.first_anchor_id, candidate.second_anchor_id}
        if anchor_ids & ambiguous:
            rejection_counts["ambiguous_endpoint"] += 1
            continue
        if anchor_ids & used:
            rejection_counts["endpoint_reused"] += 1
            continue
        used.update(anchor_ids)
        accepted.append(candidate)

    finalized = tuple(
        ContourCompletionCandidate(
            **{
                **candidate.__dict__,
                "completion_id": f"contour-link-{index}",
            }
        )
        for index, candidate in enumerate(accepted, 1)
    )
    return ContourCompletionResult(
        candidates=finalized,
        rejection_counts=dict(sorted(rejection_counts.items())),
        anchor_count=len(all_anchors),
        eligible_anchor_count=len(eligible),
        evaluated_pair_count=evaluated_pairs,
    )


__all__ = [
    "COMPLETION_BACKEND_ID",
    "CompletionBackendUnavailable",
    "ContourCompletionCandidate",
    "ContourCompletionResult",
    "ContourCompletionSettings",
    "ContourEndpointAnchor",
    "anchors_from_polyline",
    "propose_contour_completions",
    "rasterize_polylines",
]
