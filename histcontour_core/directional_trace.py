"""Optional, bounded tracing of observed ink between explicit endpoints.

This prototype uses continuous Ink support and incoming heading as search state.
It does not identify contours, choose endpoints, or approve geometry. Small
unsupported spans remain explicit in the result; larger gaps cause abstention.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from numbers import Integral, Real
from typing import Sequence


Point = tuple[float, float]
_DIRECTIONS = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))
_STEPS = tuple(math.hypot(dx, dy) for dx, dy in _DIRECTIONS)
_UNIT = tuple((dx / step, dy / step) for (dx, dy), step in zip(_DIRECTIONS, _STEPS))


class DirectionalTraceError(ValueError):
    """An explicit trace request or configuration is invalid or too large."""


@dataclass(frozen=True)
class DirectionalTraceConfig:
    max_window_size_px: int = 256
    padding_px: int = 24
    max_states: int = 100_000
    max_queue_entries: int = 200_000
    max_reference_points: int = 256
    max_corridor_evaluations: int = 8_000_000
    corridor_radius_px: float = 8.0
    minimum_support: float = 0.08
    minimum_supported_fraction: float = 0.95
    max_unsupported_run_px: float = 2.0
    max_length_ratio: float = 3.0
    support_weight: float = 2.0
    tangent_weight: float = 3.0
    turn_weight: float = 4.0
    text_weight: float = 6.0
    reference_weight: float = 2.0

    def __post_init__(self):
        limits = {
            "max_window_size_px": (4, 512), "padding_px": (0, 128),
            "max_states": (1, 2_000_000), "max_queue_entries": (1, 4_000_000),
            "max_reference_points": (2, 2048), "max_corridor_evaluations": (1, 32_000_000),
        }
        for name, (low, high) in limits.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or not low <= value <= high:
                raise DirectionalTraceError(f"{name} must be an integer in [{low}, {high}]")
        ranges = {
            "corridor_radius_px": (0.25, 64.), "minimum_support": (0.001, 1.),
            "minimum_supported_fraction": (0.5, 1.), "max_unsupported_run_px": (0., 16.),
            "max_length_ratio": (1., 10.),
            **{name: (0., 100.) for name in ("support_weight", "tangent_weight", "turn_weight", "text_weight", "reference_weight")},
        }
        for name, (low, high) in ranges.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not low <= value <= high:
                raise DirectionalTraceError(f"{name} must be finite in [{low}, {high}]")


@dataclass(frozen=True)
class UnsupportedSpan:
    """An unsupported interval measured along the returned polyline."""

    start_distance_px: float
    end_distance_px: float

    @property
    def length_px(self) -> float:
        return self.end_distance_px - self.start_distance_px


@dataclass(frozen=True)
class DirectionalTraceAudit:
    """Numerical evidence audit; corridor distance is a conservative upper bound."""

    reason: str
    window_bounds: tuple[int, int, int, int]
    discovered_states: int = 0
    expanded_states: int = 0
    queue_peak: int = 0
    cost: float | None = None
    path_length_px: float = 0.0
    supported_length_fraction: float = 0.0
    mean_support: float = 0.0
    maximum_unsupported_run_px: float = 0.0
    unsupported_spans: tuple[UnsupportedSpan, ...] = ()
    maximum_corridor_distance_px: float | None = None
    used_reference: bool = False
    used_text_avoidance: bool = False
    used_exclusion: bool = False


@dataclass(frozen=True)
class DirectionalTraceResult:
    """Copied points and evidence audit; an abstention has no output geometry.

    ``observed`` means the sampled route meets the configured ink threshold.
    ``supported_with_gaps`` explicitly retains short unsupported intervals.
    Neither status is a semantic line classification or a human decision.
    """

    points: tuple[Point, ...]
    status: str
    audit: DirectionalTraceAudit


def _point(value, name: str) -> Point:
    try:
        if isinstance(value, (str, bytes)) or len(value) != 2:
            raise ValueError
        point = float(value[0]), float(value[1])
        if any(isinstance(item, bool) for item in value) or not all(math.isfinite(item) for item in point):
            raise ValueError
        return point
    except (TypeError, ValueError, OverflowError) as error:
        raise DirectionalTraceError(f"{name} must be two finite pixel coordinates") from error


def _point_segment_distance(x, y, first, second):
    dx, dy = second[0] - first[0], second[1] - first[1]
    denominator = dx * dx + dy * dy
    if denominator == 0:
        return ((x - first[0]) ** 2 + (y - first[1]) ** 2) ** 0.5
    fraction = ((x - first[0]) * dx + (y - first[1]) * dy) / denominator
    fraction = fraction.clip(0., 1.) if hasattr(fraction, "clip") else max(0., min(1., fraction))
    return ((x - first[0] - fraction * dx) ** 2 + (y - first[1] - fraction * dy) ** 2) ** 0.5


def _sample(array, x: float, y: float) -> float:
    """Bilinear support in the local ROI, including exact subpixel endpoints."""
    x, y = max(0., min(array.shape[1] - 1., x)), max(0., min(array.shape[0] - 1., y))
    left, top = int(math.floor(x)), int(math.floor(y))
    right, bottom = min(left + 1, array.shape[1] - 1), min(top + 1, array.shape[0] - 1)
    fx, fy = x - left, y - top
    return float((1. - fy) * ((1. - fx) * array[top, left] + fx * array[top, right])
                 + fy * ((1. - fx) * array[bottom, left] + fx * array[bottom, right]))


def _segment_splits(first, second, offset=0.):
    splits = {0., 1.}
    for a, b in zip(first, second):
        if a == b:
            continue
        for index in range(math.ceil(min(a, b) - offset), math.floor(max(a, b) - offset) + 1):
            fraction = (index + offset - a) / (b - a)
            if 0. < fraction < 1.:
                splits.add(fraction)
    return sorted(splits)


def _support_intervals(array, first, second, threshold):
    """Integrate bilinear support and find threshold crossings exactly.

    On each pixel cell a straight segment has quadratic support. Splitting at
    its real roots includes white vertices and subpixel dips that fixed midpoint
    sampling could miss. Returned bounds are fractions of the whole segment.
    """
    def value(t):
        return _sample(array, first[0] + t * (second[0] - first[0]), first[1] + t * (second[1] - first[1]))

    splits = _segment_splits(first, second)
    for begin, end in zip(splits, splits[1:]):
        low, middle, high = value(begin), value((begin + end) / 2.), value(end)
        quadratic = 2. * (low + high - 2. * middle)
        linear = high - low - quadratic
        constant = low - threshold
        roots = [0., 1.]
        if abs(quadratic) < 1e-12:
            if abs(linear) >= 1e-12:
                roots.append(-constant / linear)
        else:
            discriminant = linear * linear - 4. * quadratic * constant
            if discriminant >= 0.:
                root = math.sqrt(discriminant)
                roots.extend(((-linear - root) / (2. * quadratic), (-linear + root) / (2. * quadratic)))
        roots = sorted({max(0., min(1., root)) for root in roots if -1e-12 <= root <= 1. + 1e-12})
        for a, b in zip(roots, roots[1:]):
            if b <= a:
                continue
            mean = low + linear * (a + b) / 2. + quadratic * (a * a + a * b + b * b) / 3.
            mid = (a + b) / 2.
            supported = low + linear * mid + quadratic * mid * mid >= threshold
            yield begin + a * (end - begin), begin + b * (end - begin), mean, supported


def _segment_enters_exclusion(excluded, first, second):
    # Pixel masks occupy closed unit cells. Check half-pixel transitions and
    # both sides of exact boundary touches, including subpixel endpoint edits.
    splits = _segment_splits(first, second, offset=0.5)
    positions = splits + [(a + b) / 2. for a, b in zip(splits, splits[1:])]
    for t in positions:
        x, y = (a + t * (b - a) for a, b in zip(first, second))
        for ix in {int(math.floor(x + .5 - 1e-10)), int(math.floor(x + .5 + 1e-10))}:
            for iy in {int(math.floor(y + .5 - 1e-10)), int(math.floor(y + .5 + 1e-10))}:
                if 0 <= ix < excluded.shape[1] and 0 <= iy < excluded.shape[0] and excluded[iy, ix]:
                    return True
    return False


def _corridor_bound(first, second, reference):
    """Conservative maximum distance over an entire output segment."""
    length = math.dist(first, second)
    count = max(1, int(math.ceil(length * 4.)))
    previous = None
    upper = 0.
    for index in range(count + 1):
        fraction = index / count
        x, y = (a + fraction * (b - a) for a, b in zip(first, second))
        nearest = min((_point_segment_distance(x, y, a, b), i) for i, (a, b) in enumerate(zip(reference, reference[1:])))
        upper = max(upper, nearest[0])
        if previous is not None:
            # Distance to one segment is convex. When the nearest segment
            # changes, the distance field's 1-Lipschitz bound remains safe.
            interval_bound = max(previous[0], nearest[0]) if previous[1] == nearest[1] else (previous[0] + nearest[0] + length / count) / 2.
            upper = max(upper, interval_bound)
        previous = nearest
    return upper


def trace_observed_ink(
    evidence,
    start_xy: Sequence[float],
    end_xy: Sequence[float],
    *,
    reference_path: Sequence[Sequence[float]] | None = None,
    text_avoidance_score=None,
    exclusion_mask=None,
    config: DirectionalTraceConfig = DirectionalTraceConfig(),
) -> DirectionalTraceResult:
    """Trace continuous support with finite ROI, heading, and gap budgets.

    Reference points define an optional hard distance corridor and soft route
    preference. Text scores are soft costs; caller-supplied exclusions are hard
    barriers, including diagonal corner crossings. Arrays are never mutated.
    Search failure and exhausted state/queue budgets return an audited abstention.
    Invalid inputs and requests exceeding allocation/work limits raise an error.
    """
    import numpy as np

    if not isinstance(config, DirectionalTraceConfig):
        raise DirectionalTraceError("config must be DirectionalTraceConfig")
    names = ("support_score", "tangent_x", "tangent_y", "coherence")
    try:
        arrays = {name: np.asarray(getattr(evidence, name)) for name in names}
    except (AttributeError, TypeError, ValueError) as error:
        raise DirectionalTraceError("evidence must provide continuous support, tangents, and coherence") from error
    shape = arrays["support_score"].shape
    if len(shape) != 2 or min(shape) < 1 or any(array.shape != shape for array in arrays.values()):
        raise DirectionalTraceError("Ink arrays must share a non-empty 2D shape")
    height, width = shape
    start, end = _point(start_xy, "start_xy"), _point(end_xy, "end_xy")
    if any(not (0. <= x <= width - 1 and 0. <= y <= height - 1) for x, y in (start, end)):
        raise DirectionalTraceError("explicit endpoints leave the evidence extent")
    reference = None
    if reference_path is not None:
        if not hasattr(reference_path, "__len__") or not 2 <= len(reference_path) <= config.max_reference_points:
            raise DirectionalTraceError("reference_path exceeds its point budget or has fewer than two points")
        reference = tuple(_point(point, "reference point") for point in reference_path)
        if any(not (0. <= x <= width - 1 and 0. <= y <= height - 1) for x, y in reference):
            raise DirectionalTraceError("reference_path leaves the evidence extent")
    anchors = (start, end) + (reference or ())
    left, right = math.floor(min(x for x, _ in anchors)), math.ceil(max(x for x, _ in anchors))
    top, bottom = math.floor(min(y for _, y in anchors)), math.ceil(max(y for _, y in anchors))
    if max(right - left + 1, bottom - top + 1) > config.max_window_size_px:
        raise DirectionalTraceError("explicit trace exceeds max_window_size_px")
    pad_x = min(config.padding_px, (config.max_window_size_px - (right - left + 1)) // 2)
    pad_y = min(config.padding_px, (config.max_window_size_px - (bottom - top + 1)) // 2)
    x0, y0 = max(0, left - pad_x), max(0, top - pad_y)
    x1, y1 = min(width, right + pad_x + 1), min(height, bottom + pad_y + 1)
    bounds = x0, y0, x1, y1
    try:
        local = {name: np.array(array[y0:y1, x0:x1], dtype=np.float64, copy=True) for name, array in arrays.items()}
    except (ValueError, TypeError) as error:
        raise DirectionalTraceError("Ink evidence must be numerical") from error
    if any(not np.isfinite(array).all() for array in local.values()):
        raise DirectionalTraceError("Ink evidence must be finite inside the trace ROI")
    support, tx, ty, coherence = (local[name] for name in names)
    if any(np.any((array < 0.) | (array > 1.)) for array in (support, coherence)):
        raise DirectionalTraceError("support and coherence must be in [0, 1]")
    norm = np.hypot(tx, ty)
    coherence[norm < 1e-12] = 0.
    np.divide(tx, norm, out=tx, where=norm >= 1e-12)
    np.divide(ty, norm, out=ty, where=norm >= 1e-12)

    def optional_array(value, name, *, mask=False):
        if value is None:
            return np.zeros(support.shape, dtype=bool if mask else np.float64)
        array = np.asarray(value)
        if array.shape != shape:
            raise DirectionalTraceError(f"{name} shape must match Ink evidence")
        if mask and array.dtype.kind != "b":
            raise DirectionalTraceError("exclusion_mask must be boolean")
        try:
            cropped = np.array(array[y0:y1, x0:x1], dtype=bool if mask else np.float64, copy=True)
        except (ValueError, TypeError) as error:
            raise DirectionalTraceError(f"{name} must be numerical") from error
        if not mask and (not np.isfinite(cropped).all() or np.any((cropped < 0.) | (cropped > 1.))):
            raise DirectionalTraceError(f"{name} must be finite in [0, 1]")
        return cropped

    text = optional_array(text_avoidance_score, "text_avoidance_score")
    excluded = optional_array(exclusion_mask, "exclusion_mask", mask=True)
    reference_distance = np.zeros(support.shape, dtype=np.float64)
    if reference is not None:
        if support.size * (len(reference) - 1) > config.max_corridor_evaluations:
            raise DirectionalTraceError("reference corridor exceeds max_corridor_evaluations")
        grid_y, grid_x = np.mgrid[y0:y1, x0:x1]
        reference_distance.fill(np.inf)
        for first, second in zip(reference, reference[1:]):
            np.minimum(reference_distance, _point_segment_distance(grid_x, grid_y, first, second), out=reference_distance)
    allowed = ~excluded
    if reference is not None:
        allowed &= reference_distance <= config.corridor_radius_px
    base_audit = dict(window_bounds=bounds, used_reference=reference is not None,
                      used_text_avoidance=text_avoidance_score is not None, used_exclusion=exclusion_mask is not None)

    def abstain(reason, **audit):
        return DirectionalTraceResult((), "abstained", DirectionalTraceAudit(reason=reason, **base_audit, **audit))

    if start == end:
        return abstain("coincident_endpoints")
    source = int(round(start[0])) - x0, int(round(start[1])) - y0
    target = int(round(end[0])) - x0, int(round(end[1])) - y0
    if not allowed[source[1], source[0]] or not allowed[target[1], target[0]]:
        return abstain("endpoint_excluded_or_outside_corridor")
    if not np.any(support[allowed] >= config.minimum_support):
        return abstain("no_ink_support")
    local_height, local_width = support.shape
    initial = (source[1] * local_width + source[0], 8, 0)
    distances, previous = {initial: 0.}, {}
    queue = [(math.dist(source, target), 0., initial)]
    expanded, queue_peak = 0, 1
    destination = None
    # Only unsupported intervals consume the run budget. Rounding their lengths
    # up to quarter pixels is conservative without counting an entire diagonal
    # step as a white gap when most of that step still has observed support.
    maximum_gap_units = int(math.floor(config.max_unsupported_run_px * 4. + 1e-9))
    while queue:
        _priority, cost, state = heapq.heappop(queue)
        if cost != distances.get(state):
            continue
        expanded += 1
        index, incoming, run_units = state
        y, x = divmod(index, local_width)
        if (x, y) == target:
            destination = state
            break
        for direction, (dx, dy) in enumerate(_DIRECTIONS):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < local_width and 0 <= ny < local_height) or not allowed[ny, nx]:
                continue
            if incoming != 8 and (direction - incoming) % 8 == 4:
                continue
            if dx and dy and (excluded[y, nx] or excluded[ny, x]):
                continue
            step = _STEPS[direction]
            minimum_endpoint_support = min(support[y, x], support[ny, nx])
            fully_supported = minimum_endpoint_support >= config.minimum_support * (2. if dx and dy else 1.)
            next_run = 0 if fully_supported else run_units
            if not fully_supported:
                for begin, finish, _mean, supported in _support_intervals(support, (x, y), (nx, ny), config.minimum_support):
                    next_run = 0 if supported else next_run + int(math.ceil((finish - begin) * step * 4. - 1e-12))
                    if next_run > maximum_gap_units:
                        break
            if next_run > maximum_gap_units:
                continue
            ux, uy = _UNIT[direction]
            alignment = 0.5 * ((1. - min(1., abs(ux * tx[y, x] + uy * ty[y, x]))) * coherence[y, x]
                               + (1. - min(1., abs(ux * tx[ny, nx] + uy * ty[ny, nx]))) * coherence[ny, nx])
            turn = 0. if incoming == 8 else 1. - (ux * _UNIT[incoming][0] + uy * _UNIT[incoming][1])
            displacement = reference_distance[ny, nx] / config.corridor_radius_px
            edge_cost = step * (1. + config.support_weight * (1. - 0.5 * (support[y, x] + support[ny, nx]))
                                + config.tangent_weight * alignment + config.text_weight * 0.5 * (text[y, x] + text[ny, nx])
                                + config.reference_weight * displacement * displacement)
            next_cost = cost + edge_cost + config.turn_weight * max(0., turn)
            next_state = (ny * local_width + nx, direction, next_run)
            if next_cost >= distances.get(next_state, math.inf):
                continue
            if next_state not in distances and len(distances) >= config.max_states:
                return abstain("state_budget_exhausted", discovered_states=len(distances), expanded_states=expanded, queue_peak=queue_peak)
            if len(queue) >= config.max_queue_entries:
                return abstain("queue_budget_exhausted", discovered_states=len(distances), expanded_states=expanded, queue_peak=queue_peak)
            distances[next_state], previous[next_state] = next_cost, state
            heapq.heappush(queue, (next_cost + math.hypot(target[0] - nx, target[1] - ny), next_cost, next_state))
            queue_peak = max(queue_peak, len(queue))
    search_audit = dict(discovered_states=len(distances), expanded_states=expanded, queue_peak=queue_peak)
    if destination is None:
        return abstain("no_supported_route", **search_audit)
    pixels = []
    current = destination
    while True:
        y, x = divmod(current[0], local_width)
        pixels.append((x + x0, y + y0))
        if current == initial:
            break
        current = previous[current]
    pixels.reverse()
    # Heading state can revisit a pixel; no such loop or diagonal self-crossing
    # is emitted as observed geometry, even if it was cheaper in state space.
    if len(set(pixels)) != len(pixels):
        return abstain("route_revisits_pixel", **search_audit)
    diagonals = set()
    for first, second in zip(pixels, pixels[1:]):
        if first[0] != second[0] and first[1] != second[1]:
            midpoint = first[0] + second[0], first[1] + second[1]
            if midpoint in diagonals:
                return abstain("route_self_crosses", **search_audit)
            diagonals.add(midpoint)
    points = [tuple(map(float, point)) for point in pixels]
    if len(points) == 1:
        points = [start, end]
    else:
        points[0], points[-1] = start, end
    reference_length = sum(math.dist(a, b) for a, b in zip(reference, reference[1:])) if reference else math.dist(start, end)
    length = sum(math.dist(a, b) for a, b in zip(points, points[1:]))
    if length > max(reference_length, math.dist(start, end)) * config.max_length_ratio + 1e-9:
        return abstain("route_exceeds_length_ratio", path_length_px=length, **search_audit)

    if reference is not None and (support.size + math.ceil(length * 4.) + 2 * len(points)) * (len(reference) - 1) > config.max_corridor_evaluations:
        return abstain("corridor_audit_budget_exhausted", **search_audit)
    distance, support_integral, unsupported_length = 0., 0., 0.
    spans, open_span = [], None
    max_displacement = 0. if reference is not None else None
    for first, second in zip(points, points[1:]):
        segment_length = math.dist(first, second)
        local_first, local_second = (first[0] - x0, first[1] - y0), (second[0] - x0, second[1] - y0)
        if _segment_enters_exclusion(excluded, local_first, local_second):
            return abstain("exact_endpoint_segment_enters_exclusion", **search_audit)
        if reference is not None:
            max_displacement = max(max_displacement, _corridor_bound(first, second, reference))
        for begin, end, value, supported in _support_intervals(support, local_first, local_second, config.minimum_support):
            piece_length = (end - begin) * segment_length
            support_integral += value * piece_length
            if not supported:
                unsupported_length += piece_length
                if open_span is None:
                    open_span = distance
            elif open_span is not None:
                spans.append(UnsupportedSpan(open_span, distance))
                open_span = None
            distance += piece_length
    if open_span is not None:
        spans.append(UnsupportedSpan(open_span, distance))
    fraction = max(0., 1. - unsupported_length / length)
    max_run = max((span.length_px for span in spans), default=0.)
    evidence_audit = dict(cost=float(distances[destination]), path_length_px=length,
                          supported_length_fraction=fraction, mean_support=support_integral / length,
                          maximum_unsupported_run_px=max_run, unsupported_spans=tuple(spans),
                          maximum_corridor_distance_px=max_displacement, **search_audit)
    if max_displacement is not None and max_displacement > config.corridor_radius_px + 1e-9:
        return abstain("exact_route_leaves_corridor", **evidence_audit)
    if fraction + 1e-9 < config.minimum_supported_fraction or max_run > config.max_unsupported_run_px + 1e-9:
        return abstain("insufficient_observed_ink", **evidence_audit)
    status = "supported_with_gaps" if spans else "observed"
    return DirectionalTraceResult(tuple(points), status, DirectionalTraceAudit(reason=status, **base_audit, **evidence_audit))


__all__ = ["DirectionalTraceConfig", "DirectionalTraceError", "DirectionalTraceAudit", "DirectionalTraceResult",
           "UnsupportedSpan", "trace_observed_ink"]
