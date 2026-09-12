"""Conservative vector proposals from raster line-candidate masks.

This module deliberately knows nothing about a country, map series, or CRS.
It turns a *review-only* binary mask into pixel-coordinate line proposals;
the calling application supplies the georeferencing and provenance fields.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .contours import Point, trace_polylines


SKELETON_GRAPH_VERSION = "corner-safe-skeleton/2"
DEFAULT_DIAGONAL_POLICY = "corner_safe"


class VectorizationBackendUnavailable(RuntimeError):
    """Raised when optional image-processing dependencies are unavailable."""


@dataclass(frozen=True)
class PixelLineProposal:
    """One visible linework proposal in pixel coordinates, never a final contour."""

    proposal_id: str
    points: tuple[Point, ...]
    pixel_length: float
    confidence: float


def _dependencies():
    try:
        import numpy as np
        from skimage.morphology import skeletonize
    except ImportError as error:
        raise VectorizationBackendUnavailable(
            "line vectorization requires NumPy and scikit-image in the QGIS Python environment"
        ) from error
    return np, skeletonize


def _length(points: tuple[Point, ...]) -> float:
    return sum(math.hypot(x2 - x1, y2 - y1) for (x1, y1), (x2, y2) in zip(points, points[1:]))


def _point_line_distance(point: Point, start: Point, end: Point) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    if dx == 0 and dy == 0:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    return abs(dy * point[0] - dx * point[1] + end[0] * start[1] - end[1] * start[0]) / math.hypot(dx, dy)


def simplify_polyline(points: tuple[Point, ...], tolerance_px: float) -> tuple[Point, ...]:
    """Ramer-Douglas-Peucker simplification while preserving line endpoints."""
    if tolerance_px < 0:
        raise ValueError("tolerance_px must be non-negative")
    if len(points) <= 2:
        return points
    farthest_index, farthest_distance = max(
        ((index, _point_line_distance(point, points[0], points[-1])) for index, point in enumerate(points[1:-1], 1)),
        key=lambda item: item[1],
    )
    if farthest_distance <= tolerance_px:
        return (points[0], points[-1])
    return simplify_polyline(points[: farthest_index + 1], tolerance_px)[:-1] + simplify_polyline(points[farthest_index:], tolerance_px)


def _decimate(points: tuple[Point, ...], maximum_points: int = 512) -> tuple[Point, ...]:
    """Bound simplification work on long historical-map strokes."""
    if len(points) <= maximum_points:
        return points
    stride = (len(points) - 1) / (maximum_points - 1)
    return tuple(points[round(index * stride)] for index in range(maximum_points))


def _trace_skeleton_array(source, np, *, diagonal_policy: str = DEFAULT_DIAGONAL_POLICY) -> list[tuple[Point, ...]]:
    """Trace an already-thinned array without repeatedly scanning all edges.

    The older plain-list tracer intentionally remains untouched for existing
    backends.  Ink v2 produces a much denser skeleton, so repeatedly taking
    the minimum of the complete unseen-edge set becomes prohibitively slow.
    This version visits sorted node edges first and consumes each edge once.
    A diagonal edge is redundant when an orthogonal two-step route already
    connects its pixels. Such shortcuts turn digital bends into false degree-3
    junctions and can discard an entire curved line under the length filter.
    Removing only those redundant edges preserves the foreground components;
    true T/X junctions remain split. ``full8`` reproduces the earlier graph.
    """

    if diagonal_policy not in {"corner_safe", "full8"}:
        raise ValueError("diagonal_policy must be corner_safe or full8")

    rows, columns = np.nonzero(source)
    pixels = {(int(x), int(y)) for y, x in zip(rows, columns)}
    if not pixels:
        return []
    offsets = ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1))
    neighbours = {
        point: tuple(
            sorted(
                (point[0] + dx, point[1] + dy)
                for dx, dy in offsets
                if (point[0] + dx, point[1] + dy) in pixels
                and not (
                    diagonal_policy == "corner_safe" and dx and dy
                    and ((point[0] + dx, point[1]) in pixels or (point[0], point[1] + dy) in pixels)
                )
            )
        )
        for point in pixels
    }
    nodes = {point for point, adjacent in neighbours.items() if len(adjacent) != 2}
    unseen_edges = {
        (point, neighbour) if point < neighbour else (neighbour, point)
        for point, adjacent in neighbours.items()
        for neighbour in adjacent
    }

    def edge_key(first, second):
        return (first, second) if first < second else (second, first)

    def consume(start, following):
        unseen_edges.remove(edge_key(start, following))
        path = [start, following]
        previous, current = start, following
        while current not in nodes:
            available = [
                candidate
                for candidate in neighbours[current]
                if candidate != previous and edge_key(current, candidate) in unseen_edges
            ]
            if not available:
                break
            next_point = available[0]
            unseen_edges.remove(edge_key(current, next_point))
            previous, current = current, next_point
            path.append(current)
            if current == start:
                break
        return tuple((float(x), float(y)) for x, y in path)

    lines: list[tuple[Point, ...]] = []
    for node in sorted(nodes):
        for neighbour in neighbours[node]:
            if edge_key(node, neighbour) in unseen_edges:
                lines.append(consume(node, neighbour))
    while unseen_edges:
        start, following = min(unseen_edges)
        lines.append(consume(start, following))
    return lines


def skeleton_to_pixel_line_proposals(
    skeleton,
    likelihood=None,
    *,
    minimum_length_px: float = 18.0,
    simplify_tolerance_px: float = 0.75,
    likelihood_scale: float = 1.0,
    proposal_prefix: str = "visible-line",
    diagonal_policy: str = DEFAULT_DIAGONAL_POLICY,
    junction_policy: str = "split",
) -> list[PixelLineProposal]:
    """Trace a one-pixel skeleton directly into deterministic proposals.

    Unlike :func:`mask_to_pixel_line_proposals`, this function never thins its
    input.  It is intended for backends such as Ink v2 that already guarantee
    centreline geometry.  Junctions remain split rather than being guessed
    through, and short fragments are filtered only after their pixel length
    is measured.
    """

    if minimum_length_px <= 0 or simplify_tolerance_px < 0:
        raise ValueError("minimum_length_px must be positive and simplify_tolerance_px must be non-negative")
    if likelihood_scale <= 0 or not proposal_prefix:
        raise ValueError("likelihood_scale must be positive and proposal_prefix must not be empty")
    try:
        import numpy as np
    except ImportError as error:
        raise VectorizationBackendUnavailable("skeleton vectorization requires NumPy") from error
    source = np.asarray(skeleton, dtype=bool)
    if source.ndim != 2:
        raise ValueError("skeleton must be a 2-D array")
    scores = np.asarray(likelihood, dtype=np.float32) if likelihood is not None else None
    if scores is not None and scores.shape != source.shape:
        raise ValueError("likelihood must have the same shape as skeleton")
    if junction_policy not in {"split", "tangent_pairs"}:
        raise ValueError("junction_policy must be split or tangent_pairs")
    raw_lines = _trace_skeleton_array(source, np, diagonal_policy=diagonal_policy)
    if junction_policy == "tangent_pairs":
        raw_lines = _join_tangent_pairs(raw_lines)
    proposals: list[PixelLineProposal] = []
    for raw in raw_lines:
        length = _length(raw)
        if length < minimum_length_px:
            continue
        bounded = _decimate(raw)
        points = bounded if raw[0] == raw[-1] else simplify_polyline(bounded, simplify_tolerance_px)
        if scores is None:
            confidence = 1.0
        else:
            values = [scores[int(round(y)), int(round(x))] for x, y in raw]
            confidence = float(np.clip(np.mean(values) / likelihood_scale, 0.0, 1.0))
        proposals.append(
            PixelLineProposal(
                f"{proposal_prefix}-{len(proposals) + 1}",
                points,
                length,
                confidence,
            )
        )
    return proposals


def _join_tangent_pairs(lines: list[tuple[Point, ...]]) -> list[tuple[Point, ...]]:
    """Experimental continuation through unambiguous degree-3 junctions only.

    Never creates pixels, crosses a gap, pairs four-way crossings, or changes
    the side branch. The two chosen arms must each have 8 px of evidence,
    oppose within 20 degrees and beat the next candidate by 25 degrees.
    This is visible-linework grouping, NOT semantic contour classification.
    """
    ends = {}
    for index, line in enumerate(lines):
        if line[0] != line[-1]:
            for side in (0, 1):
                ends.setdefault(line[0 if side == 0 else -1], []).append((index, side))

    def outward(key):
        index, side = key
        ordered = lines[index] if side == 0 else tuple(reversed(lines[index]))
        travelled = 0.0
        point = ordered[-1]
        for first, second in zip(ordered, ordered[1:]):
            segment = math.hypot(second[0]-first[0], second[1]-first[1])
            if travelled + segment >= 8:
                fraction = (8-travelled)/segment
                point = (first[0]+fraction*(second[0]-first[0]), first[1]+fraction*(second[1]-first[1]))
                travelled = 8
                break
            travelled += segment
        dx, dy = point[0]-ordered[0][0], point[1]-ordered[0][1]
        length = math.hypot(dx, dy)
        return (dx/length, dy/length, travelled) if length else (0., 0., 0.)

    pairs = {}
    for attached in ends.values():
        if len(attached) != 3 or len({index for index, _ in attached}) != 3:
            continue
        directions = {key: outward(key) for key in attached}
        ranked = []
        for offset, first in enumerate(attached):
            for second in attached[offset+1:]:
                a, b = directions[first], directions[second]
                if min(a[2], b[2]) < 2:
                    continue
                deviation = math.degrees(math.acos(max(-1., min(1., -a[0]*b[0]-a[1]*b[1]))))
                ranked.append((deviation, first, second))
        ranked.sort()
        # All three directions must be observed, including the side branch.
        if len(ranked) != 3:
            continue
        best, second_best = ranked[:2]
        if best[0] > 20 or second_best[0]-best[0] < 25:
            continue
        if min(directions[best[1]][2], directions[best[2]][2]) < 8:
            continue
        pairs[best[1]], pairs[best[2]] = best[2], best[1]

    consumed, result = set(), []

    def consume(start):
        index, side = start
        path = []
        while index not in consumed:
            consumed.add(index)
            ordered = lines[index] if side == 0 else tuple(reversed(lines[index]))
            path.extend(ordered if not path else ordered[1:])
            next_end = pairs.get((index, 1-side))
            if next_end is None:
                break
            index, side = next_end
        return tuple(path)

    # Start at unpaired ends so an entire open chain is consumed in one pass.
    for index, line in enumerate(lines):
        if index in consumed:
            continue
        for side in (0, 1):
            if (index, side) not in pairs:
                result.append(consume((index, side)))
                break
    for index in range(len(lines)):
        if index not in consumed:
            result.append(consume((index, 0)))
    return result


def mask_to_pixel_line_proposals(
    mask,
    likelihood=None,
    *,
    minimum_length_px: float = 18.0,
    simplify_tolerance_px: float = 0.75,
) -> list[PixelLineProposal]:
    """Thin and trace a candidate mask into conservative visible-line proposals.

    Closed loops are retained. Junctions are split rather than guessed through,
    avoiding invented topology around text and crossing map symbols.
    """
    if minimum_length_px <= 0 or simplify_tolerance_px < 0:
        raise ValueError("minimum_length_px must be positive and simplify_tolerance_px must be non-negative")
    np, skeletonize = _dependencies()
    source = np.asarray(mask, dtype=bool)
    if source.ndim != 2:
        raise ValueError("mask must be a 2-D array")
    if not source.any():
        return []
    skeleton = skeletonize(source)
    lines = trace_polylines(skeleton.tolist())
    scores = np.asarray(likelihood) if likelihood is not None else None
    if scores is not None and scores.shape != source.shape:
        raise ValueError("likelihood must have the same shape as mask")
    proposals: list[PixelLineProposal] = []
    for line in lines:
        raw = tuple(line)
        length = _length(raw)
        if length < minimum_length_px:
            continue
        bounded = _decimate(raw)
        # RDP has a degenerate baseline for a closed ring. The bounded trace is
        # already suitable for review and avoids quadratic work on large loops.
        points = bounded if raw[0] == raw[-1] else simplify_polyline(bounded, simplify_tolerance_px)
        if scores is None:
            confidence = 1.0
        else:
            values = [scores[int(round(y)), int(round(x))] for x, y in raw]
            confidence = float(np.clip(np.mean(values) / 255.0, 0.0, 1.0))
        proposals.append(PixelLineProposal(f"visible-line-{len(proposals) + 1}", points, length, confidence))
    return proposals
