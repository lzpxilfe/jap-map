"""Conservative vector proposals from raster line-candidate masks.

This module deliberately knows nothing about a country, map series, or CRS.
It turns a *review-only* binary mask into pixel-coordinate line proposals;
the calling application supplies the georeferencing and provenance fields.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .contours import Point, trace_polylines


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
