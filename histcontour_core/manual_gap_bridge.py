"""Explicit Hermite previews for gaps under labels or map symbols.

The caller supplies both endpoints.  This module never guesses endpoints,
labels, or a final contour; it only validates a short, smooth preview that a
user may accept in an editable QGIS layer. Adapted from ArchaeoTrace at
``f55d45da6228bd0c60e02618a2bb5031a55c54b4`` (GPL-2.0).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


Point = tuple[float, float]


class ManualGapBridgeError(ValueError):
    """Raised when an explicit label-gap preview is unsafe."""


@dataclass(frozen=True)
class ManualGapBridgeConfig:
    min_gap_pixels: float = 3.0
    max_gap_pixels: float = 128.0
    max_abs_tangent_slope: float = 0.35
    max_detour_ratio: float = 1.25
    min_tangent_alignment: float = math.sqrt(0.5)
    sample_spacing_pixels: float = 0.75

    def __post_init__(self) -> None:
        values = (
            self.min_gap_pixels,
            self.max_gap_pixels,
            self.max_abs_tangent_slope,
            self.max_detour_ratio,
            self.min_tangent_alignment,
            self.sample_spacing_pixels,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ManualGapBridgeError("gap bridge settings must be finite")
        if not 0 < self.min_gap_pixels <= self.max_gap_pixels:
            raise ManualGapBridgeError("gap limits are invalid")
        if not 0 <= self.max_abs_tangent_slope <= 1 or self.max_detour_ratio < 1:
            raise ManualGapBridgeError("gap bridge shape limits are invalid")
        if not 0.5 <= self.min_tangent_alignment <= 1 or self.sample_spacing_pixels <= 0:
            raise ManualGapBridgeError("gap bridge direction settings are invalid")


@dataclass(frozen=True)
class ManualGapBridge:
    points: tuple[Point, ...]
    gap_length_pixels: float
    path_length_pixels: float
    source_tangent_slope: float
    target_tangent_slope: float

    @property
    def detour_ratio(self) -> float:
        return self.path_length_pixels / self.gap_length_pixels


def _point(value: Sequence[float], name: str) -> Point:
    if isinstance(value, (str, bytes, dict)) or len(value) != 2:
        raise ManualGapBridgeError(f"{name} must contain two coordinates")
    point = float(value[0]), float(value[1])
    if not all(math.isfinite(item) for item in point):
        raise ManualGapBridgeError(f"{name} must be finite")
    return point


def _unit(value: Sequence[float], name: str) -> Point:
    x, y = _point(value, name)
    magnitude = math.hypot(x, y)
    if magnitude <= 1e-9:
        raise ManualGapBridgeError(f"{name} must be non-zero")
    return x / magnitude, y / magnitude


def _length(points: Sequence[Point]) -> float:
    return sum(math.hypot(second[0] - first[0], second[1] - first[1]) for first, second in zip(points, points[1:]))


def _oriented_slope(tangent: Point, direction: Point, normal: Point, config: ManualGapBridgeConfig) -> float:
    forward = tangent[0] * direction[0] + tangent[1] * direction[1]
    if forward < 0:
        tangent = -tangent[0], -tangent[1]
        forward = -forward
    if forward < config.min_tangent_alignment:
        raise ManualGapBridgeError("endpoint tangent is incompatible with the requested gap")
    lateral = tangent[0] * normal[0] + tangent[1] * normal[1]
    return max(-config.max_abs_tangent_slope, min(config.max_abs_tangent_slope, lateral / forward))


def build_manual_gap_bridge(
    source_xy: Sequence[float],
    target_xy: Sequence[float],
    source_tangent: Sequence[float],
    target_tangent: Sequence[float],
    *,
    config: ManualGapBridgeConfig = ManualGapBridgeConfig(),
) -> ManualGapBridge:
    """Build a bounded cubic-Hermite preview between explicit endpoints."""

    source, target = _point(source_xy, "source_xy"), _point(target_xy, "target_xy")
    direction = _unit((target[0] - source[0], target[1] - source[1]), "gap direction")
    gap = math.hypot(target[0] - source[0], target[1] - source[1])
    if not config.min_gap_pixels <= gap <= config.max_gap_pixels:
        raise ManualGapBridgeError("gap length is outside the configured range")
    normal = -direction[1], direction[0]
    source_slope = _oriented_slope(_unit(source_tangent, "source_tangent"), direction, normal, config)
    target_slope = _oriented_slope(_unit(target_tangent, "target_tangent"), direction, normal, config)
    control = gap * 0.36
    source_control = (source[0] + control * (direction[0] + source_slope * normal[0]), source[1] + control * (direction[1] + source_slope * normal[1]))
    target_control = (target[0] - control * (direction[0] + target_slope * normal[0]), target[1] - control * (direction[1] + target_slope * normal[1]))
    count = max(2, int(math.ceil(gap / config.sample_spacing_pixels)) + 1)
    points = []
    for index in range(count):
        t = index / (count - 1)
        inverse = 1.0 - t
        points.append((
            inverse ** 3 * source[0] + 3 * inverse ** 2 * t * source_control[0] + 3 * inverse * t ** 2 * target_control[0] + t ** 3 * target[0],
            inverse ** 3 * source[1] + 3 * inverse ** 2 * t * source_control[1] + 3 * inverse * t ** 2 * target_control[1] + t ** 3 * target[1],
        ))
    path_length = _length(points)
    if path_length / gap > config.max_detour_ratio:
        raise ManualGapBridgeError("gap preview makes an excessive detour")
    return ManualGapBridge(tuple(points), gap, path_length, source_slope, target_slope)


def sample_evidence_tangent(evidence, pixel_xy: Sequence[float], *, radius_pixels: float = 3.0) -> Point | None:
    """Read one unambiguous axial direction from a local Ink evidence field."""

    if radius_pixels <= 0 or radius_pixels > 32:
        raise ManualGapBridgeError("tangent radius must be in (0, 32]")
    point = _point(pixel_xy, "pixel_xy")
    try:
        import numpy as np
    except ImportError as error:
        raise ManualGapBridgeError("sampling Ink tangents requires NumPy") from error
    required = ("centerline", "center_score", "tangent_x", "tangent_y", "coherence")
    if any(not hasattr(evidence, name) for name in required):
        raise ManualGapBridgeError("evidence does not provide Ink direction fields")
    height, width = evidence.centerline.shape
    x, y = point
    if not 0 <= x < width or not 0 <= y < height:
        raise ManualGapBridgeError("endpoint leaves the Ink evidence extent")
    candidates = []
    for row in range(max(0, math.ceil(y - radius_pixels)), min(height, math.floor(y + radius_pixels) + 1)):
        for column in range(max(0, math.ceil(x - radius_pixels)), min(width, math.floor(x + radius_pixels) + 1)):
            distance = math.hypot(column - x, row - y)
            if distance > radius_pixels or not evidence.centerline[row, column]:
                continue
            if evidence.center_score[row, column] < 0.15 or evidence.coherence[row, column] < 0.15:
                continue
            tangent = _unit((float(evidence.tangent_x[row, column]), float(evidence.tangent_y[row, column])), "sampled tangent")
            candidates.append((distance, row, column, tangent, float(evidence.coherence[row, column])))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[:3])
    distance, _row, _column, reference, _coherence = candidates[0]
    nearby = [candidate for candidate in candidates if candidate[0] <= distance + 1.0]
    aligned = []
    for _distance_value, _row, _column, tangent, coherence in nearby:
        dot = tangent[0] * reference[0] + tangent[1] * reference[1]
        if dot < 0:
            tangent = -tangent[0], -tangent[1]
            dot = -dot
        if dot < math.cos(math.radians(35)):
            return None
        aligned.append((tangent, coherence / (1.0 + _distance_value) ** 2))
    total = sum(weight for _tangent, weight in aligned)
    return _unit((sum(tangent[0] * weight for tangent, weight in aligned) / total, sum(tangent[1] * weight for tangent, weight in aligned) / total), "combined tangent")


__all__ = ["ManualGapBridge", "ManualGapBridgeConfig", "ManualGapBridgeError", "build_manual_gap_bridge", "sample_evidence_tangent"]
