"""Bounded, direction-aware Ink Live-Wire for user-confirmed tracing.

Adapted as a compact, QGIS-independent boundary from ArchaeoTrace at
``f55d45da6228bd0c60e02618a2bb5031a55c54b4`` (GPL-2.0).
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Sequence

from .trace_guidance import TraceGuidance


Point = tuple[float, float]
Pixel = tuple[int, int]
_NEIGHBOURS = ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1))


class LiveWireUnavailable(RuntimeError):
    """Raised when numerical dependencies are unavailable."""


class LiveWireError(ValueError):
    """Raised for unsafe or unsupported explicit tracing requests."""


@dataclass(frozen=True)
class LiveWireConfig:
    max_window_size_px: int = 320
    hard_max_window_size_px: int = 1024
    support_weight: float = 2.4
    direction_weight: float = 0.8
    guidance_weight: float = 2.0
    minimum_score: float = 0.02

    def __post_init__(self) -> None:
        if not 16 <= self.max_window_size_px <= self.hard_max_window_size_px <= 1024:
            raise LiveWireError("Live-Wire window limits must be between 16 and 1024 pixels")
        if any(not math.isfinite(float(value)) or float(value) < 0 for value in (self.support_weight, self.direction_weight, self.guidance_weight)):
            raise LiveWireError("Live-Wire weights must be finite and non-negative")
        if not 0 <= self.minimum_score <= 1:
            raise LiveWireError("minimum_score must be in [0, 1]")


@dataclass(frozen=True)
class LiveWirePath:
    points: tuple[Point, ...]
    cost: float
    start_xy: Point
    end_xy: Point
    window_bounds: tuple[int, int, int, int]
    used_guidance: bool


def _rounded_point(value: Sequence[float], name: str) -> Pixel:
    if isinstance(value, (str, bytes)) or len(value) != 2:
        raise LiveWireError(f"{name} must have two pixel coordinates")
    x, y = float(value[0]), float(value[1])
    if not math.isfinite(x) or not math.isfinite(y):
        raise LiveWireError(f"{name} must be finite")
    return int(round(x)), int(round(y))


def _window(start: Pixel, end: Pixel, shape: tuple[int, int], limit: int) -> tuple[int, int, int, int]:
    height, width = shape
    min_x, max_x = sorted((start[0], end[0]))
    min_y, max_y = sorted((start[1], end[1]))
    span = max(max_x - min_x + 1, max_y - min_y + 1)
    if span > limit:
        raise LiveWireError(f"requested trace exceeds the {limit}px Live-Wire window")
    padding = max(12, int(math.ceil(span * 0.18)))
    x0, x1 = max(0, min_x - padding), min(width, max_x + padding + 1)
    y0, y1 = max(0, min_y - padding), min(height, max_y + padding + 1)
    if max(x1 - x0, y1 - y0) > limit:
        centre_x, centre_y = (start[0] + end[0]) // 2, (start[1] + end[1]) // 2
        half = limit // 2
        x0, x1 = max(0, centre_x - half), min(width, centre_x - half + limit)
        y0, y1 = max(0, centre_y - half), min(height, centre_y - half + limit)
    return x0, y0, x1, y1


def trace_ink_path(
    evidence,
    start_xy: Sequence[float],
    end_xy: Sequence[float],
    *,
    guidance: TraceGuidance | None = None,
    config: LiveWireConfig = LiveWireConfig(),
) -> LiveWirePath:
    """Find a bounded Ink-supported path between explicit pixel endpoints.

    The two endpoints remain exact user locations in the returned polyline.
    Guidance affects costs only, so a marked label cannot sever a valid line.
    """

    try:
        import numpy as np
    except ImportError as error:
        raise LiveWireUnavailable("Live-Wire requires NumPy") from error
    required = ("center_score", "tangent_x", "tangent_y", "coherence")
    if any(not hasattr(evidence, name) for name in required):
        raise LiveWireError("evidence must provide Ink score and direction arrays")
    score = np.asarray(evidence.center_score, dtype=np.float32)
    if score.ndim != 2 or min(score.shape, default=0) < 1:
        raise LiveWireError("Ink score must be a non-empty 2D array")
    start, end = _rounded_point(start_xy, "start_xy"), _rounded_point(end_xy, "end_xy")
    height, width = score.shape
    if not (0 <= start[0] < width and 0 <= start[1] < height and 0 <= end[0] < width and 0 <= end[1] < height):
        raise LiveWireError("trace endpoints leave the Ink evidence extent")
    if start == end:
        return LiveWirePath(((float(start[0]), float(start[1])),), 0.0, tuple(map(float, start_xy)), tuple(map(float, end_xy)), (start[0], start[1], start[0] + 1, start[1] + 1), guidance is not None)
    if guidance is not None and guidance.shape != score.shape:
        raise LiveWireError("guidance shape must match Ink evidence")
    x0, y0, x1, y1 = _window(start, end, score.shape, config.max_window_size_px)
    local_score = score[y0:y1, x0:x1]
    local_x = np.asarray(evidence.tangent_x, dtype=np.float32)[y0:y1, x0:x1]
    local_y = np.asarray(evidence.tangent_y, dtype=np.float32)[y0:y1, x0:x1]
    local_coherence = np.asarray(evidence.coherence, dtype=np.float32)[y0:y1, x0:x1]
    local_guidance = guidance.avoidance_score[y0:y1, x0:x1] if guidance is not None else None
    source = start[0] - x0, start[1] - y0
    target = end[0] - x0, end[1] - y0
    local_height, local_width = local_score.shape
    distances = np.full(local_score.shape, np.inf, dtype=np.float64)
    predecessor = np.full((local_height, local_width, 2), -1, dtype=np.int32)
    distances[source[1], source[0]] = 0.0
    queue = [(0.0, source[0], source[1])]
    visited = np.zeros(local_score.shape, dtype=bool)
    while queue:
        cost, x, y = heapq.heappop(queue)
        if visited[y, x]:
            continue
        visited[y, x] = True
        if (x, y) == target:
            break
        for dx, dy in _NEIGHBOURS:
            next_x, next_y = x + dx, y + dy
            if not (0 <= next_x < local_width and 0 <= next_y < local_height) or visited[next_y, next_x]:
                continue
            step = math.hypot(dx, dy)
            support = max(float(local_score[next_y, next_x]), config.minimum_score)
            move_x, move_y = dx / step, dy / step
            tangent_dot = abs(move_x * float(local_x[next_y, next_x]) + move_y * float(local_y[next_y, next_x]))
            direction_penalty = (1.0 - tangent_dot) * float(local_coherence[next_y, next_x])
            guidance_penalty = float(local_guidance[next_y, next_x]) if local_guidance is not None else 0.0
            next_cost = cost + step * (1.0 + config.support_weight * (1.0 - support) + config.direction_weight * direction_penalty + config.guidance_weight * guidance_penalty)
            if next_cost < distances[next_y, next_x]:
                distances[next_y, next_x] = next_cost
                predecessor[next_y, next_x] = (x, y)
                heapq.heappush(queue, (next_cost, next_x, next_y))
    if not math.isfinite(float(distances[target[1], target[0]])):
        raise LiveWireError("no finite Live-Wire route exists")
    pixels = []
    current = target
    while current != source:
        pixels.append(current)
        previous = predecessor[current[1], current[0]]
        if previous[0] < 0:
            raise LiveWireError("Live-Wire path reconstruction failed")
        current = int(previous[0]), int(previous[1])
    pixels.append(source)
    pixels.reverse()
    points = [(float(x + x0), float(y + y0)) for x, y in pixels]
    points[0] = tuple(map(float, start_xy))
    points[-1] = tuple(map(float, end_xy))
    return LiveWirePath(tuple(points), float(distances[target[1], target[0]]), tuple(map(float, start_xy)), tuple(map(float, end_xy)), (x0, y0, x1, y1), guidance is not None)


__all__ = ["LiveWireConfig", "LiveWireError", "LiveWirePath", "LiveWireUnavailable", "trace_ink_path"]
