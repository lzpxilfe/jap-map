"""Immutable soft route-avoidance evidence for Ink tracing.

Adapted from ArchaeoTrace at ``f55d45da6228bd0c60e02618a2bb5031a55c54b4``
(GPL-2.0).
"""

from __future__ import annotations

from dataclasses import dataclass


class TraceGuidanceUnavailable(RuntimeError):
    """Raised when NumPy is unavailable for a guidance raster."""


@dataclass(frozen=True)
class TraceGuidance:
    """A [0, 1] soft avoidance score; it never blocks a route outright."""

    avoidance_score: object

    def __post_init__(self) -> None:
        try:
            import numpy as np
        except ImportError as error:
            raise TraceGuidanceUnavailable("trace guidance requires NumPy") from error
        score = np.array(self.avoidance_score, dtype=np.float32, order="C", copy=True)
        if score.ndim != 2 or min(score.shape, default=0) < 1:
            raise ValueError("avoidance_score must be a non-empty 2D array")
        if not np.isfinite(score).all() or np.any((score < 0.0) | (score > 1.0)):
            raise ValueError("avoidance_score must be finite values in [0, 1]")
        score.setflags(write=False)
        object.__setattr__(self, "avoidance_score", score)

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(int(value) for value in self.avoidance_score.shape)


def guidance_from_boxes(shape: tuple[int, int], boxes, *, feather_pixels: float = 2.0) -> TraceGuidance:
    """Create max-composed soft guidance from pixel-coordinate rectangles."""

    try:
        import numpy as np
    except ImportError as error:
        raise TraceGuidanceUnavailable("trace guidance requires NumPy") from error
    if len(shape) != 2 or min(shape) < 1 or any(isinstance(value, bool) or int(value) != value for value in shape):
        raise ValueError("shape must contain two positive integers")
    if feather_pixels < 0:
        raise ValueError("feather_pixels must be non-negative")
    height, width = (int(value) for value in shape)
    score = np.zeros((height, width), dtype=np.float32)
    grid_x, grid_y = np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    for box in boxes:
        if isinstance(box, (str, bytes)) or len(box) != 4:
            raise ValueError("each guidance box needs four coordinates")
        x0, y0, x1, y1 = (float(value) for value in box)
        left, right = sorted((x0, x1))
        top, bottom = sorted((y0, y1))
        dx = np.maximum(np.maximum(left - grid_x, 0.0), grid_x - right)
        dy = np.maximum(np.maximum(top - grid_y, 0.0), grid_y - bottom)
        if feather_pixels == 0:
            current = ((grid_y[:, None] >= top) & (grid_y[:, None] <= bottom) & (grid_x[None, :] >= left) & (grid_x[None, :] <= right)).astype(np.float32)
        else:
            current = np.clip(1.0 - np.hypot(dy[:, None], dx[None, :]) / feather_pixels, 0.0, 1.0).astype(np.float32)
        np.maximum(score, current, out=score)
    return TraceGuidance(score)


__all__ = ["TraceGuidance", "TraceGuidanceUnavailable", "guidance_from_boxes"]
