"""Two-scale image context tensors around Ink segments."""

from __future__ import annotations

import math
from typing import Sequence


PATCH_CONTEXT_SCHEMA = "ink-segment-patches/1"
PATCH_SIZE = 32
PATCH_RADII = (16.0, 48.0)
PATCH_POSITIONS = (0.0, 0.5, 1.0)


class PatchContextUnavailable(RuntimeError):
    """Raised when NumPy is unavailable for patch extraction."""


def _numpy():
    try:
        import numpy as np
    except ImportError as error:
        raise PatchContextUnavailable("segment context extraction requires NumPy") from error
    return np


def _point_at_fraction(points: Sequence[tuple[float, float]], fraction: float) -> tuple[float, float]:
    lengths = [math.hypot(second[0] - first[0], second[1] - first[1]) for first, second in zip(points, points[1:])]
    total = sum(lengths)
    if total <= 0:
        raise ValueError("segment points must span a positive length")
    target, travelled = total * fraction, 0.0
    for first, second, length in zip(points, points[1:], lengths):
        if travelled + length >= target:
            ratio = (target - travelled) / length
            return first[0] + ratio * (second[0] - first[0]), first[1] + ratio * (second[1] - first[1])
        travelled += length
    return points[-1]


def _sample_polyline(points: Sequence[tuple[float, float]]):
    output = []
    for first, second in zip(points, points[1:]):
        length = math.hypot(second[0] - first[0], second[1] - first[1])
        count = max(1, int(math.ceil(length)))
        output.extend((first[0] + step / count * (second[0] - first[0]), first[1] + step / count * (second[1] - first[1])) for step in range(count))
    return output + [points[-1]]


def segment_context_tensor(image, points: Sequence[tuple[float, float]], *, patch_size: int = PATCH_SIZE, radii: Sequence[float] = PATCH_RADII, positions: Sequence[float] = PATCH_POSITIONS):
    """Return [three positions × two scales × image/mask, H, W] float32.

    Image values are normalised to dark ink = 1.  The companion binary mask
    tells a small CNN which local stroke is the segment under consideration.
    """

    np = _numpy()
    values = np.asarray(image)
    if values.ndim == 3:
        values = values[..., :3].astype(np.float32).mean(axis=2)
    if values.ndim != 2 or min(values.shape, default=0) < 1:
        raise ValueError("image must be a non-empty grayscale or RGB array")
    if len(points) < 2 or patch_size < 8 or any(not 0 <= position <= 1 for position in positions) or any(radius <= 0 for radius in radii):
        raise ValueError("invalid segment context settings")
    gray = values.astype(np.float32)
    if gray.max(initial=0) > 1.0:
        gray /= 255.0
    gray = np.clip(gray, 0.0, 1.0)
    height, width = gray.shape
    samples = _sample_polyline(points)
    output = []
    relative = (np.arange(patch_size, dtype=np.float32) + 0.5) / patch_size * 2.0 - 1.0
    for fraction in positions:
        centre_x, centre_y = _point_at_fraction(points, fraction)
        for radius in radii:
            xs = np.rint(centre_x + relative * radius).astype(int)
            ys = np.rint(centre_y + relative * radius).astype(int)
            valid_x, valid_y = (xs >= 0) & (xs < width), (ys >= 0) & (ys < height)
            patch = np.ones((patch_size, patch_size), dtype=np.float32)
            grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
            valid = valid_y[:, None] & valid_x[None, :]
            patch[valid] = gray[grid_y[valid], grid_x[valid]]
            mask = np.zeros((patch_size, patch_size), dtype=np.float32)
            for x, y in samples:
                px = int(round((x - (centre_x - radius)) / (2.0 * radius) * patch_size - 0.5))
                py = int(round((y - (centre_y - radius)) / (2.0 * radius) * patch_size - 0.5))
                if 0 <= px < patch_size and 0 <= py < patch_size:
                    mask[py, px] = 1.0
            output.extend((1.0 - patch, mask))
    return np.stack(output).astype(np.float32)


def patch_metadata() -> dict:
    return {
        "schema": PATCH_CONTEXT_SCHEMA,
        "patch_size": PATCH_SIZE,
        "radii": list(PATCH_RADII),
        "positions": list(PATCH_POSITIONS),
        "channels": len(PATCH_RADII) * len(PATCH_POSITIONS) * 2,
    }


__all__ = ["PATCH_CONTEXT_SCHEMA", "PATCH_POSITIONS", "PATCH_RADII", "PATCH_SIZE", "PatchContextUnavailable", "patch_metadata", "segment_context_tensor"]
