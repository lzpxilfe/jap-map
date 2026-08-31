"""Dependency-light Ink v2 centreline evidence for batch review candidates.

This module adapts the QGIS-independent Ink v2 detector from ArchaeoTrace
(``AI-Vectorizer-for-Archaeology``) at commit
``7960acddb4e82855e2088fdfdd2244799b63775c``.  The upstream project and this
repository are both GPL-2.0 licensed.  Adapted for jap-map on 2026-08-31.
Only the multi-scale ink evidence and binary centreline path are carried here;
interactive Live-Wire remains an explicit later integration rather than being
silently approximated.

Ink evidence is intentionally *linework* evidence, not a contour classifier.
Text, roads, rivers, frames, and symbols can all produce valid ink centres.
Callers must keep outputs review-only and preserve their backend provenance.
"""

from __future__ import annotations

from dataclasses import dataclass


ARCHAEOTRACE_UPSTREAM_COMMIT = "7960acddb4e82855e2088fdfdd2244799b63775c"
INK_BACKEND_ID = "archaeotrace_ink_v2_centerline"


class InkBackendUnavailable(RuntimeError):
    """Raised when NumPy is unavailable in the active Python runtime."""


@dataclass(frozen=True)
class InkCenterlineSettings:
    """Frozen settings matching the pinned ArchaeoTrace Ink v2 defaults."""

    scales_px: tuple[int, ...] = (9, 15, 31)
    tile_size_px: int = 128
    tile_halo_px: int = 16
    response_percentile: float = 99.0
    minimum_normalized_response: float = 0.04
    minimum_component_pixels: int = 5
    maximum_spur_length_px: float = 2.0
    centerline_score: float = 1.0
    foreground_weight: float = 0.75
    fallback_foreground_percentile: float = 10.0

    def __post_init__(self) -> None:
        if not self.scales_px or any(not isinstance(value, int) or value < 3 or value % 2 == 0 for value in self.scales_px):
            raise ValueError("scales_px must contain positive odd integers of at least 3")
        if self.tile_size_px < 16 or self.tile_halo_px < max(self.scales_px) // 2:
            raise ValueError("tile_size_px is too small or tile_halo_px does not cover the largest scale")
        if not 0 < self.response_percentile <= 100:
            raise ValueError("response_percentile must be in (0, 100]")
        if not 0 <= self.minimum_normalized_response <= 1:
            raise ValueError("minimum_normalized_response must be in [0, 1]")
        if self.minimum_component_pixels < 1 or self.maximum_spur_length_px < 0:
            raise ValueError("component and spur limits must be non-negative")
        if not 0 < self.centerline_score <= 1 or not 0 < self.foreground_weight <= 1:
            raise ValueError("evidence weights must be in (0, 1]")
        if not 0 <= self.fallback_foreground_percentile <= 100:
            raise ValueError("fallback_foreground_percentile must be in [0, 100]")


@dataclass(frozen=True)
class InkCenterlineResult:
    """Review-only centreline and continuous support on one source pixel grid."""

    centerline: object
    center_score: object
    scale_px: object
    centerline_fraction: float
    backend: str = INK_BACKEND_ID
    upstream_commit: str = ARCHAEOTRACE_UPSTREAM_COMMIT

    def __post_init__(self) -> None:
        np, _ndimage, _threshold_otsu, _skeletonize = _dependencies()
        centerline = np.array(self.centerline, dtype=bool, order="C", copy=True)
        center_score = np.array(self.center_score, dtype=np.float32, order="C", copy=True)
        scale_px = np.array(self.scale_px, dtype=np.float32, order="C", copy=True)
        if centerline.ndim != 2 or center_score.shape != centerline.shape or scale_px.shape != centerline.shape:
            raise ValueError("Ink result arrays must share one non-empty 2D shape")
        if not np.isfinite(center_score).all() or np.any((center_score < 0.0) | (center_score > 1.0)):
            raise ValueError("center_score must contain finite values in [0, 1]")
        if not np.isfinite(scale_px).all() or np.any(scale_px < 0.0):
            raise ValueError("scale_px must contain finite non-negative values")
        expected_fraction = float(centerline.mean())
        if abs(float(self.centerline_fraction) - expected_fraction) > 1e-12:
            raise ValueError("centerline_fraction does not match centerline")
        for array in (centerline, center_score, scale_px):
            array.setflags(write=False)
        object.__setattr__(self, "centerline", centerline)
        object.__setattr__(self, "center_score", center_score)
        object.__setattr__(self, "scale_px", scale_px)


def _dependencies():
    try:
        import numpy as np
    except ImportError as error:
        raise InkBackendUnavailable("Ink v2 centreline extraction requires NumPy") from error
    try:
        from scipy import ndimage
    except Exception:
        ndimage = None
    try:
        from skimage.filters import threshold_otsu
    except Exception:
        threshold_otsu = None
    try:
        from skimage.morphology import skeletonize
    except Exception:
        skeletonize = None
    return np, ndimage, threshold_otsu, skeletonize


def _normalize_values(np, values):
    source = np.asarray(values)
    array = np.asarray(source, dtype=np.float32)
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros(array.shape, dtype=np.float32)
    if np.issubdtype(source.dtype, np.bool_):
        low, high = 0.0, 1.0
    elif np.issubdtype(source.dtype, np.integer):
        limits = np.iinfo(source.dtype)
        low, high = float(limits.min), float(limits.max)
    else:
        finite_low = float(array[finite].min())
        finite_high = float(array[finite].max())
        if 0.0 <= finite_low and finite_high <= 1.0:
            low, high = 0.0, 1.0
        elif 0.0 <= finite_low and finite_high <= 255.0:
            low, high = 0.0, 255.0
        else:
            low = float(np.percentile(array[finite], 1.0))
            high = float(np.percentile(array[finite], 99.0))
    if high <= low + np.finfo(np.float32).eps:
        return np.zeros(array.shape, dtype=np.float32)
    normalized = (array - low) / (high - low)
    normalized = np.where(finite, normalized, 1.0)
    return np.ascontiguousarray(np.clip(normalized, 0.0, 1.0), dtype=np.float32)


def _evidence_sources(np, image):
    values = np.asarray(image)
    if values.ndim == 2:
        return (_normalize_values(np, values),), tuple(int(value) for value in values.shape)
    if values.ndim != 3 or values.shape[2] < 1:
        raise ValueError("Ink evidence input must be gray or RGB-like")
    shape = tuple(int(value) for value in values.shape[:2])
    if values.shape[2] < 3:
        return (_normalize_values(np, values[..., 0]),), shape
    channels = tuple(_normalize_values(np, values[..., channel]) for channel in range(3))
    luminance = (
        channels[0] * np.float32(0.299)
        + channels[1] * np.float32(0.587)
        + channels[2] * np.float32(0.114)
    ).astype(np.float32)
    return (
        np.ascontiguousarray(luminance),
        *(np.ascontiguousarray(channel) for channel in channels),
    ), shape


def _validate_tile_origin(np, tile_origin) -> tuple[int, int]:
    if (
        isinstance(tile_origin, (str, bytes))
        or not hasattr(tile_origin, "__len__")
        or len(tile_origin) != 2
        or any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            for value in tile_origin
        )
    ):
        raise ValueError("tile_origin must be an integer (x, y) pair")
    return int(tile_origin[0]), int(tile_origin[1])


def _window_filter(np, values, size: int, axis: int, reducer):
    before = int(size) // 2
    after = int(size) - before - 1
    padding = [(0, 0)] * values.ndim
    padding[axis] = (before, after)
    padded = np.pad(values, padding, mode="edge")
    sliding_window_view = getattr(np.lib.stride_tricks, "sliding_window_view", None)
    if callable(sliding_window_view):
        windows = sliding_window_view(padded, window_shape=int(size), axis=axis)
    else:  # pragma: no cover - compatibility with NumPy versions before 1.20
        window_shape = list(padded.shape)
        window_shape[axis] -= int(size) - 1
        window_shape.append(int(size))
        window_strides = list(padded.strides)
        window_strides.append(padded.strides[axis])
        windows = np.lib.stride_tricks.as_strided(
            padded,
            shape=tuple(window_shape),
            strides=tuple(window_strides),
            subok=False,
            writeable=False,
        )
    return reducer(windows, axis=-1)


def _mean_filter(np, values, size: int):
    averaged = _window_filter(np, values, size, 0, np.mean)
    return _window_filter(np, averaged, size, 1, np.mean)


def _grey_closing(np, values, size: int):
    dilated = _window_filter(np, values, size, 0, np.max)
    dilated = _window_filter(np, dilated, size, 1, np.max)
    closed = _window_filter(np, dilated, size, 0, np.min)
    return _window_filter(np, closed, size, 1, np.min)


def _remove_small_components(np, ndimage, mask, minimum_pixels: int):
    active = np.asarray(mask, dtype=bool)
    if ndimage is not None:
        labels, component_count = ndimage.label(active, structure=np.ones((3, 3), dtype=np.uint8))
        if not component_count:
            return np.zeros(labels.shape, dtype=bool)
        sizes = np.bincount(labels.ravel())
        keep = sizes >= minimum_pixels
        keep[0] = False
        return keep[labels]
    height, width = active.shape
    visited = np.zeros(active.shape, dtype=bool)
    retained = np.zeros(active.shape, dtype=bool)
    for start in np.flatnonzero(active):
        start = int(start)
        start_y, start_x = divmod(start, width)
        if visited[start_y, start_x]:
            continue
        visited[start_y, start_x] = True
        stack = [(start_y, start_x)]
        component = []
        while stack:
            y, x = stack.pop()
            component.append((y, x))
            for next_y in range(max(0, y - 1), min(height, y + 2)):
                for next_x in range(max(0, x - 1), min(width, x + 2)):
                    if active[next_y, next_x] and not visited[next_y, next_x]:
                        visited[next_y, next_x] = True
                        stack.append((next_y, next_x))
        if len(component) >= minimum_pixels:
            rows, columns = zip(*component)
            retained[rows, columns] = True
    return retained


def _thin_numpy(np, binary_mask):
    skeleton = np.asarray(binary_mask, dtype=bool).copy()
    if not skeleton.any():
        return skeleton

    def neighborhood(values):
        padded = np.pad(values, 1, mode="constant", constant_values=False)
        return (
            padded[:-2, 1:-1],
            padded[:-2, 2:],
            padded[1:-1, 2:],
            padded[2:, 2:],
            padded[2:, 1:-1],
            padded[2:, :-2],
            padded[1:-1, :-2],
            padded[:-2, :-2],
        )

    while True:
        changed = False
        neighbors = neighborhood(skeleton)
        neighbor_count = sum(neighbors)
        transitions = sum(
            (~current & following).astype(np.uint8)
            for current, following in zip(neighbors, neighbors[1:] + neighbors[:1])
        )
        p2, _p3, p4, _p5, p6, _p7, p8, _p9 = neighbors
        remove = skeleton & (neighbor_count >= 2) & (neighbor_count <= 6) & (transitions == 1) & ~(p2 & p4 & p6) & ~(p4 & p6 & p8)
        if remove.any():
            skeleton[remove] = False
            changed = True

        neighbors = neighborhood(skeleton)
        neighbor_count = sum(neighbors)
        transitions = sum(
            (~current & following).astype(np.uint8)
            for current, following in zip(neighbors, neighbors[1:] + neighbors[:1])
        )
        p2, _p3, p4, _p5, p6, _p7, p8, _p9 = neighbors
        remove = skeleton & (neighbor_count >= 2) & (neighbor_count <= 6) & (transitions == 1) & ~(p2 & p4 & p8) & ~(p2 & p6 & p8)
        if remove.any():
            skeleton[remove] = False
            changed = True
        if not changed:
            return skeleton


def _thin(np, skeletonize, binary_mask):
    binary = np.asarray(binary_mask, dtype=bool)
    if not binary.any():
        return binary
    return skeletonize(binary) if skeletonize is not None else _thin_numpy(np, binary)


def _tile_ranges(origin: int, length: int, tile_size: int):
    first = (origin // tile_size) * tile_size
    return range(first, origin + length, tile_size)


def _normalize_tiled_response(np, response, tile_origin, settings: InkCenterlineSettings):
    values = np.asarray(response, dtype=np.float32)
    height, width = values.shape
    origin_x, origin_y = tile_origin
    normalized = np.zeros(values.shape, dtype=np.float32)
    for global_y in _tile_ranges(origin_y, height, settings.tile_size_px):
        local_y0 = max(0, global_y - origin_y)
        local_y1 = min(height, global_y + settings.tile_size_px - origin_y)
        if local_y0 >= local_y1:
            continue
        halo_y0 = max(0, local_y0 - settings.tile_halo_px)
        halo_y1 = min(height, local_y1 + settings.tile_halo_px)
        for global_x in _tile_ranges(origin_x, width, settings.tile_size_px):
            local_x0 = max(0, global_x - origin_x)
            local_x1 = min(width, global_x + settings.tile_size_px - origin_x)
            if local_x0 >= local_x1:
                continue
            halo_x0 = max(0, local_x0 - settings.tile_halo_px)
            halo_x1 = min(width, local_x1 + settings.tile_halo_px)
            neighborhood = values[halo_y0:halo_y1, halo_x0:halo_x1]
            positive = neighborhood[neighborhood > 0.0]
            if positive.size == 0:
                continue
            scale = max(float(np.percentile(positive, settings.response_percentile)), np.finfo(np.float32).eps)
            normalized[local_y0:local_y1, local_x0:local_x1] = np.clip(
                values[local_y0:local_y1, local_x0:local_x1] / scale,
                0.0,
                1.0,
            )
    return normalized


def _response_threshold(np, threshold_otsu, positive_scores, settings: InkCenterlineSettings) -> float:
    continuity_threshold = float(np.percentile(positive_scores, settings.fallback_foreground_percentile))
    if threshold_otsu is not None and positive_scores.size > 1:
        try:
            threshold = min(float(threshold_otsu(positive_scores)), continuity_threshold)
        except (TypeError, ValueError):
            threshold = settings.minimum_normalized_response
    else:
        threshold = continuity_threshold
    threshold = max(settings.minimum_normalized_response, threshold)
    return min(threshold, float(np.nextafter(float(positive_scores.max()), -np.inf)))


def _tiled_centerline(np, ndimage, threshold_otsu, skeletonize, response, tile_origin, settings):
    response = np.asarray(response, dtype=np.float32)
    if response.ndim != 2:
        raise ValueError("Ink response must be a 2D array")
    height, width = response.shape
    origin_x, origin_y = tile_origin
    centerline = np.zeros(response.shape, dtype=bool)
    for global_y in _tile_ranges(origin_y, height, settings.tile_size_px):
        local_y0 = max(0, global_y - origin_y)
        local_y1 = min(height, global_y + settings.tile_size_px - origin_y)
        if local_y0 >= local_y1:
            continue
        halo_y0 = max(0, local_y0 - settings.tile_halo_px)
        halo_y1 = min(height, local_y1 + settings.tile_halo_px)
        for global_x in _tile_ranges(origin_x, width, settings.tile_size_px):
            local_x0 = max(0, global_x - origin_x)
            local_x1 = min(width, global_x + settings.tile_size_px - origin_x)
            if local_x0 >= local_x1:
                continue
            halo_x0 = max(0, local_x0 - settings.tile_halo_px)
            halo_x1 = min(width, local_x1 + settings.tile_halo_px)
            neighborhood = response[halo_y0:halo_y1, halo_x0:halo_x1]
            positive = neighborhood[neighborhood > 0.0]
            if positive.size == 0:
                continue
            scale = max(float(np.percentile(positive, settings.response_percentile)), np.finfo(np.float32).eps)
            score = np.clip(neighborhood / scale, 0.0, 1.0)
            candidate = _remove_small_components(
                np,
                ndimage,
                score >= settings.minimum_normalized_response,
                settings.minimum_component_pixels,
            )
            positive_scores = score[candidate]
            if positive_scores.size == 0:
                continue
            threshold = _response_threshold(np, threshold_otsu, positive_scores, settings)
            foreground = _remove_small_components(
                np,
                ndimage,
                candidate & (score >= threshold),
                settings.minimum_component_pixels,
            )
            thinned = _thin(np, skeletonize, foreground)
            core_y0 = local_y0 - halo_y0
            core_y1 = core_y0 + (local_y1 - local_y0)
            core_x0 = local_x0 - halo_x0
            core_x1 = core_x0 + (local_x1 - local_x0)
            centerline[local_y0:local_y1, local_x0:local_x1] = thinned[core_y0:core_y1, core_x0:core_x1]
    return centerline


def _crossing_number(np, centerline):
    active = np.asarray(centerline, dtype=bool)
    padded = np.pad(active, 1, mode="constant", constant_values=False)
    neighbors = (
        padded[:-2, 1:-1], padded[:-2, 2:], padded[1:-1, 2:], padded[2:, 2:],
        padded[2:, 1:-1], padded[2:, :-2], padded[1:-1, :-2], padded[:-2, :-2],
    )
    groups = np.zeros(active.shape, dtype=np.uint8)
    for current, following in zip(neighbors, neighbors[1:] + neighbors[:1]):
        groups += (~current & following).astype(np.uint8)
    groups[~active] = 0
    return groups


def _prune_short_spurs(np, centerline, tile_origin, settings: InkCenterlineSettings):
    active = np.asarray(centerline, dtype=bool)
    if active.ndim != 2:
        raise ValueError("centerline must be a 2D array")
    if not active.any() or settings.maximum_spur_length_px <= 0:
        return active.copy()
    topology = _crossing_number(np, active)
    endpoints = np.argwhere(active & (topology == 1))
    height, width = active.shape
    origin_x, origin_y = tile_origin
    maximum_length = float(settings.maximum_spur_length_px)
    seam_margin = int(np.ceil(maximum_length))
    removals = np.zeros(active.shape, dtype=bool)
    neighbor_offsets = ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1))
    for endpoint_y, endpoint_x in endpoints:
        endpoint_x, endpoint_y = int(endpoint_x), int(endpoint_y)
        if endpoint_x in (0, width - 1) or endpoint_y in (0, height - 1):
            continue
        tile_x = (origin_x + endpoint_x) % settings.tile_size_px
        tile_y = (origin_y + endpoint_y) % settings.tile_size_px
        if (
            tile_x <= seam_margin or tile_y <= seam_margin
            or tile_x >= settings.tile_size_px - seam_margin
            or tile_y >= settings.tile_size_px - seam_margin
        ):
            continue
        path = [(endpoint_x, endpoint_y)]
        visited = {(endpoint_x, endpoint_y)}
        previous = None
        current = (endpoint_x, endpoint_y)
        length = 0.0
        while length <= maximum_length:
            current_x, current_y = current
            candidates = []
            for offset_x, offset_y in neighbor_offsets:
                next_x, next_y = current_x + offset_x, current_y + offset_y
                candidate = (next_x, next_y)
                if 0 <= next_x < width and 0 <= next_y < height and active[next_y, next_x] and candidate not in visited:
                    candidates.append(candidate)
            if not candidates:
                break
            if previous is None:
                if len(candidates) != 1:
                    break
                next_pixel = candidates[0]
            else:
                junctions = [candidate for candidate in candidates if topology[candidate[1], candidate[0]] >= 3]
                choices = junctions or candidates
                incoming_x, incoming_y = current_x - previous[0], current_y - previous[1]
                incoming_norm = float(np.hypot(incoming_x, incoming_y))
                ranked = []
                for next_x, next_y in choices:
                    outgoing_x, outgoing_y = next_x - current_x, next_y - current_y
                    outgoing_norm = float(np.hypot(outgoing_x, outgoing_y))
                    alignment = (incoming_x * outgoing_x + incoming_y * outgoing_y) / max(incoming_norm * outgoing_norm, 1e-9)
                    ranked.append((float(alignment), (next_x, next_y)))
                best_alignment = max(item[0] for item in ranked)
                best = [pixel for alignment, pixel in ranked if abs(alignment - best_alignment) <= 1e-9]
                if len(best) != 1:
                    break
                next_pixel = best[0]
            length += float(np.hypot(next_pixel[0] - current_x, next_pixel[1] - current_y))
            if length > maximum_length + 1e-9:
                break
            next_topology = int(topology[next_pixel[1], next_pixel[0]])
            if next_topology >= 3:
                for path_x, path_y in path:
                    removals[path_y, path_x] = True
                break
            if next_topology != 2:
                break
            path.append(next_pixel)
            visited.add(next_pixel)
            previous, current = current, next_pixel
    pruned = active.copy()
    pruned[removals] = False
    return pruned


def ink_centerline_candidates(
    image,
    *,
    tile_origin=(0, 0),
    settings: InkCenterlineSettings = InkCenterlineSettings(),
) -> InkCenterlineResult:
    """Return ArchaeoTrace-compatible Ink v2 evidence for review-only use."""

    if not isinstance(settings, InkCenterlineSettings):
        raise TypeError("settings must be an InkCenterlineSettings instance")
    np, ndimage, threshold_otsu, skeletonize = _dependencies()
    origin = _validate_tile_origin(np, tile_origin)
    sources, shape = _evidence_sources(np, image)
    if min(shape, default=0) < 1:
        raise ValueError("Ink evidence input must have non-empty dimensions")
    if min(shape) < 2 or not sources:
        zeros = np.zeros(shape, dtype=np.float32)
        return InkCenterlineResult(np.zeros(shape, dtype=bool), zeros, zeros.copy(), 0.0)

    dark_supports = []
    dark_context = max(settings.scales_px)
    for source in sources:
        local_reference = (
            ndimage.uniform_filter(source, size=dark_context, mode="nearest")
            if ndimage is not None
            else _mean_filter(np, source, dark_context)
        )
        dark_supports.append(np.maximum(local_reference - source, 0.0))

    fused_response = np.zeros(shape, dtype=np.float32)
    winning_scale = np.zeros(shape, dtype=np.float32)
    for scale in settings.scales_px:
        scale_response = np.zeros(shape, dtype=np.float32)
        for source, dark_support in zip(sources, dark_supports):
            background = (
                ndimage.grey_closing(source, size=(scale, scale), mode="nearest")
                if ndimage is not None
                else _grey_closing(np, source, scale)
            )
            response = np.minimum(np.maximum(background - source, 0.0), dark_support)
            np.maximum(scale_response, response, out=scale_response)
        stronger = scale_response > fused_response
        fused_response = np.maximum(fused_response, scale_response)
        winning_scale[stronger] = float(scale)

    if not np.any(fused_response > 0.0):
        zeros = np.zeros(shape, dtype=np.float32)
        return InkCenterlineResult(np.zeros(shape, dtype=bool), zeros, zeros.copy(), 0.0)

    normalized = _normalize_tiled_response(np, fused_response, origin, settings)
    centerline = (
        _tiled_centerline(np, ndimage, threshold_otsu, skeletonize, fused_response, origin, settings)
        if np.any(normalized >= settings.minimum_normalized_response)
        else np.zeros(shape, dtype=bool)
    )
    centerline = _prune_short_spurs(np, centerline, origin, settings)
    center_score = (normalized * np.float32(settings.foreground_weight)).astype(np.float32)
    center_score[centerline] = np.float32(settings.centerline_score)
    winning_scale = np.where(normalized > 0.0, winning_scale, 0.0).astype(np.float32)
    return InkCenterlineResult(
        centerline=centerline,
        center_score=center_score,
        scale_px=winning_scale,
        centerline_fraction=float(centerline.mean()),
    )


__all__ = [
    "ARCHAEOTRACE_UPSTREAM_COMMIT",
    "INK_BACKEND_ID",
    "InkBackendUnavailable",
    "InkCenterlineResult",
    "InkCenterlineSettings",
    "ink_centerline_candidates",
]
