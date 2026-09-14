"""Experimental source-raster linework recovery, not a contour classifier.

Keep short graph edges for context rather than discarding them before tracing.
Grow strong ridge seeds only through weak, locally aligned observed ink; no
white-gap bridging or endpoint pairing takes place here. Long weak components
are preserved separately even without a strong seed. Text avoidance can move
entire *isolated* draft components to a separate review set, never erase a box.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import math

from .ink import ink_centerline_candidates
from .vectorization import _trace_skeleton_array, _length


@dataclass(frozen=True)
class ObservedLineworkConfig:
    strong_support: float = .18
    weak_support: float = .025
    minimum_coherence: float = .3
    maximum_direction_difference_deg: float = 50.
    minimum_seeded_length: float = 3.
    minimum_unseeded_length: float = 16.
    minimum_normal_contrast: float = .008
    text_avoidance_threshold: float = .4
    text_fraction_threshold: float = .65
    max_pixels: int = 2_000_000

    def validate(self):
        values = asdict(self)
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
               for v in values.values()):
            raise ValueError("linework settings must be finite and positive")
        if not 0 < self.weak_support < self.strong_support <= 1:
            raise ValueError("weak support must be below strong support")
        if any(getattr(self, k) > 1 for k in ("minimum_coherence", "minimum_normal_contrast",
                                             "text_avoidance_threshold", "text_fraction_threshold")):
            raise ValueError("evidence thresholds must not exceed one")
        if not 15 <= self.maximum_direction_difference_deg <= 65:
            raise ValueError("direction gate must remain local")
        if self.minimum_unseeded_length < self.minimum_seeded_length:
            raise ValueError("unseeded weak components need more context")
        if type(self.max_pixels) is not int or not 64 <= self.max_pixels <= 4_000_000:
            raise ValueError("max_pixels must be a bounded integer")
        return self


def _gray(image, max_pixels):
    import numpy as np
    source = np.asarray(image)
    if source.ndim != 2 or min(source.shape) < 3 or source.size > max_pixels:
        raise ValueError("expected a bounded grayscale source raster")
    array = np.asarray(source, dtype=np.float32)
    if not np.isfinite(array).all() or array.min() < 0 or array.max() > 255:
        raise ValueError("grayscale must be finite in 0..255 or 0..1")
    if np.issubdtype(source.dtype, np.integer) or array.max() > 1:
        array = array/255.
    return array.copy()


def _score(value, shape):
    import numpy as np
    if value is None:
        return np.zeros(shape, np.float32)
    a = np.asarray(value, dtype=np.float32)
    if a.shape != shape or not np.isfinite(a).all() or a.min() < 0 or a.max() > 1:
        raise ValueError("text avoidance must share the source pixel grid and lie in 0..1")
    return a.copy()


def _directional_growth(weak, strong, tx, ty, cosine):
    import numpy as np
    seen = strong.copy()
    queue = deque(zip(*np.nonzero(seen)))
    offsets = [(dx, dy, math.hypot(dx, dy)) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dx or dy]
    while queue:
        y, x = queue.popleft()
        for dx, dy, distance in offsets:
            nx, ny = x+dx, y+dy
            if not 0 <= ny < weak.shape[0] or not 0 <= nx < weak.shape[1] or seen[ny, nx] or not weak[ny, nx]:
                continue
            aligned = max(abs((tx[y, x]*dx+ty[y, x]*dy)/distance),
                          abs((tx[ny, nx]*dx+ty[ny, nx]*dy)/distance))
            orientation = abs(tx[y, x]*tx[ny, nx]+ty[y, x]*ty[ny, nx])
            if aligned < cosine or orientation < cosine:
                continue
            seen[ny, nx] = True
            queue.append((ny, nx))
    return seen


def extract_observed_linework(gray, *, text_avoidance_score=None, tile_origin=(0, 0),
                              config=ObservedLineworkConfig()):
    """Return fresh, review-only pixel paths with stage counts and masks.

    Output paths retain every skeleton vertex. An integer point is a source
    pixel centre. `short_context` edges remain available for future regional
    tracing, but are not claimed to be contours. Inferred white gaps are absent.
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter, map_coordinates, label
    from skimage.morphology import skeletonize

    config.validate()
    image = _gray(gray, config.max_pixels)
    avoidance = _score(text_avoidance_score, image.shape)
    evidence = ink_centerline_candidates(image, tile_origin=tile_origin)
    support, tx, ty = evidence.support_score, evidence.tangent_x, evidence.tangent_y
    # Ink's ``coherence`` is tensor anisotropy MULTIPLIED by local support.
    # Comparing it directly to a fixed .3 gate would reject a perfectly
    # oriented faint stroke simply because it is faint (regression tested).
    orientation_coherence = np.clip(np.divide(evidence.coherence, support,
        out=np.zeros_like(support), where=support > 1e-6), 0, 1)
    yy, xx = np.indices(image.shape, dtype=np.float32)
    # A source-normal contrast floor prevents a locally renormalized flat
    # background from becoming observed ink. The continuous score is not the
    # saturated 1.0 assigned to old skeleton centres.
    blurred = gaussian_filter(image, .55)
    left = map_coordinates(blurred, [yy-2*tx, xx+2*ty], order=1, mode="nearest")
    right = map_coordinates(blurred, [yy+2*tx, xx-2*ty], order=1, mode="nearest")
    contrast = .5*(left+right)-blurred
    weak = ((support >= config.weak_support) & (contrast >= config.minimum_normal_contrast)
            & (orientation_coherence >= config.minimum_coherence))
    strong = weak & (support >= config.strong_support)
    grown = _directional_growth(weak, strong, tx, ty,
                               math.cos(math.radians(config.maximum_direction_difference_deg)))
    # Retain isolated faint strokes if they have enough observed extent.
    # Their weaker evidence remains explicit; no new strong seed is invented.
    weak_skeleton = skeletonize(weak)
    components, count = label(weak, structure=np.ones((3, 3), dtype=bool))
    component_lengths = np.zeros(count+1, dtype=float)
    for points in _trace_skeleton_array(weak_skeleton, np):
        x, y = map(int, points[0])
        component_lengths[components[y, x]] += _length(points)
    component_has_strong = np.bincount(components[strong], minlength=count+1) > 0
    # A failed directional continuation inside a seeded component must NOT
    # sneak back through the independent faint-stroke preservation branch.
    # Use Euclidean graph length, not vertex counts (diagonals are sqrt(2)px).
    long_weak = weak & ~component_has_strong[components] & (component_lengths[components] >= config.minimum_unseeded_length)
    retained = grown | long_weak
    skeleton = skeletonize(retained)
    # Global connected-component ownership prevents a character branch at a
    # shared junction from being silently cut off using a line-level fraction.
    graph_components, graph_count = label(skeleton, structure=np.ones((3, 3), dtype=bool))
    glyph_pixels = avoidance >= config.text_avoidance_threshold
    total = np.bincount(graph_components[skeleton], minlength=graph_count+1)
    text = np.bincount(graph_components[skeleton & glyph_pixels], minlength=graph_count+1)
    text_components = np.divide(text, total, out=np.zeros_like(text, dtype=float), where=total > 0) >= config.text_fraction_threshold
    text_components[0] = False
    paths = []
    for index, points in enumerate(_trace_skeleton_array(skeleton, np), 1):
        coords = np.asarray(points, dtype=int)
        y, x = coords[:, 1], coords[:, 0]
        length = _length(points)
        has_seed = bool(grown[y, x].any())
        if length < config.minimum_seeded_length:
            role = "short_context"
        elif bool(text_components[graph_components[y, x]].all()):
            role = "suspected_text_review"
        elif not has_seed and length < config.minimum_unseeded_length:
            role = "short_context"
        else:
            role = "observed_linework_review"
        paths.append({"path_id": f"observed-{index:06d}", "points": [list(p) for p in points],
                      "length_px": length, "role": role, "strong_seed_connected": has_seed,
                      "mean_continuous_support": float(support[y, x].mean()),
                      "text_avoidance_fraction": float(glyph_pixels[y, x].mean()),
                      "human_approved": False, "contour_semantics_assigned": False,
                      "training_eligible": False, "inferred_gap": False})
    masks = {"legacy_centerline": evidence.centerline.copy(), "weak_observed": weak,
             "strong_seeds": strong, "directionally_grown": grown, "retained_observed": retained,
             "skeleton": skeleton, "suspected_text_skeleton": skeleton & text_components[graph_components]}
    for array in masks.values():
        array.setflags(write=False)
    return {"schema": "jap-map-observed-linework/1", "paths": paths, "masks": masks,
            "stage_counts": {name: int(mask.sum()) for name, mask in masks.items()},
            "config": asdict(config), "source_raster_modified": False, "human_approvals": 0,
            "training_eligible": False, "contour_semantics_assigned": False,
            "limitations": ["Long weak components can include non-contour ink; human semantics are still required.",
                            "No white-gap inference, endpoint pairing or full-network topology resolution.",
                            "Source-first paths do not overwrite or inherit approved geometry."]}
