"""Bounded source-ink ownership evidence inside machine text search regions.

This module does not erase pixels or assign text/contour truth. Its soft
avoidance is a fixed review penalty on isolated compact, dark, nonlinear ink.
Loops, near-straight strokes, connected junctions and uncertain continuations
remain protected evidence. Detector scores are recorded, never probabilities.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math


SCHEMA = "jap-map-text-ownership/1"
REVIEW_FLAGS = {"dataset_role": "review_only_not_training", "human_approved": False,
                "training_eligible": False, "automatic_promotion": False,
                "glyph_truth_assigned": False, "contour_truth_assigned": False,
                "elevation_assigned": False, "source_mutated": False}


@dataclass(frozen=True)
class TextOwnershipConfig:
    background_window_px: int = 31
    ink_contrast_floor: float = 10.0
    glyph_contrast_floor: float = 40.0
    glyph_max_gray: float = 200.0
    context_halo_px: int = 12
    min_component_pixels: int = 5
    max_glyph_pixels: int = 1024
    max_glyph_span_px: int = 64
    min_tortuosity: float = 1.45
    straight_variance_ratio: float = 0.035
    outside_extent_px: float = 4.0
    min_boundary_separation_px: float = 6.0
    continuation_angle_degrees: float = 45.0
    soft_avoidance_weight: float = 0.55
    max_image_pixels: int = 16_777_216
    max_regions: int = 1024
    max_components_per_region: int = 256
    max_region_work_pixels: int = 262_144
    max_total_work_pixels: int = 16_777_216

    def validate(self):
        integer_fields = ("background_window_px", "context_halo_px", "min_component_pixels",
                          "max_glyph_pixels", "max_glyph_span_px", "max_image_pixels", "max_regions",
                          "max_components_per_region",
                          "max_region_work_pixels", "max_total_work_pixels")
        if any(type(getattr(self, key)) is not int or getattr(self, key) <= 0 for key in integer_fields):
            raise ValueError("ownership work limits and pixel counts must be positive integers")
        if not 3 <= self.background_window_px <= 127 or self.background_window_px % 2 != 1:
            raise ValueError("background_window_px must be odd and in [3, 127]")
        if not 4 <= self.context_halo_px <= 64 or self.max_regions > 4096 or self.max_image_pixels > 67_108_864:
            raise ValueError("ownership input/context limits exceed the bounded adapter")
        if (self.max_components_per_region > 1024 or self.max_region_work_pixels > 1_048_576
                or self.max_total_work_pixels > 67_108_864 or self.max_glyph_pixels > 16_384
                or self.max_glyph_span_px > 256 or self.min_component_pixels > self.max_glyph_pixels):
            raise ValueError("ownership component/work limits exceed the bounded adapter")
        numeric = ("ink_contrast_floor", "glyph_contrast_floor", "glyph_max_gray", "min_tortuosity",
                   "straight_variance_ratio", "outside_extent_px", "min_boundary_separation_px",
                   "continuation_angle_degrees", "soft_avoidance_weight")
        if any(type(getattr(self, key)) not in (int, float) or not math.isfinite(getattr(self, key)) for key in numeric):
            raise ValueError("ownership thresholds must be finite numbers")
        if (not 0 < self.ink_contrast_floor <= self.glyph_contrast_floor <= 255
                or not 0 < self.glyph_max_gray < 255 or not 1 < self.min_tortuosity <= 4
                or not 0 < self.straight_variance_ratio < 0.5
                or not 0 < self.outside_extent_px <= self.context_halo_px
                or not 0 < self.min_boundary_separation_px <= 128
                or not 0 < self.continuation_angle_degrees <= 60
                or not 0 < self.soft_avoidance_weight <= 0.65):
            raise ValueError("ownership thresholds are outside conservative bounds")
        return self


@dataclass(frozen=True)
class TextOwnershipResult:
    source_ink: object
    text_region: object
    soft_text_avoidance: object
    ambiguous_ink: object
    throughgoing_ink: object
    region_evidence: tuple
    provenance: dict

    def __post_init__(self):
        import numpy as np
        arrays = {}
        for name in ("source_ink", "text_region", "soft_text_avoidance", "ambiguous_ink", "throughgoing_ink"):
            arrays[name] = np.array(getattr(self, name), dtype=np.float32 if name == "soft_text_avoidance" else bool, copy=True)
        shape = arrays["source_ink"].shape
        if len(shape) != 2 or not all(shape) or any(a.shape != shape for a in arrays.values()):
            raise ValueError("ownership arrays must share one non-empty source grid")
        soft, ambiguous, through = (arrays[name] for name in ("soft_text_avoidance", "ambiguous_ink", "throughgoing_ink"))
        if not np.isfinite(soft).all() or np.any((soft < 0) | (soft > .65)):
            raise ValueError("soft avoidance must be finite in [0, 0.65]")
        expected = arrays["source_ink"] & arrays["text_region"]
        if (np.any((soft > 0) & (ambiguous | through)) or np.any(ambiguous & through)
                or not np.array_equal((soft > 0) | ambiguous | through, expected)):
            raise ValueError("ownership categories must partition only source ink inside text regions")
        for name, array in arrays.items():
            array.setflags(write=False)
            object.__setattr__(self, name, array)


def _gray255(gray, config, np):
    source = np.asarray(gray)
    if (source.ndim != 2 or not all(source.shape) or source.size > config.max_image_pixels
            or source.dtype.kind not in "uif" or not np.isfinite(source).all()):
        raise ValueError("gray must be a bounded, finite numeric 2-D source image")
    low, high = float(source.min()), float(source.max())
    if low < 0 or high > 255:
        raise ValueError("gray must be 0..255, or floating-point 0..1")
    values = np.array(source, dtype=np.float32, order="C", copy=True)
    if source.dtype.kind == "f" and high <= 1:
        values *= 255.0
    return values


def _polygons(polygons, scores, shape, config, np):
    height, width = shape
    if not isinstance(polygons, (list, tuple, np.ndarray)) or len(polygons) > config.max_regions:
        raise ValueError("text_polygons must be a bounded sequence on the source pixel-centre grid")
    if scores is not None and (not isinstance(scores, (list, tuple, np.ndarray)) or len(scores) != len(polygons)):
        raise ValueError("detector score count must match polygons")
    output = []
    for index, polygon in enumerate(polygons):
        source = np.asarray(polygon)
        if (source.ndim != 2 or source.shape[1] != 2 or not 3 <= source.shape[0] <= 64
                or source.dtype.kind not in "uif" or not np.isfinite(source).all()):
            raise ValueError("each polygon needs 3..64 finite x/y vertices")
        points = np.array(source, dtype=np.float64, copy=True)
        if (np.any(points[:, 0] < -.5) or np.any(points[:, 0] > width-.5)
                or np.any(points[:, 1] < -.5) or np.any(points[:, 1] > height-.5)):
            raise ValueError("polygon leaves centre-grid source bounds; convert raw detector corners by subtracting 0.5")
        area2 = float(np.sum(points[:, 0]*np.roll(points[:, 1], -1)-np.roll(points[:, 0], -1)*points[:, 1]))
        if abs(area2) < 1e-6:
            raise ValueError("text polygon has zero area")
        score = None if scores is None else scores[index]
        if score is not None:
            if isinstance(score, (bool, np.bool_)) or not isinstance(score, (int, float, np.integer, np.floating)):
                raise ValueError("detector scores must be finite numeric raw evidence")
            score = float(score)
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("detector scores must be finite in [0, 1]")
        output.append((points, score))
    return output


def _polygon_mask(points, shape, halo, np):
    height, width = shape
    x0 = max(0, int(math.floor(float(points[:, 0].min())))-halo)
    y0 = max(0, int(math.floor(float(points[:, 1].min())))-halo)
    x1 = min(width, int(math.ceil(float(points[:, 0].max())))+halo+1)
    y1 = min(height, int(math.ceil(float(points[:, 1].max())))+halo+1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    inside = np.zeros(xx.shape, dtype=bool)
    boundary = np.zeros(xx.shape, dtype=bool)
    for first, second in zip(points, np.roll(points, -1, axis=0)):
        ax, ay = first
        bx, by = second
        cross = (xx-ax)*(by-ay)-(yy-ay)*(bx-ax)
        boundary |= ((np.abs(cross) <= 1e-8) & (xx >= min(ax, bx)-1e-8) & (xx <= max(ax, bx)+1e-8)
                     & (yy >= min(ay, by)-1e-8) & (yy <= max(ay, by)+1e-8))
        if by != ay:
            inside ^= ((ay > yy) != (by > yy)) & (xx < (bx-ax)*(yy-ay)/(by-ay)+ax)
    return (slice(y0, y1), slice(x0, x1)), inside | boundary


def _topology(component, skeletonize, np):
    skeleton = skeletonize(component)
    padded = np.pad(skeleton, 1)
    height, width = skeleton.shape
    degree = np.zeros(skeleton.shape, dtype=np.uint8)
    edge_length = 0.
    for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
        neighbor = padded[1+dy:1+dy+height, 1+dx:1+dx+width]
        connected = skeleton & neighbor
        if dx and dy:
            horizontal = padded[1:1+height, 1+dx:1+dx+width]
            vertical = padded[1+dy:1+dy+height, 1:1+width]
            connected &= ~(horizontal | vertical)
        degree += connected
        edge_length += float(connected.sum()) * (math.sqrt(2) if dx and dy else 1.) / 2.
    endpoints = np.column_stack(np.nonzero(skeleton & (degree == 1)))
    return skeleton, endpoints, int(np.count_nonzero(skeleton & (degree >= 3))), edge_length


def _continuation(component, inside, skeleton, endpoints, config, ndi, np):
    """Require two separated exits, exterior extension and compatible directions."""
    if len(endpoints) != 2 or any(inside[tuple(p)] for p in endpoints):
        return False, {"reason": "one_sided_or_unresolved_external_connection"}
    outside_labels, _ = ndi.label(component & ~inside, structure=np.ones((3, 3), bool))
    endpoint_labels = [int(outside_labels[tuple(p)]) for p in endpoints]
    if min(endpoint_labels) == 0 or endpoint_labels[0] == endpoint_labels[1]:
        return False, {"reason": "external_endpoints_do_not_cross_distinct_boundaries"}
    contacts, directions, extents = [], [], []
    for label in endpoint_labels:
        outside = outside_labels == label
        contact = skeleton & inside & ndi.binary_dilation(outside, structure=np.ones((3, 3), bool))
        contact_points = np.column_stack(np.nonzero(contact)).astype(float)
        outside_points = np.column_stack(np.nonzero(outside & skeleton)).astype(float)
        if len(contact_points) == 0 or len(outside_points) < 2:
            return False, {"reason": "insufficient_boundary_direction_evidence"}
        center = contact_points.mean(axis=0)
        distances = np.linalg.norm(outside_points-center, axis=1)
        extension = float(distances.max())
        if extension < config.outside_extent_px:
            return False, {"reason": "external_extension_too_short"}
        # Measure direction from actual exterior ink within a bounded lookback.
        selected = (distances >= min(config.outside_extent_px, extension*.5)) & (distances <= config.context_halo_px+1)
        if not selected.any():
            return False, {"reason": "insufficient_boundary_direction_evidence"}
        direction = outside_points[selected].mean(axis=0)-center
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-6:
            return False, {"reason": "external_direction_cancels"}
        contacts.append(center)
        directions.append(direction/norm)
        extents.append(extension)
    chord = contacts[1]-contacts[0]
    distance = float(np.linalg.norm(chord))
    if distance < config.min_boundary_separation_px:
        return False, {"reason": "boundary_contacts_too_close"}
    unit = chord/distance
    alignment = min(float(np.dot(directions[0], -unit)), float(np.dot(directions[1], unit)))
    opposition = -float(np.dot(directions[0], directions[1]))
    minimum = math.cos(math.radians(config.continuation_angle_degrees))
    if alignment < minimum or opposition < minimum:
        return False, {"reason": "boundary_directions_do_not_support_throughgoing_ink"}
    return True, {"reason": "two_separated_directionally_supported_boundary_crossings",
                  "boundary_separation_px": distance, "outside_extents_px": extents,
                  "minimum_direction_alignment": alignment, "exterior_direction_opposition": opposition}


def derive_text_ownership(gray, text_polygons, detector_scores=None, *, config=TextOwnershipConfig()):
    """Return source-grid ownership evidence without changing any input.

    ``gray`` is uint/integer 0..255 or float 0..1 / 0..255. Polygons MUST
    use integer pixel CENTRES, matching ``map_to_pixel``; raw OCR corner
    coordinates must first subtract 0.5. All three evidence arrays are zero
    outside observed source ink intersecting the actual polygon interiors.

    ``throughgoing_ink`` requires two separated boundary crossings with
    compatible exterior tangents. A one-sided connection or a junction is
    ambiguous. The soft weight is a provisional cost, not text probability.
    """
    import numpy as np
    from scipy import ndimage as ndi
    from skimage.morphology import skeletonize

    config.validate()
    values = _gray255(gray, config, np)
    polygons = _polygons(text_polygons, detector_scores, values.shape, config, np)
    background = ndi.maximum_filter(values, size=config.background_window_px, mode="nearest")
    contrast = background-values
    source_ink = (contrast >= config.ink_contrast_floor) & (values < 250)
    text_region = np.zeros(values.shape, bool)
    ambiguous = np.zeros(values.shape, bool)
    throughgoing = np.zeros(values.shape, bool)
    avoidance = np.zeros(values.shape, np.float32)
    records = []
    total_work = 0
    for index, (polygon, detector_score) in enumerate(polygons):
        estimated_width = min(values.shape[1], int(math.ceil(float(polygon[:, 0].max())))+config.context_halo_px+1)-max(0, int(math.floor(float(polygon[:, 0].min())))-config.context_halo_px)
        estimated_height = min(values.shape[0], int(math.ceil(float(polygon[:, 1].max())))+config.context_halo_px+1)-max(0, int(math.floor(float(polygon[:, 1].min())))-config.context_halo_px)
        if total_work+estimated_width*estimated_height > config.max_total_work_pixels:
            raise ValueError("total polygon work exceeds the configured ownership budget")
        roi, inside = _polygon_mask(polygon, values.shape, config.context_halo_px, np)
        total_work += inside.size
        text_region[roi] |= inside
        ink = source_ink[roi]
        scoped = ink & inside
        record = {"region_index": index, "detector_score_not_probability": detector_score,
                  "polygon_on_source_pixel_center_grid": polygon.tolist(), "source_ink_pixels": int(scoped.sum()),
                  "components": [], **REVIEW_FLAGS}
        if inside.size > config.max_region_work_pixels:
            ambiguous[roi] |= scoped
            record.update(status="deferred_work_bound", decision_counts={"work_bound": int(scoped.sum())})
            records.append(record)
            continue
        labels, _ = ndi.label(ink, structure=np.ones((3, 3), bool))
        component_ids = np.unique(labels[scoped])
        if len(component_ids) > config.max_components_per_region:
            ambiguous[roi] |= scoped
            record.update(status="deferred_component_bound", decision_counts={"component_bound": len(component_ids)})
            records.append(record)
            continue
        region_ambiguous = scoped.copy()
        region_through = np.zeros(inside.shape, bool)
        region_soft = np.zeros(inside.shape, np.float32)
        for component_id in component_ids:
            component = labels == component_id
            owned = component & inside
            area = int(component.sum())
            ys, xs = np.nonzero(component)
            span = max(int(xs.max()-xs.min()+1), int(ys.max()-ys.min()+1))
            info = {"component_id": int(component_id), "component_ink_pixels": area,
                    "region_ink_pixels": int(owned.sum()), "span_px": span, "decision": "ambiguous_ink"}
            if area < config.min_component_pixels:
                info["reason"] = "tiny_ink_component"
            elif (np.any(component[0]) or np.any(component[-1]) or np.any(component[:, 0]) or np.any(component[:, -1])) and not (component & ~inside).any():
                info["reason"] = "source_edge_truncation"
            else:
                skeleton, endpoints, branches, path_length = _topology(component, skeletonize, np)
                info.update(skeleton_endpoints=len(endpoints), skeleton_branch_pixels=branches)
                holes = int(ndi.label(ndi.binary_fill_holes(component) & ~component)[1])
                if branches:
                    info["reason"] = "connected_junction_ownership_unresolved"
                elif (component & ~inside).any():
                    continued, details = _continuation(component, inside, skeleton, endpoints, config, ndi, np)
                    info.update(details)
                    if continued:
                        info["decision"] = "throughgoing_ink"
                        region_through |= owned
                        region_ambiguous &= ~owned
                elif holes or len(endpoints) == 0:
                    info.update(reason="closed_loop_zero_or_contour_unresolved", hole_count=holes)
                elif area > config.max_glyph_pixels or span > config.max_glyph_span_px:
                    info["reason"] = "component_not_compact"
                elif len(endpoints) != 2:
                    info["reason"] = "component_topology_unresolved"
                else:
                    points = np.column_stack((ys, xs)).astype(float)
                    eigenvalues = np.linalg.eigvalsh(np.cov(points, rowvar=False))
                    variance_ratio = float(max(0., eigenvalues[0])/max(1e-9, eigenvalues.sum()))
                    chord = float(np.linalg.norm(endpoints[0]-endpoints[1]))
                    tortuosity = path_length/max(1., chord)
                    median_contrast = float(np.median(contrast[roi][component]))
                    median_gray = float(np.median(values[roi][component]))
                    info.update(variance_ratio=variance_ratio, tortuosity=tortuosity,
                                median_contrast=median_contrast, median_gray=median_gray)
                    if variance_ratio <= config.straight_variance_ratio or tortuosity < config.min_tortuosity:
                        info["reason"] = "near_straight_one_or_short_line_unresolved"
                    elif median_contrast < config.glyph_contrast_floor or median_gray > config.glyph_max_gray:
                        info["reason"] = "faint_compact_ink_not_owned_as_text"
                    else:
                        info.update(decision="soft_text_avoidance", reason="isolated_dark_compact_nonlinear_ink",
                                    weight_not_probability=config.soft_avoidance_weight)
                        region_soft[owned] = config.soft_avoidance_weight
                        region_ambiguous &= ~owned
            record["components"].append(info)
        ambiguous[roi] |= region_ambiguous
        throughgoing[roi] |= region_through
        np.maximum(avoidance[roi], region_soft, out=avoidance[roi])
        record.update(status="evidence_only", decision_counts=dict(Counter(c["reason"] for c in record["components"])))
        records.append(record)
    # Overlapping search regions can disagree. Unresolved ownership always
    # protects ink; supported continuation also overrides any soft penalty.
    throughgoing &= ~ambiguous
    avoidance[ambiguous | throughgoing] = 0
    provenance = {"schema": SCHEMA, "configuration": asdict(config),
                  "pixel_coordinate_convention": "integer source pixel centres; raw detector corners must subtract 0.5",
                  "source_ink_method": "raw grayscale contrast against bounded local maximum; no filled/expanded ink mask",
                  "score_policy": "detector scores recorded only; soft avoidance is a fixed provisional cost, never probability",
                  "overlap_policy": "ambiguous ownership protects first, then supported throughgoing ink, then soft avoidance",
                  "work_pixels": total_work, "counts": {"regions": len(records), "source_ink": int(source_ink.sum()),
                      "ink_in_regions": int((source_ink & text_region).sum()), "soft_text_avoidance": int((avoidance > 0).sum()),
                      "ambiguous_ink": int(ambiguous.sum()), "throughgoing_ink": int(throughgoing.sum())},
                  "limitations": ["Compact bent contours can resemble glyphs; soft evidence requires independent review.",
                                  "Closed zero/contour loops, near-straight one/line strokes and junctions remain unresolved.",
                                  "Missed detector regions receive no ownership inference.",
                                  "No pixels are erased; no elevation or semantic truth is assigned."], **REVIEW_FLAGS}
    return TextOwnershipResult(source_ink, text_region, avoidance, ambiguous, throughgoing, tuple(records), provenance)
