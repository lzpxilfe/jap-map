"""Geometric search quads from connected dark ink, independent of OCR detection.

Groups are deliberately ambiguous recognizer inputs, never glyph masks or
text/contour truth. A connected component can contain several touching glyphs.
No component is split to manufacture characters, and no source pixel is erased.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import math


SCHEMA = "jap-map-component-text-candidates/2"
FLAGS = {"dataset_role": "review_only_not_training", "human_approved": False,
         "training_eligible": False, "automatic_promotion": False, "ambiguous": True,
         "glyph_truth_assigned": False, "contour_truth_assigned": False,
         "elevation_assigned": False, "erase_mask_generated": False}


@dataclass(frozen=True)
class ComponentTextCandidateConfig:
    background_window_px: int = 31
    ink_contrast_floor: float = 40.0
    dark_gray_ceiling: float = 160.0
    min_component_pixels: int = 5
    max_component_pixels: int = 768
    min_component_span_px: int = 4
    max_component_span_px: int = 64
    min_projected_height_px: float = 4.0
    max_projected_height_px: float = 64.0
    max_height_ratio: float = 1.85
    max_width_height_ratio: float = 2.25
    max_baseline_residual_ratio: float = .35
    max_centroid_residual_ratio: float = .30
    min_gap_height_ratio: float = -.12
    max_gap_height_ratio: float = 1.10
    max_spacing_ratio: float = 2.5
    neighbor_distance_height_ratio: float = 3.0
    quad_padding_height_ratio: float = .15
    minimum_quad_padding_px: float = 1.0
    max_group_members: int = 5
    max_neighbors: int = 12
    max_components_for_grouping: int = 512
    max_total_components: int = 16_384
    max_pairs: int = 4096
    max_group_evaluations: int = 8192
    max_groups: int = 128
    max_image_pixels: int = 4_194_304
    min_enclosed_hole_pixels: int = 4
    max_enclosed_holes_per_component: int = 8
    max_hole_axis_evaluations: int = 1024
    min_hole_axis_linearity: float = .85
    max_hole_height_ratio: float = 3.0
    hole_band_margin_height_ratio: float = .5

    def validate(self):
        integers = ("background_window_px", "min_component_pixels", "max_component_pixels",
                    "min_component_span_px", "max_component_span_px", "max_group_members", "max_neighbors",
                    "max_components_for_grouping", "max_total_components", "max_pairs", "max_group_evaluations",
                    "max_groups", "max_image_pixels", "min_enclosed_hole_pixels",
                    "max_enclosed_holes_per_component", "max_hole_axis_evaluations")
        if any(type(getattr(self, k)) is not int or getattr(self, k) <= 0 for k in integers):
            raise ValueError("component counts and work limits must be positive integers")
        if (not 3 <= self.background_window_px <= 127 or self.background_window_px % 2 != 1
                or not 2 <= self.max_group_members <= 5 or self.max_neighbors > 32
                or self.max_components_for_grouping > 2048 or self.max_total_components > 65_536
                or self.max_pairs > 16_384 or self.max_group_evaluations > 65_536
                or self.max_groups > 1024 or self.max_image_pixels > 16_777_216
                or self.min_enclosed_hole_pixels > 128 or not 2 <= self.max_enclosed_holes_per_component <= 16
                or self.max_hole_axis_evaluations > 8192):
            raise ValueError("candidate settings exceed bounded work limits")
        if (not self.min_component_pixels <= self.max_component_pixels <= 16_384
                or not self.min_component_span_px <= self.max_component_span_px <= 256):
            raise ValueError("invalid compact-component size bounds")
        numeric = set(asdict(self))-set(integers)
        if any(type(getattr(self, k)) not in (int, float) or not math.isfinite(getattr(self, k)) for k in numeric):
            raise ValueError("candidate thresholds must be finite numbers")
        if (not 0 < self.ink_contrast_floor < 255 or not 0 < self.dark_gray_ceiling < 245
                or not 1 <= self.min_projected_height_px <= self.max_projected_height_px <= 256
                or not 1 < self.max_height_ratio <= 3 or not 1 <= self.max_width_height_ratio <= 4
                or not 0 < self.max_baseline_residual_ratio <= .6
                or not 0 < self.max_centroid_residual_ratio <= .5
                or not -.25 <= self.min_gap_height_ratio <= 0
                or not .1 <= self.max_gap_height_ratio <= 2
                or not 1 <= self.max_spacing_ratio <= 4
                or not 1 <= self.neighbor_distance_height_ratio <= 5
                or not 0 <= self.quad_padding_height_ratio <= .5
                or not 0 <= self.minimum_quad_padding_px <= 8
                or not .75 <= self.min_hole_axis_linearity <= 1
                or not 1 < self.max_hole_height_ratio <= 4
                or not .25 <= self.hole_band_margin_height_ratio <= 1):
            raise ValueError("candidate geometric thresholds exceed bounded search limits")
        return self


@dataclass(frozen=True)
class ComponentTextCandidateResult:
    source_ink: object
    component_labels: object
    components: tuple
    groups: tuple
    provenance: dict

    def __post_init__(self):
        import numpy as np
        ink = np.array(self.source_ink, dtype=bool, copy=True)
        labels = np.array(self.component_labels, dtype=np.int32, copy=True)
        if (ink.ndim != 2 or not all(ink.shape) or labels.shape != ink.shape
                or np.any(labels < 0) or not np.array_equal(labels > 0, ink)):
            raise ValueError("component labels must partition the original thresholded ink grid")
        ink.setflags(write=False)
        labels.setflags(write=False)
        object.__setattr__(self, "source_ink", ink)
        object.__setattr__(self, "component_labels", labels)


def _prepare(gray, origin, config, np):
    raw = np.asarray(gray)
    if (raw.ndim != 2 or not all(raw.shape) or raw.size > config.max_image_pixels
            or raw.dtype.kind not in "uif" or not np.isfinite(raw).all()
            or raw.min() < 0 or raw.max() > 255):
        raise ValueError("gray must be finite 2-D pixels in 0..255, or floating-point 0..1")
    if (not isinstance(origin, (list, tuple)) or len(origin) != 2
            or any(type(v) is not int or v < 0 for v in origin)):
        raise ValueError("source_origin_xy must be two nonnegative integer crop offsets")
    pixels = np.array(raw, dtype=np.float32, copy=True)
    if raw.dtype.kind == "f" and float(raw.max()) <= 1:
        pixels *= 255
    identity = hashlib.sha256(json.dumps({"shape": list(raw.shape), "dtype": str(raw.dtype)}, sort_keys=True).encode()
                              + np.ascontiguousarray(raw).tobytes()).hexdigest()
    return pixels, np.asarray(origin, dtype=float), identity


def _components(values, contrast, labels, count, config, ndi, np):
    records, eligible = [], {}
    for cid, window in enumerate(ndi.find_objects(labels), 1):
        mask = labels[window] == cid
        local_y, local_x = np.nonzero(mask)
        points = np.column_stack((local_x+window[1].start, local_y+window[0].start)).astype(float)
        area = len(points)
        width, height = window[1].stop-window[1].start, window[0].stop-window[0].start
        reasons = []
        if area < config.min_component_pixels: reasons.append("too_few_ink_pixels")
        if area > config.max_component_pixels: reasons.append("connected_component_too_large_not_split")
        if max(width, height) < config.min_component_span_px: reasons.append("component_span_too_small")
        if max(width, height) > config.max_component_span_px: reasons.append("long_or_large_component_not_split")
        if window[0].start == 0 or window[1].start == 0 or window[0].stop == values.shape[0] or window[1].stop == values.shape[1]:
            reasons.append("crop_edge_component_truncated")
        median_contrast = float(np.median(contrast[window][mask]))
        ink_axis, ink_linearity = _principal_axis(points, np)
        record = {"component_id": cid, "ink_pixels": area,
                  "bbox_crop_pixel_centers": [window[1].start, window[0].start, window[1].stop-1, window[0].stop-1],
                  "centroid_crop_pixel_centers": points.mean(axis=0).tolist(),
                  "width_px": width, "height_px": height,
                  "median_gray": float(np.median(values[window][mask])), "median_local_contrast": median_contrast,
                  "ink_principal_axis_degrees": float(math.degrees(math.atan2(ink_axis[1], ink_axis[0]))),
                  "ink_pca_linearity": ink_linearity,
                  "eligible_for_grouping": not reasons, "exclusion_reasons": reasons,
                  "character_type": "unknown", "component_may_contain_multiple_touching_glyphs": True, **FLAGS}
        records.append(record)
        if not reasons:
            eligible[cid] = {"points": points, "center": points.mean(axis=0), "span": max(width, height),
                             "record": record, "priority": median_contrast*math.sqrt(area)}
    return records, eligible


def _group_geometry(ids, components, config, np):
    centers = np.asarray([components[cid]["center"] for cid in ids])
    centered = centers-centers.mean(axis=0)
    covariance = centered.T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if eigenvalues[-1] <= 1e-8:
        return None, "coincident_component_centroids"
    along = eigenvectors[:, -1]
    if along[0] < -1e-10 or abs(along[0]) <= 1e-10 and along[1] < 0:
        along = -along
    normal = np.array([-along[1], along[0]])
    order = np.argsort(centers @ along, kind="stable")
    ordered_ids = tuple(ids[int(i)] for i in order)
    projected = []
    # Cell extents are computed from observed pixel centres with a half-cell
    # allowance; padded search quads can include paper and unrelated ink.
    half_u = .5*float(np.abs(along).sum())
    half_v = .5*float(np.abs(normal).sum())
    for cid in ordered_ids:
        points = components[cid]["points"]
        u, v = points @ along, points @ normal
        projected.append([float(u.min())-half_u, float(v.min())-half_v,
                          float(u.max())+half_u, float(v.max())+half_v])
    projected = np.asarray(projected)
    widths, heights = projected[:, 2]-projected[:, 0], projected[:, 3]-projected[:, 1]
    median_height = float(np.median(heights))
    height_ratio = float(heights.max()/heights.min())
    if heights.min() < config.min_projected_height_px or heights.max() > config.max_projected_height_px:
        return None, "projected_glyph_height_out_of_bounds"
    if height_ratio > config.max_height_ratio:
        return None, "inconsistent_projected_heights"
    if np.any(widths/heights > config.max_width_height_ratio):
        return None, "component_is_long_along_group_axis"
    baseline_ratio = float(np.ptp(projected[:, 3])/median_height)
    residual_ratio = float(np.ptp(centers @ normal)/median_height)
    if baseline_ratio > config.max_baseline_residual_ratio or residual_ratio > config.max_centroid_residual_ratio:
        return None, "inconsistent_rotated_baseline"
    gaps = projected[1:, 0]-projected[:-1, 2]
    normalized_gaps = gaps/median_height
    if normalized_gaps.min() < config.min_gap_height_ratio or normalized_gaps.max() > config.max_gap_height_ratio:
        return None, "overlapping_or_excessively_spaced_components"
    center_positions = centers[order] @ along
    center_steps = np.diff(center_positions)
    spacing_ratio = float(center_steps.max()/max(1e-9, center_steps.min()))
    if spacing_ratio > config.max_spacing_ratio:
        return None, "inconsistent_component_spacing"
    # A fixed geometric rank, explicitly not a text/number probability.
    height_score = max(0., 1-(height_ratio-1)/(config.max_height_ratio-1))
    baseline_score = max(0., 1-max(baseline_ratio/config.max_baseline_residual_ratio,
                                  residual_ratio/config.max_centroid_residual_ratio))
    spacing_score = 1./spacing_ratio
    count_score = (len(ids)-1)/config.max_group_members
    rank_score = .25*height_score+.25*baseline_score+.20*spacing_score+.30*count_score
    padding = max(config.minimum_quad_padding_px, config.quad_padding_height_ratio*median_height)
    lo = projected[:, :2].min(axis=0)-padding
    hi = projected[:, 2:].max(axis=0)+padding
    uv = np.asarray([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
    quad = uv[:, 0, None]*along+uv[:, 1, None]*normal
    evidence = {"projected_heights_px": heights.tolist(), "projected_widths_px": widths.tolist(),
                "height_ratio": height_ratio, "baseline_residual_height_ratio": baseline_ratio,
                "centroid_residual_height_ratio": residual_ratio, "gaps_px": gaps.tolist(),
                "gaps_height_ratio": normalized_gaps.tolist(), "center_spacing_ratio": spacing_ratio,
                "baseline_angle_degrees": float(math.degrees(math.atan2(along[1], along[0]))),
                "centroid_pca_linearity": float(eigenvalues[-1]/eigenvalues.sum()),
                "centroid_pca_two_point_degeneracy": len(ids) == 2,
                "member_ink_axes_degrees": [components[cid]["record"]["ink_principal_axis_degrees"] for cid in ordered_ids],
                "quad_padding_px": padding, "component_count_is_not_character_count": True}
    return {"ids": ordered_ids, "quad": quad, "score": rank_score, "evidence": evidence}, None


def _principal_axis(points, np):
    centered = points-points.mean(axis=0)
    values, vectors = np.linalg.eigh(centered.T @ centered)
    along = vectors[:, -1]
    if along[0] < -1e-10 or abs(along[0]) <= 1e-10 and along[1] < 0:
        along = -along
    return along, float(values[-1]/max(float(values.sum()), 1e-12))


def _hole_axis_candidates(labels, eligible, config, ndi, np):
    """Search bands through multiple enclosed holes without splitting ink.

    A touching word and an attached contour can be one connected component.
    Its neighbours' centroids then describe the space *between lines*, not a
    word baseline. Enclosed-hole centroids provide an independent search axis.
    Whole-ink PCA is retained as a control hypothesis; every rejection remains
    auditable. Neither holes nor their parent component receive a glyph label.
    """
    candidates, attempts = [], []
    evaluations, capped = 0, False
    for cid, component in sorted(eligible.items()):
        record, points = component["record"], component["points"]
        x0, y0, x1, y1 = record["bbox_crop_pixel_centers"]
        mask = labels[y0:y1+1, x0:x1+1] == cid
        holes = ndi.binary_fill_holes(np.pad(mask, 1))[1:-1, 1:-1] & ~mask
        hole_labels, hole_count = ndi.label(holes)
        usable, hole_records = [], []
        for hid, window in enumerate(ndi.find_objects(hole_labels), 1):
            y, x = np.nonzero(hole_labels[window] == hid)
            hp = np.column_stack((x+window[1].start+x0, y+window[0].start+y0)).astype(float)
            item = {"hole_id": hid, "pixels": len(hp), "centroid_crop_pixel_centers": hp.mean(axis=0).tolist(),
                    "bbox_crop_pixel_centers": [int(hp[:, 0].min()), int(hp[:, 1].min()), int(hp[:, 0].max()), int(hp[:, 1].max())],
                    "used_for_axis": False, "reason": "below_minimum_hole_area"}
            hole_records.append(item)
            if len(hp) >= config.min_enclosed_hole_pixels:
                usable.append((hid, hp, item))
        usable.sort(key=lambda item: (-len(item[1]), item[0]))
        for index, (_hid, _hp, item) in enumerate(usable):
            item.update(used_for_axis=index < config.max_enclosed_holes_per_component,
                        reason="retained" if index < config.max_enclosed_holes_per_component else "enclosed_hole_cap")
        usable = usable[:config.max_enclosed_holes_per_component]
        record["enclosed_holes"] = hole_records
        record["enclosed_hole_count"] = int(hole_count)
        if len(usable) < 2:
            continue
        centers = np.asarray([hp.mean(axis=0) for _hid, hp, _item in usable])
        hole_axis, linearity = _principal_axis(centers, np)
        ink_axis, ink_linearity = _principal_axis(points, np)
        axes = (("enclosed_hole_centroids", hole_axis), ("whole_component_ink_pca_control", ink_axis))
        used_axes = []
        for name, along in axes:
            attempt = {"component_id": cid, "orientation_hypothesis": name,
                       "baseline_angle_degrees": float(math.degrees(math.atan2(along[1], along[0]))),
                       "hole_ids": [hid for hid, _hp, _item in usable], "hole_centroid_linearity": linearity,
                       "whole_component_ink_linearity": ink_linearity, "status": "rejected"}
            attempts.append(attempt)
            if evaluations >= config.max_hole_axis_evaluations:
                capped = True
                attempt["reason"] = "hole_orientation_evaluation_cap"
                continue
            evaluations += 1
            if any(abs(float(along @ old)) > math.cos(math.radians(5.)) for old in used_axes):
                attempt["reason"] = "duplicate_orientation_within_five_degrees"
                continue
            used_axes.append(along)
            if linearity < config.min_hole_axis_linearity:
                attempt["reason"] = "hole_centroids_not_collinear"
                continue
            normal = np.array([-along[1], along[0]])
            half_u, half_v = .5*float(np.abs(along).sum()), .5*float(np.abs(normal).sum())
            projected = np.asarray([[float((hp @ along).min())-half_u, float((hp @ normal).min())-half_v,
                                     float((hp @ along).max())+half_u, float((hp @ normal).max())+half_v]
                                    for _hid, hp, _item in usable])
            heights = projected[:, 3]-projected[:, 1]
            median_height = float(np.median(heights))
            height_ratio = float(heights.max()/heights.min())
            residual = float(np.ptp(centers @ normal)/median_height)
            attempt.update(projected_hole_heights_px=heights.tolist(), hole_height_ratio=height_ratio,
                           hole_centroid_residual_height_ratio=residual)
            if height_ratio > config.max_hole_height_ratio or residual > config.max_centroid_residual_ratio:
                attempt["reason"] = "inconsistent_hole_heights_or_axis_residual"
                continue
            margin = max(config.minimum_quad_padding_px, config.hole_band_margin_height_ratio*median_height)
            v0, v1 = float(projected[:, 1].min())-margin, float(projected[:, 3].max())+margin
            u, v = points @ along, points @ normal
            in_band = (v >= v0) & (v <= v1)
            if int(in_band.sum()) < config.min_component_pixels:
                attempt["reason"] = "insufficient_component_ink_in_hole_band"
                continue
            padding = max(config.minimum_quad_padding_px, config.quad_padding_height_ratio*median_height)
            u0, u1 = float(u[in_band].min())-half_u-padding, float(u[in_band].max())+half_u+padding
            v0, v1 = v0-padding, v1+padding
            contained = (u >= u0-1e-9) & (u <= u1+1e-9) & (v >= v0-1e-9) & (v <= v1+1e-9)
            inside_count = int(contained.sum())
            attempt.update(component_ink_pixels=len(points), component_ink_inside_quad_pixels=inside_count)
            if inside_count == len(points):
                attempt["reason"] = "no_partial_component_context"
                continue
            uv = np.asarray([[u0, v0], [u1, v0], [u1, v1], [u0, v1]])
            quad = uv[:, 0, None]*along+uv[:, 1, None]*normal
            steps = np.diff(np.sort(centers @ along))
            spacing_ratio = float(steps.max()/max(1e-9, float(steps.min())))
            height_score = max(0., 1.-(height_ratio-1.)/(config.max_hole_height_ratio-1.))
            axis_score = max(0., 1.-residual/config.max_centroid_residual_ratio)
            count_score = (min(len(usable), config.max_group_members)-1)/config.max_group_members
            score = .25*height_score+.25*axis_score+.20/spacing_ratio+.30*count_score
            evidence = {**attempt, "hole_count_is_not_character_count": True,
                        "component_count_is_not_character_count": True,
                        "component_ink_inside_quad_fraction": inside_count/len(points),
                        "hole_band_margin_px": margin, "quad_padding_px": padding,
                        "source_component_pixels_unchanged": True, "whole_component_ownership_supported": False}
            evidence.pop("status")
            attempt.update(status="candidate_before_output_cap", reason="independent_hole_axis_search",
                           quad_crop_pixel_centers=quad.tolist(), ranking_score_not_probability=score)
            candidates.append({"ids": (cid,), "quad": quad, "score": score, "evidence": evidence,
                               "candidate_kind": "intra_component_hole_axis", "variant": name,
                               "partial_component_search": True, "full_component_containment": False,
                               "attempt": attempt})
            # A hole-derived band can clip a descending stroke. Keep the base
            # hypothesis and offer one bounded context alternative, never grow
            # ink ownership or chase the connected contour to its end.
            in_u = (u >= u0) & (u <= u1)
            low_contact = bool(np.any(in_u & (np.abs(v-v0) <= half_v)))
            high_contact = bool(np.any(in_u & (np.abs(v-v1) <= half_v)))
            evidence["normal_boundary_ink_contact"] = [low_contact, high_contact]
            if not (low_contact or high_contact):
                continue
            variant = name+"_bounded_normal_context"
            expansion = {"component_id": cid, "orientation_hypothesis": variant,
                         "status": "rejected", "base_orientation_hypothesis": name}
            attempts.append(expansion)
            if evaluations >= config.max_hole_axis_evaluations:
                capped = True
                expansion["reason"] = "hole_orientation_evaluation_cap"
                continue
            evaluations += 1
            extra = .5*median_height
            expanded_v0 = v0-extra if low_contact else v0
            expanded_v1 = v1+extra if high_contact else v1
            expanded_inside = in_u & (v >= expanded_v0) & (v <= expanded_v1)
            expanded_count = int(expanded_inside.sum())
            if expanded_count == inside_count:
                expansion["reason"] = "bounded_context_adds_no_component_ink"
                continue
            uv_extra = np.asarray([[u0, expanded_v0], [u1, expanded_v0], [u1, expanded_v1], [u0, expanded_v1]])
            expanded_quad = uv_extra[:, 0, None]*along+uv_extra[:, 1, None]*normal
            expanded_evidence = {**evidence, "base_quad_crop_pixel_centers": quad.tolist(),
                "component_ink_inside_quad_pixels": expanded_count,
                "component_ink_inside_quad_fraction": expanded_count/len(points),
                "bounded_normal_context_px": extra,
                "context_may_include_attached_contour": True,
                "whole_component_ownership_supported": False}
            expansion.update(status="candidate_before_output_cap", reason="bounded_boundary_contact_context",
                             quad_crop_pixel_centers=expanded_quad.tolist(), ranking_score_not_probability=score-.01)
            candidates.append({"ids": (cid,), "quad": expanded_quad, "score": score-.01,
                "evidence": expanded_evidence, "candidate_kind": "intra_component_hole_axis", "variant": variant,
                "partial_component_search": True, "full_component_containment": False, "attempt": expansion})
    return candidates, attempts, evaluations, capped


def find_component_text_candidates(gray, *, source_origin_xy=(0, 0), config=ComponentTextCandidateConfig()):
    """Find bounded inter-component and partial intra-component search quads.

    ``source_origin_xy`` translates a crop to its containing tile/sheet grid.
    Pixel-centre quads and image-corner quads (+0.5) are both explicit. Quads
    are never silently clipped: downstream recognition must abstain when
    ``evidence.quad_within_crop_bounds`` is false.
    """
    import numpy as np
    from scipy import ndimage as ndi
    from scipy.spatial import cKDTree

    config.validate()
    values, origin, source_hash = _prepare(gray, source_origin_xy, config, np)
    contrast = ndi.maximum_filter(values, size=config.background_window_px, mode="nearest")-values
    source_ink = (contrast >= config.ink_contrast_floor) & (values <= config.dark_gray_ceiling)
    labels, count = ndi.label(source_ink, structure=np.ones((3, 3), bool))
    if count > config.max_total_components:
        raise ValueError("source connected-component count exceeds bounded candidate budget")
    records, eligible = _components(values, contrast, labels, count, config, ndi, np)
    truncation = {"component_grouping_cap": False, "pair_cap": False,
                  "group_evaluation_cap": False, "output_group_cap": False}
    if len(eligible) > config.max_components_for_grouping:
        selected = set(sorted(eligible, key=lambda cid: (-eligible[cid]["priority"], cid))[:config.max_components_for_grouping])
        for cid in set(eligible)-selected:
            eligible[cid]["record"]["eligible_for_grouping"] = False
            eligible[cid]["record"]["exclusion_reasons"].append("component_grouping_cap")
        eligible = {cid: eligible[cid] for cid in sorted(selected)}
        truncation["component_grouping_cap"] = True
    ids = sorted(eligible)
    pair_candidates = set()
    if len(ids) >= 2:
        tree = cKDTree(np.asarray([eligible[cid]["center"] for cid in ids]))
        _distances, neighbors = tree.query(tree.data, k=min(len(ids), config.max_neighbors+1), workers=1)
        for i, nearby in enumerate(neighbors):
            for j in np.atleast_1d(nearby):
                j = int(j)
                if j == i: continue
                first, second = sorted((ids[i], ids[j]))
                distance = float(np.linalg.norm(eligible[first]["center"]-eligible[second]["center"]))
                if distance <= config.neighbor_distance_height_ratio*max(eligible[first]["span"], eligible[second]["span"]):
                    pair_candidates.add((first, second))
    # Prefer nearby pairs before the declared cap, deterministically.
    ordered_pairs = sorted(pair_candidates, key=lambda pair: (float(np.linalg.norm(eligible[pair[0]]["center"]-eligible[pair[1]]["center"])), pair))
    if len(ordered_pairs) > config.max_pairs:
        truncation["pair_cap"] = True
    pair_count_before_cap = len(ordered_pairs)
    ordered_pairs = ordered_pairs[:config.max_pairs]
    neighbors_by_id = {cid: set() for cid in ids}
    candidates, rejection_reasons, evaluated, group_rejections = {}, Counter(), set(), []
    for pair in ordered_pairs:
        if len(evaluated) >= config.max_group_evaluations:
            truncation["group_evaluation_cap"] = True
            break
        evaluated.add(pair)
        group, reason = _group_geometry(pair, eligible, config, np)
        if group is None:
            rejection_reasons[reason] += 1
            group_rejections.append({"group_component_ids": list(pair), "reason": reason})
            continue
        candidates[pair] = group
        neighbors_by_id[pair[0]].add(pair[1])
        neighbors_by_id[pair[1]].add(pair[0])
    queue = deque(sorted(candidates, key=lambda key: (-candidates[key]["score"], key)))
    while queue and not truncation["group_evaluation_cap"]:
        key = queue.popleft()
        if len(key) >= config.max_group_members: continue
        adjacent = set().union(*(neighbors_by_id[cid] for cid in key))-set(key)
        for cid in sorted(adjacent):
            expanded = tuple(sorted((*key, cid)))
            if expanded in evaluated: continue
            if len(evaluated) >= config.max_group_evaluations:
                truncation["group_evaluation_cap"] = True
                break
            evaluated.add(expanded)
            group, reason = _group_geometry(expanded, eligible, config, np)
            if group is None:
                rejection_reasons[reason] += 1
                group_rejections.append({"group_component_ids": list(expanded), "reason": reason})
                continue
            candidates[expanded] = group
            queue.append(expanded)
    nonmaximal = set()
    for key in candidates:
        for size in range(2, len(key)):
            nonmaximal.update(subset for subset in itertools.combinations(key, size) if subset in candidates)
    primary_order = sorted(candidates.items(), key=lambda item: (-item[1]["score"], -len(item[0]), item[0]))
    primary_ranks = {key: rank for rank, (key, _group) in enumerate(primary_order, 1)}
    hole_candidates, hole_attempts, hole_evaluations, hole_capped = _hole_axis_candidates(labels, eligible, config, ndi, np)
    ordered_groups = [(key, {**candidate, "variant": "component_centroid_axis", "candidate_kind": "inter_component_centroid_axis"})
                      for key, candidate in primary_order]
    ordered_groups += [(tuple(candidate["ids"]), candidate) for candidate in hole_candidates]
    ordered_groups.sort(key=lambda item: (-item[1]["score"], -len(item[0]), item[0], item[1]["variant"]))
    truncation["hole_orientation_evaluation_cap"] = hole_capped
    groups, omitted = [], []
    for rank, (key, candidate) in enumerate(ordered_groups, 1):
        identity_values = {"source": source_hash, "origin": list(source_origin_xy), "ids": key}
        if candidate["candidate_kind"] == "intra_component_hole_axis":
            identity_values["orientation_hypothesis"] = candidate["variant"]
        identity = hashlib.sha256(json.dumps(identity_values, sort_keys=True).encode()).hexdigest()[:16]
        group_id = "component-group-"+identity
        quad = candidate["quad"]
        if "attempt" in candidate:
            candidate["attempt"].update(group_id=group_id, rank=rank,
                                         status="returned" if rank <= config.max_groups else "output_group_cap",
                                         quad_source_pixel_centers=(quad+origin).tolist(),
                                         quad_source_image_corners=(quad+origin+.5).tolist())
        if rank > config.max_groups:
            omitted.append({"group_id": group_id, "group_component_ids": list(candidate["ids"]), "rank": rank,
                            "ranking_score_not_probability": candidate["score"], "reason": "output_group_cap",
                            "candidate_kind": candidate["candidate_kind"], "orientation_hypothesis": candidate["variant"],
                            "quad_crop_pixel_centers": quad.tolist(), "quad_source_pixel_centers": (quad+origin).tolist(),
                            "evidence": candidate["evidence"]})
            continue
        within = bool(np.all(quad[:, 0] >= -.5) and np.all(quad[:, 0] <= values.shape[1]-.5)
                      and np.all(quad[:, 1] >= -.5) and np.all(quad[:, 1] <= values.shape[0]-.5))
        reasons = ["geometry_does_not_establish_character_identity", "components_may_be_short_contours_or_symbols",
                   "one_component_may_contain_multiple_touching_glyphs"]
        if key in nonmaximal: reasons.append("subset_of_larger_geometric_candidate")
        if not within: reasons.append("crop_boundary_overrun")
        if candidate["candidate_kind"] == "intra_component_hole_axis":
            reasons.extend(("enclosed_holes_do_not_establish_text", "search_quad_covers_only_part_of_one_connected_component"))
        groups.append({"group_id": group_id, "rank": rank, "group_component_ids": list(candidate["ids"]),
                       "candidate_kind": candidate["candidate_kind"], "orientation_hypothesis": candidate["variant"],
                       "rank_among_inter_component_candidates": primary_ranks.get(key),
                       "partial_component_search": candidate.get("partial_component_search", False),
                       "full_component_containment": candidate.get("full_component_containment", True),
                       "component_count": len(key), "quad_crop_pixel_centers": quad.tolist(),
                       "quad_crop_image_corners": (quad+.5).tolist(),
                       "quad_source_pixel_centers": (quad+origin).tolist(),
                       "quad_source_image_corners": (quad+origin+.5).tolist(),
                       "ranking_score_not_probability": candidate["score"],
                       "is_maximal_geometric_candidate": len(key) >= 2 and key not in nonmaximal,
                       "evidence": {**candidate["evidence"], "quad_within_crop_bounds": within},
                       "ambiguous_reasons": reasons, "geometry_role": "recognizer_search_quad_not_glyph_mask", **FLAGS})
    truncation["output_group_cap"] = bool(omitted)
    provenance = {"schema": SCHEMA, "configuration": asdict(config), "source_origin_xy": list(source_origin_xy),
                  "source_width_px": values.shape[1], "source_height_px": values.shape[0],
                  "input_gray_payload_sha256": source_hash, "source_pixels_unchanged": True,
                  "connected_component_policy": "8-connected thresholded raw dark pixels; no dilation, closing, splitting or erasing",
                  "coordinate_convention": "integer pixel-centre grid; image-corner coordinates equal centre coordinates plus 0.5; source equals crop plus source_origin_xy",
                  "quad_policy": "unclipped padded rotated rectangle; abstain on recognition when crop bounds are exceeded",
                  "enumeration": "bounded 2..5-component centroid groups plus independent enclosed-hole-axis bands inside eligible connected components",
                  "truncation_flags": truncation, "omitted_groups": omitted,
                  "group_rejections": group_rejections, "hole_orientation_attempts": hole_attempts,
                  "rejection_reason_counts": dict(rejection_reasons),
                  "component_exclusion_reason_counts": dict(Counter(reason for record in records for reason in record["exclusion_reasons"])),
                  "counts": {"source_ink_pixels": int(source_ink.sum()), "source_components": count,
                             "components_used_for_grouping": len(eligible), "neighbor_pairs_before_cap": pair_count_before_cap,
                             "neighbor_pairs_after_cap": len(ordered_pairs), "group_evaluations": len(evaluated),
                             "inter_component_candidates": len(primary_order), "hole_orientation_evaluations": hole_evaluations,
                             "intra_component_candidates": len(hole_candidates),
                             "group_candidates_before_output_cap": len(ordered_groups), "returned_groups": len(groups)},
                  "limitations": ["A geometrically plausible group may be contour fragments, circles, marks or non-numeric text.",
                                  "A connected glyph-contour component is never split; oversized components are excluded.",
                                  "A partial hole-axis quad never certifies the whole connected component as text, even after successful recognition.",
                                  "Enclosed holes can belong to letters, symbols or contour loops; their count is not a character count.",
                                  "Threshold-dependent grouping can miss faint, touching or fragmented characters.",
                                  "Search recall is not OCR accuracy or pixel-level text ownership."], **FLAGS}
    return ComponentTextCandidateResult(source_ink, labels, tuple(records), tuple(groups), provenance)
