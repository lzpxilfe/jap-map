"""Unapproved numeric-ink hypotheses from overlapping source-quad readings.

An OCR string describes a crop, not its seed components. This module evaluates
every intersecting original dark-ink component, chooses a tight corroborated
localization, and retains boundary/external components separately. Weaker-ink
connectivity is recorded as sensitivity, never silently promoted to ownership.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re


SCHEMA = "jap-map-numeric-ink-ownership/1"
FLAGS = {"dataset_role": "review_only_not_training", "human_approved": False,
         "training_eligible": False, "automatic_promotion": False,
         "glyph_truth_assigned": False, "contour_truth_assigned": False,
         "elevation_assigned": False, "erase_mask_generated": False,
         "requires_human_pixel_review": True}


@dataclass(frozen=True)
class NumericInkOwnershipConfig:
    minimum_raw_recognition_score: float = .70
    required_scales: tuple[int, ...] = (2, 4)
    minimum_localizations: int = 2
    secondary_quad_coverage_floor: float = .80
    anchor_boundary_clearance_px: float = .5
    minimum_component_pixels: int = 5
    maximum_component_pixels: int = 1024
    maximum_component_span_px: int = 64
    minimum_height_fraction: float = .30
    maximum_height_ratio: float = 1.8
    maximum_baseline_residual_ratio: float = .35
    maximum_gap_height_ratio: float = 1.0
    maximum_word_width_per_digit_height_ratio: float = 1.5
    weak_gray_levels: tuple[int, ...] = (180, 200, 220)
    weak_contrast_floor: float = 12.0
    weak_outside_pixels: int = 8
    weak_outside_extent_px: float = 2.0
    max_components: int = 16_384
    max_groups: int = 512
    max_qualified_groups: int = 128
    max_image_pixels: int = 4_194_304
    max_point_quad_tests: int = 33_554_432

    def validate(self):
        integers = ("minimum_localizations", "minimum_component_pixels", "maximum_component_pixels",
                    "maximum_component_span_px", "weak_outside_pixels", "max_components", "max_groups",
                    "max_qualified_groups", "max_image_pixels", "max_point_quad_tests")
        if any(type(getattr(self, key)) is not int or getattr(self, key) <= 0 for key in integers):
            raise ValueError("numeric ownership counts and budgets must be positive integers")
        if (not 2 <= self.minimum_localizations <= 8 or self.max_components > 65_536
                or self.max_groups > 1024 or self.max_qualified_groups > 256 or self.max_image_pixels > 16_777_216
                or self.max_point_quad_tests > 134_217_728
                or not self.minimum_component_pixels <= self.maximum_component_pixels <= 16_384
                or self.maximum_component_span_px > 256):
            raise ValueError("numeric ownership settings exceed bounded work limits")
        if (not isinstance(self.required_scales, tuple) or set(self.required_scales) != {2, 4}
                or len(self.required_scales) != 2 or any(type(v) is not int for v in self.required_scales)
                or not isinstance(self.weak_gray_levels, tuple) or not 1 <= len(self.weak_gray_levels) <= 4
                or any(type(v) is not int or not 1 <= v < 245 for v in self.weak_gray_levels)
                or tuple(sorted(set(self.weak_gray_levels))) != self.weak_gray_levels):
            raise ValueError("expected distinct 2x/4x readings and ascending bounded weak-ink levels")
        values = set(asdict(self))-set(integers)-{"required_scales", "weak_gray_levels"}
        if any(type(getattr(self, key)) not in (int, float) or not math.isfinite(getattr(self, key)) for key in values):
            raise ValueError("numeric ownership thresholds must be finite numbers")
        if (not .5 <= self.minimum_raw_recognition_score < 1
                or not .5 <= self.secondary_quad_coverage_floor <= 1
                or not 0 <= self.anchor_boundary_clearance_px <= 3
                or not .1 <= self.minimum_height_fraction <= .7
                or not 1 < self.maximum_height_ratio <= 3
                or not 0 < self.maximum_baseline_residual_ratio <= .6
                or not 0 < self.maximum_gap_height_ratio <= 2
                or not .5 <= self.maximum_word_width_per_digit_height_ratio <= 3
                or not 0 < self.weak_contrast_floor <= 50 or not 0 < self.weak_outside_extent_px <= 16):
            raise ValueError("numeric ownership thresholds exceed conservative limits")
        return self


@dataclass(frozen=True)
class NumericInkOwnershipResult:
    source_ink: object
    glyph_candidate: object
    ambiguous_ink: object
    protected_throughgoing: object
    considered_ink: object
    hypotheses: tuple
    component_evidence: tuple
    provenance: dict

    def __post_init__(self):
        import numpy as np
        names = ("source_ink", "glyph_candidate", "ambiguous_ink", "protected_throughgoing", "considered_ink")
        arrays = {name: np.array(getattr(self, name), dtype=bool, copy=True) for name in names}
        source, glyph, ambiguous, protected, considered = (arrays[name] for name in names)
        if source.ndim != 2 or not all(source.shape) or any(a.shape != source.shape for a in arrays.values()):
            raise ValueError("numeric ownership masks must share one nonempty source grid")
        if (np.any(considered & ~source) or np.any((glyph | protected) & ~considered)
                or np.any(glyph & (ambiguous | protected)) or np.any(ambiguous & protected)
                or not np.array_equal(glyph | ambiguous | protected, source)):
            raise ValueError("numeric hypotheses must partition only the supplied source ink")
        for name, array in arrays.items():
            array.setflags(write=False)
            object.__setattr__(self, name, array)


def _quad_distances(points, quad, np):
    sign = 1 if np.sum(quad[:, 0]*np.roll(quad[:, 1], -1)-quad[:, 1]*np.roll(quad[:, 0], -1)) > 0 else -1
    distances = []
    for a, b in zip(quad, np.roll(quad, -1, axis=0)):
        vector = b-a
        distances.append(sign*(vector[0]*(points[:, 1]-a[1])-vector[1]*(points[:, 0]-a[0]))/np.linalg.norm(vector))
    return np.min(distances, axis=0)


def _validate_quad(value, np):
    q = np.asarray(value)
    if q.shape != (4, 2) or q.dtype.kind not in "uif" or not np.isfinite(q).all():
        raise ValueError("candidate needs four finite crop-local pixel-centre vertices")
    q = np.array(q, dtype=float, copy=True)
    edges = np.roll(q, -1, axis=0)-q
    turns = edges[:, 0]*np.roll(edges[:, 1], -1)-edges[:, 1]*np.roll(edges[:, 0], -1)
    if np.any(np.linalg.norm(edges, axis=1) < 1e-6) or not ((turns > 1e-8).all() or (turns < -1e-8).all()):
        raise ValueError("candidate quad must be convex and nondegenerate")
    return q


def _inputs(gray, component_labels, metadata, groups, readings, config, ndi, np):
    raw, labels = np.asarray(gray), np.asarray(component_labels)
    if (raw.ndim != 2 or not all(raw.shape) or raw.size > config.max_image_pixels or raw.dtype.kind not in "uif"
            or not np.isfinite(raw).all() or raw.min() < 0 or raw.max() > 255
            or labels.shape != raw.shape or labels.dtype.kind not in "ui" or np.any(labels < 0)):
        raise ValueError("expected finite source gray and same-grid nonnegative integer component labels")
    values = np.array(raw, dtype=np.float32, copy=True)
    if raw.dtype.kind == "f" and raw.max() <= 1: values *= 255
    if int(labels.max()) > config.max_components:
        raise ValueError("component IDs exceed the bounded component budget")
    labels = np.array(labels, dtype=np.int32, copy=True)
    source_ink = labels > 0
    contrast = ndi.maximum_filter(values, size=31, mode="nearest")-values
    if np.any(source_ink & ((values >= 245) | (contrast < 2))):
        raise ValueError("component labels include paper or unsupported source pixels")
    if not isinstance(metadata, (list, tuple)) or len(metadata) > config.max_components:
        raise ValueError("component metadata exceeds the bounded component budget")
    info = {}
    for row in metadata:
        cid = row.get("component_id")
        if type(cid) is not int or cid <= 0 or cid in info: raise ValueError("duplicate or invalid component metadata ID")
        info[cid] = row
    actual_ids = set(int(v) for v in np.unique(labels) if v > 0)
    if set(info) != actual_ids: raise ValueError("component metadata and source label IDs disagree")
    connected, number = ndi.label(source_ink, structure=np.ones((3, 3), bool))
    if number != len(actual_ids): raise ValueError("provided labels split or merge source connected components")
    component_data = {}
    for cid, window in enumerate(ndi.find_objects(labels), 1):
        if window is None: continue
        ys, xs = np.nonzero(labels[window] == cid)
        points = np.column_stack((xs+window[1].start, ys+window[0].start)).astype(float)
        if len(np.unique(connected[window][labels[window] == cid])) != 1:
            raise ValueError("one component ID covers disconnected source ink")
        bbox = [int(points[:, 0].min()), int(points[:, 1].min()), int(points[:, 0].max()), int(points[:, 1].max())]
        if info[cid].get("ink_pixels") != len(points) or info[cid].get("bbox_crop_pixel_centers") != bbox:
            raise ValueError("component area or pixel bounds differ from source metadata")
        component_data[cid] = {"points": points, "bbox": bbox, "window": window, "area": len(points)}
    if not isinstance(groups, (list, tuple)) or len(groups) > config.max_groups:
        raise ValueError("candidate group count exceeds bounded numeric ownership input")
    group_data = {}
    for row in groups:
        gid = row.get("group_id")
        if not isinstance(gid, str) or not gid or gid in group_data: raise ValueError("duplicate or invalid candidate group ID")
        if row.get("human_approved") is True or row.get("training_eligible") is True:
            raise ValueError("numeric ownership cannot reinterpret approved or training groups")
        seeds = row.get("group_component_ids")
        if (not isinstance(seeds, (list, tuple)) or not seeds or len(seeds) > 10
                or len(set(seeds)) != len(seeds) or any(type(cid) is not int or cid not in info for cid in seeds)):
            raise ValueError("candidate seed component IDs must exist on this source grid")
        quad = _validate_quad(row.get("quad_crop_pixel_centers"), np)
        area = abs(float(np.sum(quad[:, 0]*np.roll(quad[:, 1], -1)-quad[:, 1]*np.roll(quad[:, 0], -1))))/2
        partial = row.get("partial_component_search") is True or row.get("full_component_containment") is False or row.get("candidate_kind") == "intra_component_hole_axis"
        group_data[gid] = {"row": row, "quad": quad, "area": area, "partial": partial}
    if not isinstance(readings, (list, tuple)) or len(readings) > config.max_groups*len(config.required_scales):
        raise ValueError("reading count exceeds the bounded 2x/4x input")
    by_group = defaultdict(dict)
    for row in readings:
        gid, scale, text, score = (row.get(k) for k in ("group_id", "scale", "rec_text", "rec_score"))
        if (gid not in group_data or type(scale) is not int or scale not in config.required_scales
                or scale in by_group[gid] or not isinstance(text, str)
                or type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError("reading IDs/scales/text/scores are malformed or duplicated")
        by_group[gid][scale] = {"group_id": gid, "scale": scale, "rec_text": text, "rec_score": float(score)}
    return values, labels, source_ink, contrast, component_data, group_data, by_group


def _families(qualified, group_data, component_data, config, np):
    if len(qualified) > config.max_qualified_groups:
        raise ValueError("qualified numeric groups exceed bounded family comparison budget")
    covered = {}
    for gid in qualified:
        q = group_data[gid]["quad"]
        covered[gid] = {cid for cid, c in component_data.items() if (_quad_distances(c["points"], q, np) >= 0).mean() >= .5}
    remaining = set(qualified)
    families = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        family, frontier = [first], [first]
        while frontier:
            current = frontier.pop()
            for other in sorted(remaining):
                if qualified[current] != qualified[other] or not (covered[current] & covered[other]): continue
                a, b = group_data[current]["quad"], group_data[other]["quad"]
                ua, ub = a[1]-a[0], b[1]-b[0]
                if abs(float(np.dot(ua, ub)/np.linalg.norm(ua)/np.linalg.norm(ub))) < math.cos(math.radians(45)): continue
                family.append(other)
                frontier.append(other)
                remaining.remove(other)
        families.append(family)
    return families


def infer_numeric_ink_ownership(gray, component_labels, components, candidate_groups, readings, *, config=NumericInkOwnershipConfig()):
    """Partition supplied dark ink into candidate, ambiguous and protected masks.

    Candidate quads MUST be crop-local integer-centre-grid coordinates. Flat
    readings are ``{group_id, scale: 2|4, rec_text, rec_score}``. Repeated
    readings are correlated model evidence, never independent validation.
    Seed IDs never determine ownership; all intersecting source CCs are tested.
    """
    import numpy as np
    from scipy import ndimage as ndi

    config.validate()
    values, labels, source_ink, contrast, cc, groups, by_group = _inputs(
        gray, component_labels, components, candidate_groups, readings, config, ndi, np)
    qualified, reading_audit = {}, []
    for gid in sorted(groups):
        passes = by_group.get(gid, {})
        texts = {row["rec_text"].strip() for row in passes.values()}
        if set(passes) != set(config.required_scales): reason = "missing_2x_or_4x_reading"
        elif len(texts) != 1: reason = "scale_readings_disagree"
        elif not re.fullmatch(r"(?:[0-9]{2,6}|[0-9]{1,4}[.,][0-9]{1,2})", next(iter(texts))): reason = "not_a_multidigit_numeric_string"
        elif min(row["rec_score"] for row in passes.values()) < config.minimum_raw_recognition_score: reason = "raw_score_below_provisional_evidence_gate"
        else:
            reason = "numeric_string_consistent_across_correlated_scales"
            qualified[gid] = next(iter(texts))
        reading_audit.append({"group_id": gid, "reason": reason, "raw_readings": list(passes.values())})
    if len(qualified)*int(source_ink.sum()) > config.max_point_quad_tests:
        raise ValueError("numeric component/quad comparison budget exceeded")
    hypotheses, component_evidence = [], []
    proposed_by_component = defaultdict(set)
    protected_ids = set()
    weak_labelings = {}
    for level in config.weak_gray_levels:
        broad = source_ink | ((values <= level) & (contrast >= config.weak_contrast_floor))
        weak_labelings[level] = ndi.label(broad, structure=np.ones((3, 3), bool))[0]
    for family_index, family in enumerate(_families(qualified, groups, cc, config, np), 1):
        full = [gid for gid in family if not groups[gid]["partial"]]
        text = qualified[family[0]]
        digit_sequence = text.replace('.', '').replace(',', '')
        identity = "numeric-hypothesis-"+hashlib.sha256(json.dumps([text, sorted(family)]).encode()).hexdigest()[:16]
        anchor_id = min(full or family, key=lambda gid: (groups[gid]["area"], gid))
        anchor = groups[anchor_id]["quad"]
        along = (anchor[1]-anchor[0])/np.linalg.norm(anchor[1]-anchor[0])
        normal = np.array([-along[1], along[0]])
        word_height = float(np.ptp(anchor @ normal))
        h = {"hypothesis_id": identity, "numeric_string_hypothesis": text,
             "numeric_separator_present": any(c in text for c in '.,'),
             "separator_interpretation": "unresolved_not_an_elevation_value",
             "punctuation_ink_ownership_assigned": False,
             "source_group_ids": sorted(family), "reference_group_id": anchor_id,
             "reference_quad_crop_pixel_centers": anchor.tolist(),
             "reference_selection": "smallest-area full-component localization reading the same string; no seed ownership",
             "partial_component_search_group_ids": sorted(set(family)-set(full)),
             "correlated_scale_and_duplicate_evidence_not_independent_validation": True,
             "glyph_candidate_component_ids": [], "protected_component_ids": [], "evaluated_component_ids": [],
             "component_decisions": [], **FLAGS}
        sufficient = len(full) >= config.minimum_localizations
        # A full word can be corroborated by overlapping prefix/suffix crops,
        # but only with an exact ordered one-component-per-digit correspondence.
        # This deliberately abstains for touching/fragmented characters and does
        # not let a short reading claim an entire longer word.
        h["ordered_subword_support"] = []
        if full and not sufficient:
            def contained_run(gid):
                q = groups[gid]["quad"]
                return sorted((cid for cid, c in cc.items()
                               if _quad_distances(c["points"], q, np).min() >= config.anchor_boundary_clearance_px
                               and (not h['numeric_separator_present'] or
                                    float(np.ptp(c['points'] @ normal)+1) >= max(4.,config.minimum_height_fraction*word_height))),
                              key=lambda cid: float(np.mean(cc[cid]["points"] @ along)))
            anchor_run = contained_run(anchor_id)
            coverage = {cid: {anchor_id} for cid in anchor_run}
            if len(anchor_run) == len(digit_sequence):
                for gid, short in sorted(qualified.items()):
                    short_digits = short.replace('.', '').replace(',', '')
                    if groups[gid]["partial"] or not 2 <= len(short_digits) < len(digit_sequence):
                        continue
                    axis = groups[gid]["quad"][1]-groups[gid]["quad"][0]
                    if float(np.dot(along, axis)/np.linalg.norm(axis)) < math.cos(math.radians(15)):
                        continue
                    run = contained_run(gid)
                    if len(run) != len(short_digits):
                        continue
                    offsets = [i for i in range(len(digit_sequence)-len(short_digits)+1)
                               if digit_sequence[i:i+len(short_digits)] == short_digits and anchor_run[i:i+len(short_digits)] == run]
                    if len(offsets) != 1:
                        continue
                    for cid in run:
                        coverage[cid].add(gid)
                    h["ordered_subword_support"].append({"group_id": gid, "numeric_string_hypothesis": short,
                        "character_offset_hypothesis": offsets[0], "ordered_component_ids": run})
                sufficient = bool(coverage) and all(len(gids) >= config.minimum_localizations for gids in coverage.values())
            h["ordered_subword_corroboration_sufficient"] = sufficient
        eligible = []
        for cid, component in cc.items():
            fractions = {gid: float((_quad_distances(component["points"], groups[gid]["quad"], np) >= -1e-8).mean()) for gid in family}
            if not any(fractions.values()): continue
            distances = _quad_distances(component["points"], anchor, np)
            bbox = component["bbox"]
            seed_for = [gid for gid in family if cid in groups[gid]["row"]["group_component_ids"]]
            row = {"hypothesis_id": identity, "component_id": cid, "ink_pixels": component["area"],
                   "considered_by_hypotheses": [identity],
                   "seed_for_group_ids": seed_for, "evaluated_even_if_not_a_seed": not bool(seed_for),
                   "coverage_by_group": fractions, "minimum_anchor_boundary_distance_px": float(distances.min()),
                   "decision": "ambiguous_ink", "weak_connectivity_review_required": False, **FLAGS}
            h["evaluated_component_ids"].append(cid)
            if min(bbox[:2]) == 0 or bbox[2] == values.shape[1]-1 or bbox[3] == values.shape[0]-1:
                row.update(decision="protected_throughgoing", reason="source_crop_boundary_contact_not_glyph_truth")
            elif distances.max() >= 0 and distances.min() < config.anchor_boundary_clearance_px:
                row.update(decision="protected_throughgoing", reason="component_crosses_or_contacts_reference_boundary")
            elif not full:
                row["reason"] = "partial_component_search_never_assigns_whole_component"
            elif not sufficient:
                row["reason"] = "insufficient_repeated_localizations_for_ownership_hypothesis"
            elif fractions[anchor_id] < 1:
                row["reason"] = "outside_tight_numeric_localization"
            elif min(fractions[gid] for gid in full) < config.secondary_quad_coverage_floor:
                row["reason"] = "not_supported_by_other_numeric_localizations"
            elif not config.minimum_component_pixels <= component["area"] <= config.maximum_component_pixels or max(bbox[2]-bbox[0]+1, bbox[3]-bbox[1]+1) > config.maximum_component_span_px:
                row["reason"] = "noncompact_component_or_noise"
            else:
                u, v = component["points"] @ along, component["points"] @ normal
                height, width = float(np.ptp(v)+1), float(np.ptp(u)+1)
                row.update(projected_height_px=height, projected_width_px=width, baseline_position=float(v.max()),
                           along_start=float(u.min()), along_end=float(u.max()))
                if height < max(4., config.minimum_height_fraction*word_height):
                    row["reason"] = "component_too_short_for_numeric_word_band"
                else:
                    row["reason"] = "inside_numeric_word_band_pending_run_check"
                    eligible.append(row)
            if row["decision"] == "protected_throughgoing":
                protected_ids.add(cid)
                h["protected_component_ids"].append(cid)
            h["component_decisions"].append(row)
            component_evidence.append(row)
        accepted = []
        if eligible:
            median_height = float(np.median([row["projected_height_px"] for row in eligible]))
            baseline = float(np.median([row["baseline_position"] for row in eligible]))
            for row in eligible:
                ratio = max(row["projected_height_px"]/median_height, median_height/row["projected_height_px"])
                if ratio > config.maximum_height_ratio or abs(row["baseline_position"]-baseline) > config.maximum_baseline_residual_ratio*median_height:
                    row["reason"] = "component_height_or_baseline_disagrees_with_numeric_run"
                else: accepted.append(row)
            accepted.sort(key=lambda row: row["along_start"])
            run_reason = None
            if not 2 <= len(accepted) <= len(digit_sequence): run_reason = "component_run_count_unresolved_not_character_count_truth"
            elif any(b["along_start"]-a["along_end"] > config.maximum_gap_height_ratio*median_height for a, b in zip(accepted, accepted[1:])):
                run_reason = "numeric_run_contains_an_unexplained_large_gap"
            elif (accepted[-1]["along_end"]-accepted[0]["along_start"])/(len(digit_sequence)*median_height) > config.maximum_word_width_per_digit_height_ratio:
                run_reason = "word_extent_inconsistent_with_numeric_hypothesis"
            if run_reason:
                for row in accepted: row["reason"] = run_reason
                accepted = []
        for row in accepted:
            cid = row["component_id"]
            row.update(decision="glyph_candidate", reason="complete_dark_component_in_correlated_numeric_localizations_and_aligned_run")
            sensitivities = []
            for level, broad_labels in weak_labelings.items():
                related = np.unique(broad_labels[labels == cid])
                related = related[related > 0]
                ys, xs = np.nonzero(np.isin(broad_labels, related))
                distances = _quad_distances(np.column_stack((xs, ys)), anchor, np)
                outside = distances < 0
                sensitive = int(outside.sum()) >= config.weak_outside_pixels and float(-distances.min()) >= config.weak_outside_extent_px
                sensitivities.append({"gray_ceiling": level, "connected_pixels": len(xs),
                                      "outside_reference_pixels": int(outside.sum()),
                                      "maximum_outside_extent_px": max(0., float(-distances.min())),
                                      "weaker_ink_external_connection": bool(sensitive)})
            row["connectivity_sensitivity"] = sensitivities
            row["weak_connectivity_review_required"] = any(item["weaker_ink_external_connection"] for item in sensitivities)
            if row["weak_connectivity_review_required"]:
                row["additional_review_reason"] = "dark_component_hypothesis_changes_when_weaker_connected_ink_is_included"
            proposed_by_component[cid].add(identity)
            h["glyph_candidate_component_ids"].append(cid)
        h["status"] = "numeric_glyph_hypothesis" if accepted else "ambiguous_numeric_localization"
        inferred = set(h["glyph_candidate_component_ids"])
        h["seed_set_corrections"] = [{"group_id": gid, "original_seed_component_ids": list(groups[gid]["row"]["group_component_ids"]),
            "added_nonseed_component_ids": sorted(inferred-set(groups[gid]["row"]["group_component_ids"])),
            "seed_ids_not_owned_as_glyphs": sorted(set(groups[gid]["row"]["group_component_ids"])-inferred)} for gid in family]
        hypotheses.append(h)
    # Conflicting words and protection override provisional dark-ink ownership.
    words = {h["hypothesis_id"]: h["numeric_string_hypothesis"] for h in hypotheses}
    conflicting = {cid for cid, owners in proposed_by_component.items() if len({words[owner] for owner in owners}) > 1}
    glyph_ids = set(proposed_by_component)-protected_ids-conflicting
    for row in component_evidence:
        cid = row["component_id"]
        if row["decision"] == "glyph_candidate" and cid not in glyph_ids:
            row.update(decision="protected_throughgoing" if cid in protected_ids else "ambiguous_ink",
                       reason="another_numeric_localization_requires_protection" if cid in protected_ids else "conflicting_numeric_string_hypotheses")
    for h in hypotheses:
        h["glyph_candidate_component_ids"] = [cid for cid in h["glyph_candidate_component_ids"] if cid in glyph_ids]
        h["protected_component_ids"] = sorted({row["component_id"] for row in h["component_decisions"] if row["decision"] == "protected_throughgoing"})
        inferred = set(h["glyph_candidate_component_ids"])
        h["seed_set_corrections"] = [{"group_id": gid, "original_seed_component_ids": list(groups[gid]["row"]["group_component_ids"]),
            "added_nonseed_component_ids": sorted(inferred-set(groups[gid]["row"]["group_component_ids"])),
            "seed_ids_not_owned_as_glyphs": sorted(set(groups[gid]["row"]["group_component_ids"])-inferred)} for gid in h["source_group_ids"]]
        if not h["glyph_candidate_component_ids"]: h["status"] = "ambiguous_numeric_localization"
    glyph = np.isin(labels, sorted(glyph_ids)) & source_ink
    protected = np.isin(labels, sorted(protected_ids)) & source_ink
    ambiguous = source_ink & ~glyph & ~protected
    considered = np.isin(labels, sorted({row["component_id"] for row in component_evidence})) & source_ink
    provenance = {"schema": SCHEMA, "configuration": asdict(config),
                  "source_grid": "crop-local integer pixel centres; source_ink is exactly the supplied connected dark-ink labels",
                  "source_pixels_unchanged": True, "provided_component_metadata_and_connectivity_verified": True,
                  "source_gray_payload_sha256": hashlib.sha256(np.ascontiguousarray(np.asarray(gray)).tobytes()).hexdigest(),
                  "component_labels_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
                  "seed_ids_used_as_pixel_ownership": False,
                  "raw_ocr_score_policy": "unadjusted evidence gate only; not a calibrated glyph probability",
                  "protected_class_meaning": "conservative source-boundary/external-connection protection; not contour truth",
                  "weak_connectivity_policy": "separate sensitivity on broader original ink; does not extend the dark glyph mask",
                  "ambiguous_ink_scope": "includes unexamined default unknown; only considered_ink/component_evidence marks actual numeric evaluation",
                  "reading_audit": reading_audit, "decision_reason_counts": dict(Counter(row["reason"] for row in component_evidence)),
                  "counts": {"source_ink_pixels": int(source_ink.sum()), "glyph_candidate_pixels": int(glyph.sum()),
                             "ambiguous_ink_pixels": int(ambiguous.sum()), "protected_throughgoing_pixels": int(protected.sum()),
                             "considered_ink_pixels": int(considered.sum()),
                             "numeric_hypotheses": len(hypotheses), "glyph_candidate_components": len(glyph_ids),
                             "weak_connectivity_sensitive_glyph_components": len({row["component_id"] for row in component_evidence if row["decision"] == "glyph_candidate" and row["weak_connectivity_review_required"]})},
                  "limitations": ["Correct repeated OCR strings do not prove the ownership of individual pixels.",
                                  "Dark-component candidates can connect to contour ink at weaker thresholds; inspect sensitivity records.",
                                  "Partial one-component searches never promote the whole component to glyph ownership.",
                                  "Fragmented characters, a single touching component and conflicting words remain unresolved.",
                                  "No source pixels are removed and no numeric string is assigned as elevation."], **FLAGS}
    return NumericInkOwnershipResult(source_ink, glyph, ambiguous, protected, considered, tuple(hypotheses), tuple(component_evidence), provenance)
