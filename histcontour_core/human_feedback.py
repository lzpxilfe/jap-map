"""Keep human decisions, corrected paths, and evaluation references distinct.

This validates explicit attestations; it cannot prove who operated a UI. No
automatic model score or provisional AI label is a human approval. Unknown
image pixels are not negative training examples, and inferred gaps are not
observed contour ink.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math

BINARY_CLASSES = ("contour", "text", "road_river", "symbol")
REVIEW_CLASSES = ("unreviewed", *BINARY_CLASSES, "mixed", "unsure")
GEOMETRY_DECISIONS = ("unreviewed", "accept_original", "replace_with_trace", "reject", "unsure")
TRACE_KINDS = ("observed_contour", "inferred_gap", "hard_negative")


def geometry_digest(geometry):
    return hashlib.sha256(json.dumps(geometry, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def map_to_pixel(tile, coordinates):
    west, south, east, north = tile["bounds"]
    width, height = tile["pixel_bounds"][2:]
    return [[(float(x)-west)*width/(east-west)-.5, (north-float(y))*height/(north-south)-.5] for x, y in coordinates]


def source_geometry_matches(case, tile, geometry, *, tolerance_px=1e-5):
    """Allow only subpixel serialization round-off, not changed source paths."""
    if geometry.get("type") != "LineString":
        return False
    actual = map_to_pixel(tile, geometry.get("coordinates", []))
    expected = case["pixel_points"]
    return len(actual) == len(expected) and all(math.isfinite(v) for point in actual for v in point) and all(math.dist(a, b) <= tolerance_px for a, b in zip(actual, expected))


def validate_pixel_points(points, tile, *, polygon=False):
    width, height = tile["pixel_bounds"][2:]
    minimum = 4 if polygon else 2
    if not isinstance(points, (list, tuple)) or len(points) < minimum:
        raise ValueError("human geometry has too few points")
    if any(len(point) != 2 or any(type(value) not in (float, int) or not math.isfinite(value) for value in point) for point in points):
        raise ValueError("human geometry must contain finite 2-D pixel points")
    if any(not -.501 <= x <= width-.499 or not -.501 <= y <= height-.499 for x, y in points):
        raise ValueError("human geometry leaves its source tile; check the case ID/CRS")
    if sum(math.dist(a, b) for a, b in zip(points, points[1:])) <= 0:
        raise ValueError("human geometry has zero length")
    if polygon and math.dist(points[0], points[-1]) > 1e-5:
        raise ValueError("ignore polygon must be closed")
    return [[float(x), float(y)] for x, y in points]


def validate_packet(packet):
    if packet.get("schema") != "jap-map-contour-human-packet/1" or packet.get("holdout_used") is not False:
        raise ValueError("unsupported or held-out human review packet")
    cases = {row["case_id"]: row for row in packet["cases"]}
    tiles = {row["tile_id"]: row for row in packet["tiles"]}
    if len(cases) != len(packet["cases"]) or len(tiles) != len(packet["tiles"]):
        raise ValueError("duplicate packet case/source IDs")
    for tile in tiles.values():
        bounds = tile.get("bounds", [])
        pixels = tile.get("pixel_bounds", [])
        if (len(bounds) != 4 or any(type(v) not in (int, float) or not math.isfinite(v) for v in bounds)
                or not bounds[0] < bounds[2] or not bounds[1] < bounds[3] or len(pixels) != 4
                or any(type(v) is not int or v <= 0 for v in pixels[2:])):
            raise ValueError("review source needs finite increasing bounds and positive pixel dimensions")
    for row in cases.values():
        if (row["tile_id"] not in tiles or tiles[row["tile_id"]].get("split") != "development"
                or row["sheet_id"] == "178-gongju" or row["sheet_id"] != tiles[row["tile_id"]]["sheet_id"]
                or row["dataset_role"] not in ("training", "evaluation_only")
                or row["source_raster_sha256"] != tiles[row["tile_id"]]["source_raster_sha256"]
                or row["original_geometry_sha256"] != geometry_digest(row["original_geometry"])):
            raise ValueError("packet source, role or original-geometry integrity failed")
        if not source_geometry_matches(row, tiles[row["tile_id"]], row["original_geometry"]):
            raise ValueError("packet pixel path differs from its original map geometry")
        validate_pixel_points(row["pixel_points"], tiles[row["tile_id"]])
        if row["sample_id"].startswith("E") and row["dataset_role"] != "evaluation_only":
            raise ValueError("E-series human corrections must remain evaluation-only")
    return cases, tiles


def collect_feedback(packet, decisions, traces=(), ignore_regions=(), *, minimum_per_class=20):
    """Validate one saved human-edit snapshot without changing any source file."""
    cases, tiles = validate_packet(packet)
    decisions_by_id = {row["case_id"]: row for row in decisions}
    if len(decisions_by_id) != len(decisions) or set(decisions_by_id) != set(cases):
        raise ValueError("saved decision rows must match every packet case exactly once")
    if minimum_per_class < 2:
        raise ValueError("training readiness requires at least two labels per class")
    seen_trace_ids = set()
    for row in (*traces, *ignore_regions):
        if row["case_id"] not in cases or row["record_id"] in seen_trace_ids:
            raise ValueError("unknown case or duplicate human geometry record ID")
        seen_trace_ids.add(row["record_id"])
    binary, geometry_references, masks, snapshots, unresolved = [], [], [], [], []
    for case_id, original in cases.items():
        decision = decisions_by_id[case_id]
        for name in ("sample_id", "dataset_role", "segment_uid", "source_raster_sha256", "original_geometry_sha256"):
            if decision.get(name) != original[name]:
                raise ValueError(f"immutable case metadata changed: {case_id}/{name}")
        status, action = decision.get("review_status"), decision.get("geometry_decision")
        if status not in REVIEW_CLASSES or action not in GEOMETRY_DECISIONS:
            raise ValueError(f"unknown human review status/action: {case_id}")
        approved = decision.get("human_approved")
        if type(approved) not in (bool, int) or approved not in (0, 1, False, True):
            raise ValueError("approval must be explicit 0/1")
        snapshot = {key: decision.get(key) for key in ("case_id", "sample_id", "dataset_role", "review_status", "geometry_decision", "human_approved", "annotator", "review_note")}
        snapshots.append(snapshot)
        if not approved:
            unresolved.append(case_id)
            continue
        annotator = decision.get("annotator")
        if not isinstance(annotator, str) or not annotator.strip() or annotator.strip().upper() == "NULL":
            raise ValueError(f"explicitly approved case needs a named annotator: {case_id}")
        if status == "unreviewed" or action == "unreviewed":
            raise ValueError(f"approved case is still unreviewed: {case_id}")
        note = decision.get("review_note") or ""
        if not isinstance(note, str):
            raise ValueError("review note must be text")
        if status in ("mixed", "unsure") and not note.strip():
            raise ValueError(f"mixed/unsure human decision needs a reason: {case_id}")
        if ((status == "contour" and action not in ("accept_original", "replace_with_trace"))
                or (status in ("text", "road_river", "symbol") and action not in ("reject", "replace_with_trace"))
                or (status == "unsure" and action != "unsure")
                or (status == "mixed" and action not in ("replace_with_trace", "reject", "unsure"))):
            raise ValueError(f"class and geometry decision conflict: {case_id}")
        case_traces = [row for row in traces if row["case_id"] == case_id]
        if action == "replace_with_trace" and not case_traces:
            raise ValueError(f"replacement decision needs a separate drawn trace: {case_id}")
        common = {"case_id": case_id, "sample_id": original["sample_id"], "dataset_role": original["dataset_role"],
                  "segment_uid": original["segment_uid"], "tile_id": original["tile_id"], "sheet_id": original["sheet_id"],
                  "source_raster_sha256": original["source_raster_sha256"], "original_geometry_sha256": original["original_geometry_sha256"],
                  "original_pixel_points": original["pixel_points"], "annotator": annotator.strip(), "review_note": note,
                  "human_approved": True, "reference_origin": "human_explicitly_approved"}
        # A partly replaced positive line is not silently relabeled as a wholly
        # correct original candidate; its new geometry is separate supervision.
        if status in BINARY_CLASSES and (status != "contour" or action == "accept_original"):
            binary.append({**common, "class": status, "geometry_decision": action})
        if status == "contour" and action == "accept_original":
            geometry_references.append({**common, "record_id": f"{case_id}:accepted-original", "trace_kind": "observed_contour",
                                        "human_pixel_points": original["pixel_points"], "geometry_origin": "human_accepted_candidate_not_independent_drawing"})
        if status in ("text", "road_river", "symbol"):
            geometry_references.append({**common, "record_id": f"{case_id}:rejected-original", "trace_kind": "hard_negative",
                                        "human_pixel_points": original["pixel_points"], "geometry_origin": "human_rejected_candidate"})
        for trace in case_traces:
            kind = trace.get("trace_kind")
            if kind not in TRACE_KINDS or (status == "unsure" and kind != "inferred_gap"):
                raise ValueError(f"drawn trace must explicitly distinguish observed/inferred/negative: {case_id}")
            points = validate_pixel_points(trace["pixel_points"], tiles[original["tile_id"]])
            geometry_references.append({**common, "record_id": trace["record_id"], "trace_kind": kind, "human_pixel_points": points,
                                        "geometry_origin": "human_drawn", "trace_note": trace.get("note") or ""})
        for mask in [row for row in ignore_regions if row["case_id"] == case_id]:
            if not isinstance(mask.get("reason"), str) or not mask["reason"].strip():
                raise ValueError(f"ignore region needs a reason: {case_id}")
            rings = [validate_pixel_points(ring, tiles[original["tile_id"]], polygon=True) for ring in mask["pixel_rings"]]
            if not rings:
                raise ValueError("ignore region needs a polygon")
            masks.append({**common, "record_id": mask["record_id"], "pixel_rings": rings, "reason": mask["reason"]})
    training_labels = [row for row in binary if row["dataset_role"] == "training"]
    evaluation_labels = [row for row in binary if row["dataset_role"] == "evaluation_only"]
    positives = sum(row["class"] == "contour" for row in training_labels)
    negatives = len(training_labels)-positives
    groups = sorted({row["sheet_id"] for row in training_labels})
    ready = positives >= minimum_per_class and negatives >= minimum_per_class and len(groups) >= 3
    return {"schema": "jap-map-contour-human-feedback/1", "status": "validated_human_edit_snapshot", "human_approval_count": len(cases)-len(unresolved),
            "pending_case_ids": unresolved, "decisions": snapshots, "training_labels": training_labels, "evaluation_labels": evaluation_labels,
            "geometry_references": geometry_references, "ignore_regions": masks,
            "training_readiness": {"status": "ready" if ready else "waiting_for_human_labels", "positive": positives, "negative": negatives,
                                   "sheet_groups": groups, "minimum_per_class": minimum_per_class,
                                   "evaluation_labels_excluded": len(evaluation_labels), "corrected_positive_paths_are_not_original_segment_labels": True},
            "geometry_reference_counts": dict(Counter(row["trace_kind"] for row in geometry_references)),
            "automatic_model_fit": False, "automatic_promotion": False,
            "interpretation": "Explicit named human attestations, not identity verification. Accepted candidates and independently drawn paths have distinct origins. E-series stays evaluation-only; inferred gaps are not observed ink."}


def rasterize_sparse_training_feedback(packet, feedback, output):
    """Export sparse labels only; unmarked background remains UNKNOWN."""
    import numpy as np
    from PIL import Image, ImageDraw
    output.mkdir(parents=True, exist_ok=False)
    cases, tiles = validate_packet(packet)
    records = []
    for row in (*feedback["geometry_references"], *feedback["ignore_regions"]):
        case = cases.get(row["case_id"])
        if (case is None or row.get("dataset_role") != case["dataset_role"] or row.get("tile_id") != case["tile_id"]
                or row.get("source_raster_sha256") != case["source_raster_sha256"] or row.get("human_approved") is not True):
            raise ValueError("geometry supervision lost its approved source or immutable training/evaluation role")
    refs = [row for row in feedback["geometry_references"] if row["dataset_role"] == "training" and row["trace_kind"] != "inferred_gap"]
    for tile_id, tile in tiles.items():
        selected = [row for row in refs if row["tile_id"] == tile_id]
        if not selected:
            continue
        width, height = tile["pixel_bounds"][2:]
        masks = {name: Image.new("1", (width, height), 0) for name in ("positive", "negative", "ignore")}
        for row in selected:
            key = "positive" if row["trace_kind"] == "observed_contour" else "negative"
            ImageDraw.Draw(masks[key]).line([tuple(point) for point in row["human_pixel_points"]], fill=1, width=1)
        for row in feedback["ignore_regions"]:
            if row["dataset_role"] == "training" and row["tile_id"] == tile_id:
                region = Image.new("1", (width, height), 0)
                draw = ImageDraw.Draw(region)
                for number, ring in enumerate(row["pixel_rings"]):
                    draw.polygon([tuple(point) for point in ring], fill=1 if number == 0 else 0)
                masks["ignore"] = Image.fromarray(np.asarray(masks["ignore"]) | np.asarray(region))
        positive, negative, ignored = (np.asarray(masks[name], dtype=bool) for name in ("positive", "negative", "ignore"))
        conflict = positive & negative
        ignored = ignored | conflict
        known = (positive | negative) & ~ignored
        # 255=unknown/ignored, 0=explicit negative, 1=explicit positive.
        labels = np.full((height, width), 255, dtype=np.uint8)
        labels[known & negative] = 0
        labels[known & positive] = 1
        filename = f"{tile_id}-sparse-labels.png"
        Image.fromarray(labels).save(output/filename)
        records.append({"tile_id": tile_id, "path": filename, "source_raster_sha256": tile["source_raster_sha256"],
                        "positive_pixels": int((labels == 1).sum()), "negative_pixels": int((labels == 0).sum()),
                        "unknown_or_ignored_pixels": int((labels == 255).sum()), "conflicting_pixels_ignored": int(conflict.sum())})
    return {"schema": "jap-map-sparse-human-contour-labels/1", "values": {"0": "explicit negative", "1": "explicit positive", "255": "unknown or ignored"},
            "evaluation_and_inferred_gap_records_excluded": True, "tiles": records}
