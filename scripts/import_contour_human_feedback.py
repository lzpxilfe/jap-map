#!/usr/bin/env python3
"""Read a saved human-review copy into versioned, role-separated feedback.

Reads only the NEW packet GeoPackage, never the older annotation database.
No model is fitted, no source/project is saved, and no approval is created.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from histcontour_core.human_feedback import collect_feedback, map_to_pixel, source_geometry_matches, validate_packet
from histcontour_core.provenance import sha256_file
from scripts.run_contour_context_experiment import read, write


def read_review(packet_path, gpkg_path):
    from qgis.core import QgsVariantUtils, QgsVectorLayer
    packet = read(packet_path)
    cases, tiles = validate_packet(packet)
    digest_before = sha256_file(gpkg_path)

    def load(name):
        value = QgsVectorLayer(f"{gpkg_path}|layername={name}", name, "ogr")
        if not value.isValid():
            raise ValueError(f"the packet review database has no valid {name} layer")
        value.setReadOnly(True)
        return value

    metadata = load("review_metadata")
    rows = list(metadata.getFeatures())
    if len(rows) != 1 or str(rows[0]["packet_sha256"]) != sha256_file(packet_path):
        raise ValueError("review database does not belong to this immutable packet")
    del metadata, rows
    for tile in tiles.values():
        if sha256_file(packet_path.parent/tile["raster_path"]) != tile["source_raster_sha256"]:
            raise ValueError("review source image changed")

    def attributes(layer, feature):
        return {field.name(): None if QgsVariantUtils.isNull(feature[field.name()]) else feature[field.name()] for field in layer.fields()}

    originals = load("source_candidates")
    seen = set()
    for feature in originals.getFeatures():
        case_id = str(feature["case_id"])
        if case_id not in cases or case_id in seen:
            raise ValueError("original reference case IDs changed")
        seen.add(case_id)
        geometry = json.loads(feature.geometry().asJson(17))
        if not source_geometry_matches(cases[case_id], tiles[cases[case_id]["tile_id"]], geometry):
            raise ValueError(f"immutable blue source geometry changed: {case_id}; draw in human_traces instead")
    if seen != set(cases):
        raise ValueError("original reference cases were deleted")
    del originals
    layer = load("review_cases")
    decisions = []
    for feature in layer.getFeatures():
        row = attributes(layer, feature)
        case_id = row.get("case_id")
        if case_id not in cases:
            raise ValueError("unknown edited case ID")
        polygon = feature.geometry().asPolygon()
        if feature.geometry().isMultipart() or len(polygon) != 1 or len(polygon[0]) != 5:
            raise ValueError("review rectangles are immutable; draw correction geometry separately")
        actual = map_to_pixel(tiles[cases[case_id]["tile_id"]], [(p.x(), p.y()) for p in polygon[0][:-1]])
        x1, y1, x2, y2 = cases[case_id]["pixel_box"]
        expected = [(x1-.5, y1-.5), (x2-.5, y1-.5), (x2-.5, y2-.5), (x1-.5, y2-.5)]
        if sorted((round(x, 3), round(y, 3)) for x, y in actual) != sorted(expected):
            raise ValueError(f"review area moved: {case_id}; restore it before exporting feedback")
        decisions.append(row)
    del layer
    traces = []
    layer = load("human_traces")
    for feature in layer.getFeatures():
        row = attributes(layer, feature)
        case_id = row.get("case_id")
        geometry = feature.geometry()
        if case_id not in cases or geometry.isNull() or geometry.isEmpty() or geometry.isMultipart():
            raise ValueError("every human trace needs one nonempty line and an existing case ID")
        points = [(p.x(), p.y()) for p in geometry.asPolyline()]
        traces.append({"case_id": case_id, "record_id": f"human_traces:{feature.id()}", "trace_kind": row.get("trace_kind"),
                       "pixel_points": map_to_pixel(tiles[cases[case_id]["tile_id"]], points), "note": row.get("note")})
    del layer
    masks = []
    layer = load("human_ignore")
    for feature in layer.getFeatures():
        row = attributes(layer, feature)
        case_id = row.get("case_id")
        geometry = feature.geometry()
        if case_id not in cases or geometry.isNull() or geometry.isEmpty() or geometry.isMultipart() or not geometry.isGeosValid():
            raise ValueError("every unreadable-area mark needs a valid polygon and an existing case ID")
        rings = [map_to_pixel(tiles[cases[case_id]["tile_id"]], [(p.x(), p.y()) for p in ring]) for ring in geometry.asPolygon()]
        masks.append({"case_id": case_id, "record_id": f"human_ignore:{feature.id()}", "pixel_rings": rings, "reason": row.get("reason")})
    del layer
    if sha256_file(gpkg_path) != digest_before:
        raise RuntimeError("review file changed while reading; save/close QGIS and retry a new export")
    return packet, decisions, traces, masks, digest_before


def import_feedback(packet_path, gpkg_path, output, *, previous_path=None):
    packet, decisions, traces, masks, review_digest = read_review(packet_path, gpkg_path)
    feedback = collect_feedback(packet, decisions, traces, masks)
    packet_digest = sha256_file(packet_path)
    feedback.update(packet_sha256=packet_digest, source_review_gpkg_sha256=review_digest, previous_feedback_sha256=None)
    if previous_path is not None:
        previous = read(previous_path)
        if previous.get("packet_sha256") != packet_digest:
            raise ValueError("previous feedback belongs to a different review packet")
        feedback["previous_feedback_sha256"] = sha256_file(previous_path)
        before = {row["case_id"]: row for row in previous["decisions"]}
        feedback["changed_decision_case_ids"] = [row["case_id"] for row in feedback["decisions"] if before.get(row["case_id"]) != row]
    else:
        feedback["changed_decision_case_ids"] = []
    output.mkdir(parents=True, exist_ok=False)
    write(output/"feedback.json", feedback)
    write(output/"export-complete.json", {"status": "complete", "feedback_sha256": sha256_file(output/"feedback.json"),
                                          "next_step": "prepare_contour_feedback_training.py in a NumPy/SciPy/Pillow environment; QGIS itself needs no extra packages"})
    print({"human_approvals": feedback["human_approval_count"], "pending": len(feedback["pending_case_ids"]),
           "training": feedback["training_readiness"], "drawn_reference_count": len(feedback["geometry_references"])}, flush=True)
    return feedback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path)
    parser.add_argument("--gpkg", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous-feedback", type=Path)
    args = parser.parse_args()
    from qgis.testing import start_app
    app = start_app()
    try:
        import_feedback(args.packet, args.gpkg or args.packet.parent/"human-review.gpkg", args.output, previous_path=args.previous_feedback)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Feedback export stopped: {error}", file=sys.stderr)
        error.__traceback__ = None
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
