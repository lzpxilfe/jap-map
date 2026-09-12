#!/usr/bin/env python3
"""Propose explicit native-pixel margin crops from public scan previews.

Line geometry is used only to suggest image regions. This is not a map
registration, a GCP generator, a human approval, or an OCR accuracy result.
Inspect the overlays and edit the emitted manifest before running prepare.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from histcontour_core.margin_ocr import MANIFEST_SCHEMA, MarginOcrError, digest_file
from margin_ocr_pilot import _bounds, read_json, write_json


def outer_corners(preview):
    import cv2
    import numpy as np

    gray = np.asarray(preview.convert("L"))
    height, width = gray.shape
    edges = cv2.Canny(gray, 60, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 1800, threshold=int(min(width, height) * 0.10),
                            minLineLength=int(min(width, height) * 0.28), maxLineGap=int(min(width, height) * 0.05))
    if lines is None:
        raise MarginOcrError("no long frame-line candidates; supply manual crop boxes")
    dark = cv2.dilate((gray < 135).astype("uint8"), np.ones((3, 3), dtype="uint8"))
    groups = {"left": [], "right": [], "top": [], "bottom": []}
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = map(float, line)
        xs = np.rint(np.linspace(x1, x2, 192)).astype(int).clip(0, width - 1)
        ys = np.rint(np.linspace(y1, y2, 192)).astype(int).clip(0, height - 1)
        if float(dark[ys, xs].mean()) < 0.88:
            continue
        x, y = (x1 + x2) / 2, (y1 + y2) / 2
        if abs(y2 - y1) <= 0.02 * abs(x2 - x1) and abs(x2 - x1) > width * 0.30:
            if height * 0.035 < y < height * 0.32:
                groups["top"].append((y, line))
            if height * 0.65 < y < height * 0.98:
                groups["bottom"].append((y, line))
        if abs(x2 - x1) <= 0.02 * abs(y2 - y1) and abs(y2 - y1) > height * 0.35:
            if width * 0.065 < x < width * 0.3:
                groups["left"].append((x, line))
            if width * 0.7 < x < width * 0.965:
                groups["right"].append((x, line))
    if any(not group for group in groups.values()):
        # A curved or folded frame can be longer than any one straight Hough
        # segment. A bounded projection fallback is specific to this archive
        # layout; its result still requires visual review, never GIS use.
        rows = dark[:, int(width * 0.3):int(width * 0.7)].mean(axis=1)
        columns = dark[int(height * 0.3):int(height * 0.7), :].mean(axis=0)

        def peak(values, low, high, last=False):
            start, end = int(len(values) * low), int(len(values) * high)
            section = values[start:end]
            maximum = float(section.max())
            if maximum < 0.55:
                raise MarginOcrError("frame projection is weak; supply manual crop boxes")
            locations = np.flatnonzero(section >= maximum * 0.99)
            return float(start + locations[-1 if last else 0])

        left = peak(columns, 0.065, 0.3)
        right = peak(columns, 0.7, 0.965, True)
        top = peak(rows, 0.035, 0.32)
        bottom = peak(rows, 0.65, 0.98, True)
        return [[left, top], [right, top], [right, bottom], [left, bottom]]
    selected = {name: sorted(group, key=lambda pair: pair[0], reverse=name in {"bottom", "right"})[0][1]
                for name, group in groups.items()}

    def intersect(first, second):
        a, b = np.asarray(first[:2], dtype=float), np.asarray(first[2:], dtype=float)
        c, d = np.asarray(second[:2], dtype=float), np.asarray(second[2:], dtype=float)
        amount = np.linalg.solve(np.column_stack((b - a, -(d - c))), c - a)[0]
        return (a + amount * (b - a)).tolist()

    corners = [intersect(selected["left"], selected["top"]), intersect(selected["right"], selected["top"]),
               intersect(selected["right"], selected["bottom"]), intersect(selected["left"], selected["bottom"])]
    if any(not (0 <= x < width and 0 <= y < height) for x, y in corners):
        raise MarginOcrError("proposed frame leaves the image")
    return corners


def proposed_regions(corners, width: int, height: int) -> list[dict]:
    regions = []
    for name, (x, y) in zip(("nw", "ne", "se", "sw"), corners):
        xs = (x + 0.002 * width, x + 0.024 * width) if name.endswith("w") else (x - 0.024 * width, x - 0.002 * width)
        ys = (y + 0.002 * height, y + 0.026 * height) if name.startswith("n") else (y - 0.026 * height, y - 0.002 * height)
        box = [round(xs[0]), round(ys[0]), round(xs[1]), round(ys[1])]
        _bounds(box, width, height)
        regions.append({"region_id": name, "kind": "corner_" + name, "pixel_box": box})
    # Leave room below the scale header: several archive layouts place the
    # glyph baseline lower than the narrow v4 preview proposal suggested.
    for name, indices, offsets in (("title", (0, 1), (-0.035, -0.001)), ("scale", (2, 3), (0.003, 0.033))):
        x = sum(corners[index][0] for index in indices) / 2
        y = sum(corners[index][1] for index in indices) / 2
        box = [round(x - 0.13 * width), round(y + offsets[0] * height),
               round(x + 0.13 * width), round(y + offsets[1] * height)]
        _bounds(box, width, height)
        regions.append({"region_id": name, "kind": name, "pixel_box": box})
    return regions


def propose(sources_path: Path, output: Path, *, notes_path: Path | None = None) -> dict:
    from PIL import Image, ImageDraw

    sources = read_json(sources_path)
    if sources.get("schema") != "jap-map-public-margin-sources/1" or sources.get("downloaded_sheets") != 20:
        raise MarginOcrError("this cohort needs 20 verified source records")
    notes = read_json(notes_path) if notes_path is not None else None
    indexed_notes = {}
    if notes is not None:
        if notes.get("schema") != "jap-map-margin-visual-notes/1" or notes.get("human_approved") is not False:
            raise MarginOcrError("visual notes must explicitly remain non-human-approved")
        indexed_notes = {row["record_id"]: row for row in notes["sheets"]}
        if len(indexed_notes) != len(notes["sheets"]) or set(indexed_notes) != {row["record_id"] for row in sources["sources"]}:
            raise MarginOcrError("visual notes must match the source cohort exactly")
    output.mkdir(parents=True, exist_ok=False)
    sheets, audit = [], []
    for source in sources["sources"]:
        sid = source["sheet_id"]
        if source.get("status") != "downloaded" or source["regional_group"] == "D11":
            raise MarginOcrError("failed or held-out source in crop proposal cohort")
        preview_path = Path(source["preview_path"])
        if digest_file(preview_path) != source["preview_sha256"]:
            raise MarginOcrError("preview changed after acquisition")
        with Image.open(preview_path) as preview:
            width, height = source["image_size"]
            try:
                points = outer_corners(preview)
                corners = [[x * width / preview.width, y * height / preview.height] for x, y in points]
                regions = proposed_regions(corners, width, height)
                note = indexed_notes.get(source["record_id"], {})
                if note.get("legend_left") is True:
                    left, top = corners[0]
                    frame_height = corners[3][1] - top
                    for name, y1, y2 in (("legend_top", top - .025 * height, top + .085 * height),
                                         ("legend_middle", top + .38 * frame_height, top + .48 * frame_height)):
                        box = [round(left - .085 * width), round(y1), round(left - .010 * width), round(y2)]
                        _bounds(box, width, height)
                        regions.append({"region_id": name, "kind": "legend", "pixel_box": box})
                error = None
            except (MarginOcrError, ValueError) as failure:
                corners, regions, error = None, [], str(failure)
            overlay = preview.convert("RGB")
            draw = ImageDraw.Draw(overlay)
            for region in regions:
                box = region["pixel_box"]
                preview_box = [box[0] * preview.width / width, box[1] * preview.height / height,
                               box[2] * preview.width / width, box[3] * preview.height / height]
                draw.rectangle(preview_box, outline="#d21863", width=2)
                draw.text((preview_box[0], max(0, preview_box[1] - 13)), region["region_id"], fill="#d21863")
            overlay.save(output / f"{sid}-proposal.jpg", quality=92)
            overlay.close()
        sheets.append({
            "sheet_id": sid, "split": "development", "scenario": "old_type",
            "image_path": source["image_path"], "scan_source": source["source_page"],
            "rights": f"Public Domain Mark 1.0; {source['license']}; courtesy Stanford University Libraries",
            "source_index_label": source["index_label"], "source_manifest_sha256": source["source_manifest_sha256"],
            "crop_selection_status": "unreviewed_geometry_proposal", "regions": regions,
            "legend_status": ("two_sample_patches_only" if indexed_notes.get(source["record_id"], {}).get("legend_left") else
                              "not_seen_in_ai_preview_inspection") if notes is not None else "not_selected; inspect original before declaring absent",
            "visual_note": indexed_notes.get(source["record_id"]),
            "scenario_note": "Provisional old-type cohort; not a verified 7/7/6 quality-stratified corpus.",
        })
        audit.append({"sheet_id": sid, "outer_frame_pixel_corners": corners, "error": error})
    manifest = {"schema": MANIFEST_SCHEMA, "pilot_id": "stanford-korea-margin-20", "corpus_kind": "historical_scan",
                "source_catalog_sha256": digest_file(sources_path), "sheets": sheets}
    if notes_path is not None:
        manifest["visual_notes_sha256"] = digest_file(notes_path)
    write_json(output / "manifest.proposed.json", manifest)
    write_json(output / "crop-proposal-audit.json", {"algorithm": "preview-dark-frame-lines/3", "crop_layout": "margin-padding/2", "sheets": audit,
                                                   "gis_applied": False, "human_approvals": 0})
    return {"sheets": len(sheets), "needs_manual_boxes": sum(not sheet["regions"] for sheet in sheets)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visual-notes", type=Path, help="optional explicit source-inspection notes; never human approvals")
    args = parser.parse_args(argv)
    try:
        result = propose(args.sources, args.output, notes_path=args.visual_notes)
    except (MarginOcrError, OSError, ImportError, ValueError, KeyError, TypeError) as error:
        print(f"Crop proposal stopped: {error}", file=sys.stderr)
        return 2
    print(f"{result['sheets']} unreviewed crop proposals; {result['needs_manual_boxes']} need manual boxes. Inspect before OCR.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
