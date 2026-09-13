#!/usr/bin/env python3
"""Make a separately reviewable curvature revision, keeping approved endpoints.

The original packet is immutable. This is a local requested edit, not a model
update, a new endpoint match, or a human-approved replacement geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from histcontour_core.completion import _line_pixels
from histcontour_core.human_feedback import map_to_pixel
from scripts.generate_assisted_contour_drawing import world
from scripts.screen_assisted_contour_drawing import screening_reasons


def subtle_curve_offset(points, side):
    """Small initial magnitude; direction still requires a local human/AI choice."""
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 2 or not np.isfinite(p).all():
        raise ValueError("need finite endpoints for a subtle curvature proposal")
    if side not in ("positive", "negative"):
        raise ValueError("choose the bend side from the local map context")
    gap = float(np.linalg.norm(p[-1]-p[0]))
    if gap <= 0:
        raise ValueError("zero-length gap")
    return min(.5, gap*.05)*(1 if side == "positive" else -1)


def curved_revision(points, offset_pixels, *, samples=81):
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 4 or not np.isfinite(p).all():
        raise ValueError("need at least four finite 2-D cubic samples")
    chord = p[-1]-p[0]
    length = float(np.linalg.norm(chord))
    if (length <= 0 or not math.isfinite(offset_pixels)
            or abs(offset_pixels) > min(4., length*.25)):
        raise ValueError("curvature change must be finite and modest relative to the gap")
    if type(samples) is not int or not 5 <= samples <= 4097 or samples % 2 == 0:
        raise ValueError("use an odd sample count between 5 and 4097")
    t = np.linspace(0, 1, len(p))[:, None]
    basis = np.hstack([3*(1-t)**2*t, 3*(1-t)*t*t])
    controls = np.linalg.lstsq(basis, p-(1-t)**3*p[0]-t**3*p[-1], rcond=None)[0]
    fit = (1-t)**3*p[0]+basis@controls+t**3*p[-1]
    error = float(np.linalg.norm(fit-p, axis=1).max())
    if error > 1e-6:
        raise ValueError("original samples are not an evenly sampled cubic; review manually")
    u = np.linspace(0, 1, samples)[:, None]
    base = ((1-u)**3*p[0]+3*(1-u)**2*u*controls[0]
            +3*(1-u)*u*u*controls[1]+u**3*p[-1])
    normal = np.array([-chord[1], chord[0]])/length
    # Value AND first derivative vanish at both endpoints. Curvature is free
    # to change while the endpoint position and analytic tangent stay fixed.
    bump = 16*u*u*(1-u)**2*offset_pixels
    refined = base+bump*normal
    refined[0], refined[-1] = p[0], p[-1]
    progress = (refined-p[0])@chord/length
    if not (np.diff(progress) > 0).all():
        raise ValueError("revision doubles back along the gap")
    return refined.tolist(), {
        "method": "original_cubic_plus_endpoint_flat_quartic_normal_bump",
        "original_cubic_controls": [p[0].tolist(), *controls.tolist(), p[-1].tolist()],
        "original_cubic_fit_max_error_pixels": error,
        "normal_xy": normal.tolist(), "signed_midpoint_offset_pixels": offset_pixels,
        "maximum_displacement_pixels": float(np.linalg.norm(refined-base, axis=1).max()),
        "endpoint_positions_unchanged": True, "analytic_endpoint_tangents_unchanged": True,
        "endpoint_derivatives_xy": [(3*(controls[0]-p[0])).tolist(), (3*(p[-1]-controls[1])).tolist()],
        "forward_progress_monotone": True, "sample_count": samples,
        "gap_pixels": length,
        "revised_length_pixels": float(np.linalg.norm(np.diff(refined, axis=0), axis=1).sum()),
    }


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def render_comparison(source, box, before, after, output, *, proposal_id, revision_id):
    from PIL import Image, ImageDraw, ImageFont
    factor, header, gutter = 10, 36, 16
    crop = source.crop(box).convert("RGB")
    w, h = crop.width*factor, crop.height*factor
    board = Image.new("RGB", (w*2+gutter, h+header), "white")
    draw = ImageDraw.Draw(board)
    for index, (label, points, colour) in enumerate((
            (f"{proposal_id} / previous", before, (230, 115, 0)),
            (f"{revision_id} / curve draft", after, (0, 155, 85)))):
        x0 = index*(w+gutter)
        board.paste(crop.resize((w, h), Image.Resampling.NEAREST), (x0, header))
        draw.text((x0+8, 8), label, fill="black", font=ImageFont.load_default(size=18))
        convert = lambda p: (x0+(p[0]-box[0]+.5)*factor-.5, header+(p[1]-box[1]+.5)*factor-.5)
        draw.line([convert(p) for p in points], fill=colour, width=4)
        for point in (before[0], before[-1]):
            x, y = convert(point)
            draw.ellipse((x-5, y-5, x+5, y+5), outline=(0, 100, 235), width=2)
    board.save(output)


def run(packet, request_path, output, *, offset_pixels=None, subtle_side=None):
    from PIL import Image
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    request, report = read(request_path), read(packet/"drawing-report.json")
    if (request.get("review_origin") != "explicit_user_chat"
            or request.get("requested_action") != "refine_curvature"
            or request.get("endpoint_pairing_decision") != "accept"
            or request.get("source_report_sha256") != digest(packet/"drawing-report.json")):
        raise ValueError("need a recorded user curvature request tied to this packet")
    row = next(p for p in report["proposals"] if p["proposal_id"] == request["proposal_id"])
    if row["mode"] != "contextual_gap":
        raise ValueError("this helper only revises the cubic inferred-gap geometry")
    tile = next(t for t in report["tiles"] if t["tile_id"] == row["tile_id"])
    if digest(packet/tile["raster_path"]) != tile["source_raster_sha256"]:
        raise ValueError("source raster changed")
    if (offset_pixels is None) == (subtle_side is None):
        raise ValueError("supply either an explicit offset or a context-chosen subtle bend side")
    if subtle_side is not None:
        offset_pixels = subtle_curve_offset(row["pixel_points"], subtle_side)
    points, audit = curved_revision(row["pixel_points"], offset_pixels)
    audit["magnitude_policy"] = "subtle_max_0.5px_or_5_percent_gap" if subtle_side is not None else "explicit_offset"
    width, height = tile["pixel_bounds"][2:]
    if any(not 0 <= x < width or not 0 <= y < height for x, y in points):
        raise ValueError("revision leaves the source tile")
    base = read(packet/"base-lines.geojson")
    owner, identities = np.zeros((height, width), np.int32), {}
    for feature in base["features"]:
        if feature["properties"]["tile_id"] != tile["tile_id"]:
            continue
        uid = feature["properties"]["segment_uid"]
        number = len(identities)+1
        identities[uid] = number
        for x, y in _line_pixels(map_to_pixel(tile, feature["geometry"]["coordinates"])):
            if 0 <= x < width and 0 <= y < height:
                owner[y, x] = number if owner[y, x] in (0, number) else -1
    hazards = screening_reasons({**row, "pixel_points": points}, owner, identities)
    all_proposals = read(packet/"ai-proposals.geojson")
    if all_proposals["crs"]["properties"]["name"] != "EPSG:5132":
        raise ValueError("unexpected source CRS")
    curve_pixels = set(_line_pixels(points))
    collisions = []
    for other in report["proposals"]:
        if other["tile_id"] != row["tile_id"] or other["proposal_id"] == row["proposal_id"]:
            continue
        other_pixels = set(_line_pixels(other["pixel_points"]))
        if any((x+dx, y+dy) in other_pixels for x, y in curve_pixels
               for dx in (-1, 0, 1) for dy in (-1, 0, 1)):
            collisions.append(other["proposal_id"])
    if hazards or collisions:
        raise ValueError(f"local geometry conflict: {hazards}, {collisions}")
    revision_id = row["proposal_id"]+"-R1"
    properties = {"proposal_id": revision_id, "parent_proposal_id": row["proposal_id"],
                  "tile_id": row["tile_id"], "revision": 1, "review_status": "needs_user_review",
                  "dataset_role": "review_only_not_training", "reference_kind": "inferred_gap_not_observed_ink",
                  "human_approved": False, "endpoint_pairing_human_approved": True,
                  "geometry_revision_human_approved": False, "scope": "gap_only"}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/"revised-curve.geojson", {"type": "FeatureCollection", "crs": all_proposals["crs"],
              "features": [{"type": "Feature", "properties": properties,
                            "geometry": {"type": "LineString", "coordinates": world(tile, points)}}]})
    metadata = {"schema": "jap-map-assisted-curve-refinement/1", **properties,
                "request_record_sha256": digest(request_path),
                "source_report_sha256": digest(packet/"drawing-report.json"),
                "source_proposals_sha256": digest(packet/"ai-proposals.geojson"),
                "source_base_lines_sha256": digest(packet/"base-lines.geojson"),
                "source_raster_sha256": tile["source_raster_sha256"],
                "source_packet_modified": False, "model_fitted": False,
                "offset_choice_origin": "assistant_local_draft_not_user_numeric_instruction",
                "original_pixel_points": row["pixel_points"], "revised_pixel_points": points,
                "local_geometry_audit": {"source_screening_reasons": hazards, "other_proposal_collisions": collisions},
                "curve_audit": audit}
    write_json(output/"revision.json", metadata)
    with Image.open(packet/tile["raster_path"]) as source:
        cx = (row["start"][0]+row["end"][0])/2
        cy = (row["start"][1]+row["end"][1])/2
        box = [int(cx)-24, int(cy)-24, int(cx)+24, int(cy)+24]
        render_comparison(source, box, row["pixel_points"], points, output/"comparison.png",
                          proposal_id=row["proposal_id"], revision_id=revision_id)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    curve = parser.add_mutually_exclusive_group(required=True)
    curve.add_argument("--offset-pixels", type=float)
    curve.add_argument("--subtle-side", choices=("positive", "negative"),
                       help="Start with at most 0.5 px / 5%% of the gap; choose the normal direction from the map")
    args = parser.parse_args()
    result = run(args.packet, args.request, args.output, offset_pixels=args.offset_pixels, subtle_side=args.subtle_side)
    print(json.dumps({"revision_id": result["proposal_id"], "audit": result["curve_audit"],
                      "geometry_audit": result["local_geometry_audit"]}, ensure_ascii=False))
