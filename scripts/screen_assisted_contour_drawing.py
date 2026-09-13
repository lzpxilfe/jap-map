#!/usr/bin/env python3
"""Remove evident topological hazards from the exploratory route pool.

This is a post-hoc development safeguard, NOT an independent accuracy test.
The complete original attempt pool stays on disk; only a new draft is written.
"""

import argparse
from collections import Counter
import copy
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from histcontour_core.completion import _line_pixels
from histcontour_core.assisted_drawing import native_feature_collection, validate_machine_drawing_report
from histcontour_core.human_feedback import map_to_pixel
from histcontour_core.provenance import sha256_file
from scripts.generate_assisted_contour_drawing import draw_overlay, read, write


def screening_reasons(row, owner, identities):
    """Reject an interior third-line hit or an overlong automatic gap."""
    import numpy as np
    reasons = []
    if row["mode"] == "contextual_gap" and row["gap_pixels"] > 16:
        reasons.append("automatic_inferred_gap_over_16px_requires_explicit_anchor_review")
    allowed = {identities[row["source_uid"]], identities[row["target_uid"]], 0}
    points = row["pixel_points"]
    for x, y in _line_pixels(points):
        if min(math.dist((x, y), row["start"]), math.dist((x, y), row["end"])) <= 3:
            continue
        if not 0 <= x < owner.shape[1] or not 0 <= y < owner.shape[0]:
            reasons.append("outside_source")
            break
        neighbors = owner[max(0, y-1):y+2, max(0, x-1):x+2]
        if any(int(value) not in allowed for value in np.unique(neighbors)):
            reasons.append("crosses_or_touches_third_retained_source_line")
            break
    return reasons


def screen(report_path, output):
    import numpy as np
    from PIL import Image
    report_path, output = report_path.resolve(), output.resolve()
    original = read(report_path)
    validate_machine_drawing_report(original)
    if original.get("holdout_used") is not False or original.get("human_approvals") != 0:
        raise ValueError("only untouched development proposals may be screened")
    output.mkdir(parents=True, exist_ok=False)
    for directory in ("sources",):
        shutil.copytree(report_path.parent/directory, output/directory)
    for name in ("configuration.json", "upstream-pin.json", "attempts.json"):
        shutil.copyfile(report_path.parent/name, output/name)
    (output/"images").mkdir()
    (output/"tile-previews").mkdir()
    base = read(report_path.parent/"base-lines.geojson")
    write(output/"base-lines.geojson", native_feature_collection(base["features"], original["tiles"], source_crs=base.get("crs")))
    source_proposals = read(report_path.parent/"ai-proposals.geojson")
    features = {feature["properties"]["proposal_id"]: feature for feature in source_proposals["features"]}
    decisions, kept = [], []
    for tile in original["tiles"]:
        height, width = tile["pixel_bounds"][3], tile["pixel_bounds"][2]
        lines = [feature for feature in base["features"] if feature["properties"]["tile_id"] == tile["tile_id"]]
        owner = np.zeros((height, width), dtype=np.int32)
        identities, pixel_lines = {}, []
        for number, feature in enumerate(lines, 1):
            identities[feature["properties"]["segment_uid"]] = number
            points = map_to_pixel(tile, feature["geometry"]["coordinates"])
            pixel_lines.append(points)
            for x, y in _line_pixels(points):
                if 0 <= x < width and 0 <= y < height:
                    owner[y, x] = number if owner[y, x] in (0, number) else -1
        eligible = []
        for row in [row for row in original["proposals"] if row["tile_id"] == tile["tile_id"]]:
            reasons = screening_reasons(row, owner, identities)
            decision = {"proposal_id": row["proposal_id"], "reasons": reasons, "retained_in_draft": not reasons,
                        "human_approved": False, "decision_origin": "machine_geometry_screen"}
            decisions.append(decision)
            if not reasons:
                eligible.append((row, decision))
        # A drawing draft cannot take two alternatives at one endpoint. Order
        # by supported vs inferred, then shorter gaps, without any E labels.
        eligible.sort(key=lambda item: (item[0]["mode"] == "contextual_gap", item[0]["gap_pixels"], item[0]["proposal_id"]))
        used = set()
        new_owner = np.zeros((height, width), dtype=bool)
        for row, decision in eligible:
            ends = {row["source_endpoint"], row["target_endpoint"]}
            pixels = [(x, y) for x, y in _line_pixels(row["pixel_points"]) if 0 <= x < width and 0 <= y < height]
            if used.intersection(ends):
                decision["reasons"].append("competing_endpoint_already_has_shorter_draft_route")
            elif any(new_owner[max(0,y-1):y+2, max(0,x-1):x+2].any() for x,y in pixels):
                decision["reasons"].append("intersects_another_draft_route")
            if decision["reasons"]:
                decision["retained_in_draft"] = False
                continue
            used.update(ends)
            for x,y in pixels:
                new_owner[y,x] = True
            kept.append(copy.deepcopy(row))
        subset = [row for row in kept if row["tile_id"] == tile["tile_id"]]
        with Image.open(output/tile["raster_path"]) as image:
            source = image.convert("RGB")
        before = draw_overlay(source, pixel_lines, (0,130,180), 1)
        after = before.copy()
        for mode, colour in (("ink_livewire",(0,170,70)),("smart_recovery",(180,0,200)),("contextual_gap",(240,115,0))):
            after = draw_overlay(after, [row["pixel_points"] for row in subset if row["mode"] == mode], colour, 2)
        source.save(output/"tile-previews"/f"{tile['tile_id']}-source.png")
        before.save(output/"tile-previews"/f"{tile['tile_id']}-before.png")
        after.save(output/"tile-previews"/f"{tile['tile_id']}-after.png")
    priority = []
    for tile in original["tiles"]:
        subset = [row for row in kept if row["tile_id"] == tile["tile_id"]]
        for inferred in (False, True):
            choices = [row for row in subset if (row["mode"] == "contextual_gap") == inferred]
            if choices:
                priority.append(max(choices, key=lambda row: row["quality"]["path_length_pixels"]*row["quality"]["new_fraction_outside_original_1_5px"])["proposal_id"])
    for row in kept:
        row["priority"] = 1 if row["proposal_id"] in priority else 2
        for suffix in ("source", "proposal"):
            name=f"{row['proposal_id']}-{suffix}.png"
            shutil.copyfile(report_path.parent/"images"/name, output/"images"/name)
    selected_features = [features[row["proposal_id"]] for row in kept]
    for feature in selected_features:
        feature["properties"]["priority"] = 1 if feature["properties"]["proposal_id"] in priority else 2
    write(output/"ai-proposals.geojson", native_feature_collection(selected_features, original["tiles"], source_crs=source_proposals.get("crs")))
    screening = {"schema":"jap-map-assisted-drawing-screen/1","source_report_sha256":sha256_file(report_path),
                 "scope":"post-hoc development topology screen; not new accuracy evidence",
                 "automatic_inferred_gap_limit_pixels":16, "source_pool":len(original["proposals"]),
                 "kept":len(kept), "removed":len(original["proposals"])-len(kept),
                 "reason_counts":dict(Counter(reason for row in decisions for reason in row["reasons"])),
                 "human_approvals":0,"decisions":decisions}
    write(output/"screening.json",screening)
    result={**original,"proposals":kept,"proposal_count":len(kept),"priority_ids":priority,
            "mode_counts":dict(Counter(row["mode"] for row in kept)),
            "screening":{key:value for key,value in screening.items() if key!="decisions"},
            "screening_sha256":sha256_file(output/"screening.json")}
    write(output/"drawing-report.json",result)
    blocks=[]
    for row in sorted(kept,key=lambda row:(row["priority"],row["proposal_id"])):
        sid=row["proposal_id"]
        blocks.append(f'<section id="{sid}"><h2>{sid} · {row["tile_id"]} · {row["mode"]} · 우선 {row["priority"]}</h2><p>{row["question"]}</p><div class="pair"><img src="images/{sid}-source.png" loading="lazy"><img src="images/{sid}-proposal.png" loading="lazy"></div></section>')
    document=('<!doctype html><html lang="ko"><meta charset="utf-8"><title>위험한 연결을 걸러낸 AI 초안</title><style>body{font:16px system-ui;margin:24px;max-width:1200px}.pair{display:flex;gap:20px}.pair img{max-width:45%;image-rendering:pixelated}section{border-top:1px solid #aaa;padding:15px}</style>'
              f'<h1>AI 추가 경로 초안 {len(kept)}개</h1><p>처음 만든 {len(original["proposals"])}개 중 교차·긴 자동 공백·끝점 충돌을 제외했습니다. 왼쪽 원본 / 오른쪽 제안입니다. 모든 선은 미승인이고 정확도 주장이 아닙니다. 초록·보라는 잉크 지지 경로, 주황은 공백 추론입니다. 먼저 우선순위 1의 {len(priority)}개를 보세요.</p>'+"".join(blocks)+"</html>")
    with (output/"report.html").open("x",encoding="utf-8") as handle:
        handle.write(document)
    print({key:screening[key] for key in ("source_pool","kept","removed","reason_counts")},flush=True)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    screen(args.report,args.output)
