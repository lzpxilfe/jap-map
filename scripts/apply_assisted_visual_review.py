#!/usr/bin/env python3
"""Create a new AI-screened draft without pretending its decisions are human."""

import argparse
from collections import Counter
import html
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from histcontour_core.human_feedback import map_to_pixel
from histcontour_core.assisted_drawing import native_feature_collection, validate_machine_drawing_report
from histcontour_core.provenance import sha256_file
from scripts.generate_assisted_contour_drawing import draw_overlay,read,write


def apply(report_path, review_path, output):
    from PIL import Image
    report_path,review_path,output=report_path.resolve(),review_path.resolve(),output.resolve()
    report,review=read(report_path),read(review_path)
    validate_machine_drawing_report(report)
    if (review.get("schema")!="jap-map-assisted-visual-screen/1" or review.get("source_report_sha256")!=sha256_file(report_path)
            or review.get("human_approved") is not False or review.get("reference_origin")!="ai_visual_provisional"):
        raise ValueError("AI visual decisions must bind this exact draft and remain non-human")
    existing={row["proposal_id"]:row for row in report["proposals"]}
    items={row["proposal_id"]:row for row in review["items"]}
    if len(items)!=len(review["items"]) or any(sid not in existing for sid in items):
        raise ValueError("unknown or duplicate visual-review IDs")
    if any(row["decision"] not in ("ask_human","exclude_from_draft") for row in items.values()):
        raise ValueError("AI screen is not a human approval mechanism")
    removed={sid for sid,row in items.items() if row["decision"]=="exclude_from_draft"}
    kept=[row for row in report["proposals"] if row["proposal_id"] not in removed]
    priority=[sid for sid,row in items.items() if row["decision"]=="ask_human"]
    output.mkdir(parents=True,exist_ok=False)
    for name in ("configuration.json","upstream-pin.json","attempts.json","screening.json"):
        shutil.copyfile(report_path.parent/name,output/name)
    shutil.copyfile(review_path,output/"ai-visual-review.json")
    shutil.copytree(report_path.parent/"sources",output/"sources")
    (output/"images").mkdir()
    (output/"tile-previews").mkdir()
    source_base=read(report_path.parent/"base-lines.geojson")
    write(output/"base-lines.geojson",native_feature_collection(source_base["features"],report["tiles"],source_crs=source_base.get("crs")))
    for row in kept:
        row["priority"]=1 if row["proposal_id"] in priority else 2
        row["ai_visually_inspected"]=row["proposal_id"] in items
        if row["proposal_id"] in items:
            row["question"]=items[row["proposal_id"]]["note"]
        for suffix in ("source","proposal"):
            name=f"{row['proposal_id']}-{suffix}.png"
            shutil.copyfile(report_path.parent/"images"/name,output/"images"/name)
    source_proposals=read(report_path.parent/"ai-proposals.geojson")
    features=[feature for feature in source_proposals["features"] if feature["properties"]["proposal_id"] not in removed]
    for feature in features:
        sid=feature["properties"]["proposal_id"]
        feature["properties"]["priority"]=1 if sid in priority else 2
    write(output/"ai-proposals.geojson",native_feature_collection(features,report["tiles"],source_crs=source_proposals.get("crs")))
    base=read(output/"base-lines.geojson")
    for tile in report["tiles"]:
        lines=[map_to_pixel(tile,feature["geometry"]["coordinates"]) for feature in base["features"] if feature["properties"]["tile_id"]==tile["tile_id"]]
        with Image.open(output/tile["raster_path"]) as image:
            source=image.convert("RGB")
        before=draw_overlay(source,lines,(0,130,180),1)
        after=before.copy()
        for mode,colour in (("ink_livewire",(0,170,70)),("smart_recovery",(180,0,200)),("contextual_gap",(240,115,0))):
            after=draw_overlay(after,[row["pixel_points"] for row in kept if row["tile_id"]==tile["tile_id"] and row["mode"]==mode],colour,2)
        for kind,image in (("source",source),("before",before),("after",after)):
            image.save(output/"tile-previews"/f"{tile['tile_id']}-{kind}.png")
    result={**report,"proposals":kept,"proposal_count":len(kept),"priority_ids":priority,
            "mode_counts":dict(Counter(row["mode"] for row in kept)),
            "ai_visual_review":{"source_report_sha256":sha256_file(report_path),"review_sha256":sha256_file(review_path),
                                "inspected_count":len(items),"excluded_count":len(removed),"human_questions":len(priority),
                                "human_approved":False,"remaining_uninspected_count":len(kept)-len(priority)}}
    write(output/"drawing-report.json",result)
    blocks=[]
    for row in sorted(kept,key=lambda row:(row["priority"],row["proposal_id"])):
        sid=row["proposal_id"]
        blocks.append(f'<section id="{sid}"><h2>{sid} · {row["tile_id"]} · {row["mode"]} · 우선 {row["priority"]}</h2><p>{html.escape(row["question"])}</p><div class="pair"><img src="images/{sid}-source.png" loading="lazy"><img src="images/{sid}-proposal.png" loading="lazy"></div></section>')
    document=('<!doctype html><html lang="ko"><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="default-src \'none\'; img-src \'self\'; style-src \'unsafe-inline\'"><title>AI가 먼저 그린 검수 초안</title><style>body{font:16px system-ui;margin:24px;max-width:1200px}.pair{display:flex;gap:20px}.pair img{max-width:45%;image-rendering:pixelated}section{border-top:1px solid #aaa;padding:15px}</style>'
              f'<h1>AI 추가 경로 초안 {len(kept)}개 · 먼저 질문 {len(priority)}개</h1><p>{report["attempt_count"]:,}쌍을 시도해 만든 {report["screening"]["source_pool"]:,}개에서 기하 위험 {report["screening"]["removed"]}개, AI 육안 판독으로 {len(removed)}개를 추가 제외했습니다. 왼쪽 원본 / 오른쪽 제안입니다. 모든 선은 사람 미승인입니다.</p><p>초록·보라: 잉크 지지 후보, 주황: 공백 추론. 먼저 아래 우선순위 1의 질문에 답해 주세요. 나머지 경로는 개별 육안 판독까지 끝난 것이 아닙니다. QGIS에서 채택·거절·수정 필요·판독 불가를 기록할 수 있습니다.</p>'+"".join(blocks)+"</html>")
    with (output/"report.html").open("x",encoding="utf-8") as handle:
        handle.write(document)
    print({"draft_routes":len(kept),"priority_questions":len(priority),"ai_excluded":len(removed),"human_approvals":0},flush=True)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report",type=Path)
    parser.add_argument("review",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    apply(args.report,args.review,args.output)
