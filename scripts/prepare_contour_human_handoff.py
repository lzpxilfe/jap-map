#!/usr/bin/env python3
"""Prepare source-first human judgments and geometry corrections, not AI truth.

All ambiguous S/E samples plus E decision changes form the priority queue.
Additional S-only class-stratified controls support later human-supervised
training. E cases remain evaluation-only regardless of subsequent correction.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from histcontour_core.provenance import sha256_file
from scripts.run_contour_context_experiment import CLASSES, DEVELOPMENT_SHEETS, read, resolve, write


def geometry_digest(geometry):
    return hashlib.sha256(json.dumps(geometry, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def choose_controls(samples, labels, selected, *, count_per_class=21, seed=20260912):
    result = []
    for positive in (True, False):
        candidates = [row for row in samples if row["sample_id"] not in selected and labels[row["sample_id"]]["class"] in CLASSES
                      and (labels[row["sample_id"]]["class"] == "contour") == positive]
        buckets = {}
        for sheet in sorted(DEVELOPMENT_SHEETS):
            buckets[sheet] = sorted((row for row in candidates if row["sheet_id"] == sheet), key=lambda row: hashlib.sha256(f"{seed}:{row['segment_uid']}".encode()).hexdigest())
        chosen = []
        while len(chosen) < count_per_class and any(buckets.values()):
            for sheet in sorted(buckets):
                if buckets[sheet] and len(chosen) < count_per_class:
                    chosen.append(buckets[sheet].pop(0))
        if len(chosen) != count_per_class:
            raise ValueError("not enough old-series controls for the declared queue")
        result.extend(chosen)
    return result


def prepare(index_path, vector_path, training_path, training_labels_path, forward_path, forward_labels_path, evaluation_path, output, *, legacy_path=None):
    from PIL import Image, ImageDraw
    index, vectors, training, old_labels, forward, new_labels, evaluation = (read(path) for path in
        (index_path, vector_path, training_path, training_labels_path, forward_path, forward_labels_path, evaluation_path))
    if (evaluation.get("schema") != "jap-map-contour-forward-evaluation/1" or evaluation.get("status") != "completed_without_refitting"
            or evaluation.get("source_index_sha256") != sha256_file(index_path) or evaluation.get("vector_index_sha256") != sha256_file(vector_path)
            or evaluation.get("forward_manifest_sha256") != sha256_file(forward_path) or evaluation.get("forward_labels_sha256") != sha256_file(forward_labels_path)
            or forward.get("excluded_manifest_sha256") != sha256_file(training_path) or old_labels.get("sample_manifest_sha256") != sha256_file(training_path)):
        raise ValueError("handoff inputs do not match the frozen studies")
    protocol_path = evaluation_path.parent/"frozen-protocol.json"
    protocol = read(protocol_path)
    legacy_path = legacy_path or ROOT/"examples/ink_synthetic_legacy_model.json"
    if sha256_file(protocol_path) != evaluation["protocol_sha256"] or sha256_file(legacy_path) != protocol["legacy_model_sha256"]:
        raise ValueError("handoff legacy prior does not match the frozen study")
    tiles = {row["tile_id"]: row for row in index["tiles"] if row["split"] == "development" and row["sheet_id"] in DEVELOPMENT_SHEETS}
    entries = {row["tile_id"]: row for row in vectors["tiles"]}
    if len(tiles) != 9 or set(entries) != set(tiles) or vectors.get("holdout_included"):
        raise ValueError("handoff is restricted to the nine development tiles")
    old_by_id = {row["sample_id"]: row for row in old_labels["items"]}
    new_by_id = {row["sample_id"]: row for row in new_labels["items"]}
    old_samples, new_samples = {row["sample_id"]: row for row in training["samples"]}, {row["sample_id"]: row for row in forward["samples"]}
    selected = {}
    for name, samples, labels in (("training", old_samples, old_by_id), ("evaluation_only", new_samples, new_by_id)):
        for sid, row in samples.items():
            if labels[sid]["class"] in ("unsure", "mixed"):
                selected[sid] = (row, name, "ambiguous_source", 1)
    for row in evaluation["samples"]:
        if row["legacy_retained"] != row["candidate_retained"]:
            selected.setdefault(row["sample_id"], (new_samples[row["sample_id"]], "evaluation_only", "changed_decision", 1))
    controls = choose_controls(list(old_samples.values()), old_by_id, set(selected))
    for row in controls:
        selected[row["sample_id"]] = (row, "training", "source_control", 2)
    output.mkdir(parents=True, exist_ok=False)
    (output/"sources").mkdir()
    (output/"reference-vectors").mkdir()
    (output/"images").mkdir()
    shutil.copyfile(legacy_path, output/"legacy-model.json")
    collections, copied_tiles = {}, []
    for tile_id, tile in tiles.items():
        path = resolve(tile["raster_path"])
        if sha256_file(path) != entries[tile_id]["source_raster_sha256"]:
            raise ValueError("source raster changed before handoff")
        copied = output/"sources"/f"{tile_id}.tif"
        shutil.copyfile(path, copied)
        if sha256_file(copied) != sha256_file(path):
            raise ValueError("handoff source copy differs")
        collection_path = resolve(entries[tile_id]["ink_vector_path"])
        copied_vector = output/"reference-vectors"/f"{tile_id}.geojson"
        shutil.copyfile(collection_path, copied_vector)
        if sha256_file(copied_vector) != sha256_file(collection_path):
            raise ValueError("handoff original vector copy differs")
        copied_tiles.append({**tile, "raster_path": f"sources/{tile_id}.tif", "source_raster_sha256": sha256_file(path),
                             "reference_vector_path": f"reference-vectors/{tile_id}.geojson", "source_vector_sha256": sha256_file(collection_path)})
        collections[tile_id] = {feature["properties"]["segment_uid"]: feature for feature in read(collection_path)["features"]}
    cases, audit, blocks = [], [], []
    ordered = sorted(selected.values(), key=lambda item: (item[3], item[1] != "training", item[0]["sheet_id"], item[0]["sample_id"]))
    for number, (sample, role, reason, priority) in enumerate(ordered, 1):
        case_id, sid = f"H{number:03}", sample["sample_id"]
        tile = tiles[sample["tile_id"]]
        feature = collections[sample["tile_id"]][sample["segment_uid"]]
        with Image.open(output/"sources"/f"{sample['tile_id']}.tif") as image:
            source = image.crop(sample["pixel_box"]).convert("RGB")
        source_path = f"images/{case_id}-source.png"
        target_path = f"images/{case_id}-target.png"
        source.save(output/source_path)
        target = source.copy()
        x1, y1, x2, y2 = sample["pixel_box"]
        draw = ImageDraw.Draw(target)
        draw.line([(x-x1, y-y1) for x, y in sample["pixel_points"]], fill=(0, 100, 235), width=2)
        target.save(output/target_path)
        prompt = "대상 선 전체의 종류를 판단하세요. 일부만 맞거나 경로가 다르면 실제 등고선을 별도 레이어에 그려 주세요. 절벽·둑·도로·문자라고 판단한 근거도 남겨 주세요."
        cases.append({"case_id": case_id, "sample_id": sid, "dataset_role": role, "priority": priority,
                      "tile_id": sample["tile_id"], "sheet_id": sample["sheet_id"], "segment_uid": sample["segment_uid"],
                      "source_raster_sha256": sample["source_raster_sha256"], "source_vector_sha256": sample["source_vector_sha256"],
                      "original_geometry": feature["geometry"], "original_geometry_sha256": geometry_digest(feature["geometry"]),
                      "pixel_points": sample["pixel_points"], "pixel_box": sample["pixel_box"], "source_image": source_path, "target_image": target_path,
                      "prompt": prompt, "review_status": "unreviewed", "geometry_decision": "unreviewed", "human_approved": False, "annotator": "", "review_note": ""})
        reference = (old_by_id if role == "training" else new_by_id)[sid]
        audit.append({"case_id": case_id, "sample_id": sid, "selection_reason": reason, "provisional_ai_class_not_a_human_label": reference["class"]})
        role_label = "나중 학습용" if role == "training" else "평가 전용 · 학습 금지"
        blocks.append(f'<section id="{case_id}"><h2>{case_id} · {sid} · 우선순위 {priority} · {role_label}</h2><p>{html.escape(prompt)}</p>'
                      f'<p>{html.escape(sample["tile_id"])}</p><div class="pair"><figure><figcaption>동일 입력 타일 원본</figcaption><img src="{source_path}" loading="lazy"></figure>'
                      f'<figure><figcaption>판단할 대상 선만 파랑</figcaption><img src="{target_path}" loading="lazy"></figure></div></section>')
    packet = {"schema": "jap-map-contour-human-packet/1", "status": "waiting_for_human_judgment", "human_approvals": 0, "holdout_used": False,
              "source_index_sha256": sha256_file(index_path), "source_vector_index_sha256": sha256_file(vector_path), "forward_evaluation_sha256": sha256_file(evaluation_path),
              "training_manifest_sha256": sha256_file(training_path), "training_ai_labels_sha256": sha256_file(training_labels_path),
              "forward_manifest_sha256": sha256_file(forward_path), "forward_ai_labels_sha256": sha256_file(forward_labels_path),
              "legacy_model_path": "legacy-model.json", "legacy_model_sha256": sha256_file(legacy_path),
              "tiles": copied_tiles, "cases": cases, "counts": dict(Counter(row["dataset_role"] for row in cases)),
              "priority_counts": dict(Counter(str(row["priority"]) for row in cases)),
              "rules": ["AI labels and model scores are not copied into editable human decision fields.",
                        "E-series cases remain evaluation-only; never train or calibrate on them.",
                        "Original target geometry is immutable evidence. Draw corrections separately.",
                        "Observed contours, inferred gap links, negative traces and unreadable masks are separate supervision types.",
                        "Nothing is approved at packet creation; named explicit human approval is required."]}
    write(output/"packet.json", packet)
    write(output/"selection-audit.json", {"purpose": "selection provenance only; inspect after human judgment to reduce suggestion bias", "items": audit})
    document = ('<!doctype html><html lang="ko"><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="default-src \'none\'; img-src \'self\'; style-src \'unsafe-inline\'">'
                '<title>사람의 등고선 판단·수정 검수</title><style>body{font:16px system-ui;margin:24px;max-width:1200px}.pair{display:flex;gap:20px}figure{flex:1;margin:0}img{max-width:100%;width:auto;min-width:260px;image-rendering:pixelated}section{border-top:1px solid #bbb;padding:14px 0}figcaption{padding:8px}</style>'
                f'<h1>사람의 판단과 선 긋기</h1><p>핵심 {sum(row["priority"]==1 for row in cases)}개 + 학습용 비교 표본 {sum(row["priority"]==2 for row in cases)}개입니다. 현재 사람 승인 0개입니다.</p>'
                '<p>먼저 원본을 보고 판단한 뒤 필요하면 대상 파란 선을 확인하세요. 모델 점수와 AI 정답 제안은 이 화면에 표시하지 않습니다.</p>'
                '<p>판단·수정은 함께 제공한 QGIS 프로젝트에서 저장합니다. 평가 전용 E 표본은 수정 후에도 학습에 넣지 않습니다.</p>'
                +''.join(blocks)+'</html>')
    with (output/"source-first-review.html").open("x", encoding="utf-8") as handle:
        handle.write(document)
    print({"cases": len(cases), "roles": packet["counts"], "priorities": packet["priority_counts"], "human_approvals": 0}, flush=True)
    return packet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--training-samples", type=Path, required=True)
    parser.add_argument("--training-labels", type=Path, required=True)
    parser.add_argument("--forward-samples", type=Path, required=True)
    parser.add_argument("--forward-labels", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy-model", type=Path)
    args = parser.parse_args()
    prepare(args.index, args.vectors, args.training_samples, args.training_labels, args.forward_samples, args.forward_labels, args.evaluation, args.output, legacy_path=args.legacy_model)


if __name__ == "__main__":
    main()
