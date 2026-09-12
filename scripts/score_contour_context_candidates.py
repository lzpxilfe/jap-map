#!/usr/bin/env python3
"""Export and visually compare out-of-sheet context candidates, without erasing lines.

Uses the two *other* development sheets' model for each output tile. Every raw
candidate survives in the all-score layer. Uncertain lines are a separate layer;
neither AI class labels nor scores become human approvals or elevation values.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from histcontour_core.contour_context import NeighborhoodFeatures, probability, validate_classifier
from histcontour_core.provenance import sha256_file
from histcontour_core.segment_review import geometry_features
from scripts.compare_contour_vector_outputs import overlay, pixel_proposals, selected_sources
from scripts.generate_ink_centerline_candidates import proposal_mask
from scripts.run_contour_context_experiment import DEVELOPMENT_SHEETS, read, resolve, write
from scripts.score_ink_segments import _context_features, load_model

METHODS = ("legacy", "balanced", "conservative", "uncertain")
LABELS = {"legacy": "기존 합성 점수 ≥ 0.1", "balanced": "주변 문맥 · 선별 후보", "conservative": "주변 문맥 · 보수 후보", "uncertain": "불확실 · 반드시 재검수"}


def portable(path):
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def select_flags(legacy_score, score, balanced_threshold, conservative_threshold):
    values = (legacy_score, score, balanced_threshold, conservative_threshold)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError("selection requires finite [0, 1] scores and thresholds")
    if conservative_threshold > balanced_threshold:
        raise ValueError("conservative threshold cannot be more restrictive")
    legacy, balanced, conservative = legacy_score >= 0.1, score >= balanced_threshold, score >= conservative_threshold
    return {"legacy": legacy, "balanced": balanced, "conservative": conservative,
            "uncertain": (legacy or conservative) and not balanced}


def load_experiments(balanced_path, conservative_path):
    reports = [read(path) for path in (balanced_path, conservative_path)]
    required = ("source_index_sha256", "sample_manifest_sha256", "labels_sha256", "legacy_model_sha256", "feature_schema", "l2_fixed_for_both_models")
    if any(not isinstance(report.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", report[key]) for report in reports for key in required[:4]):
        raise ValueError("experiments require explicit SHA256 input fingerprints")
    if any(reports[0].get(key) != reports[1].get(key) for key in required):
        raise ValueError("threshold alternatives must share identical samples, sources and feature fitting")
    if reports[0].get("target_inner_contour_retention") != 0.95 or reports[1].get("target_inner_contour_retention") != 1.0:
        raise ValueError("this comparison requires the explicit 0.95 and 1.0 inner-threshold policies")
    for report in reports:
        if (report.get("schema") != "jap-map-contour-context-experiment/1" or report.get("status") != "completed_research_only"
                or report.get("human_approved") is not False or report.get("holdout_used") is not False or report.get("reference_origin") != "ai_visual_provisional"):
            raise ValueError("only explicitly non-human development experiments are allowed")
    alternatives = [report["alternatives"]["real_weak_neighborhood"] for report in reports]
    folds = [{row["test_sheet"]: row for row in alternative["folds"]} for alternative in alternatives]
    if any(set(rows) != DEVELOPMENT_SHEETS for rows in folds):
        raise ValueError("all three out-of-sheet models are required")
    for sheet in DEVELOPMENT_SHEETS:
        first, second = (rows[sheet] for rows in folds)
        validate_classifier(first["model"])
        if first["model"] != second["model"] or second["threshold"] > first["threshold"]:
            raise ValueError("threshold comparison must not change fitted models")
        if set(first["training_sample_ids"]) & set(first["test_sample_ids"]):
            raise ValueError("test identities entered the training set")
    return reports, folds


def score(index_path, vector_path, baseline_path, balanced_path, conservative_path, samples_path, labels_path, probes_path, output, *, illustration_ids=()):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from scipy.ndimage import distance_transform_edt

    reports, folds = load_experiments(balanced_path, conservative_path)
    sources, vectors, samples, probes = (read(path) for path in (index_path, vector_path, samples_path, probes_path))
    tiles = selected_sources(sources, probes)
    expected = reports[0]
    if (expected["source_index_sha256"] != sha256_file(index_path) or expected["legacy_model_sha256"] != sha256_file(baseline_path)
            or expected["sample_manifest_sha256"] != sha256_file(samples_path) or samples["vector_index_sha256"] != sha256_file(vector_path)
            or expected["labels_sha256"] != sha256_file(labels_path)):
        raise ValueError("experiment source, baseline, sample or raw vector fingerprints differ")
    entries = {row["tile_id"]: row for row in vectors["tiles"]}
    if len(entries) != len(vectors["tiles"]) or set(entries) != set(tiles) or vectors.get("holdout_included") or {row["sheet_id"] for row in tiles.values()} != DEVELOPMENT_SHEETS:
        raise ValueError("only the same nine development tiles may be scored")
    sample_by_id = {row["sample_id"]: row for row in samples["samples"]}
    labels = read(labels_path)
    if (samples.get("schema") != "jap-map-contour-semantic-samples/1" or samples.get("holdout_used") is not False
            or labels.get("human_approved") is not False or labels.get("reference_origin") != "ai_visual_provisional"):
        raise ValueError("sample and label files must explicitly remain non-human development evidence")
    labels_by_id = {row["sample_id"]: row for row in labels["items"]}
    if (len(sample_by_id) != len(samples["samples"]) or any(not re.fullmatch(r"S[0-9]{3}", sid) for sid in sample_by_id)
            or set(labels_by_id) != set(sample_by_id) or len(labels_by_id) != len(labels["items"])):
        raise ValueError("duplicate frozen sample IDs")
    if len(set(illustration_ids)) != len(illustration_ids) or any(sid not in sample_by_id for sid in illustration_ids):
        raise ValueError("illustration IDs must be distinct existing samples")
    for by_sheet in folds:
        for sheet, fold in by_sheet.items():
            if (any(sid not in sample_by_id or sample_by_id[sid]["sheet_id"] == sheet for sid in fold["training_sample_ids"])
                    or any(sid not in sample_by_id or sample_by_id[sid]["sheet_id"] != sheet for sid in fold["test_sample_ids"])):
                raise ValueError("displayed source sheet entered its model fitting")
    legacy_kind, legacy = load_model(baseline_path)
    if legacy_kind != "logistic":
        raise ValueError("the declared legacy baseline must be its frozen logistic model")
    output.mkdir(parents=True, exist_ok=False)
    (output/"review-images").mkdir()
    write(output/"balanced-experiment.json", reports[0])
    write(output/"conservative-experiment.json", reports[1])
    records, region_rows, sample_predictions, sections, sample_sections, illustration_rows = [], [], {}, [], [], {}
    sample_by_uid = {row["segment_uid"]: row for row in samples["samples"]}
    expected_predictions = {row["sample_id"]: row for row in expected["alternatives"]["real_weak_neighborhood"]["outer_predictions"]}
    started = time.perf_counter()
    for tile_id, tile in tiles.items():
        entry = entries[tile_id]
        raster_path, collection_path = resolve(tile["raster_path"]), resolve(entry["ink_vector_path"])
        digest = sha256_file(raster_path)
        vector_digest = sha256_file(collection_path)
        if entry["source_raster_sha256"] != digest:
            raise ValueError("raw candidate source raster changed")
        subset_samples = [row for row in samples["samples"] if row["tile_id"] == tile_id]
        if any(row["source_raster_sha256"] != digest or row["source_vector_sha256"] != vector_digest for row in subset_samples):
            raise ValueError("sample raster/vector fingerprints do not match current candidates")
        with Image.open(raster_path) as image:
            gray = np.asarray(image.convert("L")).copy()
        collection = read(collection_path)
        uids = [feature["properties"]["segment_uid"] for feature in collection["features"]]
        if len(set(uids)) != len(uids):
            raise ValueError("raw vector identities must be unique")
        proposals = pixel_proposals(tile, collection)
        cache = NeighborhoodFeatures(gray)
        sheet = tile["sheet_id"]
        model = folds[0][sheet]["model"]
        thresholds = (folds[0][sheet]["threshold"], folds[1][sheet]["threshold"])
        scored, selected_proposals = [], {name: [] for name in METHODS}
        selected_features = {name: [] for name in METHODS}
        for feature, proposal in zip(collection["features"], proposals):
            uid = feature["properties"]["segment_uid"]
            descriptor = {**geometry_features(proposal.points), **_context_features(gray, proposal.points), **cache.describe(proposal.points)}
            current, previous = probability(model, descriptor), legacy.probability(descriptor)
            flags = select_flags(previous, current, *thresholds)
            properties = {**feature["properties"], "context_score": current, "legacy_contour_score": previous,
                          "context_balanced_threshold": thresholds[0], "context_conservative_threshold": thresholds[1],
                          "context_model_scope": "trained on the two other development sheets; AI provisional labels",
                          "context_test_sheet": sheet, "context_review_only": True, "human_approved": False,
                          **{f"context_{name}": selected for name, selected in flags.items()}}
            output_feature = {**feature, "properties": properties}  # geometry object is unchanged
            scored.append(output_feature)
            for name, included in flags.items():
                if included:
                    selected_features[name].append(output_feature)
                    selected_proposals[name].append(proposal)
            if uid in sample_by_uid:
                sid = sample_by_uid[uid]["sample_id"]
                if sid in expected_predictions and abs(current-expected_predictions[sid]["score"]) > 1e-7:
                    raise ValueError("full-tile scoring no longer reproduces its out-of-sheet sample score")
                sample_predictions[sid] = {"context_score": current, "legacy_score": previous, **flags}
        all_path = output/f"{tile_id}-all-scores.geojson"
        write(all_path, {**collection, "name": f"context_all_{tile_id}", "features": scored})
        record = {"tile_id": tile_id, "sheet_id": sheet, "source_raster_sha256": digest, "source_vector_sha256": vector_digest,
                  "all_scores_path": portable(all_path), "all_count": len(scored), "balanced_threshold": thresholds[0], "conservative_threshold": thresholds[1], "methods": {}}
        masks = {}
        for name in METHODS:
            destination = output/f"{tile_id}-{name}.geojson"
            write(destination, {**collection, "name": f"context_{name}_{tile_id}", "selection": {"mode": name, "review_only": True, "human_approved": False}, "features": selected_features[name]})
            record["methods"][name] = {"path": portable(destination), "count": len(selected_features[name])}
            masks[name] = proposal_mask(selected_proposals[name], gray.shape)
        records.append(record)
        source_png = f"{tile_id}-source.png"
        Image.fromarray(gray).save(output/source_png)
        figures = [f'<figure><figcaption>원본</figcaption><img src="{source_png}" loading="lazy"></figure>']
        for name in METHODS[:3]:
            filename = f"{tile_id}-{name}.png"
            overlay(gray, masks[name]).save(output/filename)
            figures.append(f'<figure><figcaption>{LABELS[name]} · {len(selected_features[name])}개</figcaption><img src="{filename}" loading="lazy"></figure>')
        sections.append(f'<h2>{html.escape(tile_id)}</h2><div class="comparison">{"".join(figures)}</div>')
        for sample in subset_samples:
            sid = sample["sample_id"]
            x1, y1, x2, y2 = sample["pixel_box"]
            target = Image.fromarray(gray[y1:y2, x1:x2]).convert("RGB")
            draw = ImageDraw.Draw(target)
            draw.line([(x-x1, y-y1) for x, y in sample["pixel_points"]], fill=(0, 90, 235), width=2)
            source_name = f"review-images/{sid}-source-target.png"
            target.save(output/source_name)
            figure_rows = [f'<figure><figcaption>원본 · 대상 선은 파랑</figcaption><img src="{source_name}" loading="lazy"></figure>']
            images = [target]
            for name in METHODS[:3]:
                filename = f"review-images/{sid}-{name}.png"
                preview = overlay(gray[y1:y2, x1:x2], masks[name][y1:y2, x1:x2])
                preview.save(output/filename)
                figure_rows.append(f'<figure><figcaption>{LABELS[name]} · {"유지" if sample_predictions[sid][name] else "제외"}</figcaption><img src="{filename}" loading="lazy"></figure>')
                images.append(preview)
            label = labels_by_id[sid]
            sample_sections.append(f'<details id="{sid}"><summary>{sid} · AI {html.escape(label["class"])} · 기존 {int(sample_predictions[sid]["legacy"])} / 선별 {int(sample_predictions[sid]["balanced"])} / 보수 {int(sample_predictions[sid]["conservative"])}</summary><p>{html.escape(label["note"])}</p><div class="comparison sample">{"".join(figure_rows)}</div></details>')
            if sid in illustration_ids:
                panel = Image.new("RGB", (990, 320), "white")
                draw = ImageDraw.Draw(panel)
                font = ImageFont.load_default(size=17)
                statuses = (f"Source: blue target; AI {label['class']}", f"Legacy: {'kept' if sample_predictions[sid]['legacy'] else 'excluded'}", f"Context: {'kept' if sample_predictions[sid]['balanced'] else 'excluded'}")
                for column, (preview, caption) in enumerate(zip(images[:3], statuses)):
                    draw.text((column*330+8, 7), f"{sid}  {caption}", fill="black", font=font)
                    factor = min(310/preview.width, 275/preview.height)
                    enlarged = preview.resize((round(preview.width*factor), round(preview.height*factor)), Image.Resampling.NEAREST)
                    panel.paste(enlarged, (column*330+(330-enlarged.width)//2, 38))
                illustration_rows[sid] = panel
        distances = {name: distance_transform_edt(~mask) if mask.any() else np.full(gray.shape, np.inf) for name, mask in masks.items()}
        for region in (row for row in probes["regions"] if row["tile_id"] == tile_id):
            x1, y1, x2, y2 = region["pixel_box"]
            ink = gray[y1:y2, x1:x2] <= 176
            ink[:2] = ink[-2:] = False
            ink[:, :2] = ink[:, -2:] = False
            counts = {name: int(((distances[name][y1:y2, x1:x2] <= 2.0) & ink).sum()) for name in METHODS}
            region_rows.append({**region, "dark_pixels": int(ink.sum()), "covered_dark_pixels": counts})
        print(f"{tile_id}: {len(scored)} raw; " + "; ".join(f"{name}={len(selected_features[name])}" for name in METHODS), flush=True)
    if set(sample_predictions) != {row["sample_id"] for row in samples["samples"]}:
        raise ValueError("not every frozen sample appeared exactly once in scoring")
    aggregates = {}
    for kind in ("contour", "text", "road_river"):
        chosen = [row for row in region_rows if row["kind"] == kind]
        denominator = sum(row["dark_pixels"] for row in chosen)
        aggregates[kind] = {"regions": len(chosen), "dark_pixels": denominator,
                            "coverage": {name: sum(row["covered_dark_pixels"][name] for row in chosen)/denominator if denominator else None for name in METHODS}}
    summary = {"schema": "jap-map-contour-context-candidates/1", "review_only": True, "human_approved": False, "holdout_included": False,
               "scoring_mode": "leave_one_development_sheet_out", "automatic_promotion": False,
               "source_index_sha256": sha256_file(index_path), "source_vector_index_sha256": sha256_file(vector_path),
               "balanced_experiment_sha256": sha256_file(balanced_path), "conservative_experiment_sha256": sha256_file(conservative_path),
               "sample_manifest_sha256": sha256_file(samples_path), "probe_sha256": sha256_file(probes_path),
               "tiles": records, "counts": {name: sum(row["methods"][name]["count"] for row in records) for name in METHODS},
               "raw_candidates": sum(row["all_count"] for row in records), "sample_predictions": sample_predictions,
               "regions": region_rows, "probe_aggregates": aggregates, "elapsed_seconds": time.perf_counter()-started,
               "limitations": "Three development sheets and provisional AI labels only; small ROI ink coverage is not formal contour accuracy. All original geometries remain in all-score layers; no gap joining or elevation assignment."}
    write(output/"context_candidate_index.json", summary)
    if illustration_ids:
        board = Image.new("RGB", (990, 320*len(illustration_ids)), "white")
        for row, sid in enumerate(illustration_ids):
            board.paste(illustration_rows[sid], (0, row*320))
        board.save(output/"context-tradeoff-examples.png")
        write(output/"illustration-selection.json", {"sample_ids": list(illustration_ids), "selection": "post-experiment explanatory examples only; not a new evaluation subset"})
    class_counts = expected["class_counts"]
    positive_count = class_counts.get("contour", 0)
    negative_count = sum(class_counts.get(kind, 0) for kind in ("text", "road_river", "symbol"))
    table = f'<table><tr><th>AI 판독 표본</th><th>등고선 유지 / {positive_count}</th><th>비등고선 제외 / {negative_count}</th><th>도로·하천 제외 / {class_counts.get("road_river", 0)}</th></tr>'
    experiment_metrics = [(LABELS["legacy"], expected["legacy_fixed_010"]["metrics"]),
                          (LABELS["balanced"], expected["alternatives"]["real_weak_neighborhood"]["metrics"]),
                          (LABELS["conservative"], reports[1]["alternatives"]["real_weak_neighborhood"]["metrics"])]
    for label, values in experiment_metrics:
        roads = values["by_class"]["road_river"]
        table += f'<tr><td>{label}</td><td>{values["contours_retained"]}</td><td>{values["noncontours_rejected"]}</td><td>{roads["total"]-roads["retained"]}</td></tr>'
    table += '</table>'
    document = ('<!doctype html><html lang="ko"><meta charset="utf-8">'
                '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; img-src \'self\'; style-src \'unsafe-inline\'">'
                '<title>주변 문맥 등고선 후보 비교</title><style>body{font:16px system-ui;margin:24px}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #aaa}.comparison{display:flex;gap:12px}figure{margin:0;flex:1;min-width:0}img{width:100%;height:auto}.sample img{max-height:320px;object-fit:contain;image-rendering:pixelated}figcaption{padding:8px}h2{margin-top:32px}details{border-bottom:1px solid #ccc;padding:10px}summary{cursor:pointer}</style>'
                '<h1>주변 선 방향·간격을 이용한 등고선 후보 비교</h1>'
                '<p>각 도엽 출력은 다른 두 개발 도엽으로 학습한 모델입니다. 빨강은 실제 내보낸 후보 벡터입니다. 선 연결·높이 부여·사람 승인은 하지 않았습니다.</p>'
                f'<p>{len(samples["samples"])}개 중 명확한 AI 임시 판독 {positive_count+negative_count}개만 수치 비교. 사람 정답이나 전체 지도 정확도가 아닙니다. 보수 기준은 첫 실험 후 추가한 탐색 비교입니다. 기본 추출 설정은 변경하지 않았습니다.</p>'
                +table+f'<p>선별 후보는 혼입을 줄이지만 등고선 누락도 늘었습니다. 불확실 레이어는 기존 또는 보수 필터가 유지했으나 선별 필터가 뺀 선입니다. 원본 후보 {summary["raw_candidates"]:,}개는 all-scores 레이어에 모두 보존됩니다.</p>'
                +'<h2>고정 표본 전체 — 펼쳐서 원본과 실제 출력 비교</h2><p>1=유지, 0=제외. AI 라벨은 사람 정답이 아닙니다. 파랑은 선택된 대상 선, 빨강은 실제 출력선입니다. 모호한 표본도 숨기지 않습니다.</p>'
                +''.join(sample_sections)+'<h2>9개 타일 전체</h2>'+''.join(sections)+'</html>')
    with (output/"report.html").open("x", encoding="utf-8") as handle:
        handle.write(document)
    print(output/"context_candidate_index.json", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--legacy-model", type=Path, required=True)
    parser.add_argument("--balanced-experiment", type=Path, required=True)
    parser.add_argument("--conservative-experiment", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--illustration-ids", nargs="*", default=[], help="optional post-experiment explanatory sample board; does not change evaluation")
    args = parser.parse_args()
    score(args.index, args.vectors, args.legacy_model, args.balanced_experiment, args.conservative_experiment, args.samples, args.labels, args.probes, args.output, illustration_ids=args.illustration_ids)


if __name__ == "__main__":
    main()
