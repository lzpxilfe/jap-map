#!/usr/bin/env python3
"""Draw new review-only routes using a byte-pinned ArchaeoTrace snapshot.

Machine-selected endpoint pairs are NOT user-confirmed contour endpoints.
No classifier fitting, network, source edits, or human approvals occur.
Local short-gap postprocessing incorporates development human feedback;
its geometry heuristics are not independent accuracy evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import html
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from histcontour_core.assisted_drawing import DrawingConfig, assess_path, endpoint_pairs, native_feature_collection
from histcontour_core.human_feedback import map_to_pixel
from histcontour_core.provenance import sha256_file
from histcontour_core.gap_refinement import GapRefinementConfig, REFINEMENT_VERSION, regularize_gap_path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def resolve(path):
    value = Path(path)
    return value if value.is_absolute() else ROOT/value


def world(tile, points):
    west, south, east, north = tile["bounds"]
    width, height = tile["pixel_bounds"][2:]
    return [[west+(x+.5)*(east-west)/width, north-(y+.5)*(north-south)/height] for x, y in points]


def load_upstream(source, pin_path, output, model_cache):
    pin = read(pin_path)
    if pin.get("schema") != "jap-map-assisted-tracing-upstream/1":
        raise ValueError("unsupported upstream pin")
    for name, digest in pin["files_sha256"].items():
        path = source/name
        if not path.resolve().is_relative_to(source.resolve()) or sha256_file(path) != digest:
            raise ValueError(f"upstream source fingerprint differs: {name}")
    destination = output/"upstream-source"
    destination.mkdir()
    for name in [*pin["files_sha256"], "LICENSE"]:
        target = destination/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source/name, target)
    shutil.copyfile(pin_path, output/"upstream-pin.json")
    sys.path.insert(0, str(destination))
    from benchmarks import worker
    from ai_vectorizer.core import manual_gap_bridge, smart_recovery
    if not Path(worker.__file__).resolve().is_relative_to(destination.resolve()):
        raise ValueError("a different upstream package was already imported")
    if model_cache is None:
        pipeline = worker._load_ink_pipeline(worker.INK_LIVEWIRE_V2_BACKEND, 1)
    else:
        pipeline = worker._load_efficientsam_pipeline(worker.INK_V2_EFFICIENTSAM_RECOVERY_BACKEND, 1, model_cache.resolve())
    config = {"edge_weight": 1.0, "livewire_window_px": 320, "target_snap_radius_px": 6,
              "smoothing_window_px": 5, "smoothing_profile": worker.INK_LIVEWIRE_SMOOTHING_PROFILE,
              "recovery_policy_id": smart_recovery.RECOVERY_POLICY_ID, "recovery_provisional": True,
              "recovery_thresholds": asdict(smart_recovery.DEFAULT_RECOVERY_CONFIG),
              "recovery_configuration_sha256": smart_recovery.DEFAULT_RECOVERY_CONFIG.sha256}
    return pipeline, worker, manual_gap_bridge, pin, config


def draw_overlay(image, lines, colour, width=2, offset=(0, 0)):
    from PIL import ImageDraw
    result = image.convert("RGB").copy()
    draw = ImageDraw.Draw(result)
    for points in lines:
        draw.line([(x-offset[0], y-offset[1]) for x, y in points], fill=colour, width=width)
    return result


def run(args):
    import numpy as np
    from PIL import Image, ImageDraw
    from histcontour_core.completion import rasterize_polylines

    started = time.perf_counter()
    index, evaluated = read(args.index), read(args.baseline)
    if sha256_file(args.index) != evaluated["source_index_sha256"] or evaluated.get("holdout_used") is not False:
        raise ValueError("baseline/source index mismatch or held-out input")
    tiles = [tile for tile in index["tiles"] if tile["split"] == "development"]
    if len(tiles) != 9 or any(tile["sheet_id"] == "178-gongju" for tile in tiles):
        raise ValueError("only the existing nine development tiles may be drawn")
    protected = {str(path.relative_to(ROOT)): sha256_file(path) for path in (
        ROOT/"data/derived/annotation_package/contour_annotations.gpkg",
        ROOT/"data/derived/contour-chain-2026-09-12/human-review-ready/human-review.gpkg")}
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ("sources", "images", "tile-previews"):
        (args.output/name).mkdir()
    pipeline, worker, bridge, pin, tracing_config = load_upstream(args.upstream.resolve(), args.pin, args.output, args.model_cache)
    planning_config = DrawingConfig().validate()
    write(args.output/"configuration.json", {"drawing": asdict(planning_config), "tracing": tracing_config,
          "local_gap_postprocess": {"version": REFINEMENT_VERSION, "config": asdict(GapRefinementConfig()),
                                    "kind": "fixed_endpoints_only", "human_approval_inferred": False},
          "max_per_tile": args.max_per_tile, "endpoint_origin": "machine_selected_not_human_confirmed",
          "baseline_sha256": sha256_file(args.baseline), "upstream_pin_sha256": sha256_file(args.pin),
          "no_fitting_or_E_label_use": True})
    entries = {tile["tile_id"]: tile for tile in evaluated["tiles"]}
    records, attempts, base_features, extra_features, copied_tiles, audits = [], [], [], [], [], []
    for tile in tiles:
        tile_id = tile["tile_id"]
        raster = resolve(tile["raster_path"])
        source_entry = entries[tile_id]["methods"]["all"]
        vectors = resolve(source_entry["path"])
        if sha256_file(raster) != entries[tile_id]["source_raster_sha256"] or sha256_file(vectors) != source_entry["sha256"]:
            raise ValueError("source pixels or baseline geometry changed")
        collection = read(vectors)
        lines = []
        for feature in collection["features"]:
            properties = feature["properties"]
            points = [[round(x, 5), round(y, 5)] for x, y in map_to_pixel(tile, feature["geometry"]["coordinates"])]
            lines.append({"uid": properties["segment_uid"], "points": points,
                          "retained": properties["candidate_retained"], "score": properties["forward_candidate_score"]})
            if properties["candidate_retained"]:
                base_features.append(feature)
        pairs, audit = endpoint_pairs(lines, planning_config)
        if args.max_per_tile:
            pairs = pairs[:args.max_per_tile]
        audit.update(tile_id=tile_id, attempted_pairs=len(pairs))
        audits.append(audit)
        write(args.output/f"{tile_id}-endpoint-plan.json", {"config": asdict(planning_config), "audit": audit, "pairs": pairs})
        copied = args.output/"sources"/f"{tile_id}.tif"
        shutil.copyfile(raster, copied)
        copied_tiles.append({**tile, "raster_path": f"sources/{tile_id}.tif",
                             "source_raster_sha256": sha256_file(raster), "baseline_vector_sha256": source_entry["sha256"]})
        width, height = tile["pixel_bounds"][2:]
        cached = pipeline.load_image(raster, width, height)
        cached["source_tile_origin_xy"] = tuple(tile["pixel_bounds"][:2])
        _, evidence = pipeline._prepare_ink(cached)
        base_mask = rasterize_polylines((height, width), [row["points"] for row in lines])
        with Image.open(raster) as image:
            source_image = image.convert("RGB")
        tile_records = []
        for pair_index, pair in enumerate(pairs, 1):
            start, end = pair["first"]["point"], pair["second"]["point"]
            prompt = worker.TracePrompt(tuple(start), tuple(end), previous_xy=tuple(pair["first"]["previous"]))
            attempt = {"tile_id": tile_id, "pair_id": pair["pair_id"], "endpoint_origin": "machine",
                       "source_uid": pair["first"]["uid"], "target_uid": pair["second"]["uid"],
                       "gap_pixels": pair["gap_pixels"], "routes": {}, "selected_mode": None}
            routes = {}
            try:
                path = [list(map(float, point)) for point in pipeline.predict(cached, prompt, tracing_config)]
                quality = assess_path(path, start, end, evidence.center_score, base_mask)
                recovery = pipeline.prediction_evidence() if args.model_cache is not None else None
                mode = "smart_recovery" if recovery and recovery["selected_route"] == "challenger" else "ink_livewire"
                attempt["recovery"] = recovery
                attempt["routes"][mode] = quality
                if (quality["endpoint_error_pixels"] <= 1.0 and quality["detour_ratio"] <= 1.5
                        and quality["supported_fraction"] >= .8 and quality["new_fraction_outside_original_1_5px"] >= .1):
                    routes[mode] = (path, quality)
            except (ValueError, RuntimeError) as error:
                attempt["ink_error"] = str(error)[:500]
            try:
                # The strengthened connected-outward sampler is mandatory here,
                # even if a three-pixel tangent alone would permit a bridge.
                tangents = bridge.sample_manual_gap_bridge_tangents(evidence, start, end)
                if tangents is None:
                    attempt["gap_rejection"] = "no_unambiguous_connected_outward_support"
                else:
                    gap = bridge.build_manual_gap_bridge(start, end, *tangents)
                    path = [list(point) for point in gap.points_xy]
                    path, refinement = regularize_gap_path(path)
                    attempt["gap_geometry_refinement"] = refinement
                    quality = assess_path(path, start, end, evidence.center_score, base_mask)
                    attempt["routes"]["contextual_gap"] = quality
                    if quality["new_fraction_outside_original_1_5px"] >= .1:
                        routes["contextual_gap"] = (path, quality)
            except (ValueError, RuntimeError) as error:
                attempt["gap_rejection"] = str(error)[:500]
            chosen = next((mode for mode in ("smart_recovery", "ink_livewire", "contextual_gap") if mode in routes), None)
            attempt["selected_mode"] = chosen
            attempts.append(attempt)
            if chosen is None:
                continue
            path, quality = routes[chosen]
            proposal_id = f"A{len(records)+1:04}"
            center_x, center_y = (start[0]+end[0])/2, (start[1]+end[1])/2
            half_x, half_y = max(80, abs(start[0]-end[0])/2+40), max(80, abs(start[1]-end[1])/2+40)
            box = [max(0, int(center_x-half_x)), max(0, int(center_y-half_y)),
                   min(width, int(center_x+half_x+1)), min(height, int(center_y+half_y+1))]
            record = {"proposal_id": proposal_id, "tile_id": tile_id, "sheet_id": tile["sheet_id"],
                      "pair_id": pair["pair_id"], "mode": chosen, "pixel_points": path, "pixel_box": box,
                      "start": start, "end": end, "source_uid": pair["first"]["uid"], "target_uid": pair["second"]["uid"],
                      "source_endpoint": pair["first"]["id"], "target_endpoint": pair["second"]["id"],
                      "competing_endpoint_pair": pair["competing_endpoint_pair"], "gap_pixels": pair["gap_pixels"],
                      "quality": quality, "dataset_role": "review_only_not_training",
                      "reference_origin": "machine_generated_not_human", "human_approved": False,
                      "review_status": "unreviewed", "annotator": "", "review_note": "",
                      "question": "두 끝점을 이 경로로 이어도 될까요? 등고선이 아닌 획이거나 이웃 선으로 넘어가면 거절해 주세요."}
            if chosen == "contextual_gap":
                record["geometry_refinement"] = attempt["gap_geometry_refinement"]
            records.append(record)
            tile_records.append(record)
            extra_features.append({"type": "Feature", "geometry": {"type": "LineString", "coordinates": world(tile, path)},
                                   "properties": {key: value for key, value in record.items() if key not in ("pixel_points", "pixel_box", "quality", "start", "end")}})
            crop = source_image.crop(box)
            overlay = draw_overlay(crop, [path], (230, 115, 0) if chosen == "contextual_gap" else (0, 160, 70), 2, box[:2])
            for image in (crop, overlay):
                draw = ImageDraw.Draw(image)
                for x, y in (start, end):
                    x, y = x-box[0], y-box[1]
                    draw.ellipse((x-3, y-3, x+3, y+3), outline=(0, 100, 235), width=1)
            crop.save(args.output/"images"/f"{proposal_id}-source.png")
            overlay.save(args.output/"images"/f"{proposal_id}-proposal.png")
            if pair_index % 25 == 0:
                print(f"{tile_id}: {pair_index}/{len(pairs)} endpoint pairs; {len(tile_records)} proposed paths", flush=True)
        base_preview = draw_overlay(source_image, [line["points"] for line in lines if line["retained"]], (0, 130, 180), 1)
        new_preview = base_preview.copy()
        for mode, colour in (("ink_livewire", (0, 170, 70)), ("smart_recovery", (180, 0, 200)), ("contextual_gap", (240, 115, 0))):
            new_preview = draw_overlay(new_preview, [row["pixel_points"] for row in tile_records if row["mode"] == mode], colour, 2)
        source_image.save(args.output/"tile-previews"/f"{tile_id}-source.png")
        base_preview.save(args.output/"tile-previews"/f"{tile_id}-before.png")
        new_preview.save(args.output/"tile-previews"/f"{tile_id}-after.png")
        print(f"{tile_id}: finished {len(pairs)} attempts, {len(tile_records)} new review-only routes", flush=True)
    priority = []
    for tile in tiles:
        subset = [row for row in records if row["tile_id"] == tile["tile_id"]]
        for inferred in (False, True):
            candidates = [row for row in subset if (row["mode"] == "contextual_gap") == inferred]
            if candidates:
                priority.append(max(candidates, key=lambda row: row["quality"]["path_length_pixels"]*row["quality"]["new_fraction_outside_original_1_5px"])["proposal_id"])
    for row in records:
        row["priority"] = 1 if row["proposal_id"] in priority else 2
    for row in extra_features:
        row["properties"]["priority"] = 1 if row["properties"]["proposal_id"] in priority else 2
    write(args.output/"base-lines.geojson", native_feature_collection(base_features, tiles))
    write(args.output/"ai-proposals.geojson", native_feature_collection(extra_features, tiles))
    write(args.output/"attempts.json", attempts)
    if any(sha256_file(ROOT/path) != digest for path, digest in protected.items()):
        raise RuntimeError("a protected review file changed during the run; no file was restored or overwritten")
    report = {"schema": "jap-map-assisted-contour-drawing/1", "status": "review_only_candidates_ready",
              "upstream_commit": pin["commit"], "upstream_pin_sha256": sha256_file(args.pin),
              "configuration_sha256": sha256_file(args.output/"configuration.json"),
              "baseline_sha256": sha256_file(args.baseline), "protected_review_sha256": protected,
              "model_cache_used": args.model_cache is not None, "backend": asdict(pipeline.info),
              "endpoint_selection": "machine heuristic; not equivalent to upstream human-confirmed endpoints",
              "holdout_used": False, "human_approvals": 0, "automatic_promotion": False,
              "model_fitted": False, "formal_accuracy": None, "tiles": copied_tiles, "planning_audits": audits,
              "baseline_line_count": len(base_features), "attempt_count": len(attempts), "proposal_count": len(records),
              "mode_counts": dict(Counter(row["mode"] for row in records)), "priority_ids": priority,
              "recovery_trigger_count": sum(bool(row.get("recovery", {}).get("gate", {}).get("trigger")) for row in attempts if row.get("recovery")),
              "recovery_accepted_count": sum(row.get("recovery", {}).get("selected_route") == "challenger" for row in attempts if row.get("recovery")),
              "elapsed_seconds": time.perf_counter()-started, "proposals": records}
    write(args.output/"drawing-report.json", report)
    blocks = []
    ordered = sorted(records, key=lambda row: (row["priority"], row["proposal_id"]))
    for row in ordered:
        sid = row["proposal_id"]
        blocks.append(f'<section id="{sid}"><h2>{sid} · {html.escape(row["tile_id"])} · {row["mode"]} · 우선순위 {row["priority"]}</h2><p>{html.escape(row["question"])}</p><div class="pair"><figure><figcaption>원본 · 파란 점은 기계가 선택한 끝점</figcaption><img src="images/{sid}-source.png" loading="lazy"></figure><figure><figcaption>AI 제안 · 사람 승인 없음</figcaption><img src="images/{sid}-proposal.png" loading="lazy"></figure></div></section>')
    document = ('<!doctype html><html lang="ko"><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="default-src \'none\'; img-src \'self\'; style-src \'unsafe-inline\'"><title>AI가 먼저 그린 등고선 후보</title><style>body{font:16px system-ui;margin:24px;max-width:1200px}.pair{display:flex;gap:20px}figure{flex:1;margin:0}img{max-width:100%;image-rendering:pixelated}section{border-top:1px solid #bbb;padding:14px 0}</style>'
                f'<h1>AI가 먼저 그린 추가 경로 {len(records)}개</h1><p>기존 후보 {len(base_features)}개는 그대로 보존했습니다. 초록=Ink 지지 경로, 보라=Recovery 채택, 주황=공백 추론입니다. 색은 등고선 정답이나 사람 승인이 아닙니다.</p><p>전체를 다시 그리지 말고 함께 제공한 QGIS에서 채택·거절·수정 필요·판독 불가를 기록하세요. 먼저 우선순위 1의 {len(priority)}개를 제안합니다. 모든 새 자료는 학습에 넣지 않는 검토용입니다.</p>'
                + "".join(blocks) + "</html>")
    with (args.output/"report.html").open("x", encoding="utf-8") as handle:
        handle.write(document)
    print({key: report[key] for key in ("attempt_count", "proposal_count", "mode_counts", "recovery_trigger_count", "recovery_accepted_count", "elapsed_seconds")}, flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=ROOT/"data/derived/annotation_package/index.json")
    parser.add_argument("--baseline", type=Path, default=ROOT/"data/derived/contour-chain-2026-09-12/forward-evaluation-final/forward_evaluation.json")
    parser.add_argument("--pin", type=Path, default=ROOT/"examples/contour_assisted_upstream.v1.json")
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, help="explicit existing verified cache; never downloads a model")
    parser.add_argument("--max-per-tile", type=int, default=0, help="0 runs every bounded endpoint pair")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_per_tile < 0:
        parser.error("--max-per-tile must be nonnegative")
    args.output = args.output.resolve()
    run(args)


if __name__ == "__main__":
    main()
