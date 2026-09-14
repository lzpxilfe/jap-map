#!/usr/bin/env python3
"""Recognize source-component word quads without a text detector.

The fixed R004/R015/R009 development crops run by default. Every group retains
component IDs, geometric evidence and selection/omission reasons. Separate 2x
and 4x readings never assign elevation, deletion masks or human approval.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from histcontour_core.map_text_detection import EVIDENCE_FLAGS, MapTextDetectionError, inspect_raster, raw_recognition
from histcontour_core.margin_ocr import digest_file
from histcontour_core.paddle_margin_ocr import local_inference_only, verify_models
from scripts.detect_map_text_multiscale import DEFAULT_REGIONS, DEFAULT_SCALES, _pixel_digest, _selection, map_coordinates
from scripts.prepare_contour_reconstruction_review import read, write, validate_regions

SCHEMA = "jap-map-component-text-recognition/1"
MAX_CROPPED_PIXELS_PER_PASS = 8_000_000


def create_recognizer(lock, base, cpu_threads):
    """Use only the pinned recognizer and geometric cropper; no detector."""
    directories = verify_models(lock, base)
    from paddleocr import TextRecognition
    from paddlex.inference.pipelines.components import CropByPolys
    recognizer = TextRecognition(model_name=lock["models"]["recognition"]["model_name"],
                                 model_dir=directories["recognition"], device="cpu", engine="paddle_static",
                                 enable_hpi=False, enable_mkldnn=False, cpu_threads=cpu_threads)
    return recognizer, CropByPolys(det_box_type="quad")


def create_candidates(gray, origin):
    from histcontour_core.component_text_candidates import ComponentTextCandidateConfig, find_component_text_candidates
    config = ComponentTextCandidateConfig()
    return find_component_text_candidates(gray, source_origin_xy=origin, config=config), asdict(config)


def _numeric_runtime():
    from importlib import metadata
    return {"python_executable": str(Path(sys.executable).resolve()), "python_version": sys.version,
            "packages": {name: metadata.version(name) for name in ("numpy", "scipy", "Pillow")}}


def candidate_worker(request_path, output):
    """Numeric-only subprocess: validate the fixed source again and emit data."""
    import numpy as np
    from PIL import Image
    request_path, output = Path(request_path).resolve(), Path(output).resolve()
    request_hash = digest_file(request_path)
    request = read(request_path)
    report_path, manifest_path = Path(request["drawing_report"]), Path(request["manifest"])
    if output.exists() or output.is_relative_to(report_path.parent):
        raise MapTextDetectionError("numeric worker output must be new and outside the source snapshot")
    if digest_file(report_path) != request["drawing_report_sha256"] or digest_file(manifest_path) != request["manifest_sha256"]:
        raise MapTextDetectionError("numeric worker source pins changed")
    report, manifest = read(report_path), read(manifest_path)
    tiles = validate_regions(manifest, report)
    if manifest["drawing_report_sha256"] != request["drawing_report_sha256"]:
        raise MapTextDetectionError("numeric worker manifest/report pins disagree")
    region = _selection(manifest, (request["region_id"],), DEFAULT_SCALES)[0]
    tile = tiles[region["tile_id"]]
    relative = Path(tile["raster_path"])
    source = (report_path.parent/relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or not source.is_relative_to(report_path.parent):
        raise MapTextDetectionError("numeric worker raster path escapes source snapshot")
    inspect_raster(source, tile)
    runtime_before = _numeric_runtime()
    output.mkdir(parents=True, exist_ok=False)
    with local_inference_only(cache_directory=output/"runtime-cache"):
        with Image.open(source) as image:
            gray = np.asarray(image.crop(region["box"]).convert("L"))
        pixels_hash = _pixel_digest(gray)
        candidates, config = create_candidates(gray, tuple(region["box"][:2]))
        if _pixel_digest(gray) != pixels_hash:
            raise MapTextDetectionError("numeric worker changed source pixels")
    runtime_after = _numeric_runtime()
    if (runtime_after != runtime_before or digest_file(source) != tile["source_raster_sha256"]
            or digest_file(report_path) != request["drawing_report_sha256"] or digest_file(manifest_path) != request["manifest_sha256"]
            or digest_file(request_path) != request_hash):
        raise MapTextDetectionError("numeric worker inputs or runtime changed")
    arrays_path = output/"component-arrays.npz"
    np.savez_compressed(arrays_path, source_ink=candidates.source_ink, component_labels=candidates.component_labels)
    write(output/"candidates.json", {"components": list(candidates.components), "groups": list(candidates.groups),
          "provenance": candidates.provenance, "configuration": config, "source_crop_gray_payload_sha256": pixels_hash,
          "arrays_sha256": digest_file(arrays_path), "runtime_before": runtime_before, "runtime_after": runtime_after,
          "source_files_unchanged": True, "network_connections_blocked": True})


def candidates_from_worker(python, report_path, manifest_path, rid, output, expected_gray_hash):
    import numpy as np
    from histcontour_core.component_text_candidates import ComponentTextCandidateResult
    folder = output/"candidate-workers"/rid
    folder.mkdir(parents=True, exist_ok=False)
    request = folder/"request.json"
    write(request, {"drawing_report": str(report_path), "manifest": str(manifest_path), "region_id": rid,
                   "drawing_report_sha256": digest_file(report_path), "manifest_sha256": digest_file(manifest_path)})
    completed = subprocess.run([str(python), "-I", str(Path(__file__).resolve()), "--candidate-worker",
                                "--worker-request", str(request), "--worker-output", str(folder/"result")],
                               capture_output=True, text=True, timeout=180, check=False)
    if completed.returncode:
        raise MapTextDetectionError(f"numeric candidate worker failed: {completed.stderr[-4000:]}")
    packet_path, arrays_path = folder/"result/candidates.json", folder/"result/component-arrays.npz"
    packet = read(packet_path)
    if (packet["source_crop_gray_payload_sha256"] != expected_gray_hash or packet["arrays_sha256"] != digest_file(arrays_path)
            or packet["runtime_before"] != packet["runtime_after"]):
        raise MapTextDetectionError("numeric candidate packet pixels, arrays or runtime differ")
    with np.load(arrays_path, allow_pickle=False) as arrays:
        result = ComponentTextCandidateResult(arrays["source_ink"], arrays["component_labels"], tuple(packet["components"]),
                                              tuple(packet["groups"]), packet["provenance"])
    execution = {"mode": "isolated_numeric_subprocess", "runtime": packet["runtime_before"],
                 "runtime_verified_before_and_after": True, "network_connections_blocked": True,
                 "packet_path": str(packet_path.relative_to(output)), "packet_sha256": digest_file(packet_path),
                 "array_packet_sha256": packet["arrays_sha256"]}
    return result, packet["configuration"], execution


def group_input_quad(group, scale):
    """Convert centre to corner BEFORE enlargement: (0, 0) becomes (2, 2) at 4x."""
    return [[(float(x) + .5) * scale, (float(y) + .5) * scale] for x, y in group["quad_crop_pixel_centers"]]


def validate_group(group, box, sheet_origin):
    import math
    left, top, right, bottom = box
    quad = group.get("quad_crop_pixel_centers")
    if (not isinstance(quad, (list, tuple)) or len(quad) != 4
            or any(not isinstance(point, (list, tuple)) or len(point) != 2
                   or any(type(value) not in (int, float) or not math.isfinite(value) for value in point) for point in quad)):
        raise MapTextDetectionError("component group needs a finite source-centre quad")
    area = sum(a[0]*b[1] - a[1]*b[0] for a, b in zip(quad, [*quad[1:], quad[0]]))
    if abs(area) <= 1e-8:
        raise MapTextDetectionError("component group quad is degenerate")
    if not isinstance(group.get("group_id"), str) or not group["group_id"]:
        raise MapTextDetectionError("component group ID is required")
    if type(group.get("rank")) is not int or group["rank"] < 1:
        raise MapTextDetectionError("component group needs its original positive rank")
    score = group.get("ranking_score_not_probability")
    if type(score) not in (float, int) or not math.isfinite(score):
        raise MapTextDetectionError("component geometric ranking score must be finite")
    if (group.get("human_approved") is not False or group.get("training_eligible") is not False
            or group.get("ambiguous") is not True):
        raise MapTextDetectionError("component group must remain ambiguous and unapproved")
    mapped = map_coordinates([group_input_quad(group, 2)], box, 2, sheet_origin)
    for actual, expected, name in (
        (group.get("quad_source_pixel_centers"), mapped["tile_center_grid_polygons"][0], "centre"),
        (group.get("quad_source_image_corners"), mapped["tile_corner_polygons"][0], "corner"),
    ):
        if (not isinstance(actual, (tuple, list)) or len(actual) != 4
                or any(not isinstance(a, (tuple, list)) or len(a) != 2
                       or any(type(v) not in (int, float) or not math.isfinite(v) for v in a)
                       or any(abs(x-y) > 1e-8 for x, y in zip(a, b)) for a, b in zip(actual, expected))):
            raise MapTextDetectionError(f"core and caller source {name} transforms disagree")
    within = all(-.5 <= x <= right-left-.5 and -.5 <= y <= bottom-top-.5 for x, y in quad)
    return mapped, within


def run(drawing_report, manifest_path, model_lock, output, *, region_ids=DEFAULT_REGIONS,
        scales=DEFAULT_SCALES, max_recognition_groups_per_region=128, cpu_threads=1, candidate_python=None):
    import numpy as np
    from PIL import Image

    if type(max_recognition_groups_per_region) is not int or not 1 <= max_recognition_groups_per_region <= 128:
        raise MapTextDetectionError("recognition group cap must be an integer in [1, 128]")
    if type(cpu_threads) is not int or not 1 <= cpu_threads <= 8:
        raise MapTextDetectionError("cpu_threads must be an integer in [1, 8]")
    report_path, manifest_path, lock_path, output = (Path(path).resolve() for path in (drawing_report, manifest_path, model_lock, output))
    if output.exists():
        raise FileExistsError("component recognition output must be a NEW directory")
    if output.is_relative_to(report_path.parent):
        raise MapTextDetectionError("output must be outside the source snapshot")
    inputs = {"drawing_report": report_path, "region_manifest": manifest_path, "model_lock": lock_path,
              "recognition_python_executable": Path(sys.executable).resolve()}
    if candidate_python is not None:
        candidate_python = Path(candidate_python).absolute()
        if not candidate_python.is_file():
            raise MapTextDetectionError("candidate Python must be an existing local interpreter")
        inputs["candidate_python_executable"] = candidate_python.resolve()
    before = {key: digest_file(path) for key, path in inputs.items()}
    report, manifest, lock = read(report_path), read(manifest_path), read(lock_path)
    tiles = validate_regions(manifest, report)
    if before["drawing_report"] != manifest.get("drawing_report_sha256"):
        raise MapTextDetectionError("drawing report differs from the fixed region manifest")
    selected = _selection(manifest, tuple(region_ids), tuple(scales))
    sources, metadata = {}, {}
    for tid, tile in tiles.items():
        relative = Path(tile.get("raster_path", ""))
        source = (report_path.parent / relative).resolve()
        if (relative.is_absolute() or ".." in relative.parts or not relative.parts
                or not source.is_relative_to(report_path.parent) or (report_path.parent / relative).is_symlink()):
            raise MapTextDetectionError("source raster path escapes the immutable snapshot")
        metadata[tid] = inspect_raster(source, tile)
        sources[tid], inputs[tid], before[tid] = source, source, tile["source_raster_sha256"]
    for region in selected:
        left, top, right, bottom = region["box"]
        if any(max(right-left, bottom-top)*scale > 4096 or (right-left)*(bottom-top)*scale*scale > 4_000_000 for scale in scales):
            raise MapTextDetectionError("enlarged region exceeds its finite input-pixel budget")
    directories = verify_models(lock, lock_path.parent)
    if any(output.is_relative_to(Path(directory)) for directory in directories.values()):
        raise MapTextDetectionError("output must be outside pinned model directories")
    implementations = (Path(__file__), ROOT/"histcontour_core/component_text_candidates.py",
                       ROOT/"histcontour_core/map_text_detection.py", ROOT/"histcontour_core/paddle_margin_ocr.py",
                       ROOT/"histcontour_core/margin_ocr.py", ROOT/"scripts/detect_map_text_multiscale.py",
                       ROOT/"scripts/prepare_contour_reconstruction_review.py")
    implementation_before = {str(path.resolve()): digest_file(path) for path in implementations}
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    rows, features, image_artifacts, component_artifacts = [], [], [], []
    with local_inference_only(cache_directory=output/"runtime-cache"):
        recognizer, cropper = create_recognizer(lock, lock_path.parent, cpu_threads)
        try:
            for region in selected:
                rid, tid, box = region["region_id"], region["tile_id"], region["box"]
                tile = tiles[tid]
                with Image.open(sources[tid]) as source:
                    crop = source.crop(box).convert("RGB")
                gray = np.asarray(crop.convert("L"))
                source_payload = _pixel_digest(gray)
                if candidate_python is None:
                    candidates, candidate_config = create_candidates(gray, tuple(box[:2]))
                    candidate_execution = {"mode": "same_process", "python_executable": str(Path(sys.executable).resolve())}
                else:
                    candidates, candidate_config, candidate_execution = candidates_from_worker(
                        candidate_python, report_path, manifest_path, rid, output, source_payload)
                if _pixel_digest(gray) != source_payload:
                    raise MapTextDetectionError("component generation changed source pixels")
                components = copy.deepcopy(list(candidates.components))
                groups = copy.deepcopy(list(candidates.groups))
                if len({group["group_id"] for group in groups}) != len(groups):
                    raise MapTextDetectionError("duplicate component group IDs")
                if len({group["rank"] for group in groups}) != len(groups):
                    raise MapTextDetectionError("duplicate component group ranks")
                component_ids = {component["component_id"] for component in components}
                ranked = sorted(groups, key=lambda group: group["rank"])
                records = []
                for group in ranked:
                    rank = group["rank"]
                    ids = group.get("group_component_ids")
                    partial_single = (isinstance(ids, list) and len(ids) == 1
                        and group.get("candidate_kind") == "intra_component_hole_axis"
                        and group.get("partial_component_search") is True
                        and group.get("full_component_containment") is False)
                    if (not isinstance(ids, list) or not (2 <= len(ids) <= 5 or partial_single) or len(set(ids)) != len(ids)
                            or any(type(cid) is not int or cid not in component_ids for cid in ids)):
                        raise MapTextDetectionError("group IDs must refer to 2..5 source components or an explicit partial intra-component search")
                    mapped, within = validate_group(group, box, tile["pixel_bounds"][:2])
                    reason = "selected_by_geometric_rank" if rank <= max_recognition_groups_per_region else "recognition_group_cap"
                    if not within:
                        reason = "quad_outside_crop_no_clipping"
                    records.append({"candidate": group, "selection_rank": rank, "selection_reason": reason,
                                    "selected_for_recognition": reason == "selected_by_geometric_rank",
                                    "mapped_source_coordinates": mapped, "passes": [], **EVIDENCE_FLAGS})
                    quad = mapped["tile_corner_polygons"][0]
                    gt = metadata[tid]["geotransform"]
                    native = [[gt[0]+x*gt[1]+y*gt[2], gt[3]+x*gt[4]+y*gt[5]] for x, y in quad]
                    features.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[*native, native[0][:]]]},
                                     "properties": {"region_id": rid, "tile_id": tid, "group_id": group["group_id"],
                                                    "group_component_ids": group["group_component_ids"],
                                                    "ranking_score_not_probability": group["ranking_score_not_probability"],
                                                    "selection_rank": rank, "selection_reason": reason,
                                                    "geometry_role": "component_group_recognition_search_quad_not_glyph_mask", **EVIDENCE_FLAGS}})
                for scale in scales:
                    pixels = np.asarray(crop.resize((crop.width*scale, crop.height*scale), Image.Resampling.BICUBIC))[:, :, ::-1].copy()
                    payload = _pixel_digest(pixels)
                    inference_crops, inference_records, crop_hashes = [], [], []
                    cropped_pixels, exhausted = 0, False
                    for record in records:
                        group = record["candidate"]
                        pass_row = {"scale": scale, "enlarged_input_size_px": [pixels.shape[1], pixels.shape[0]],
                                    "input_bgr_payload_sha256": payload, "resize_filter": "Pillow BICUBIC",
                                    "quad_on_enlarged_crop_image_corners": group_input_quad(group, scale),
                                    "status": "not_run", "reason": record["selection_reason"], **EVIDENCE_FLAGS}
                        record["passes"].append(pass_row)
                        if not record["selected_for_recognition"]:
                            continue
                        if exhausted:
                            pass_row["reason"] = "recognition_pixel_budget"
                            continue
                        cropped = list(cropper(pixels, [copy.deepcopy(pass_row["quad_on_enlarged_crop_image_corners"])]))
                        if len(cropped) != 1:
                            raise MapTextDetectionError("one component group must yield exactly one recognition crop")
                        patch = np.asarray(cropped[0])
                        if patch.ndim != 3 or patch.shape[2] != 3 or min(patch.shape[:2]) < 1:
                            raise MapTextDetectionError("quad rectifier returned an invalid BGR image")
                        if cropped_pixels + patch.shape[0]*patch.shape[1] > MAX_CROPPED_PIXELS_PER_PASS:
                            exhausted = True
                            pass_row["reason"] = "recognition_pixel_budget"
                            continue
                        cropped_pixels += patch.shape[0]*patch.shape[1]
                        inference_crops.append(patch)
                        inference_records.append(record)
                        crop_hashes.append(_pixel_digest(patch))
                        pass_row.update(status="pending", reason="selected_by_geometric_rank",
                                        rectified_size_px=[patch.shape[1], patch.shape[0]], rectified_bgr_payload_sha256=crop_hashes[-1])
                    ids = [f"{rid}:{record['candidate']['group_id']}:{scale}x" for record in inference_records]
                    recognition = raw_recognition(list(recognizer.predict(inference_crops)) if inference_crops else [], ids)
                    if _pixel_digest(pixels) != payload or any(_pixel_digest(patch) != digest for patch, digest in zip(inference_crops, crop_hashes)):
                        raise MapTextDetectionError("recognition mutated its supplied image payload")
                    for record, patch, reading in zip(inference_records, inference_crops, recognition["readings"]):
                        pass_row = record["passes"][-1]
                        path = f"{rid}/recognition-crops/group-{record['selection_rank']:03d}-{scale}x.png"
                        pass_row.update(status="completed", recognition=reading, recognition_crop_image=path, crop_method=recognition["crop_method"])
                        image_artifacts.append((path, patch[:, :, ::-1].copy()))
                    print(f"{rid} {scale}x: {len(groups)} retained geometric groups, {len(inference_records)} recognized; no detector used", flush=True)
                component_artifacts.append((rid, np.array(candidates.source_ink, copy=True), np.array(candidates.component_labels, copy=True)))
                rows.append({**region, "sheet_id": tile["sheet_id"], "split": "development",
                             "source_raster_sha256": tile["source_raster_sha256"], "source_absolute_path": str(sources[tid]),
                             "source_crop_gray_payload_sha256": source_payload, "crop_size_px": [crop.width, crop.height],
                             "source_sheet_crop_origin_px": [tile["pixel_bounds"][0]+box[0], tile["pixel_bounds"][1]+box[1]],
                             "crs_authid": tile["crs_authid"], "geotransform": metadata[tid]["geotransform"],
                             "component_generation_configuration": candidate_config,
                             "component_generation_execution": candidate_execution,
                             "component_generation_provenance": copy.deepcopy(candidates.provenance),
                             "components": components, "groups": records, "component_pixel_arrays": f"{rid}/component-pixels.npz", **EVIDENCE_FLAGS})
        finally:
            recognizer.close()
        verify_models(lock, lock_path.parent)
    if any(digest_file(path) != before[key] for key, path in inputs.items()):
        raise MapTextDetectionError("input changed during inference; no success report published")
    if any(digest_file(Path(path)) != digest for path, digest in implementation_before.items()):
        raise MapTextDetectionError("implementation changed during inference; no success report published")
    result = {"schema": SCHEMA, "status": "unapproved_component_recognition_evidence", "regions": rows,
              "holdout_used": False, "human_approvals": 0, "native_crs": next(iter(tiles.values()))["crs_authid"],
              "counts": {"regions": len(rows), "retained_groups": sum(len(row["groups"]) for row in rows),
                         "recognition_attempts": sum(p["status"] == "completed" for row in rows for g in row["groups"] for p in g["passes"]),
                         "recognition_skips": sum(p["status"] == "not_run" for row in rows for g in row["groups"] for p in g["passes"])},
              "provenance": {"input_files": {key: {"path": str(path), "sha256": before[key]} for key, path in inputs.items()},
                             "implementation_sha256": implementation_before, "model_lock": lock,
                             "scales": list(scales), "selected_region_ids": list(region_ids), "cpu_threads": cpu_threads,
                             "max_recognition_groups_per_region": max_recognition_groups_per_region,
                             "max_cropped_pixels_per_pass": MAX_CROPPED_PIXELS_PER_PASS,
                             "model_and_runtime_verified_before_and_after": True, "source_files_unchanged": True,
                             "network_connections_blocked": True, "source_raster_resized_on_disk": False,
                             "device": "cpu", "inference_engine": "paddle_static", "text_detector_used": False,
                             "python_executable": str(Path(sys.executable).resolve())},
              "limitations": ["Geometric groups are ambiguous source-ink hypotheses, not text ground truth.",
                              "All retained groups and rank/cap/boundary skips are preserved; core enumeration/output caps remain in generation provenance.",
                              "Recognition score never ranks or filters candidate selection and is not a geometric detector score.",
                              "Source quads are mapped from centre to corner coordinates before enlargement; no clipping or global rotation occurs.",
                              "Perspective/tall-crop rectification is recognition-only; no glyph coordinates are inferred from its rotated image.",
                              "Numeral strings never assign contour elevation, erase ink, or approve geometry.",
                              "Three selected development crops do not estimate general OCR recall or independent accuracy."],
              "wall_seconds": time.perf_counter()-started, **EVIDENCE_FLAGS, "dataset_role": manifest["dataset_role"]}
    for rid, ink, labels in component_artifacts:
        folder = output/rid
        folder.mkdir()
        np.savez_compressed(folder/"component-pixels.npz", source_ink=ink, component_labels=labels)
    for relative, image in image_artifacts:
        path = output/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(path)
    for row in rows:
        write(output/row["region_id"]/"candidates-and-readings.json", row)
    if (any(digest_file(path) != before[key] for key, path in inputs.items())
            or any(digest_file(Path(path)) != digest for path, digest in implementation_before.items())):
        raise MapTextDetectionError("input or implementation changed while writing artifacts; no success report published")
    write(output/"component-text-recognition.json", result)
    write(output/"component-search-regions.geojson", {"type": "FeatureCollection", "features": features,
          "crs": {"type": "name", "properties": {"name": result["native_crs"]}},
          "coordinate_contract": "native CRS GeoJSON extension; candidate source corners transformed without another half-pixel shift", **EVIDENCE_FLAGS})
    return result


def main():
    if "--candidate-worker" in sys.argv[1:]:
        parser = argparse.ArgumentParser(description="Internal numeric-only source candidate worker")
        parser.add_argument("--candidate-worker", action="store_true")
        parser.add_argument("--worker-request", required=True, type=Path)
        parser.add_argument("--worker-output", required=True, type=Path)
        args = parser.parse_args()
        candidate_worker(args.worker_request, args.worker_output)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drawing-report", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model-lock", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--regions", nargs="+", default=DEFAULT_REGIONS)
    parser.add_argument("--scales", nargs="+", type=int, choices=(2, 4), default=DEFAULT_SCALES)
    parser.add_argument("--max-recognition-groups-per-region", type=int, default=128)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--candidate-python", type=Path, help="Existing NumPy/SciPy/Pillow interpreter; uses a data-only worker, without installs")
    args = parser.parse_args()
    result = run(args.drawing_report, args.manifest, args.model_lock, args.output, region_ids=args.regions,
                 scales=args.scales, max_recognition_groups_per_region=args.max_recognition_groups_per_region,
                 cpu_threads=args.cpu_threads, candidate_python=args.candidate_python)
    print(json.dumps({"output": str(args.output.resolve()), "counts": result["counts"], "status": result["status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
