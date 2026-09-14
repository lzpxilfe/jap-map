#!/usr/bin/env python3
"""Compare pinned local text detection/recognition on explicit 2x/4x crops.

Only R004, R015 and R009 run by default. This is a three-region development
diagnostic, not an OCR accuracy evaluation. Input GeoTIFFs are read unchanged;
resizing takes place in memory and all detector polygons/scores stay separate
for each pass. OCR numerals are never interpreted as elevation or erase masks.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from histcontour_core.map_text_detection import (
    DetectionConfig, EVIDENCE_FLAGS, MapTextDetectionError, create_engines,
    evidence_features, inspect_raster, polygons_on_pixel_center_grid,
    raw_detection, raw_recognition,
)
from histcontour_core.margin_ocr import digest_file, digest_value
from histcontour_core.paddle_margin_ocr import local_inference_only, verify_models
from scripts.prepare_contour_reconstruction_review import read, render_panel, validate_regions, write


SCHEMA = "jap-map-map-text-multiscale/1"
DEFAULT_REGIONS = ("R004", "R015", "R009")
DEFAULT_SCALES = (2, 4)


def map_coordinates(polygons, box, scale, sheet_origin):
    """Invert the explicit crop/resize transform without changing raw polygons."""
    left, top, _right, _bottom = box
    ox, oy = sheet_origin
    tile = [[[float(x) / scale + left, float(y) / scale + top] for x, y in polygon] for polygon in polygons]
    sheet = [[[x + ox, y + oy] for x, y in polygon] for polygon in tile]
    return {
        "tile_corner_polygons": tile,
        "tile_center_grid_polygons": polygons_on_pixel_center_grid(tile),
        "source_sheet_corner_polygons": sheet,
        "source_sheet_center_grid_polygons": polygons_on_pixel_center_grid(sheet),
        "detector_corner_to_tile_corner_affine": [[1. / scale, 0., left], [0., 1. / scale, top], [0., 0., 1.]],
        "detector_corner_to_source_sheet_corner_affine": [[1. / scale, 0., left + ox], [0., 1. / scale, top + oy], [0., 0., 1.]],
        "center_grid_conversion": "Apply the inverse resize/crop affine to raw detector corners, then subtract 0.5 from x and y.",
        "padding_px": [0, 0, 0, 0], "global_rotation_degrees": 0,
        "recognition_geometry": "Recognition rectifies detector quads only for string inference; no coordinates from rectified/rotated recognition images are mapped or reported.",
    }


def _pixel_digest(pixels):
    return digest_value({"shape": list(pixels.shape), "dtype": str(pixels.dtype),
                         "bytes_sha256": hashlib.sha256(pixels.tobytes()).hexdigest()})


def _selection(manifest, region_ids, scales):
    regions = {row["region_id"]: row for row in manifest["regions"]}
    if (not region_ids or len(region_ids) != len(set(region_ids))
            or any(type(rid) is not str or rid not in regions for rid in region_ids)):
        raise MapTextDetectionError("select distinct region IDs from the fixed development manifest")
    if not scales or len(scales) != len(set(scales)) or any(type(scale) is not int or scale not in (2, 4) for scale in scales):
        raise MapTextDetectionError("scales must be distinct values from 2 and 4")
    return [copy.deepcopy(regions[rid]) for rid in region_ids]


def run(drawing_report, manifest_path, model_lock, output, *, region_ids=DEFAULT_REGIONS,
        scales=DEFAULT_SCALES, config=DetectionConfig(recognize=True)):
    """Write a new diagnostic folder after strict source/model/runtime checks."""
    import numpy as np
    from PIL import Image, ImageDraw

    config.validate()
    if not config.recognize:
        raise MapTextDetectionError("this comparison requires the separate recognition pass")
    report_path, manifest_path, lock_path, output = (Path(path).resolve() for path in (drawing_report, manifest_path, model_lock, output))
    if output.exists():
        raise FileExistsError("multiscale output must be a NEW directory")
    if output.is_relative_to(report_path.parent):
        raise MapTextDetectionError("output must be outside the source snapshot")
    started = time.perf_counter()
    inputs = {"drawing_report": report_path, "region_manifest": manifest_path, "model_lock": lock_path}
    before = {key: digest_file(path) for key, path in inputs.items()}
    report, manifest, lock = read(report_path), read(manifest_path), read(lock_path)
    tiles = validate_regions(manifest, report)
    if before["drawing_report"] != manifest.get("drawing_report_sha256"):
        raise MapTextDetectionError("drawing report differs from the fixed manifest hash")
    selected = _selection(manifest, tuple(region_ids), tuple(scales))
    sources, metadata = {}, {}
    # Check every declared development source before running any selected crop.
    for tid, tile in tiles.items():
        relative = Path(tile.get("raster_path", ""))
        source = (report_path.parent / relative).resolve()
        if (relative.is_absolute() or ".." in relative.parts or not relative.parts
                or not source.is_relative_to(report_path.parent) or (report_path.parent / relative).is_symlink()):
            raise MapTextDetectionError("source raster path must remain inside the snapshot")
        metadata[tid] = inspect_raster(source, tile)
        sources[tid], inputs[tid], before[tid] = source, source, tile["source_raster_sha256"]
    for region in selected:
        left, top, right, bottom = region["box"]
        for scale in scales:
            width, height = (right - left) * scale, (bottom - top) * scale
            if max(width, height) > config.limit_side_len or width * height > 4_000_000:
                raise MapTextDetectionError("enlarged crop exceeds detector side or 4,000,000-pixel work limit")
    directories = verify_models(lock, lock_path.parent)
    if any(output.is_relative_to(Path(directory)) for directory in directories.values()):
        raise MapTextDetectionError("output must be outside pinned model directories")
    implementations = (Path(__file__), ROOT/"histcontour_core/map_text_detection.py",
                       ROOT/"histcontour_core/paddle_margin_ocr.py", ROOT/"scripts/prepare_contour_reconstruction_review.py")
    implementation_before = {str(path.resolve()): digest_file(path) for path in implementations}
    output.mkdir(parents=True, exist_ok=False)
    rows, features = [], []
    with local_inference_only(cache_directory=output/"runtime-cache"):
        detector, recognizer, cropper = create_engines(lock, lock_path.parent, config)
        try:
            if recognizer is None or cropper is None:
                raise MapTextDetectionError("local recognition engine and quad cropper are required")
            for region in selected:
                tid, rid, box = region["tile_id"], region["region_id"], region["box"]
                tile = tiles[tid]
                passes = []
                with Image.open(sources[tid]) as source_image:
                    crop = source_image.crop(box).convert("RGB")
                    crop_payload = _pixel_digest(np.asarray(crop))
                    for scale in scales:
                        tick = time.perf_counter()
                        size = crop.width * scale, crop.height * scale
                        pixels = np.asarray(crop.resize(size, Image.Resampling.BICUBIC))[:, :, ::-1].copy()
                        payload = _pixel_digest(pixels)
                        results = list(detector.predict(pixels))
                        if len(results) != 1:
                            raise MapTextDetectionError("one enlarged crop must yield exactly one detector result")
                        detection = raw_detection(results[0], *size)
                        mapped = map_coordinates(detection["dt_polys"], box, scale, tile["pixel_bounds"][:2])
                        mapped_detection = {"dt_polys": mapped["tile_corner_polygons"], "dt_scores": detection["dt_scores"]}
                        pass_features = evidence_features(tile, metadata[tid], mapped_detection)
                        ids = []
                        for index, feature in enumerate(pass_features):
                            region_id = f"{rid}:{scale}x:detector:{index:04d}"
                            ids.append(region_id)
                            feature["properties"].update(region_id=region_id, development_region_id=rid, inference_scale=scale,
                                                         raw_detector_coordinate_space="enlarged_crop_corner_grid")
                        crops = cropper(pixels, copy.deepcopy(detection["dt_polys"])) if detection["dt_polys"] else []
                        recognition = raw_recognition(list(recognizer.predict(crops)) if crops else [], ids)
                        if _pixel_digest(pixels) != payload or _pixel_digest(np.asarray(crop)) != crop_payload:
                            raise MapTextDetectionError("inference mutated its supplied image payload")
                        row = {"scale": scale, "input_size_px": list(size), "resize_filter": "Pillow BICUBIC",
                               "input_bgr_payload_sha256": payload,
                               "raw_detector": {"dt_polys": detection["dt_polys"], "dt_scores": detection["dt_scores"],
                                                "status": detection["status"], "output_stage": detection["output_stage"],
                                                "score_meaning": detection["score_meaning"],
                                                "coordinate_space": "enlarged crop image boundaries; unmodified model postprocessor polygons"},
                               "mapped_source_coordinates": mapped, "recognition": recognition,
                               "wall_seconds": time.perf_counter() - tick, **EVIDENCE_FLAGS}
                        passes.append(row)
                        features.extend(pass_features)
                        print(f"{rid} {scale}x: {len(ids)} raw regions; {len(recognition['readings'])} unfiltered readings", flush=True)
                rows.append({**region, "sheet_id": tile["sheet_id"], "split": "development",
                             "source_raster_sha256": tile["source_raster_sha256"], "source_absolute_path": str(sources[tid]),
                             "source_crop_rgb_payload_sha256": crop_payload, "source_crop_size_px": [crop.width, crop.height],
                             "source_sheet_crop_origin_px": [tile["pixel_bounds"][0]+box[0], tile["pixel_bounds"][1]+box[1]],
                             "crs_authid": tile["crs_authid"], "geotransform": metadata[tid]["geotransform"],
                             "passes": passes, **EVIDENCE_FLAGS})
        finally:
            detector.close()
            if recognizer is not None:
                recognizer.close()
        verify_models(lock, lock_path.parent)
    if any(digest_file(path) != before[key] for key, path in inputs.items()):
        raise MapTextDetectionError("input changed during inference; no success report published")
    if any(digest_file(Path(path)) != expected for path, expected in implementation_before.items()):
        raise MapTextDetectionError("implementation changed during inference; no success report published")
    result = {"schema": SCHEMA, "status": "unapproved_multiscale_text_evidence", "regions": rows,
              "holdout_used": False, "human_approvals": 0,
              "counts": {"regions": len(rows), "passes": sum(len(row["passes"]) for row in rows),
                         "raw_detection_regions": len(features), "recognition_readings": len(features)},
              "native_crs": next(iter(tiles.values()))["crs_authid"],
              "provenance": {"input_files": {key: {"path": str(path), "sha256": before[key]} for key, path in inputs.items()},
                             "implementation_sha256": implementation_before, "model_lock": lock,
                             "configuration": asdict(config), "scales": list(scales), "selected_region_ids": list(region_ids),
                             "model_and_runtime_verified_before_and_after": True,
                             "source_files_unchanged": True, "source_raster_resized_on_disk": False,
                             "network_connections_blocked": True, "device": "cpu", "inference_engine": "paddle_static",
                             "python_executable": str(Path(sys.executable).resolve()),
                             "raw_detector_probability_maps_retained": False},
              "limitations": ["Selected development diagnostics do not estimate general or holdout OCR accuracy.",
                              "Different scales remain separate: no merged regions or silently discarded scores/readings.",
                              "Detector boxes are text search regions, not glyph ownership or deletion masks.",
                              "Recognized digits do not assign contour elevation or approve geometry.",
                              "Perspective/tall-crop rotation is recognition-only; no recognized glyph coordinates are claimed.",
                              "Resizing cannot recover information absent from the original scan."],
              "wall_seconds": time.perf_counter() - started, **EVIDENCE_FLAGS,
              "dataset_role": manifest["dataset_role"]}
    collection = {"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": result["native_crs"]}},
                  "coordinate_contract": "native CRS GeoJSON extension; raw image-corner polygons transformed without half-pixel shift",
                  "features": features, **EVIDENCE_FLAGS}
    for row in rows:
        folder = output/row["region_id"]
        folder.mkdir()
        for pass_row in row["passes"]:
            write(folder/f"{pass_row['scale']}x.json", pass_row)
        with Image.open(sources[row["tile_id"]]) as source:
            panels = [("Original", render_panel(source, row["box"]))]
            panels += [(f"{p['scale']}x detector: {len(p['raw_detector']['dt_polys'])} regions",
                        render_panel(source, row["box"], polygons=p["mapped_source_coordinates"]["tile_center_grid_polygons"])) for p in row["passes"]]
        width, height = panels[0][1].size
        board = Image.new("RGB", ((width+12)*len(panels)+12, height+64), "white")
        pen = ImageDraw.Draw(board)
        pen.text((12, 8), row["region_id"] + " | separate scales; unapproved text regions", fill="black")
        for index, (label, panel) in enumerate(panels):
            x = 12+index*(width+12)
            pen.text((x, 30), label, fill="black")
            board.paste(panel, (x, 52))
        board.save(folder/"comparison.png")
    # Recheck source hashes after diagnostic rendering too.
    if any(digest_file(path) != before[key] for key, path in inputs.items()):
        raise MapTextDetectionError("input changed during rendering; no success report published")
    write(output/"multiscale-text-detection.json", result)
    write(output/"text-regions.geojson", collection)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drawing-report", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model-lock", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--regions", nargs="+", default=DEFAULT_REGIONS)
    parser.add_argument("--scales", nargs="+", type=int, choices=(2, 4), default=DEFAULT_SCALES)
    parser.add_argument("--det-max-side", type=int, default=1024)
    parser.add_argument("--det-thresh", type=float, default=.3)
    parser.add_argument("--det-box-thresh", type=float, default=.6)
    parser.add_argument("--det-unclip-ratio", type=float, default=1.5)
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args()
    config = DetectionConfig(args.det_max_side, args.det_thresh, args.det_box_thresh, args.det_unclip_ratio, args.cpu_threads, True)
    result = run(args.drawing_report, args.manifest, args.model_lock, args.output,
                 region_ids=args.regions, scales=args.scales, config=config)
    print(json.dumps({"output": str(args.output.resolve()), "counts": result["counts"], "status": result["status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
