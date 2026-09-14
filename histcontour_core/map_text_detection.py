"""Local map-interior text *region evidence*, never glyph or contour truth.

The standalone Paddle detector retains its own polygons and scores. Optional
recognition is a separate pass over those regions, with no filtering by the
recognized string or score. Imports of Paddle remain inside the CPU worker.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import time
import xml.etree.ElementTree as ET

from .margin_ocr import digest_file, digest_value
from .paddle_margin_ocr import local_inference_only, verify_models


SCHEMA = "jap-map-map-text-detection/1"
DEVELOPMENT_SHEETS = frozenset(("173-buyeo", "174-cheongyang", "177-nonsan"))
EVIDENCE_FLAGS = {
    "dataset_role": "review_only_not_training", "human_approved": False,
    "training_eligible": False, "automatic_promotion": False,
    "glyph_pixels_identified": False, "contour_semantics_assigned": False,
    "elevation_assigned": False, "erase_mask_generated": False,
}


class MapTextDetectionError(ValueError):
    pass


@dataclass(frozen=True)
class DetectionConfig:
    limit_side_len: int = 1024
    thresh: float = 0.3
    box_thresh: float = 0.6
    unclip_ratio: float = 1.5
    cpu_threads: int = 1
    recognize: bool = False

    def validate(self):
        if type(self.limit_side_len) is not int or not 128 <= self.limit_side_len <= 4096:
            raise MapTextDetectionError("limit_side_len must be an integer in [128, 4096]")
        if type(self.cpu_threads) is not int or not 1 <= self.cpu_threads <= 8:
            raise MapTextDetectionError("cpu_threads must be an integer in [1, 8]")
        for name in ("thresh", "box_thresh"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value < 1:
                raise MapTextDetectionError(f"{name} must be finite and strictly between zero and one")
        if (type(self.unclip_ratio) not in (int, float) or not math.isfinite(self.unclip_ratio)
                or not 0 < self.unclip_ratio <= 3 or type(self.recognize) is not bool):
            raise MapTextDetectionError("invalid region expansion or recognition setting")
        return self


def validate_report(report):
    """Validate all declarations before opening any raster, including holdout."""
    if (report.get("schema") != "jap-map-assisted-contour-drawing/1"
            or report.get("holdout_used") is not False):
        raise MapTextDetectionError("expected an explicitly development-only drawing report")
    tiles = report.get("tiles")
    if not isinstance(tiles, list) or not tiles:
        raise MapTextDetectionError("drawing report needs development tiles")
    ids, crs_ids = set(), set()
    for tile in tiles:
        tid, sheet = tile.get("tile_id"), tile.get("sheet_id")
        if (sheet not in DEVELOPMENT_SHEETS or tile.get("split") != "development"
                or not isinstance(tid, str) or not tid.startswith(sheet + "-")
                or not re.fullmatch(r"[a-z0-9-]+", tid) or tid in ids):
            raise MapTextDetectionError("only distinct declared development tiles are allowed; no 178-gongju")
        ids.add(tid)
        bounds, pixels = tile.get("bounds"), tile.get("pixel_bounds")
        if (not isinstance(pixels, list) or len(pixels) != 4
                or any(type(v) is not int or v < 0 for v in pixels) or min(pixels[2:]) <= 0):
            raise MapTextDetectionError("pixel_bounds must give source-sheet x, y, width, height")
        if (not isinstance(bounds, list) or len(bounds) != 4
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in bounds)
                or bounds[0] >= bounds[2] or bounds[1] >= bounds[3]):
            raise MapTextDetectionError("tile needs finite increasing native bounds")
        crs = tile.get("crs_authid")
        if not isinstance(crs, str) or not re.fullmatch(r"EPSG:[1-9][0-9]*", crs):
            raise MapTextDetectionError("tile needs an explicit native EPSG CRS")
        crs_ids.add(crs)
        if not re.fullmatch(r"[0-9a-f]{64}", str(tile.get("source_raster_sha256", ""))):
            raise MapTextDetectionError("source raster SHA-256 is required")
    if len(crs_ids) != 1:
        raise MapTextDetectionError("one native CRS is required for the evidence collection")
    return copy.deepcopy(tiles)


def inspect_raster(path, tile):
    """Read actual dimensions, GeoTIFF keys and transform; never infer from labels.

    This bounded adapter supports the north-up, PixelIsArea GeoTIFFs used by
    this development packet. Unsupported georeferencing is rejected explicitly.
    """
    from PIL import Image

    path = Path(path)
    if digest_file(path) != tile["source_raster_sha256"]:
        raise MapTextDetectionError(f"source raster hash changed: {tile['tile_id']}")
    with Image.open(path) as image:
        if image.format != "TIFF" or getattr(image, "n_frames", 1) != 1:
            raise MapTextDetectionError("source must be a single-frame GeoTIFF")
        width, height = image.size
        if [width, height] != tile["pixel_bounds"][2:]:
            raise MapTextDetectionError("actual source-pixel dimensions differ from report")
        tags = image.tag_v2
        keys = tags.get(34735, ())
        if len(keys) < 4 or len(keys) != 4 + 4 * keys[3]:
            raise MapTextDetectionError("GeoTIFF needs a complete GeoKeyDirectory")
        geo = {keys[i]: keys[i+3] for i in range(4, len(keys), 4)
               if keys[i+1] == 0 and keys[i+2] == 1}
        epsg = geo.get(3072) if geo.get(1024) == 1 else geo.get(2048)
        if geo.get(1024) not in (1, 2) or geo.get(1025) != 1 or f"EPSG:{epsg}" != tile["crs_authid"]:
            raise MapTextDetectionError("actual raster CRS or PixelIsArea convention differs from report")
        scale, tie = tags.get(33550, ()), tags.get(33922, ())
        if tags.get(34264) is not None or len(scale) != 3 or len(tie) != 6:
            raise MapTextDetectionError("only north-up scale/tiepoint GeoTIFF transforms are supported")
        sx, sy = float(scale[0]), float(scale[1])
        if not all(math.isfinite(v) for v in (*scale, *tie)) or min(sx, sy) <= 0:
            raise MapTextDetectionError("raster transform must be finite with positive pixel scale")
        x0, y0 = tie[3] - tie[0] * sx, tie[4] + tie[1] * sy
        actual_bounds = [x0, y0-height*sy, x0+width*sx, y0]
        if any(abs(a-b) > max(sx, sy)*1e-4 for a, b in zip(actual_bounds, tile["bounds"])):
            raise MapTextDetectionError("actual GeoTIFF bounds differ from report")
        metadata = tags.get(42112)
        if metadata:
            try:
                fields = {item.attrib.get("name"): item.text for item in ET.fromstring(metadata)}
            except ET.ParseError as error:
                raise MapTextDetectionError("malformed GeoTIFF development metadata") from error
            for field in ("tile_id", "sheet_id", "split"):
                if field in fields and fields[field] != tile[field]:
                    raise MapTextDetectionError("GeoTIFF identity/split metadata differs from declaration")
        return {"source_width_px": width, "source_height_px": height,
                "source_mode": image.mode, "source_format": image.format,
                "actual_crs_authid": f"EPSG:{epsg}", "actual_bounds": actual_bounds,
                "geotransform": [x0, sx, 0.0, y0, 0.0, -sy],
                "coordinate_convention": "detector image coordinates; top-left pixel corner origin; x right, y down; affine without half-pixel shift"}


def _raw_mapping(result):
    if isinstance(result, Mapping):
        value = result
    else:
        value = result.json
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, Mapping) and "res" in value:
        value = value["res"]
    if not isinstance(value, Mapping):
        raise MapTextDetectionError("inference result must be a mapping")
    return value


def _plain(value):
    return value.tolist() if hasattr(value, "tolist") else value


def _score(value):
    value = _plain(value)
    if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
        raise MapTextDetectionError("model confidence must be a finite number in [0, 1]")
    return float(value)


def polygons_on_pixel_center_grid(polygons):
    """Translate region boundary coordinates to the repository centre grid.

    Raw detector coordinates use the image boundary origin (0..width/height).
    Existing stroke arrays and map_to_pixel put the first pixel CENTRE at 0.
    Subtracting 0.5 expresses the same region on that grid; it does not modify
    the raw model output or identify any glyph pixels inside the region.
    """
    return [[[float(x)-0.5, float(y)-0.5] for x, y in polygon] for polygon in polygons]


def raw_detection(result, width, height):
    """Preserve postprocessor order/geometry and detection scores independently."""
    raw = _raw_mapping(result)
    if "dt_polys" not in raw or "dt_scores" not in raw:
        raise MapTextDetectionError("standalone detector must expose both dt_polys and dt_scores")
    polygons, scores = _plain(raw["dt_polys"]), _plain(raw["dt_scores"])
    if not isinstance(polygons, (list, tuple)) or not isinstance(scores, (list, tuple)) or len(polygons) != len(scores):
        raise MapTextDetectionError("raw polygon and detector-score counts differ")
    converted = []
    for polygon in polygons:
        polygon = _plain(polygon)
        if not isinstance(polygon, (list, tuple)) or len(polygon) != 4:
            raise MapTextDetectionError("pinned quad detector must return four polygon vertices")
        points = []
        for point in polygon:
            point = _plain(point)
            if (not isinstance(point, (list, tuple)) or len(point) != 2
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in point)
                    or not 0 <= point[0] <= width or not 0 <= point[1] <= height):
                raise MapTextDetectionError("detector vertex is malformed or outside source dimensions")
            points.append(list(point))
        area = sum(a[0]*b[1]-b[0]*a[1] for a, b in zip(points, points[1:]+points[:1]))
        if abs(area) <= 1e-8:
            raise MapTextDetectionError("detector polygon is degenerate")
        converted.append(points)
    return {"dt_polys": converted, "dt_scores": [_score(score) for score in scores],
            "polygons_on_source_pixel_center_grid": polygons_on_pixel_center_grid(converted),
            "raw_coordinate_convention": "image boundary origin (top-left corner), x right/y down; native affine applied without a half-pixel shift",
            "center_grid_conversion": "subtract 0.5 from each raw x/y; compatible with repository map_to_pixel and integer-centre stroke arrays",
            "status": "detected" if converted else "no_regions_detected",
            "output_stage": "raw_detector_postprocessor_output_before_recognition",
            "score_meaning": "DB detector region score; not calibrated text, glyph, contour, or elevation probability"}


def raw_recognition(results, region_ids):
    if len(results) != len(region_ids):
        raise MapTextDetectionError("recognition results do not match ordered detector crops")
    rows = []
    for result, region_id in zip(results, region_ids):
        raw = _raw_mapping(result)
        if not isinstance(raw.get("rec_text"), str) or "rec_score" not in raw:
            raise MapTextDetectionError("recognizer must expose raw rec_text and rec_score")
        rows.append({"region_id": region_id, "rec_text": raw["rec_text"], "rec_score": _score(raw["rec_score"]),
                     **EVIDENCE_FLAGS})
    return {"status": "completed", "readings": rows,
            "filtering": "none; empty and low-confidence strings preserved",
            "crop_method": "pinned PaddleX CropByPolys quad perspective rectification; tall crops rotate 90 degrees; no learned orientation classifier"}


def evidence_features(tile, metadata, detection):
    transform = metadata["geotransform"]
    features = []
    for index, (polygon, score) in enumerate(zip(detection["dt_polys"], detection["dt_scores"])):
        coordinates = [[transform[0]+x*transform[1]+y*transform[2],
                        transform[3]+x*transform[4]+y*transform[5]] for x, y in polygon]
        coordinates.append(coordinates[0][:])
        features.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [coordinates]},
                         "properties": {"region_id": f"{tile['tile_id']}:text-region:{index:04d}",
                                        "tile_id": tile["tile_id"], "sheet_id": tile["sheet_id"],
                                        "split": "development", "source_raster_sha256": tile["source_raster_sha256"],
                                        "raw_detector_index": index, "detection_score": score,
                                        "geometry_role": "text_search_region_not_glyph_mask", **EVIDENCE_FLAGS}})
    return features


def create_engines(lock, base, config):
    config.validate()
    directories = verify_models(lock, base)
    from paddleocr import TextDetection, TextRecognition
    options = {"device": "cpu", "engine": "paddle_static", "enable_hpi": False,
               "enable_mkldnn": False, "cpu_threads": config.cpu_threads}
    detector = TextDetection(model_name=lock["models"]["detection"]["model_name"],
                             model_dir=directories["detection"], limit_type="max",
                             limit_side_len=config.limit_side_len, thresh=config.thresh,
                             box_thresh=config.box_thresh, unclip_ratio=config.unclip_ratio, **options)
    recognizer, cropper = None, None
    if config.recognize:
        from paddlex.inference.pipelines.components import CropByPolys
        recognizer = TextRecognition(model_name=lock["models"]["recognition"]["model_name"],
                                     model_dir=directories["recognition"], **options)
        cropper = CropByPolys(det_box_type="quad")
    return detector, recognizer, cropper


def run(report_path, lock_path, output, config=DetectionConfig()):
    """Create a new evidence directory; perform no mutation of source or labels."""
    import numpy as np
    from PIL import Image

    config.validate()
    report_path, lock_path, output = (Path(p).resolve() for p in (report_path, lock_path, output))
    if output.exists():
        raise FileExistsError("text-detection output directory must be new")
    if output.is_relative_to(report_path.parent):
        raise MapTextDetectionError("output must be outside the source report snapshot")
    started = time.perf_counter()
    paths = {"drawing_report": report_path, "model_lock": lock_path}
    initial_hashes = {name: digest_file(path) for name, path in paths.items()}
    report, lock = (json.loads(path.read_text(encoding="utf-8")) for path in (report_path, lock_path))
    tiles = validate_report(report)
    sources = []
    for tile in tiles:
        relative = Path(tile.get("raster_path", ""))
        source = (report_path.parent / relative).resolve()
        if (relative.is_absolute() or ".." in relative.parts or not relative.parts
                or not source.is_relative_to(report_path.parent)
                or (report_path.parent / relative).is_symlink()):
            raise MapTextDetectionError("raster path must stay inside the source snapshot")
        paths[tile["tile_id"]] = source
        initial_hashes[tile["tile_id"]] = tile["source_raster_sha256"]
        sources.append((tile, source, inspect_raster(source, tile)))
    directories = verify_models(lock, lock_path.parent)
    if any(output.is_relative_to(Path(directory)) for directory in directories.values()):
        raise MapTextDetectionError("output must be outside pinned model directories")
    output.mkdir(parents=True, exist_ok=False)
    rows, features = [], []
    with local_inference_only(cache_directory=output/"runtime-cache"):
        detector, recognizer, cropper = create_engines(lock, lock_path.parent, config)
        try:
            for tile, source, metadata in sources:
                tick = time.perf_counter()
                with Image.open(source) as source_image:
                    # PaddleX ReadImage receives ndarray in BGR order. This
                    # conversion changes only the in-memory inference input.
                    pixels = np.asarray(source_image.convert("RGB"))[:, :, ::-1].copy()
                original_pixels_hash = digest_value({"shape": list(pixels.shape),
                                                     "bytes_sha256": hashlib.sha256(pixels.tobytes()).hexdigest()})
                results = list(detector.predict(pixels))
                if len(results) != 1:
                    raise MapTextDetectionError("one source tile must produce exactly one detector result")
                detection = raw_detection(results[0], metadata["source_width_px"], metadata["source_height_px"])
                tile_features = evidence_features(tile, metadata, detection)
                recognition = {"status": "not_run", "readings": []}
                if recognizer is not None:
                    crops = cropper(pixels, detection["dt_polys"]) if detection["dt_polys"] else []
                    recognition = raw_recognition(list(recognizer.predict(crops)) if crops else [],
                                                  [f["properties"]["region_id"] for f in tile_features])
                rows.append({**tile, **metadata, "source_absolute_path": str(source),
                             "input_bgr_pixel_payload_sha256": original_pixels_hash,
                             "detection": detection, "recognition": recognition,
                             "wall_seconds": time.perf_counter()-tick, **EVIDENCE_FLAGS})
                features.extend(tile_features)
                print(f"{tile['tile_id']}: {len(tile_features)} raw detector regions", flush=True)
        finally:
            detector.close()
            if recognizer is not None:
                recognizer.close()
        verify_models(lock, lock_path.parent)
    if any(digest_file(path) != initial_hashes[name] for name, path in paths.items()):
        raise MapTextDetectionError("input changed during inference; no successful evidence was published")
    implementation = {str(Path(__file__).resolve()): digest_file(Path(__file__))}
    cli = Path(__file__).resolve().parents[1]/"scripts"/"detect_map_text_regions.py"
    implementation[str(cli)] = digest_file(cli)
    provenance = {"input_files": {name: {"path": str(path), "sha256": initial_hashes[name]} for name, path in paths.items()},
                  "model_lock": lock, "model_lock_sha256": initial_hashes["model_lock"],
                  "model_and_runtime_verified_before_and_after": True,
                  "implementation_sha256": implementation, "configuration": asdict(config),
                  "configuration_sha256": digest_value(asdict(config)),
                  "device": "cpu", "inference_engine": "paddle_static", "network_connections_blocked": True,
                  "inference_api": "paddleocr.TextDetection; optional separate paddleocr.TextRecognition",
                  "source_files_unchanged": True, "source_raster_resized_on_disk": False,
                  "learned_orientation_classifier_used": False,
                  "rotation_handling": "detector quadrilaterals preserved; optional recognition rectifies each quad",
                  "raw_model_probability_maps_retained": False}
    result = {"schema": SCHEMA, "status": "unapproved_detection_evidence", "holdout_used": False,
              "model_fitted": False, "human_approvals": 0, "native_crs": tiles[0]["crs_authid"],
              "tiles": rows, "counts": {"tiles": len(rows), "detection_regions": len(features),
                                        "recognition_readings": sum(len(row["recognition"]["readings"]) for row in rows)},
              "provenance": provenance, "wall_seconds": time.perf_counter()-started, **EVIDENCE_FLAGS,
              "limitations": ["Text regions are search evidence; overlapping contour ink remains unresolved.",
                              "Detector scores are not semantic ground truth or calibrated probabilities.",
                              "Detector postprocessing is thresholded; no claim of complete text recall.",
                              "No OCR string is interpreted as elevation or used to erase source pixels.",
                              "Recognition of historical numerals and characters needs independent human validation."]}
    collection = {"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": result["native_crs"]}},
                  "coordinate_contract": "native CRS GeoJSON extension; not RFC 7946 longitude/latitude",
                  "features": features, **EVIDENCE_FLAGS}
    for name, value in (("text-detection.json", result), ("text-regions.geojson", collection)):
        with (output/name).open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    return result
