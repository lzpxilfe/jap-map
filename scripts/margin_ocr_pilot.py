#!/usr/bin/env python3
"""Standalone, optional margin OCR pilot. Never imports QGIS or edits GIS data."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import shutil
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from histcontour_core.margin_ocr import (
    MarginOcrError, RESULT_SCHEMA, digest_file, digest_value, evaluate,
    pixel_box, reading_from_paddle, review_template, validate_manifest,
)
from histcontour_core.paddle_margin_ocr import (
    create_engine, detection_options, local_inference_only, pin_local_models, verify_models,
)

BUNDLE_SCHEMA = "jap-map-margin-bundle/1"
MAX_CROP_PIXELS = 4_000_000
MAX_CROP_FRACTION = 0.25
MAX_SOURCE_PIXELS = 250_000_000
_PIL_LIMIT_LOCK = threading.RLock()


@contextmanager
def open_scan(path: Path):
    """Allow documented large archive scans, with an explicit finite cap.

    This is a standalone worker helper. Restore Pillow's default immediately
    after opening metadata, and reject oversize rasters before pixel decoding.
    """
    from PIL import Image

    with _PIL_LIMIT_LOCK:
        previous = Image.MAX_IMAGE_PIXELS
        try:
            Image.MAX_IMAGE_PIXELS = MAX_SOURCE_PIXELS
            try:
                scan = Image.open(path)
            except Image.DecompressionBombError as error:
                raise MarginOcrError("source scan exceeds the bounded large-image worker limit") from error
        finally:
            Image.MAX_IMAGE_PIXELS = previous
    with scan:
        if scan.width * scan.height > MAX_SOURCE_PIXELS:
            raise MarginOcrError("source scan exceeds the 250-million-pixel worker limit")
        yield scan


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise MarginOcrError(f"JSON document must be an object: {path}")
    return value


def write_json(path: Path, value: dict) -> None:
    # Never replace an existing OCR result, source-lock, or human review.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _new_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=False)


def _bounds(box, width: int, height: int) -> tuple[int, int, int, int]:
    left, top, right, bottom = pixel_box(box)
    area = (right - left) * (bottom - top)
    if right > width or bottom > height:
        raise MarginOcrError("crop is outside the source image")
    if area > MAX_CROP_PIXELS or area > width * height * MAX_CROP_FRACTION:
        raise MarginOcrError("crop exceeds 4 million pixels or 25% of the scan; whole-map OCR is forbidden")
    return left, top, right, bottom


def _safe_crop_path(root: Path, row: dict) -> Path:
    expected = f"crops/{row['sheet_id']}/{row['region_id']}.png"
    if row.get("crop_path") != expected:
        raise MarginOcrError("crop path differs from its manifest identity")
    candidate = root / expected
    if not candidate.resolve().is_relative_to(root.resolve()) or candidate.is_symlink():
        raise MarginOcrError("crop must remain inside its local bundle")
    return candidate


def render_report(rows: list[dict], title: str) -> str:
    escape = lambda value: html.escape(str(value), quote=True)
    sections = []
    for row in rows:
        # Paths come from validated identifiers, not OCR output or remote URLs.
        expected = f"crops/{row['sheet_id']}/{row['region_id']}.png"
        spans = "".join(f"<li><code>{escape(span['text'])}</code> — engine score {escape(span['confidence'])}</li>" for span in row.get("spans", []))
        details = "" if "raw_text" not in row else (
            f"<h3>OCR 원문 — {escape(row['status'])}</h3><pre>{escape(row['raw_text'])}</pre><ul>{spans}</ul>"
            f"<p>{escape(row.get('error', ''))}</p>"
        )
        sections.append(
            f"<section><h2>{escape(row['crop_uid'])} · {escape(row['kind'])} · {escape(row['scenario'])}</h2>"
            f"<p>원본 픽셀 영역: {escape(row['pixel_box'])} · SHA256: {escape(row['crop_sha256'])}</p>"
            f'<img src="{escape(expected)}" alt="{escape(row["crop_uid"])} original crop">{details}</section>'
        )
    return (
        '<!doctype html><html lang="ko"><meta charset="utf-8">'
        '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; img-src \'self\'; style-src \'unsafe-inline\'; base-uri \'none\'">'
        f"<title>{escape(title)}</title><style>body{{font:16px system-ui;max-width:1100px;margin:2rem auto;padding:1rem}}"
        "section{border-top:1px solid #aaa;padding:1rem 0}img{max-width:100%;height:auto}pre{white-space:pre-wrap}p{overflow-wrap:anywhere}</style>"
        f"<h1>{escape(title)}</h1><p>읽기 후보만 제공합니다. 엔진 신뢰도는 정답 확률이 아닙니다. 모든 승인은 별도 검수 기록이 필요하며 좌표·CRS·GCP는 적용하지 않습니다.</p>"
        + "".join(sections) + "</html>"
    )


def _write_report(path: Path, rows: list[dict], title: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(render_report(rows, title))


def prepare(manifest_path: Path, output: Path) -> dict:
    from PIL import Image

    manifest = read_json(manifest_path)
    sheets = validate_manifest(manifest)
    # Validate every image and box before creating any output.
    sources = {}
    source_hashes = set()
    for sheet in sheets:
        source = Path(sheet["image_path"])
        source = source if source.is_absolute() else manifest_path.parent / source
        source = source.resolve(strict=True)
        source_hash = digest_file(source)
        if source_hash in source_hashes:
            raise MarginOcrError("the same source scan cannot count as multiple pilot sheets")
        source_hashes.add(source_hash)
        with open_scan(source) as scan:
            for region in sheet["regions"]:
                _bounds(region["pixel_box"], *scan.size)
            sources[sheet["sheet_id"]] = {
                "image_path": str(source), "image_sha256": source_hash,
                "image_size": list(scan.size), "scan_source": sheet["scan_source"], "rights": sheet["rights"],
            }
    _new_directory(output)
    rows = []
    for sheet in sheets:
        sid = sheet["sheet_id"]
        source = sources[sid]
        with open_scan(Path(source["image_path"])) as scan:
            for region in sheet["regions"]:
                rid = region["region_id"]
                relative = f"crops/{sid}/{rid}.png"
                destination = output / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                # Pixel coordinates refer to stored raster order. Do not EXIF
                # rotate, georeference, downsample, or enhance the source.
                with scan.crop(tuple(region["pixel_box"])) as cropped:
                    with cropped.convert("RGB") as rgb:
                        with destination.open("xb") as handle:
                            rgb.save(handle, format="PNG")
                        size = list(rgb.size)
                rows.append({
                    "crop_uid": f"{sid}/{rid}", "sheet_id": sid, "region_id": rid,
                    "kind": region["kind"], "scenario": sheet["scenario"], "pixel_box": region["pixel_box"],
                    "crop_path": relative, "crop_size": size, "crop_sha256": digest_file(destination),
                    "source_sha256": source["image_sha256"],
                })
        if digest_file(Path(source["image_path"])) != source["image_sha256"]:
            raise MarginOcrError(f"source changed while preparing: {sid}; discard this incomplete output")
    bundle = {"schema": BUNDLE_SCHEMA, "pilot_id": manifest["pilot_id"], "corpus_kind": manifest["corpus_kind"], "manifest": manifest,
              "sources": sources, "crops": rows, "pixel_policy": "stored_raster_no_resizing_rgb_png"}
    _write_report(output / "crops.html", rows, "여백 crop — OCR를 보기 전 독립 전사")
    # This completion marker is written last. An interrupted directory without
    # bundle.json is incomplete and cannot be passed to run.
    write_json(output / "bundle.json", bundle)
    return bundle


def verify_bundle(bundle: dict, base: Path) -> list[dict]:
    from PIL import Image

    if bundle.get("schema") != BUNDLE_SCHEMA:
        raise MarginOcrError("unknown crop bundle schema")
    sheets = validate_manifest(bundle.get("manifest"))
    if bundle.get("pilot_id") != bundle["manifest"]["pilot_id"]:
        raise MarginOcrError("bundle pilot ID differs from its manifest")
    if bundle.get("corpus_kind") != bundle["manifest"]["corpus_kind"]:
        raise MarginOcrError("bundle corpus kind differs from its manifest")
    expected = {f"{sheet['sheet_id']}/{region['region_id']}": (sheet, region) for sheet in sheets for region in sheet["regions"]}
    rows = bundle.get("crops")
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise MarginOcrError("bundle crop count differs from its manifest")
    for sheet in sheets:
        source = bundle["sources"][sheet["sheet_id"]]
        source_path = Path(source["image_path"])
        if digest_file(source_path) != source["image_sha256"]:
            raise MarginOcrError(f"source scan changed: {sheet['sheet_id']}")
        with open_scan(source_path) as scan:
            if list(scan.size) != source["image_size"]:
                raise MarginOcrError("source size differs from its record")
    if len({bundle["sources"][sheet["sheet_id"]]["image_sha256"] for sheet in sheets}) != len(sheets):
        raise MarginOcrError("duplicate source scans in crop bundle")
    seen = set()
    for row in rows:
        uid = row.get("crop_uid")
        if uid not in expected or uid in seen:
            raise MarginOcrError("unknown or duplicate bundle crop")
        seen.add(uid)
        sheet, region = expected[uid]
        for field, value in {"sheet_id": sheet["sheet_id"], "region_id": region["region_id"], "kind": region["kind"],
                             "scenario": sheet["scenario"], "pixel_box": region["pixel_box"]}.items():
            if row.get(field) != value:
                raise MarginOcrError(f"crop metadata differs from manifest: {uid}.{field}")
        source = bundle["sources"][sheet["sheet_id"]]
        box = _bounds(row["pixel_box"], *source["image_size"])
        if row.get("source_sha256") != source["image_sha256"]:
            raise MarginOcrError("crop is bound to a different source")
        crop = _safe_crop_path(base, row)
        if digest_file(crop) != row["crop_sha256"]:
            raise MarginOcrError(f"crop changed: {uid}")
        with Image.open(crop) as scan:
            if list(scan.size) != row["crop_size"] or scan.size != (box[2] - box[0], box[3] - box[1]) or scan.format != "PNG":
                raise MarginOcrError("crop image size or format differs from manifest")
    return rows


def run(bundle_path: Path, lock_path: Path, output: Path, *, det_max_side: int | None = None) -> dict:
    detector_options = detection_options(det_max_side)
    bundle, lock = read_json(bundle_path), read_json(lock_path)
    rows = verify_bundle(bundle, bundle_path.parent)
    verify_models(lock, lock_path.parent)
    _new_directory(output)
    for row in rows:
        destination = output / row["crop_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        with _safe_crop_path(bundle_path.parent, row).open("rb") as source, destination.open("xb") as target:
            shutil.copyfileobj(source, target)
        if digest_file(destination) != row["crop_sha256"]:
            raise MarginOcrError("crop changed while copying; discard this incomplete output")
    readings = []
    with local_inference_only(cache_directory=output / ".paddlex-cache"):
        engine_started = time.monotonic()
        engine = create_engine(lock, lock_path.parent, det_max_side=det_max_side)
        initialization_seconds = time.monotonic() - engine_started
        try:
            for row in rows:
                started = time.monotonic()
                try:
                    # Only the frozen, preselected PNG is ever passed to OCR.
                    raw_results = list(engine.predict(input=str(output / row["crop_path"])))
                    if len(raw_results) != 1:
                        raise MarginOcrError("expected one OCR image result per crop")
                    reading = reading_from_paddle(raw_results[0], *row["crop_size"])
                except Exception as error:
                    reading = {"status": "error", "raw_text": "", "spans": [], "error": f"{type(error).__name__}: {error}"}
                readings.append({**row, **reading, "inference_seconds": time.monotonic() - started})
        finally:
            close = getattr(engine, "close", None)
            if callable(close):
                close()
    errors = sum(row["status"] == "error" for row in readings)
    result = {"schema": RESULT_SCHEMA, "pilot_id": bundle["pilot_id"], "created_utc": datetime.now(timezone.utc).isoformat(),
              "corpus_kind": bundle["corpus_kind"],
              "status": "completed_with_errors" if errors else "completed", "runtime_errors": errors,
              "bundle_sha256": digest_value(bundle), "model_lock_sha256": digest_value(lock),
              "engine": {"name": "PaddleOCR", "packages": lock["packages"], "device": "cpu", "score_threshold": 0.0,
                         "detector_overrides": detector_options,
                         "initialization_seconds": initialization_seconds},
              "sources": bundle["sources"], "crops": readings, "gis_applied": False}
    write_json(output / "model-lock.json", lock)
    write_json(output / "bundle.json", bundle)
    write_json(output / "review.template.json", review_template(result))
    _write_report(output / "report.html", readings, "여백 OCR 판독 후보 — 수동 검수 필요")
    write_json(output / "result.json", result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_cmd = commands.add_parser("prepare", help="extract only researcher-specified margin crops")
    prepare_cmd.add_argument("manifest", type=Path)
    prepare_cmd.add_argument("--output", type=Path, required=True)
    pin_cmd = commands.add_parser("pin-models", help="fingerprint existing local models with supplied provenance; never download")
    pin_cmd.add_argument("spec", type=Path)
    pin_cmd.add_argument("--output", type=Path, required=True)
    check_cmd = commands.add_parser("check-models")
    check_cmd.add_argument("lock", type=Path)
    run_cmd = commands.add_parser("run")
    run_cmd.add_argument("bundle", type=Path)
    run_cmd.add_argument("--model-lock", type=Path, required=True)
    run_cmd.add_argument("--output", type=Path, required=True)
    run_cmd.add_argument("--det-max-side", type=int, help="optional detector max-side limit, 128..4096; does not modify stored crops")
    evaluate_cmd = commands.add_parser("evaluate", help="score original OCR against independently transcribed references")
    evaluate_cmd.add_argument("result", type=Path)
    evaluate_cmd.add_argument("--reviews", type=Path, required=True)
    evaluate_cmd.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            bundle = prepare(args.manifest, args.output)
            print(f"Prepared {len(bundle['crops'])} crops; no OCR or GIS operation performed.")
        elif args.command == "pin-models":
            write_json(args.output, pin_local_models(read_json(args.spec), args.spec.parent))
            print("Recorded supplied provenance and local hashes. This is not independent proof of origin or rights.")
        elif args.command == "check-models":
            verify_models(read_json(args.lock), args.lock.parent)
            print("Local model inventory and three pinned package versions match.")
        elif args.command == "run":
            result = run(args.bundle, args.model_lock, args.output, det_max_side=args.det_max_side)
            print(f"{result['status']}: {len(result['crops'])} crops, {result['runtime_errors']} errors; all unreviewed.")
            return 2 if result["runtime_errors"] else 0
        else:
            result = evaluate(read_json(args.result), read_json(args.reviews))
            write_json(args.output, result)
            print(f"{result['status']}; {len(result['evaluated_sheets'])}/20 sheets evaluated; promotion_passed=false.")
        return 0
    except (MarginOcrError, OSError, ImportError, KeyError, TypeError, ValueError, RuntimeError) as error:
        print(f"Margin OCR stopped: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
