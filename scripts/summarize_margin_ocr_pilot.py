#!/usr/bin/env python3
"""Summarize a real local run; keep provisional AI title probes separate from accuracy."""

from __future__ import annotations

import argparse
from collections import Counter
from importlib import metadata
from pathlib import Path
import platform
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from histcontour_core.margin_ocr import MarginOcrError, digest_file, digest_value, evaluate, validate_result
from histcontour_core.paddle_margin_ocr import verify_models
from margin_ocr_pilot import read_json, write_json


def summarize(run_directory: Path, notes_path: Path, sources_path: Path) -> dict:
    result = read_json(run_directory / "result.json")
    rows = validate_result(result)
    bundle = read_json(run_directory / "bundle.json")
    lock = read_json(run_directory / "model-lock.json")
    verify_models(lock, run_directory)
    notes = read_json(notes_path)
    sources = read_json(sources_path)
    if (result["bundle_sha256"] != digest_value(bundle) or result["model_lock_sha256"] != digest_value(lock)
            or bundle["manifest"]["visual_notes_sha256"] != digest_file(notes_path)
            or bundle["manifest"]["source_catalog_sha256"] != digest_file(sources_path)):
        raise MarginOcrError("summary inputs differ from the frozen run provenance")
    if notes.get("reference_origin") != "ai_visual_provisional" or notes.get("human_approved") is not False:
        raise MarginOcrError("this report only accepts explicitly provisional AI notes")
    indexed = {f"stanford-{item['record_id']}": item for item in notes["sheets"]}
    if len(indexed) != len(notes["sheets"]) or set(indexed) != {row["sheet_id"] for row in rows}:
        raise MarginOcrError("visual notes do not match the run cohort")
    title_comparisons = []
    for row in rows:
        if row["kind"] != "title":
            continue
        note = indexed[row["sheet_id"]]
        reference = note.get("title_visual_ltr")
        excluded = "uncertain_reference" if reference is None else ("calibration_ocr_already_seen" if note.get("ocr_seen_before_reference") else None)
        title_comparisons.append({
            "sheet_id": row["sheet_id"], "raw_text": row["raw_text"],
            "scores": [span["confidence"] for span in row.get("spans", [])],
            "provisional_visual_ltr": reference, "exclusion_reason": excluded,
            "raw_exact": None if excluded else row["status"] != "error" and row["raw_text"] == reference,
            # This extra title-only probe ignores layout whitespace, never DMS
            # punctuation or character order. It is not the formal evaluator.
            "without_whitespace_exact": None if excluded else row["status"] != "error" and "".join(row["raw_text"].split()) == "".join(reference.split()),
        })
    eligible = [row for row in title_comparisons if row["exclusion_reason"] is None]
    durations = [row["inference_seconds"] for row in rows]
    review = read_json(run_directory / "review.template.json")
    formal = evaluate(result, review)
    if (formal["review_status_counts"] != {"unreviewed": len(rows)}
            or any(item.get("reference_text") is not None or item.get("reference_independent") for item in review["items"])):
        raise MarginOcrError("initial-run summary requires an untouched, entirely unreviewed reference template")
    return {
        "schema": "jap-map-margin-run-summary/1", "created_utc": result["created_utc"],
        "pilot_id": result["pilot_id"], "result_sha256": digest_value(result),
        "bundle_sha256": result["bundle_sha256"], "model_lock_sha256": result["model_lock_sha256"],
        "visual_notes_sha256": digest_file(notes_path), "source_catalog_sha256": digest_file(sources_path),
        "status": result["status"], "corpus_kind": result["corpus_kind"],
        "sheet_count": len({row["sheet_id"] for row in rows}), "crop_count": len(rows),
        "crop_status_counts": dict(Counter(row["status"] for row in rows)),
        "crop_kind_counts": dict(Counter(row["kind"] for row in rows)),
        "scenario_sheet_counts": dict(Counter(sheet["scenario"] for sheet in bundle["manifest"]["sheets"])),
        "engine": result["engine"],
        "summary_runtime": {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
                            "packages": dict(sorted((item.metadata["Name"], item.version) for item in metadata.distributions())),
                            "interpretation": "Observed during summary generation in the matching three-package OCR runtime, not an attestation of all earlier transitive files."},
        "inference_time": {"total_crop_seconds": sum(durations), "mean_crop_seconds": statistics.mean(durations),
                           "median_crop_seconds": statistics.median(durations), "max_crop_seconds": max(durations),
                           "excludes": "download, source decoding/cropping, source hashing, human review; single CPU run, not a benchmark distribution"},
        "models": {role: {key: entry[key] for key in ("model_name", "source_url", "source_revision", "weight_license", "files")}
                   for role, entry in lock["models"].items()},
        "original_bytes": sum(source["image_bytes"] for source in sources["sources"]),
        "sources": [{key: source[key] for key in ("sheet_id", "record_id", "regional_group", "index_label", "source_page", "source_url",
                                                    "iiif_manifest_url", "image_size", "image_bytes", "image_sha256", "source_manifest_sha256", "license")}
                    for source in sources["sources"]],
        "provisional_title_probe": {
            "status": "exploratory_not_human_ground_truth", "reference_origin": "ai_visual_provisional_with_catalog_visible",
            "count": len(eligible), "raw_exact_count": sum(row["raw_exact"] for row in eligible),
            "without_whitespace_exact_count": sum(row["without_whitespace_exact"] for row in eligible),
            "comparisons": title_comparisons,
            "interpretation": "Not full-sheet accuracy. The calibration sheet and an uncertain title are excluded before comparison. Historical reading order is not corrected in raw OCR. No human approvals are created.",
        },
        "formal_evaluation": formal, "human_approvals": 0, "gis_applied": False, "promotion_passed": False,
        "crops": [{key: row[key] for key in ("crop_uid", "kind", "scenario", "pixel_box", "crop_size", "crop_sha256", "status", "raw_text", "spans", "inference_seconds")}
                  for row in rows],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--visual-notes", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = summarize(args.run_directory, args.visual_notes, args.sources)
        write_json(args.output, report)
    except (MarginOcrError, OSError, ValueError, KeyError, TypeError) as error:
        print(f"Summary stopped: {error}", file=sys.stderr)
        return 2
    print(f"{report['sheet_count']} actual sheets, {report['crop_count']} crops; formal evaluation {report['formal_evaluation']['status']}; no approvals.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
