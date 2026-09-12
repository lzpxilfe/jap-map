"""Crop-only marginal OCR records and evaluation, without GIS side effects.

An OCR reading is evidence, never a coordinate or a control point. Reviews
live in a separate document, and evaluation always scores the original OCR
text against an independently transcribed reference.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata


MANIFEST_SCHEMA = "jap-map-margin-crops/1"
RESULT_SCHEMA = "jap-map-margin-ocr/1"
REVIEW_SCHEMA = "jap-map-margin-review/1"
EVALUATION_SCHEMA = "jap-map-margin-evaluation/1"
KINDS = frozenset(("corner_nw", "corner_ne", "corner_se", "corner_sw", "title", "scale", "legend"))
SCENARIOS = frozenset(("clear", "old_type", "degraded"))
CORPUS_KINDS = frozenset(("historical_scan", "synthetic_smoke"))
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}\Z")


class MarginOcrError(ValueError):
    """A malformed or incompatible OCR pilot artifact."""


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_value(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def identifier(value, name: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise MarginOcrError(f"{name} must be a short ASCII identifier")
    return value


def required_text(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MarginOcrError(f"{name} must be non-empty text")
    return value


def pixel_box(value, name: str = "pixel_box") -> tuple[int, int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4 or any(type(item) is not int for item in value):
        raise MarginOcrError(f"{name} must be integer [left, top, right, bottom]")
    left, top, right, bottom = value
    if left < 0 or top < 0 or right <= left or bottom <= top:
        raise MarginOcrError(f"{name} is empty or outside positive image coordinates")
    return left, top, right, bottom


def validate_manifest(value: dict) -> list[dict]:
    if not isinstance(value, dict) or value.get("schema") != MANIFEST_SCHEMA:
        raise MarginOcrError("unknown margin-crop manifest schema")
    identifier(value.get("pilot_id"), "pilot_id")
    if value.get("corpus_kind") not in CORPUS_KINDS:
        raise MarginOcrError("corpus_kind must distinguish historical_scan from synthetic_smoke")
    sheets = value.get("sheets")
    if not isinstance(sheets, list) or not sheets:
        raise MarginOcrError("a pilot needs at least one explicitly described sheet")
    seen = set()
    for sheet in sheets:
        if not isinstance(sheet, dict):
            raise MarginOcrError("each sheet must be a mapping")
        sid = identifier(sheet.get("sheet_id"), "sheet_id")
        if sid in seen:
            raise MarginOcrError(f"duplicate sheet: {sid}")
        seen.add(sid)
        # The existing pilot reserves Gongju. An explicit excluded split is
        # required even for new corpora, rather than silently assuming train.
        if sheet.get("split") != "development" or sid == "178-gongju":
            raise MarginOcrError(f"margin OCR pilot excludes holdout sheets: {sid}")
        if sheet.get("scenario") not in SCENARIOS:
            raise MarginOcrError(f"{sid}: scenario must be clear, old_type, or degraded")
        for name in ("image_path", "scan_source", "rights"):
            required_text(sheet.get(name), f"{sid}.{name}")
        regions = sheet.get("regions")
        if not isinstance(regions, list) or not regions:
            raise MarginOcrError(f"{sid}: specify margin regions explicitly")
        region_ids = set()
        for region in regions:
            if not isinstance(region, dict):
                raise MarginOcrError("each region must be a mapping")
            rid = identifier(region.get("region_id"), "region_id")
            if rid in region_ids or region.get("kind") not in KINDS:
                raise MarginOcrError(f"{sid}: duplicate region or unsupported kind: {rid}")
            region_ids.add(rid)
            pixel_box(region.get("pixel_box"))
    return sheets


def reading_from_paddle(result, width: int, height: int) -> dict:
    """Normalize one PaddleOCR 3.x image result without losing empty reads."""
    if hasattr(result, "json"):
        result = result.json
    if isinstance(result, str):
        result = json.loads(result)
    if not isinstance(result, dict):
        raise MarginOcrError("PaddleOCR result must be a mapping")
    result = result.get("res", result)
    try:
        texts, scores, polygons = result["rec_texts"], result["rec_scores"], result["rec_polys"]
    except (KeyError, TypeError) as error:
        raise MarginOcrError("PaddleOCR result lacks rec_texts/rec_scores/rec_polys") from error
    if not len(texts) == len(scores) == len(polygons):
        raise MarginOcrError("PaddleOCR text, score, and polygon counts disagree")
    spans = []
    for text, score, polygon in zip(texts, scores, polygons):
        if not isinstance(text, str) or isinstance(score, bool):
            raise MarginOcrError("OCR text must be a string and confidence must be numeric")
        score = float(score)
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise MarginOcrError("OCR confidence must be finite and in [0, 1]")
        points = []
        if len(polygon) < 3:
            raise MarginOcrError("OCR polygon must contain at least three points")
        for point in polygon:
            if len(point) != 2:
                raise MarginOcrError("OCR polygon point must be [x, y]")
            x, y = map(float, point)
            if not math.isfinite(x) or not math.isfinite(y) or not 0 <= x <= width or not 0 <= y <= height:
                raise MarginOcrError("OCR polygon leaves its crop image")
            points.append([x, y])
        spans.append({"text": text, "confidence": score, "polygon": points})
    return {
        "status": "read" if spans else "no_text",
        "raw_text": "\n".join(span["text"] for span in spans),
        "spans": spans,
        # This is an engine score, not a calibrated probability of a correct
        # coordinate. No confidence threshold can create approval.
        "confidence_kind": "paddle_recognition_score_not_calibrated_accuracy",
    }


def validate_result(result: dict) -> list[dict]:
    if not isinstance(result, dict) or result.get("schema") != RESULT_SCHEMA:
        raise MarginOcrError("unknown OCR result schema")
    identifier(result.get("pilot_id"), "pilot_id")
    if result.get("corpus_kind") not in CORPUS_KINDS:
        raise MarginOcrError("OCR result needs an explicit corpus_kind")
    rows = result.get("crops")
    if not isinstance(rows, list) or not rows:
        raise MarginOcrError("OCR result needs crop records")
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise MarginOcrError("each OCR crop must be a mapping")
        sid = identifier(row.get("sheet_id"), "sheet_id")
        rid = identifier(row.get("region_id"), "region_id")
        uid = row.get("crop_uid")
        if uid != f"{sid}/{rid}" or uid in seen:
            raise MarginOcrError("invalid or duplicate crop UID")
        seen.add(uid)
        if row.get("kind") not in KINDS or row.get("scenario") not in SCENARIOS:
            raise MarginOcrError("invalid OCR crop kind or scenario")
        if row.get("status") not in {"read", "no_text", "error"} or not isinstance(row.get("raw_text"), str):
            raise MarginOcrError("invalid OCR status or raw_text")
        if row["status"] != "read" and (row["raw_text"] or row.get("spans")):
            raise MarginOcrError("no_text/error records cannot contain readings")
    return rows


def review_template(result: dict) -> dict:
    rows = validate_result(result)
    return {
        "schema": REVIEW_SCHEMA,
        "result_sha256": digest_value(result),
        "items": [
            {"crop_uid": row["crop_uid"], "review_status": "unreviewed", "reviewer": "", "reviewed_text": None,
             "reference_text": None, "reference_independent": False, "review_seconds": None}
            for row in rows
        ],
    }


def validate_reviews(result: dict, reviews: dict) -> dict[str, dict]:
    rows = validate_result(result)
    if not isinstance(reviews, dict) or reviews.get("schema") != REVIEW_SCHEMA or reviews.get("result_sha256") != digest_value(result):
        raise MarginOcrError("review file belongs to a different OCR result")
    known = {row["crop_uid"]: row for row in rows}
    items = {}
    if not isinstance(reviews.get("items"), list):
        raise MarginOcrError("review items must be a list")
    for item in reviews["items"]:
        if not isinstance(item, dict):
            raise MarginOcrError("each review item must be a mapping")
        uid = item.get("crop_uid")
        if uid not in known or uid in items:
            raise MarginOcrError("unknown or duplicate crop in review file")
        status = item.get("review_status")
        if status not in {"unreviewed", "approved", "corrected", "rejected", "unreadable"}:
            raise MarginOcrError("unknown review status")
        if status != "unreviewed":
            required_text(item.get("reviewer"), "reviewer")
        if status in {"approved", "corrected"}:
            if not isinstance(item.get("reviewed_text"), str):
                raise MarginOcrError("approved/corrected readings require explicit reviewed_text")
            if known[uid]["status"] == "error":
                raise MarginOcrError("an OCR runtime error cannot be approved")
            if status == "approved" and item["reviewed_text"] != known[uid]["raw_text"]:
                raise MarginOcrError("changed OCR text must be marked corrected")
        reference = item.get("reference_text")
        if reference is not None and not isinstance(reference, str):
            raise MarginOcrError("reference_text must be text or null")
        if type(item.get("reference_independent")) is not bool:
            raise MarginOcrError("reference_independent must be explicit boolean")
        if item["reference_independent"]:
            required_text(item.get("reviewer"), "reference reviewer")
            if reference is None:
                raise MarginOcrError("independent reference flag needs reference_text")
        seconds = item.get("review_seconds")
        if seconds is not None and (isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0):
            raise MarginOcrError("review_seconds must be a finite non-negative duration")
        items[uid] = item
    return items


def _normal(text: str) -> str:
    # Preserve punctuation, minus signs, seconds, and historical characters.
    # We only compose Unicode and normalize whitespace for a secondary metric.
    return " ".join(unicodedata.normalize("NFC", text).split())


def _edit_distance(first: str, second: str) -> int:
    previous = list(range(len(second) + 1))
    for i, a in enumerate(first, 1):
        current = [i]
        for j, b in enumerate(second, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def evaluate(result: dict, reviews: dict, *, target_sheets: int = 20) -> dict:
    if type(target_sheets) is not int or target_sheets < 20:
        raise MarginOcrError("the margin OCR pilot needs at least 20 sheets")
    indexed = validate_reviews(result, reviews)
    eligible = []
    unreadable = 0
    for row in result["crops"]:
        item = indexed.get(row["crop_uid"], {})
        if item.get("review_status") == "unreadable":
            unreadable += 1
        elif item.get("reference_independent") and item.get("reference_text") is not None:
            eligible.append((row, item))

    def metrics(pairs):
        if not pairs:
            return {"count": 0, "raw_exact_match": None, "normalized_exact_match": None, "character_error_rate": None}
        errors = reference_chars = raw_exact = normalized_exact = 0
        for row, item in pairs:
            reference, predicted = _normal(item["reference_text"]), _normal(row["raw_text"])
            raw_exact += int(row["status"] != "error" and row["raw_text"] == item["reference_text"])
            normalized_exact += int(row["status"] != "error" and predicted == reference)
            errors += _edit_distance(reference, predicted)
            reference_chars += len(reference)
        return {"count": len(pairs), "raw_exact_match": raw_exact / len(pairs), "normalized_exact_match": normalized_exact / len(pairs),
                "character_error_rate": errors / reference_chars if reference_chars else None,
                "character_errors": errors, "reference_characters": reference_chars}

    evaluated_sheets = sorted({row["sheet_id"] for row, _ in eligible})
    scenario_sheets = {scenario: len({row["sheet_id"] for row, _ in eligible if row["scenario"] == scenario}) for scenario in sorted(SCENARIOS)}
    measured = result["corpus_kind"] == "historical_scan" and len(evaluated_sheets) >= target_sheets and len(eligible) + unreadable == len(result["crops"]) and all(scenario_sheets.values())
    durations = [item["review_seconds"] for item in indexed.values() if item.get("review_seconds") is not None]
    return {
        "schema": EVALUATION_SCHEMA,
        "corpus_kind": result["corpus_kind"],
        "result_sha256": digest_value(result),
        "target_sheets": target_sheets,
        "available_sheet_count": len({row["sheet_id"] for row in result["crops"]}),
        "evaluated_sheets": evaluated_sheets,
        "status": "synthetic_smoke_only" if result["corpus_kind"] == "synthetic_smoke" else ("pilot_measured" if measured else "pilot_incomplete"),
        "scenario_evaluated_sheet_counts": scenario_sheets,
        "total_crops": len(result["crops"]),
        "runtime_errors": sum(row["status"] == "error" for row in result["crops"]),
        "unreadable_reference_count": unreadable,
        "review_status_counts": dict(Counter(item["review_status"] for item in indexed.values())),
        "review_time": {"timed_crops": len(durations), "total_seconds": sum(durations),
                        "mean_seconds": sum(durations) / len(durations) if durations else None},
        "overall": metrics(eligible),
        "by_kind": {kind: metrics([pair for pair in eligible if pair[0]["kind"] == kind]) for kind in sorted(KINDS)},
        "by_scenario": {scenario: metrics([pair for pair in eligible if pair[0]["scenario"] == scenario]) for scenario in sorted(SCENARIOS)},
        "promotion_passed": False,
        "interpretation": "Scores original OCR, not corrected text. Partial/unreadable coverage is explicit. No coordinates, CRS or GCPs are applied.",
    }
