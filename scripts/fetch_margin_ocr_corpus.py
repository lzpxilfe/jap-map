#!/usr/bin/env python3
"""Explicitly acquire a bounded public-domain Stanford Korean map pilot.

This network-enabled acquisition command is separate from the offline OCR
worker. It never runs OCR, applies GIS data, or creates human approvals.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import sys
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from histcontour_core.margin_ocr import MarginOcrError, digest_file, digest_value
from margin_ocr_pilot import open_scan, read_json, write_json

INDEX_URL = "https://stacks.stanford.edu/object/ng525ny5879"
INDEX_PAGE = "https://earthworks.stanford.edu/catalog/stanford-ng525ny5879"
# One source per regional grid. D11 (the Gongju regional group) is excluded
# conservatively, not just the particular Gongju scan in the existing pilot.
GROUPS = ("A1", "B2", "B3", "C4", "D4", "E5", "F7", "B9", "B10", "B12",
          "C9", "C10", "C12", "D9", "D10", "D12", "D13", "E9", "E13", "E15")
PUBLIC_DOMAIN = "https://creativecommons.org/publicdomain/mark/1.0/"
RECORD_ID = re.compile(r"[a-z]{2}[0-9]{3}[a-z]{2}[0-9]{4}\Z")


def select_sources(index: dict) -> list[dict]:
    features = index.get("features", [])
    selected = []
    for position, group in enumerate(GROUPS):
        candidates = [feature for feature in features
                      if feature["properties"].get("available") is True
                      and re.search(r"SHEET " + group + r"-", feature["properties"].get("call_num", ""))]
        # Alternate preference for index variants carrying the R suffix. This
        # is a sampling rule, not an assertion about scan quality or date.
        candidates.sort(key=lambda feature: (
            bool(re.search(r"[0-9]R(?:$|[. /])", feature["properties"]["label"])) != bool(position % 2),
            feature["properties"]["recid"],
        ))
        if not candidates:
            raise MarginOcrError(f"source index has no available map for {group}")
        chosen = candidates[0]
        if group == "E9":
            chosen = next((item for item in candidates if item["properties"]["recid"] == "pr492sf6025"), chosen)
        entry = chosen["properties"]
        record_id = entry["recid"]
        if not RECORD_ID.fullmatch(record_id):
            raise MarginOcrError("unexpected Stanford record ID")
        if entry.get("iiifUrl") != f"https://purl.stanford.edu/{record_id}/iiif/manifest":
            raise MarginOcrError("index manifest URL differs from the record's public Stanford endpoint")
        selected.append({
            "sheet_id": f"stanford-{record_id}", "record_id": record_id, "regional_group": group,
            "index_label": entry["label"], "call_number": entry["call_num"],
            "source_page": entry["websiteUrl"], "iiif_manifest_url": entry["iiifUrl"],
            "index_geometry": chosen["geometry"],
        })
    if len({item["record_id"] for item in selected}) != len(GROUPS):
        raise MarginOcrError("selection contains duplicate source records")
    return selected


def _archive_url(url: str) -> None:
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.hostname not in {"purl.stanford.edu", "stacks.stanford.edu"}
            or parsed.username is not None or parsed.password is not None or parsed.port not in (None, 443)):
        raise MarginOcrError("acquisition only accepts the selected public Stanford hosts")


class _ArchiveRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, newurl):
        # Check BEFORE urllib follows a redirect, not after an outside request.
        _archive_url(newurl)
        return super().redirect_request(request, response, code, message, headers, newurl)


_OPENER = build_opener(_ArchiveRedirectHandler())


def _request(url: str, *, maximum: int):
    _archive_url(url)
    response = _OPENER.open(Request(url, headers={"User-Agent": "jap-map-margin-pilot/1"}), timeout=60)
    if response.status != 200:
        response.close()
        raise MarginOcrError("source did not return a complete HTTP 200 response")
    try:
        _archive_url(response.url)
        length = response.headers.get("Content-Length")
        if length is not None and not 0 < int(length) <= maximum:
            raise MarginOcrError("archive response exceeds the download limit or is empty")
    except (MarginOcrError, ValueError):
        response.close()
        raise
    return response


def fetch_bytes(url: str, maximum: int) -> bytes:
    with _request(url, maximum=maximum) as response:
        body = response.read(maximum + 1)
        length = response.headers.get("Content-Length")
    if not body or len(body) > maximum:
        raise MarginOcrError("empty or oversized archive response")
    if length is not None and len(body) != int(length):
        raise MarginOcrError("archive response was truncated")
    return body


def image_source(manifest: dict, record_id: str) -> tuple[dict, str, str]:
    if manifest.get("license") != PUBLIC_DOMAIN:
        raise MarginOcrError("this pilot requires an explicit Public Domain Mark on each source manifest")
    canvases = manifest["sequences"][0]["canvases"]
    if len(canvases) != 1:
        raise MarginOcrError("select a single-sheet source, not a multi-image volume")
    canvas = canvases[0]
    originals = [item["@id"] for item in canvas.get("rendering", []) if item.get("format") == "image/jp2"]
    if len(originals) != 1 or not originals[0].startswith(f"https://stacks.stanford.edu/file/{record_id}/"):
        raise MarginOcrError("source manifest does not expose one matching original JP2")
    service = canvas["images"][0]["resource"]["service"]["@id"]
    if not service.startswith("https://stacks.stanford.edu/image/iiif/"):
        raise MarginOcrError("unexpected preview service")
    return canvas, originals[0], service


def acquire_source(source: dict, output: Path, reuse: Path | None) -> dict:
    record_id = source["record_id"]
    manifest_bytes = fetch_bytes(source["iiif_manifest_url"], 2_000_000)
    manifest = json.loads(manifest_bytes)
    canvas, original_url, service = image_source(manifest, record_id)
    directory = output / record_id
    directory.mkdir()
    (directory / "source-manifest.json").write_bytes(manifest_bytes)
    destination = directory / f"{record_id}.jp2"
    reused = reuse / f"{record_id}.jp2" if reuse is not None else None
    maximum = 100_000_000
    if reused is not None and reused.is_file() and not reused.is_symlink():
        if not 0 < reused.stat().st_size <= maximum:
            raise MarginOcrError("reused original exceeds the download cap or is empty")
        with reused.open("rb") as incoming, destination.open("xb") as target:
            shutil.copyfileobj(incoming, target)
        method = "reuse_local_original_and_verify_manifest_dimensions"
    else:
        with _request(original_url, maximum=maximum) as response, destination.open("xb") as target:
            length = 0
            for block in iter(lambda: response.read(1 << 20), b""):
                length += len(block)
                if length > maximum:
                    raise MarginOcrError("original exceeds the 100 MB download cap")
                target.write(block)
            declared = response.headers.get("Content-Length")
            if not length or (declared is not None and length != int(declared)):
                raise MarginOcrError("original response was empty or truncated")
        method = "download_original_from_manifest_rendering"
    with open_scan(destination) as scan:
        if scan.format != "JPEG2000" or list(scan.size) != [canvas["width"], canvas["height"]]:
            raise MarginOcrError("downloaded original differs from the manifest format or dimensions")
        scan.verify()
        size = list(scan.size)
    # A service-generated preview is only for visual selection, never OCR.
    preview = directory / "preview.jpg"
    preview.write_bytes(fetch_bytes(service + "/full/1600,/0/default.jpg", 3_000_000))
    with open_scan(preview) as image:
        if image.format != "JPEG":
            raise MarginOcrError("preview is not JPEG")
        image.verify()
    record = {**source, "status": "downloaded", "downloaded_utc": datetime.now(timezone.utc).isoformat(),
              "source_url": original_url, "image_path": str(destination.resolve()), "image_size": size,
              "image_bytes": destination.stat().st_size, "image_sha256": digest_file(destination),
              "preview_path": str(preview.resolve()), "preview_sha256": digest_file(preview),
              "source_manifest_sha256": digest_file(directory / "source-manifest.json"),
              "license": manifest["license"], "attribution": manifest.get("attribution", ""),
              "acquisition_method": method, "human_review_status": "unreviewed",
              "interpretation": "Index geometry is source metadata, never applied as coordinates or GCPs. Preview is not an OCR input."}
    write_json(directory / "source.json", record)
    return record


def acquire(index_path: Path, output: Path, reuse: Path | None = None) -> dict:
    selected = select_sources(read_json(index_path))
    output.mkdir(parents=True, exist_ok=False)
    selection = {"schema": "jap-map-public-margin-source-selection/1", "index_page": INDEX_PAGE,
                 "index_archive_url": INDEX_URL, "index_geojson_sha256": digest_file(index_path),
                 "selection_rule": "One map per explicit regional group, alternating R-suffix preference then record-ID order; E9 uses the independently inspected pr492sf6025. All D11/Gongju sources excluded. No OCR scores used for selection.",
                 "sources": selected}
    write_json(output / "selection.json", selection)
    by_id = {}
    # Bound concurrency for the public archive and local bandwidth.
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(acquire_source, entry, output, reuse): entry for entry in selected}
        for future in as_completed(futures):
            entry = futures[future]
            try:
                record = future.result()
            except Exception as error:
                record = {**entry, "status": "download_failed", "error": f"{type(error).__name__}: {error}"}
            by_id[entry["record_id"]] = record
            print(f"{len(by_id)}/{len(selected)} {entry['record_id']} {record['status']}", flush=True)
    records = [by_id[entry["record_id"]] for entry in selected]
    report = {"schema": "jap-map-public-margin-sources/1", "selection_sha256": digest_value(selection),
              "sources": records, "downloaded_sheets": sum(row["status"] == "downloaded" for row in records),
              "target_sheets": len(selected), "ocr_performed": False, "human_approvals": 0}
    write_json(output / "sources.json", report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path, help="index_map.geojson from the public Stanford index archive")
    parser.add_argument("--output", type=Path, required=True, help="new acquisition directory")
    parser.add_argument("--reuse-originals", type=Path, help="optional known local originals named by record ID")
    args = parser.parse_args(argv)
    try:
        report = acquire(args.index, args.output, args.reuse_originals)
    except (MarginOcrError, OSError, ValueError, KeyError, TypeError) as error:
        print(f"Acquisition stopped: {error}", file=sys.stderr)
        return 2
    print(f"Verified {report['downloaded_sheets']}/{report['target_sheets']} original sheets. No OCR or approvals.")
    return 0 if report["downloaded_sheets"] == report["target_sheets"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
