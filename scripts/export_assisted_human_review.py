#!/usr/bin/env python3
"""Export accepted contour additions separately from tentative or rejected work."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from histcontour_core.assisted_review import build_review_outputs
from histcontour_core.provenance import sha256_file


def export(packet, ledger_path, output, *, revisions_path=None, next_id=None):
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
    ledger = read(ledger_path)
    for name, key in (("drawing-report.json", "source_report_sha256"), ("ai-proposals.geojson", "source_proposals_sha256")):
        if sha256_file(packet/name) != ledger.get(key):
            raise ValueError("review ledger does not belong to these exact source files")
    if revisions_path is not None and sha256_file(revisions_path) != ledger.get("revision_collection_sha256"):
        raise ValueError("revision collection byte digest differs from the reviewed snapshot")
    result = build_review_outputs(read(packet/"drawing-report.json"), read(packet/"ai-proposals.geojson"),
                                  ledger, read(revisions_path) if revisions_path else None)
    if next_id is not None and next_id not in result["queue"]["eligible_ids"]:
        raise ValueError("next question must be an unreviewed, non-deferred original case")
    chosen = next_id or next(iter(result["queue"]["eligible_ids"]), None)
    result["queue"]["next_batch_ids"] = [chosen] if chosen else []
    result["queue"]["batch_size"] = 1
    result["summary"]["ledger_sha256"] = sha256_file(ledger_path)
    result["summary"]["source_report_sha256"] = ledger["source_report_sha256"]
    result["summary"]["source_proposals_sha256"] = ledger["source_proposals_sha256"]
    output.mkdir(parents=True, exist_ok=False)

    def write(name, value):
        with (output/name).open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")

    for bucket, collection in result["collections"].items():
        write(bucket.replace("_", "-")+".geojson", collection)
    write("reviewed-proposals.geojson", result["reviewed"])
    write("review-summary.json", result["summary"])
    write("remaining-review-queue.json", result["queue"])
    print(json.dumps(result["summary"], ensure_ascii=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path)
    parser.add_argument("ledger", type=Path)
    parser.add_argument("--revisions", type=Path)
    parser.add_argument("--next-id")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        export(args.packet, args.ledger, args.output, revisions_path=args.revisions, next_id=args.next_id)
    except (OSError, ValueError, KeyError) as error:
        print(f"Assisted review export stopped: {error}", file=sys.stderr)
        raise SystemExit(2)
