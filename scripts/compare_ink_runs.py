"""Create an auditable summary before accepting a new Ink candidate run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/derived/annotation_package/ink_run_comparison.json"))
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY / path


def compare(baseline: dict, candidate: dict) -> dict:
    old_tiles = {tile["tile_id"]: tile for tile in baseline["tiles"]}
    new_tiles = {tile["tile_id"]: tile for tile in candidate["tiles"]}
    rows = []
    for tile_id in sorted(set(old_tiles) | set(new_tiles)):
        old, new = old_tiles.get(tile_id), new_tiles.get(tile_id)
        if old is None or new is None:
            rows.append({"tile_id": tile_id, "status": "missing_from_one_run"})
            continue
        old_pixels, new_pixels = int(old["centerline_pixels"]), int(new["centerline_pixels"])
        old_proposals, new_proposals = int(old["proposal_count"]), int(new["proposal_count"])
        rows.append({
            "tile_id": tile_id,
            "centerline_pixels_before": old_pixels,
            "centerline_pixels_after": new_pixels,
            "centerline_pixel_delta": new_pixels - old_pixels,
            "centerline_pixel_delta_fraction": (new_pixels - old_pixels) / max(1, old_pixels),
            "proposals_before": old_proposals,
            "proposals_after": new_proposals,
            "proposal_delta": new_proposals - old_proposals,
        })
    return {
        "baseline": {"version": baseline.get("version"), "backend": baseline.get("backend"), "upstream": baseline.get("upstream")},
        "candidate": {"version": candidate.get("version"), "backend": candidate.get("backend"), "upstream": candidate.get("upstream")},
        "review_required": True,
        "tiles": rows,
        "notes": "This compares recorded output counts only. Accepting a candidate also requires fixed synthetic regression and visual review.",
    }


def main():
    args = parse_args()
    result = compare(json.loads(_resolve(args.baseline).read_text(encoding="utf-8")), json.loads(_resolve(args.candidate).read_text(encoding="utf-8")))
    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
