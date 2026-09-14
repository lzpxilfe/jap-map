#!/usr/bin/env python3
"""Run pinned local CPU text detection on declared development GeoTIFFs.

Example (isolated Paddle runtime, no install/download performed):
  data/derived/margin-ocr-runtime/bin/python scripts/detect_map_text_regions.py \
    --drawing-report data/derived/contour-assisted-2026-09-13/review-ready/drawing-report.json \
    --model-lock data/derived/margin-ocr/model-lock-mobile-v5.json \
    --output data/derived/contour-reconstruction-2026-09-14/text-detection-v1 --recognize
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from histcontour_core.map_text_detection import DetectionConfig, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drawing-report", required=True, type=Path)
    parser.add_argument("--model-lock", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Must be a NEW directory outside the source snapshot")
    parser.add_argument("--recognize", action="store_true", help="Add separate raw recognition of detector regions")
    parser.add_argument("--det-max-side", type=int, default=1024)
    parser.add_argument("--det-thresh", type=float, default=0.3)
    parser.add_argument("--det-box-thresh", type=float, default=0.6)
    parser.add_argument("--det-unclip-ratio", type=float, default=1.5)
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args()
    result = run(args.drawing_report, args.model_lock, args.output,
                 DetectionConfig(args.det_max_side, args.det_thresh, args.det_box_thresh,
                                 args.det_unclip_ratio, args.cpu_threads, args.recognize))
    print(json.dumps({"output": str(args.output.resolve()), "counts": result["counts"],
                      "status": result["status"], "source_files_unchanged": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
