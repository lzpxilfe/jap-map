#!/usr/bin/env python3
"""Render source/proposal boards for concrete human decisions."""

import argparse
import json
from pathlib import Path


def render(root, ids=None, output_name="decision-sheets", case_columns=2, *, output=None):
    from PIL import Image, ImageDraw, ImageFont
    root = Path(root)
    report = json.loads((root/"drawing-report.json").read_text(encoding="utf-8"))
    ids = ids or report["priority_ids"]
    rows = {row["proposal_id"]: row for row in report["proposals"]}
    if len(set(ids)) != len(ids) or any(sid not in rows for sid in ids):
        raise ValueError("unknown or repeated proposal IDs")
    if Path(output_name).name != output_name or output_name in (".", ".."):
        raise ValueError("output_name must be one new subdirectory name")
    if case_columns not in (1, 2):
        raise ValueError("case_columns must be 1 or 2")
    # A chat batch can live outside the immutable original delivery packet.
    output = Path(output) if output is not None else root/output_name
    output.mkdir(parents=True, exist_ok=False)
    files = []
    for page, first in enumerate(range(0, len(ids), 6), 1):
        page_ids = ids[first:first+6]
        row_count = (len(page_ids)+case_columns-1)//case_columns
        board = Image.new("RGB", (512*case_columns, 300*row_count), "white")
        draw = ImageDraw.Draw(board)
        font = ImageFont.load_default(size=15)
        for position, sid in enumerate(page_ids):
            x0, y0 = (position%case_columns)*512, (position//case_columns)*300
            row = rows[sid]
            draw.text((x0+10, y0+7), f"{sid}  {row['mode']}  {row['gap_pixels']:.0f}px", fill="black", font=font)
            for column, suffix in enumerate(("source", "proposal")):
                with Image.open(root/"images"/f"{sid}-{suffix}.png") as image:
                    image = image.convert("RGB")
                    factor = min(240/image.width, 252/image.height)
                    size = (round(image.width*factor), round(image.height*factor))
                    image = image.resize(size, Image.Resampling.NEAREST)
                x = x0+column*256+(256-size[0])//2
                y = y0+35+(252-size[1])//2
                board.paste(image, (x,y))
            draw.line((x0, y0+298, x0+510, y0+298), fill="#cccccc")
        path = output/f"decisions-{page:02}.png"
        board.save(path)
        files.append(str(path))
    print(json.dumps(files))
    return files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--ids", nargs="+")
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--output-name", default="decision-sheets")
    destination.add_argument("--output", type=Path, help="New output folder, optionally outside the source packet")
    parser.add_argument("--case-columns", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    render(args.root, args.ids, args.output_name, args.case_columns, output=args.output)
