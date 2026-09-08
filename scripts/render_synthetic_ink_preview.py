"""Render a compact, deterministic visual companion to the synthetic report."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

from histcontour_core.synthetic_ink import build_synthetic_case


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("docs/assets/synthetic-ink-preview.png"))
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY / path


def overlay(case, mask, colour):
    import numpy as np
    from PIL import Image
    values = case.image.astype(np.float32).copy()
    values[mask] = values[mask] * 0.25 + np.asarray(colour, dtype=np.float32) * 0.75
    return Image.fromarray(values.astype(np.uint8))


def main():
    from PIL import Image, ImageDraw
    args = parse_args()
    # Four deterministic combinations cover fading, labels, crossings, and
    # stain/blur without presenting this as a real historical-map benchmark.
    cases = [build_synthetic_case(terrain, variant, size=192) for terrain, variant in ((4, 0), (27, 3), (66, 5), (109, 7))]
    panel_width, panel_height = 192, 214
    image = Image.new("RGB", (len(cases) * panel_width, panel_height * 3), "white")
    draw = ImageDraw.Draw(image)
    for index, case in enumerate(cases):
        left = index * panel_width
        image.paste(Image.fromarray(case.image), (left, 0))
        image.paste(overlay(case, case.visible_contour, (225, 29, 72)), (left, panel_height))
        image.paste(overlay(case, case.complete_contour, (22, 163, 74)), (left, panel_height * 2))
        draw.text((left + 4, 194), case.case_id, fill="black")
    for row, label in enumerate(("source", "visible contour truth", "complete contour truth")):
        draw.rectangle((0, row * panel_height, 132, row * panel_height + 13), fill="white")
        draw.text((2, row * panel_height + 1), label, fill="black")
    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    print(output)


if __name__ == "__main__":
    main()
