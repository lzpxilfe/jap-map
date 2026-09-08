"""Deterministic synthetic historical-map cases for Ink regression work.

These fixtures are for algorithm development only.  They deliberately keep
their complete and visible contour masks separate, so a classifier is not
credited for reconstructing a label gap that was invisible in the input.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


SYNTHETIC_DATASET_VERSION = "synthetic-ink-terrain/1"
SCENES_PER_TERRAIN = 8


@dataclass(frozen=True)
class SyntheticInkCase:
    terrain_id: int
    variant_id: int
    split: str
    image: object
    visible_contour: object
    complete_contour: object
    non_contour: object
    label_gap: object

    @property
    def case_id(self) -> str:
        return f"terrain-{self.terrain_id:03d}-variant-{self.variant_id}"


def terrain_split(terrain_id: int) -> str:
    if not 0 <= terrain_id < 120:
        raise ValueError("terrain_id must be in [0, 120)")
    return "train" if terrain_id < 80 else "validation" if terrain_id < 100 else "test"


def _draw_curve(draw, points, *, fill, width: int) -> None:
    draw.line([(int(round(x)), int(round(y))) for x, y in points], fill=fill, width=width, joint="curve")


def _curve_points(width: int, height: int, terrain_id: int, contour_index: int):
    phase = terrain_id * 0.173 + contour_index * 0.71
    amplitude = 4.0 + (terrain_id * 7 + contour_index * 5) % 13
    base = height * (0.18 + contour_index * 0.115) + ((terrain_id * 11) % 17 - 8)
    slope = ((terrain_id * 3 + contour_index) % 9 - 4) * 0.025
    return [
        (x, base + slope * (x - width / 2) + amplitude * math.sin(x / (17 + terrain_id % 13) + phase) + amplitude * 0.22 * math.sin(x / 7.0 + phase * 1.7))
        for x in range(8, width - 8)
    ]


def _mask_from_draw(size, painter):
    from PIL import Image, ImageDraw

    image = Image.new("L", size, 0)
    painter(ImageDraw.Draw(image))
    import numpy as np
    return np.asarray(image, dtype=np.uint8) > 0


def build_synthetic_case(terrain_id: int, variant_id: int, *, size: int = 192) -> SyntheticInkCase:
    """Generate one map-like raster with contour/non-contour truth masks."""

    if not 0 <= terrain_id < 120 or not 0 <= variant_id < SCENES_PER_TERRAIN or size < 96:
        raise ValueError("invalid synthetic terrain, variant, or image size")
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter

    random = np.random.default_rng(20260908 + terrain_id * 19 + variant_id)
    paper = np.full((size, size, 3), (238, 231, 211), dtype=np.int16)
    paper += random.normal(0.0, 3.5 + variant_id % 3, size=(size, size, 1)).astype(np.int16)
    paper = np.clip(paper, 0, 255).astype(np.uint8)
    complete = Image.new("L", (size, size), 0)
    complete_draw = ImageDraw.Draw(complete)
    contours = [_curve_points(size, size, terrain_id, index) for index in range(6)]
    thickness = 1 if variant_id not in (2, 6) else 3
    for points in contours:
        _draw_curve(complete_draw, points, fill=255, width=thickness)
    complete_mask = np.asarray(complete, dtype=np.uint8) > 0
    visible_mask = complete_mask.copy()
    label_gap_mask = np.zeros((size, size), dtype=bool)
    target_points = contours[2 + terrain_id % 2]
    if variant_id in (1, 3, 5, 7):
        start = 66 + terrain_id % 18
        end = start + (13 if variant_id != 3 else 30)
        label_gap_mask[:, start:end] = True
        visible_mask &= ~label_gap_mask
    non_contour = Image.new("L", (size, size), 0)
    non_draw = ImageDraw.Draw(non_contour)
    # Printed labels and values.  These are intentionally dark and overlap
    # a contour in variants 1/3/5/7.
    label_x = 69 + terrain_id % 16
    label_y = int(round(target_points[label_x][1])) - 12
    if variant_id in (1, 3, 5, 7):
        for digit, offset in zip("128", (0, 9, 18)):
            non_draw.text((label_x + offset, label_y), digit, fill=255, stroke_width=0)
    if variant_id in (2, 4, 6, 7):
        road_x = 22 + (terrain_id * 13) % (size - 44)
        non_draw.line((road_x, 7, road_x + (terrain_id % 9 - 4), size - 8), fill=255, width=2)
    if variant_id in (4, 5, 6, 7):
        non_draw.arc((size - 58, 28 + terrain_id % 36, size - 32, 54 + terrain_id % 36), 0, 330, fill=255, width=2)
        non_draw.rectangle((16 + terrain_id % 28, size - 46, 26 + terrain_id % 28, size - 36), outline=255, width=2)
    non_mask = np.asarray(non_contour, dtype=np.uint8) > 0
    image = Image.fromarray(paper, "RGB")
    image_draw = ImageDraw.Draw(image)
    # Vary darkness and local fading.  Completed-but-erased gap stays blank.
    contour_color = (65, 56, 45) if variant_id not in (0, 6) else (102, 91, 74)
    for index, points in enumerate(contours):
        _draw_curve(image_draw, points, fill=contour_color, width=thickness)
        if variant_id in (0, 2, 6) and index == 2:
            for x, y in points[42:71]:
                image_draw.point((int(x), int(y)), fill=(174, 158, 134))
    if label_gap_mask.any():
        gap_image = Image.new("L", (size, size), 0)
        gap_draw = ImageDraw.Draw(gap_image)
        gap_draw.rectangle((66 + terrain_id % 18, 0, 66 + terrain_id % 18 + (13 if variant_id != 3 else 30), size), fill=255)
        gap_array = np.asarray(gap_image, dtype=np.uint8) > 0
        raw = np.asarray(image).copy()
        raw[gap_array & complete_mask] = paper[gap_array & complete_mask]
        image = Image.fromarray(raw, "RGB")
        image_draw = ImageDraw.Draw(image)
    non_overlay = np.asarray(non_contour, dtype=np.uint8) > 0
    raw = np.asarray(image).copy()
    raw[non_overlay] = (37, 31, 27)
    image = Image.fromarray(raw, "RGB")
    image_draw = ImageDraw.Draw(image)
    if variant_id in (6, 7):
        for x, y, radius in ((36, 41, 12), (130, 148, 16), (162, 55, 10)):
            image_draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(193, 180, 153))
        image = image.filter(ImageFilter.GaussianBlur(radius=0.8 if variant_id == 6 else 1.25))
    return SyntheticInkCase(terrain_id, variant_id, terrain_split(terrain_id), np.asarray(image), visible_mask, complete_mask, non_mask, label_gap_mask & complete_mask)


def iter_synthetic_cases(*, terrain_count: int = 120, variants: int = SCENES_PER_TERRAIN, size: int = 192):
    if not 1 <= terrain_count <= 120 or not 1 <= variants <= SCENES_PER_TERRAIN:
        raise ValueError("terrain_count and variants exceed the fixed benchmark design")
    for terrain_id in range(terrain_count):
        for variant_id in range(variants):
            yield build_synthetic_case(terrain_id, variant_id, size=size)


__all__ = ["SCENES_PER_TERRAIN", "SYNTHETIC_DATASET_VERSION", "SyntheticInkCase", "build_synthetic_case", "iter_synthetic_cases", "terrain_split"]
