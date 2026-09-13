#!/usr/bin/env python3
"""Exact pixel-centred comparison: preserve raster pixels, antialias vectors.

This renderer is intentionally separate from historical review-sheet layouts,
whose old pixel transforms are still needed to register user screenshots.
"""
from __future__ import annotations

import math
from pathlib import Path


def render_panel(source, box, *, points=(), endpoints=(), joins=(), size=(360, 360), colour="#e66b00"):
    from PIL import Image, ImageDraw
    left, top, right, bottom = box
    left, top = max(0, math.floor(left)), max(0, math.floor(top))
    right, bottom = min(source.width, math.ceil(right)), min(source.height, math.ceil(bottom))
    if not 0 <= left < right <= source.width or not 0 <= top < bottom <= source.height:
        raise ValueError("review crop must intersect source pixels")
    factor = min(size[0]/(right-left), size[1]/(bottom-top))
    width, height = max(1, round((right-left)*factor)), max(1, round((bottom-top)*factor))
    crop = source.crop((left, top, right, bottom)).convert("RGBA").resize((width, height), Image.Resampling.NEAREST)
    aa = 4
    overlay = Image.new("RGBA", (width*aa, height*aa), (0, 0, 0, 0)); pen = ImageDraw.Draw(overlay)
    def transform(point):
        return ((point[0]-left+.5)*width/(right-left)*aa, (point[1]-top+.5)*height/(bottom-top)*aa)
    if len(points) >= 2:
        pen.line([transform(p) for p in points], fill=colour, width=2*aa, joint="curve")
    for group, color, radius in ((endpoints, "#007cda", 3), (joins, "#079345", 3)):
        for point in group:
            x, y = transform(point)
            pen.ellipse((x-radius*aa, y-radius*aa, x+radius*aa, y+radius*aa), outline=color, width=aa)
    crop = Image.alpha_composite(crop, overlay.resize((width, height), Image.Resampling.LANCZOS)).convert("RGB")
    panel = Image.new("RGB", size, "white")
    panel.paste(crop, ((size[0]-width)//2, (size[1]-height)//2))
    return panel


def render_comparison(source_path, row, new_points, output, *, title=None):
    from PIL import Image, ImageDraw, ImageFont
    output = Path(output)
    if output.exists():
        raise FileExistsError("comparison image already exists")
    endpoints = [row["pixel_points"][0], row["pixel_points"][-1]]
    all_points = [*row["pixel_points"], *new_points]
    xs, ys = [p[0] for p in all_points], [p[1] for p in all_points]
    cx, cy = (min(xs)+max(xs))/2, (min(ys)+max(ys))/2
    radius = max(24, (max(xs)-min(xs))/2+8, (max(ys)-min(ys))/2+8)
    box = (cx-radius, cy-radius, cx+radius, cy+radius)
    board = Image.new("RGB", (1160, 470), "white"); pen = ImageDraw.Draw(board)
    font = ImageFont.load_default(size=17)
    pen.text((20, 10), title or row["proposal_id"], fill="black", font=font)
    with Image.open(source_path) as source:
        for x, crop_box, points, joins, colour, label in (
            (20, row["pixel_box"], (), (), "#e66b00", "Source context"),
            (400, box, row["pixel_points"], (), "#e66b00", "Before | original machine draft"),
            (780, box, new_points, (new_points[0], new_points[-1]), "#079345", "After | needs human review")):
            board.paste(render_panel(source, crop_box, points=points, endpoints=endpoints, joins=joins, colour=colour), (x, 68))
            pen.text((x, 42), label, fill="black", font=font)
    pen.text((20, 444), "Blue circles: original tips. Green circles: proposed joins. Source raster pixels are unchanged.", fill="#555555", font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        board.save(handle, format="PNG")
    return output
