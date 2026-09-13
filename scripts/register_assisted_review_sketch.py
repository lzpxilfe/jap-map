#!/usr/bin/env python3
"""Register a user's red screenshot marks as a review-only raster overlay.

Uses blue endpoint rings to initialize a scale/translation fit, then matches
the uncoloured map background. Does not join red marks, vectorize contours,
change source proposals, infer approvals, or fit a classification model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage, optimize


def colour_masks(rgb):
    r, g, b = np.asarray(rgb, dtype=np.float32).transpose(2, 0, 1)
    blue = (b > 130) & (r < 100) & (g < 170) & (b > g+60)
    # Keep the red pen separate from the original orange AI proposal.
    red = (r > 100) & (g < 90) & (b < 90) & (r > 1.8*g) & (r > 1.8*b)
    coloured = np.maximum.reduce([r, g, b])-np.minimum.reduce([r, g, b]) > 25
    return blue, red, coloured


def split_red_annotations(red, boxes):
    """Separate explicitly located annotation glyphs, never infer their meaning.

    A box marks only the symbol pixels, not the geographic region of doubt.
    Remaining pen pixels are still unverified sketches, not accepted paths.
    """
    red = np.asarray(red, dtype=bool)
    if red.ndim != 2:
        raise ValueError("red mask must be two-dimensional")
    annotation = np.zeros_like(red)
    for box in boxes:
        if len(box) != 4 or any(type(v) is not int for v in box):
            raise ValueError("annotation box needs four integer pixel bounds")
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= red.shape[1] and 0 <= y1 < y2 <= red.shape[0]):
            raise ValueError("annotation box outside the registered panel")
        if not red[y1:y2, x1:x2].any():
            raise ValueError("annotation box contains no red mark")
        annotation[y1:y2, x1:x2] |= red[y1:y2, x1:x2]
    return red & ~annotation, annotation


def blue_centres(rgb):
    blue = colour_masks(rgb)[0]
    radius = max(1, round(min(blue.shape)*.008))
    labels, _ = ndimage.label(ndimage.binary_dilation(blue, iterations=radius))
    centres = []
    for label in range(1, int(labels.max())+1):
        ys, xs = np.where((labels == label) & blue)
        if len(xs) >= 10:
            centres.append([(float(xs.min())+float(xs.max()))/2,
                            (float(ys.min())+float(ys.max()))/2])
    if len(centres) != 2:
        raise ValueError(f"expected exactly two blue endpoint rings, found {len(centres)}")
    return np.array(sorted(centres, key=lambda p: (p[1], p[0])))


def register(reference, screenshot, endpoints):
    reference = np.asarray(reference, dtype=np.uint8)
    screenshot = np.asarray(screenshot, dtype=np.uint8)
    endpoints = np.array(sorted(endpoints, key=lambda p: (p[1], p[0])), dtype=float)
    anchors = blue_centres(screenshot)
    source_distance = np.linalg.norm(endpoints[1]-endpoints[0])
    if source_distance < 3:
        raise ValueError("endpoint separation too small for registration")
    scale = np.linalg.norm(anchors[1]-anchors[0])/source_distance
    if not 1 <= scale <= 32:
        raise ValueError("unsupported screenshot scale")
    shift = anchors.mean(axis=0)-scale*endpoints.mean(axis=0)
    _, red, coloured = colour_masks(screenshot)
    if red.sum() < 20:
        raise ValueError("no substantial red user sketch found")
    target_mask = ndimage.binary_dilation(coloured, iterations=max(2, round(scale)))
    source_mask = ~ndimage.binary_dilation(colour_masks(reference)[2], iterations=1)
    yy, xx = np.where(source_mask)
    source_grey = np.array(Image.fromarray(reference).convert("L"), dtype=float)/255
    target_grey = np.array(Image.fromarray(screenshot).convert("L"), dtype=float)/255
    reference_values = source_grey[yy, xx]

    def score(parameters):
        sx, sy, tx, ty = np.array(parameters)*scale
        u, v = xx*sx+tx, yy*sy+ty
        valid = ((u >= 0) & (v >= 0) & (u < screenshot.shape[1]-1) & (v < screenshot.shape[0]-1))
        valid &= ndimage.map_coordinates(target_mask.astype(np.uint8), [v, u], order=0, mode="constant", cval=1) == 0
        if valid.sum() < .35*len(xx):
            return 2.
        values = ndimage.map_coordinates(target_grey, [v[valid], u[valid]], order=1)
        original = reference_values[valid]
        if original.std() < .02 or values.std() < .02:
            return 2.
        return 1-float(np.corrcoef(original, values)[0, 1])

    initial = [1., 1., shift[0]/scale, shift[1]/scale]
    bounds = [(.88, 1.12), (.88, 1.12),
              (initial[2]-6, initial[2]+6), (initial[3]-6, initial[3]+6)]
    fit = optimize.minimize(score, initial, method="Powell", bounds=bounds,
                            options={"maxiter": 60, "xtol": 1e-5, "ftol": 1e-6})
    sx, sy, tx, ty = fit.x*scale
    ncc = 1-score(fit.x)
    mapped_anchors = (anchors-np.array([tx, ty]))/np.array([sx, sy])
    errors = np.linalg.norm(mapped_anchors-endpoints, axis=1)
    if not fit.success or ncc < .85 or errors.max() > .75:
        raise ValueError(f"registration needs manual review: NCC={ncc:.4f}, anchor error={errors.max():.3f}px")
    return {"screenshot_from_crop": {"scale_x": float(sx), "scale_y": float(sy),
                                     "offset_x": float(tx), "offset_y": float(ty)},
            "background_alignment_ncc": ncc,
            "blue_anchor_error_source_pixels": errors.tolist(),
            "quality_gate": {"minimum_background_ncc": .85, "maximum_anchor_error_source_pixels": .75},
            "registration_model": "axis_aligned_scale_translation; no rotation or perspective",
            "red_core_pixel_count": int(red.sum())}


def register_board_panel(reference, screenshot, board_reference, row):
    """Use the exact renderer mapping when a user annotates a delivered board.

    Unlike isolated screenshots, very close or painted-over endpoint rings
    need not be recoverable. Verify the source panel and unchanged background
    before using the known resize/paste transform; never infer row identity.
    """
    if type(row) is not int or row < 0:
        raise ValueError("board row must be a nonnegative integer")
    if (screenshot.size != board_reference.size or screenshot.width != 512
            or screenshot.height % 300 or (row+1)*300 > screenshot.height):
        raise ValueError("expected unchanged single-column review board dimensions")
    factor = min(240/reference.width, 252/reference.height)
    size = (round(reference.width*factor), round(reference.height*factor))
    x, y = (256-size[0])//2, 35+(252-size[1])//2
    panel_box = (0, row*300, 256, (row+1)*300)
    original = board_reference.crop(panel_box).convert("RGB")
    panel = screenshot.crop(panel_box).convert("RGB")
    expected = reference.resize(size, Image.Resampling.NEAREST)
    if not np.array_equal(np.asarray(original.crop((x, y, x+size[0], y+size[1]))), np.asarray(expected)):
        raise ValueError("board row does not exactly match this proposal's source panel")
    a, b = np.asarray(panel), np.asarray(original)
    red = colour_masks(a)[1]
    if red.sum() < 20:
        raise ValueError("no substantial red user sketch found")
    valid = ~ndimage.binary_dilation(colour_masks(a)[2] | colour_masks(b)[2], iterations=2)
    inside = np.zeros(valid.shape, dtype=bool)
    inside[y:y+size[1], x:x+size[0]] = True
    valid &= inside
    if valid.sum() < .25*size[0]*size[1]:
        raise ValueError("too little unmarked map background to verify board alignment")
    delta = np.abs(a.astype(float)-b.astype(float))[valid]
    exact_fraction = float((delta.max(axis=1) == 0).mean())
    mean_error = float(delta.mean())
    if exact_fraction < .995 or mean_error > .25:
        raise ValueError("board background changed; do not assume its old pixel mapping")
    sx, sy = size[0]/reference.width, size[1]/reference.height
    return panel, {
        "screenshot_from_crop": {"scale_x": sx, "scale_y": sy,
                                 "offset_x": x+(sx-1)/2, "offset_y": y+(sy-1)/2},
        "registration_model": "verified_original_board_nearest_resize_pixel_centres",
        "board_panel_pixel_box": list(panel_box),
        "source_image_box_in_panel": [x, y, x+size[0], y+size[1]],
        "board_unchanged_background_fraction": exact_fraction,
        "board_background_mean_absolute_channel_error": mean_error,
        "quality_gate": {"minimum_unchanged_background_fraction": .995,
                         "maximum_mean_absolute_channel_error": .25},
        "red_core_pixel_count": int(red.sum()),
    }


def geotransform(tile, box, fit):
    west, south, east, north = tile["bounds"]
    width, height = tile["pixel_bounds"][2:]
    dx, dy = (east-west)/width, (north-south)/height
    sx, sy = fit["scale_x"], fit["scale_y"]
    tx, ty = fit["offset_x"], fit["offset_y"]
    a, e = dx/sx, -dy/sy
    c = west+(box[0]+.5-tx/sx)*dx
    f = north-(box[1]+.5-ty/sy)*dy
    return [c-a/2, a, 0., f-e/2, 0., e]


def save_json(path, data):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def render_registered_preview(reference, red, fit, proposal_id):
    factor = 5
    preview = reference.resize((reference.width*factor, reference.height*factor), Image.Resampling.NEAREST)
    yy, xx = np.indices((preview.height, preview.width), dtype=float)
    sx = ((xx+.5)/factor-.5)*fit["scale_x"]+fit["offset_x"]
    sy = ((yy+.5)/factor-.5)*fit["scale_y"]+fit["offset_y"]
    # Inverse sampling preserves the footprint of a low-resolution pen pixel.
    # Forward-scattering one dot per input pixel would falsely show dashed ink.
    mask = ndimage.map_coordinates(red.astype(np.uint8), [sy, sx], order=0,
                                   mode="constant", cval=0).astype(bool)
    pixels = np.array(preview)
    pixels[mask] = [190, 35, 10]
    preview = Image.fromarray(pixels)
    draw = ImageDraw.Draw(preview)
    draw.rectangle((0, 0, preview.width, 21), fill="white")
    draw.text((5, 4), f"{proposal_id} | USER SKETCH - PROVISIONAL REFERENCE", fill="black")
    return preview


def run(packet, proposal_id, screenshot_path, output, *, board_reference=None, board_row=None):
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    report = json.loads((packet/"drawing-report.json").read_text(encoding="utf-8"))
    row = next(p for p in report["proposals"] if p["proposal_id"] == proposal_id)
    tile = next(t for t in report["tiles"] if t["tile_id"] == row["tile_id"])
    if digest(packet/tile["raster_path"]) != tile["source_raster_sha256"]:
        raise ValueError("source raster differs from packet")
    crs = json.loads((packet/"ai-proposals.geojson").read_text())["crs"]["properties"]["name"]
    if crs != "EPSG:5132":
        raise ValueError("unexpected packet CRS")
    original_sha = digest(screenshot_path)
    reference = Image.open(packet/"images"/f"{proposal_id}-source.png").convert("RGB")
    screenshot = Image.open(screenshot_path).convert("RGB")
    original_size = list(screenshot.size)
    box = row["pixel_box"]
    endpoints = [[p[0]-box[0], p[1]-box[1]] for p in [row["start"], row["end"]]]
    if (board_reference is None) != (board_row is None):
        raise ValueError("board reference and row must be supplied together")
    if board_reference is None:
        result = register(reference, screenshot, endpoints)
    else:
        board_digest = digest(board_reference)
        screenshot, result = register_board_panel(
            reference, screenshot, Image.open(board_reference).convert("RGB"), board_row)
        if digest(board_reference) != board_digest:
            raise ValueError("reference board changed during registration")
        result["reference_board_sha256"] = board_digest
    if digest(screenshot_path) != original_sha:
        raise ValueError("screenshot changed during registration")
    result.update(schema="jap-map-user-sketch-registration/1", proposal_id=proposal_id,
                  screenshot_sha256=original_sha, source_raster_sha256=tile["source_raster_sha256"],
                  drawing_report_sha256=digest(packet/"drawing-report.json"),
                  screenshot_size=original_size, registered_raster_size=list(screenshot.size), original_pixel_box=box,
                  tile_id=tile["tile_id"], native_crs=crs,
                  dataset_role="review_only_not_training", reference_origin="user_red_screenshot_marks",
                  source_packet_modified=False, vector_geometry_created=False, model_fitted=False,
                  semantic_label_assigned=False, human_approval_inferred=False,
                  red_strokes_joined=False, visible_vs_inferred_split="not_annotated",
                  limitation="Raster registration preserves pen marks, not verified contour centreline geometry or semantic truth.")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(screenshot_path, output/"user-screenshot.png")
    if digest(output/"user-screenshot.png") != original_sha:
        raise RuntimeError("screenshot copy digest mismatch")
    if board_reference is not None:
        screenshot.save(output/"registered-panel.png")
    red = colour_masks(np.asarray(screenshot))[1]
    overlay = np.zeros((screenshot.height, screenshot.width, 4), np.uint8)
    overlay[:, :, :3] = np.asarray(screenshot)
    overlay[:, :, 3] = red.astype(np.uint8)*255
    Image.fromarray(overlay).save(output/"red-marks.png")
    gt = geotransform(tile, box, result["screenshot_from_crop"])
    result["red_marks_gdal_geotransform"] = gt
    vrt = [f'<VRTDataset rasterXSize="{screenshot.width}" rasterYSize="{screenshot.height}">',
           f'  <SRS>{crs}</SRS>', '  <GeoTransform>'+', '.join(map(str, gt))+'</GeoTransform>']
    for band, colour in enumerate(("Red", "Green", "Blue", "Alpha"), 1):
        vrt += [f'  <VRTRasterBand dataType="Byte" band="{band}">', f'    <ColorInterp>{colour}</ColorInterp>',
                '    <SimpleSource><SourceFilename relativeToVRT="1">red-marks.png</SourceFilename>',
                f'      <SourceBand>{band}</SourceBand></SimpleSource>', '  </VRTRasterBand>']
    vrt.append('</VRTDataset>')
    with (output/"red-marks.vrt").open("x", encoding="utf-8") as handle:
        handle.write('\n'.join(vrt)+'\n')
    # A red-only overlay on the original black map makes registration inspectable.
    preview = render_registered_preview(reference, red, result["screenshot_from_crop"], proposal_id)
    preview.save(output/"registered-preview.png")
    save_json(output/"registration.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path)
    parser.add_argument("--proposal-id", required=True)
    parser.add_argument("--screenshot", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--board-reference", type=Path, help="Exact original single-column board before user annotation")
    parser.add_argument("--board-row", type=int, help="Zero-based row in that original board")
    args = parser.parse_args()
    try:
        result = run(args.packet, args.proposal_id, args.screenshot, args.output,
                     board_reference=args.board_reference, board_row=args.board_row)
        print(json.dumps(result, ensure_ascii=False))
    except (OSError, ValueError, StopIteration) as error:
        print(f"Sketch registration stopped: {error}", file=sys.stderr)
        raise SystemExit(2)
