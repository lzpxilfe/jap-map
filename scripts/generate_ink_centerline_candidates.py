"""Generate separate ArchaeoTrace Ink v2 vector proposals for A/B review.

The existing grayscale-ridge candidates and mutable review GeoPackage are
never overwritten.  Holdout tiles are excluded unless ``--include-holdout``
is explicitly supplied.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

from histcontour_core.ink import (
    ARCHAEOTRACE_UPSTREAM_COMMIT,
    INK_BACKEND_ID,
    InkCenterlineSettings,
    ink_centerline_candidates,
)
from histcontour_core.provenance import INK_ADAPTER_VERSION, execution_id, polyline_geometry_id, sha256_file
from histcontour_core.vectorization import DEFAULT_DIAGONAL_POLICY, SKELETON_GRAPH_VERSION, skeleton_to_pixel_line_proposals


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=Path, help="annotation_package/index.json")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/derived/annotation_package/ink_candidate_vectors_corner_safe_v3"),
    )
    parser.add_argument("--minimum-length-px", type=float, default=18.0)
    parser.add_argument("--simplify-tolerance-px", type=float, default=0.75)
    parser.add_argument("--diagonal-policy", choices=("corner_safe", "full8"), default=DEFAULT_DIAGONAL_POLICY,
                        help="corner_safe removes redundant diagonal shortcuts; full8 reproduces the earlier vector graph")
    parser.add_argument("--junction-policy", choices=("split", "tangent_pairs"), default="split",
                        help="optional research grouping at unambiguous existing three-way junctions; no gap filling")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument(
        "--include-holdout",
        action="store_true",
        help="Process holdout_test tiles too; off by default to prevent tuning leakage",
    )
    return parser.parse_args()


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY / path


def _portable_path(path: Path) -> str:
    try:
        value = path.resolve().relative_to(REPOSITORY)
    except ValueError:
        value = path.resolve()
    return str(value).replace("\\", "/")


def _atomic_json(path: Path, value) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def select_tiles(index: dict, include_holdout: bool = False) -> list[dict]:
    """Select development tiles by default; holdout access must be explicit."""

    return [
        tile
        for tile in index["tiles"]
        if include_holdout or tile["split"] != "holdout_test"
    ]


def _map_coordinates(tile: dict, width: int, height: int, points) -> list[list[float]]:
    west, south, east, north = (float(value) for value in tile["bounds"])
    if not west < east or not south < north:
        raise ValueError(f"Invalid bounds for {tile['tile_id']}")
    return [
        [
            west + (float(x) + 0.5) * (east - west) / width,
            north - (float(y) + 0.5) * (north - south) / height,
        ]
        for x, y in points
    ]


def _feature(tile: dict, proposal, width: int, height: int, *, run_id: str, raster_digest: str, ink_support: float, direction_coherence: float) -> dict:
    return {
        "type": "Feature",
        "properties": {
            "proposal_id": proposal.proposal_id,
            "segment_geometry_id": polyline_geometry_id(proposal.points),
            "segment_uid": f"{run_id}:{polyline_geometry_id(proposal.points)}",
            "ink_run_id": run_id,
            "source_raster_sha256": raster_digest,
            "tile_id": tile["tile_id"],
            "sheet_id": tile["sheet_id"],
            "split": tile["split"],
            "proposal_kind": "visible_linework_review_only",
            "backend": INK_BACKEND_ID,
            "upstream_commit": ARCHAEOTRACE_UPSTREAM_COMMIT,
            "adapter_version": INK_ADAPTER_VERSION,
            "pixel_length": round(proposal.pixel_length, 3),
            "point_count": len(proposal.points),
            "ink_support": round(ink_support, 4),
            "direction_coherence": round(direction_coherence, 4),
            # Compatibility only. This describes retained Ink membership, not
            # a contour probability; consumers should use ink_support.
            "confidence": round(proposal.confidence, 4),
            "review_status": "unreviewed",
        },
        "geometry": {
            "type": "LineString",
            "coordinates": _map_coordinates(tile, width, height, proposal.points),
        },
    }


def proposal_mask(proposals, shape):
    """Rasterize the EXPORTED geometry, not the unfiltered Ink skeleton."""
    import numpy as np
    from PIL import Image, ImageDraw

    height, width = shape
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for proposal in proposals:
        draw.line([(round(x), round(y)) for x, y in proposal.points], fill=255, width=1)
    return np.asarray(mask) > 0


def _preview(source, proposals, output_path: Path) -> None:
    import numpy as np
    from PIL import Image

    values = np.asarray(source)
    if values.ndim == 2:
        rgb = np.repeat(values[..., None], 3, axis=2)
    else:
        rgb = np.asarray(values[..., :3])
    rgb = np.clip(rgb, 0, 255).astype(np.uint8, copy=True)
    centerline = proposal_mask(proposals, rgb.shape[:2])
    colour = np.array((6, 182, 212), dtype=np.float32)
    rgb[centerline] = (rgb[centerline].astype(np.float32) * 0.25 + colour * 0.75).astype(np.uint8)
    Image.fromarray(rgb).resize((384, 384)).save(output_path)


def process_tile(task: tuple) -> dict:
    tile, output_directory, minimum_length_px, simplify_tolerance_px, *extra = task
    diagonal_policy = extra[0] if extra else DEFAULT_DIAGONAL_POLICY
    junction_policy = extra[1] if len(extra) > 1 else "split"
    try:
        import numpy as np
        from PIL import Image
    except ImportError as error:
        raise RuntimeError(f"Missing Ink candidate dependency: {error.name}") from error

    raster_path = _resolve(tile["raster_path"])
    raster_digest = sha256_file(raster_path)
    with Image.open(raster_path) as image:
        source = np.asarray(image).copy()
    if source.ndim not in (2, 3):
        raise ValueError(f"Unsupported raster dimensions for {tile['tile_id']}: {source.shape}")
    height, width = source.shape[:2]
    _x, _y, expected_width, expected_height = tile["pixel_bounds"]
    if (width, height) != (expected_width, expected_height):
        raise ValueError(
            f"Raster dimensions for {tile['tile_id']} are {(width, height)}, "
            f"expected {(expected_width, expected_height)}"
        )
    evidence = ink_centerline_candidates(source, tile_origin=tuple(tile["pixel_bounds"][:2]))
    proposals = skeleton_to_pixel_line_proposals(
        evidence.centerline,
        evidence.center_score,
        minimum_length_px=minimum_length_px,
        simplify_tolerance_px=simplify_tolerance_px,
        likelihood_scale=1.0,
        proposal_prefix="ink-line",
        diagonal_policy=diagonal_policy,
        junction_policy=junction_policy,
    )
    output_dir = Path(output_directory)
    run_id = execution_id(
        raster_sha256=raster_digest,
        backend=INK_BACKEND_ID,
        upstream_commit=ARCHAEOTRACE_UPSTREAM_COMMIT,
        settings=asdict(InkCenterlineSettings()),
        vectorization={"minimum_length_px": minimum_length_px, "simplify_tolerance_px": simplify_tolerance_px,
                       "graph_version": SKELETON_GRAPH_VERSION, "diagonal_policy": diagonal_policy, "junction_policy": junction_policy},
    )
    vector_path = output_dir / f"{tile['tile_id']}-ink-proposals.geojson"
    preview_path = output_dir / f"{tile['tile_id']}-ink-preview.png"
    collection = {
        "type": "FeatureCollection",
        "name": f"ink_candidate_proposals_{tile['tile_id']}",
        "crs": {"type": "name", "properties": {"name": f"urn:ogc:def:crs:{tile['crs_authid'].replace(':', '::')}"}},
        "features": [
            _feature(
                tile,
                proposal,
                width,
                height,
                run_id=run_id,
                raster_digest=raster_digest,
                ink_support=float(sum(
                    evidence.support_score[int(round(y)), int(round(x))]
                    for x, y in proposal.points
                ) / len(proposal.points)),
                direction_coherence=float(sum(
                    evidence.coherence[int(round(y)), int(round(x))]
                    for x, y in proposal.points
                ) / len(proposal.points)),
            )
            for proposal in proposals
        ],
    }
    _atomic_json(vector_path, collection)
    _preview(source, proposals, preview_path)
    return {
        "tile_id": tile["tile_id"],
        "sheet_id": tile["sheet_id"],
        "split": tile["split"],
        "scene_type": tile["scene_type"],
        "ink_vector_path": _portable_path(vector_path),
        "ink_preview_path": _portable_path(preview_path),
        "centerline_fraction": evidence.centerline_fraction,
        "centerline_pixels": int(evidence.centerline.sum()),
        "ink_run_id": run_id,
        "source_raster_sha256": raster_digest,
        "proposal_count": len(proposals),
        "exported_vector_pixels": int(proposal_mask(proposals, (height, width)).sum()),
        "preview_geometry": "exported_polylines",
    }


def _write_contact_sheet(results: list[dict], output_path: Path) -> None:
    from PIL import Image, ImageDraw

    columns = 3
    rows = (len(results) + columns - 1) // columns
    contact = Image.new("RGB", (columns * 384, rows * 420), "white")
    draw = ImageDraw.Draw(contact)
    for index, result in enumerate(results):
        column, row = index % columns, index // columns
        left, top = column * 384, row * 420
        with Image.open(_resolve(result["ink_preview_path"])) as preview_source:
            preview = preview_source.convert("RGB")
        contact.paste(preview, (left, top))
        draw.text(
            (left + 6, top + 390),
            f"{result['tile_id']} | {result['proposal_count']} proposals",
            fill="black",
        )
    contact.save(output_path)


def main():
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.minimum_length_px <= 0 or args.simplify_tolerance_px < 0:
        raise SystemExit("length must be positive and simplify tolerance must be non-negative")
    index_path = _resolve(args.index)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    selected = select_tiles(index, args.include_holdout)
    if not selected:
        raise SystemExit("No tiles selected")
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        (tile, str(output_dir), args.minimum_length_px, args.simplify_tolerance_px, args.diagonal_policy, args.junction_policy)
        for tile in selected
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(process_tile, tasks))

    result_index = {
        "version": "3",
        "graph_version": SKELETON_GRAPH_VERSION,
        "diagonal_policy": args.diagonal_policy,
        "junction_policy": args.junction_policy,
        "backend": INK_BACKEND_ID,
        "adapter_version": INK_ADAPTER_VERSION,
        "upstream": {
            "repository": "https://github.com/lzpxilfe/AI-Vectorizer-for-Archaeology",
            "commit": ARCHAEOTRACE_UPSTREAM_COMMIT,
            "algorithm": "Ink v2 source-grid evidence with direction fields; batch vectors remain review-only",
        },
        "review_only": True,
        "holdout_included": bool(args.include_holdout),
        "settings": asdict(InkCenterlineSettings()),
        "minimum_length_px": args.minimum_length_px,
        "simplify_tolerance_px": args.simplify_tolerance_px,
        "tiles": results,
    }
    index_output = output_dir / "ink_candidate_vector_index.json"
    contact_output = output_dir / "ink_candidate_contact_sheet.png"
    _atomic_json(index_output, result_index)
    _write_contact_sheet(results, contact_output)
    for result in results:
        print(f"{result['ink_vector_path']} ({result['proposal_count']} proposals)")
    print(index_output)
    print(contact_output)


if __name__ == "__main__":
    main()
