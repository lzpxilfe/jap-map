#!/usr/bin/env python3
"""Separate numeric ink hypotheses from a frozen draft, keeping ALL geometry.

Consumes historical source-coupled OCR/CC evidence, not the current producer's
code or a fresh OCR run. Exact source pixels, CC labels and metadata are checked.
The retained/number layers are alternative review partitions; no source, human
approval, elevation, or default contour layer is changed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import importlib.metadata
from pathlib import Path
import shutil
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from histcontour_core.human_feedback import geometry_digest, map_to_pixel, validate_pixel_points
from histcontour_core.map_text_detection import inspect_raster
from histcontour_core.numeric_ink_ownership import infer_numeric_ink_ownership
from histcontour_core.numeric_line_partition import partition_pixel_line, native_part_coordinates, rasterize_line_cells
from histcontour_core.provenance import sha256_file
from histcontour_core.vectorization import _length
from scripts.detect_map_text_multiscale import _pixel_digest
from scripts.prepare_contour_reconstruction_review import read, write, validate_regions, render_panel, intersects


def verified_components(row, gray, recognition_dir):
    """Reconstruct the frozen thresholded CC grid, without reranking groups."""
    import numpy as np
    from scipy import ndimage
    if _pixel_digest(gray) != row["source_crop_gray_payload_sha256"]:
        raise ValueError("recognition crop differs from original source pixels")
    relative = Path(row["component_pixel_arrays"])
    path = (recognition_dir/relative).resolve()
    if relative.is_absolute() or recognition_dir not in path.parents:
        raise ValueError("component arrays escape the recognition packet")
    with zipfile.ZipFile(path) as archive:
        if set(archive.namelist()) != {"source_ink.npy", "component_labels.npy"} or any(i.file_size > gray.size*8+4096 for i in archive.infolist()):
            raise ValueError("component arrays exceed the declared crop budget")
    with np.load(path, allow_pickle=False) as arrays:
        ink, labels = arrays["source_ink"], arrays["component_labels"]
    if ink.shape != gray.shape or labels.shape != gray.shape or ink.dtype.kind != 'b' or labels.dtype.kind not in 'iu':
        raise ValueError("component arrays must match the original grayscale crop")
    config = row["component_generation_configuration"]
    window = config["background_window_px"]
    floor, ceiling = config["ink_contrast_floor"], config["dark_gray_ceiling"]
    if type(window) is not int or not 3 <= window <= 127 or window % 2 != 1 or not 0 < floor < 255 or not 0 < ceiling < 245:
        raise ValueError("unsupported source CC threshold contract")
    values = gray.astype(np.float32)
    expected_ink = ((ndimage.maximum_filter(values, size=window, mode='nearest')-values) >= floor) & (values <= ceiling)
    expected_labels, _ = ndimage.label(expected_ink, structure=np.ones((3, 3), bool))
    if not np.array_equal(ink, expected_ink) or not np.array_equal(labels, expected_labels):
        raise ValueError("component labels no longer represent the declared original ink")
    return labels, path


def merge_evidence(target, box, result):
    """Unknown outside a considered hypothesis is NOT a contradictory label."""
    l, t, r, b = box
    target['glyph'][t:b, l:r] |= result.glyph_candidate
    target['protected'][t:b, l:r] |= result.protected_throughgoing
    target['ambiguous'][t:b, l:r] |= result.ambiguous_ink & result.considered_ink
    target['considered'][t:b, l:r] |= result.considered_ink


def run(packet, manifest_path, recognition_path, vectors_path, approved_path, output):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    started = time.perf_counter()
    packet, recognition_path, vectors_path, approved_path, output = [Path(p).resolve() for p in (packet, recognition_path, vectors_path, approved_path, output)]
    if output.exists():
        raise FileExistsError("numeric separation output must be new")
    if any(output == p or p in output.parents for p in (packet, recognition_path.parent, vectors_path.parent)):
        raise ValueError("numeric output must be outside its source snapshots")
    report_path = packet/'drawing-report.json'
    report, manifest, recognition, vectors, approved = map(read, (report_path, manifest_path, recognition_path, vectors_path, approved_path))
    tiles = validate_regions(manifest, report)
    if sha256_file(report_path) != manifest['drawing_report_sha256']:
        raise ValueError("drawing report differs from fixed diagnostic sources")
    if (recognition.get('schema') != 'jap-map-component-text-recognition/1'
            or recognition.get('holdout_used') is not False or recognition.get('human_approvals') != 0):
        raise ValueError("expected unapproved development-only component recognition")
    crs = {'type': 'name', 'properties': {'name': next(iter(tiles.values()))['crs_authid']}}
    if vectors.get('crs') != crs or approved.get('crs') != crs:
        raise ValueError("all vectors must retain their declared native CRS")
    identities = set()
    for f in vectors['features']:
        props = f['properties']; identity = props.get('path_id') or props.get('line_id')
        key = (props.get('tile_id'), identity)
        if not isinstance(identity, str) or not identity or key in identities or key[0] not in tiles or f['geometry']['type'] != 'LineString':
            raise ValueError("source vectors need distinct native tile/path line identities")
        identities.add(key)
    regions = {r['region_id']: r for r in manifest['regions']}
    seen = set()
    for row in recognition['regions']:
        rid = row['region_id']
        if rid in seen or rid not in regions or any(row[k] != regions[rid][k] for k in ('tile_id', 'box')):
            raise ValueError("duplicate or changed numeric diagnostic region")
        seen.add(rid)
        tile = tiles[row['tile_id']]
        if row.get('split') != 'development' or any(row[k] != tile[k] for k in ('source_raster_sha256', 'crs_authid', 'sheet_id')):
            raise ValueError("numeric evidence source identity differs")
    sources, images, metadata = {}, {}, {}
    paths = [report_path, Path(manifest_path), recognition_path, vectors_path, approved_path]
    for tid in sorted({r['tile_id'] for r in recognition['regions']}):
        tile = tiles[tid]
        relative = Path(tile['raster_path']); source = (packet/relative).resolve()
        if relative.is_absolute() or packet not in source.parents:
            raise ValueError("source raster escapes the drawing packet")
        metadata[tid] = inspect_raster(source, tile)
        with Image.open(source) as image:
            images[tid] = np.asarray(image.convert('L'))
        sources[tid] = source; paths.append(source)
    prepared = []
    for row in recognition['regions']:
        l, t, r, b = row['box']
        gray = images[row['tile_id']][t:b, l:r]
        labels, array_path = verified_components(row, gray, recognition_path.parent)
        paths.append(array_path)
        prepared.append((row, gray, labels))
    before = {str(p.resolve()): sha256_file(p) for p in paths}
    code_paths = [Path(__file__), ROOT/'scripts/prepare_contour_reconstruction_review.py', ROOT/'scripts/detect_map_text_multiscale.py',
                  *[ROOT/'histcontour_core'/name for name in ('numeric_ink_ownership.py', 'numeric_line_partition.py',
                    'map_text_detection.py', 'human_feedback.py', 'vectorization.py', 'margin_ocr.py', 'provenance.py')]]
    code = {str(p.resolve()): sha256_file(p) for p in code_paths}
    masks = {tid: {name: np.zeros(image.shape, bool) for name in ('glyph', 'protected', 'ambiguous', 'considered')}
             for tid, image in images.items()}
    output.mkdir(parents=True)
    region_results = []
    for row, gray, labels in prepared:
        groups = [g['candidate'] for g in row['groups']]
        readings = [{'group_id': g['candidate']['group_id'], 'scale': p['scale'],
                     'rec_text': p['recognition']['rec_text'], 'rec_score': p['recognition']['rec_score']}
                    for g in row['groups'] for p in g['passes'] if p['status'] == 'completed']
        ownership = infer_numeric_ink_ownership(gray, labels, row['components'], groups, readings)
        merge_evidence(masks[row['tile_id']], row['box'], ownership)
        folder = output/row['region_id']; folder.mkdir()
        np.savez_compressed(folder/'numeric-masks.npz', source_ink=ownership.source_ink, glyph_candidate=ownership.glyph_candidate,
                            protected=ownership.protected_throughgoing, ambiguous=ownership.ambiguous_ink, considered=ownership.considered_ink)
        result = {k: copy.deepcopy(row[k]) for k in ('region_id', 'tile_id', 'box', 'source_raster_sha256', 'crs_authid')}
        result.update(hypotheses=list(ownership.hypotheses), component_evidence=list(ownership.component_evidence),
                      provenance=ownership.provenance, mask_path=str((folder/'numeric-masks.npz').relative_to(output)),
                      mask_sha256=sha256_file(folder/'numeric-masks.npz'), human_approved=False, training_eligible=False)
        write(folder/'ownership.json', result)
        region_results.append(result)
        print(row['region_id'], ownership.provenance['counts'], flush=True)
    # Exact approved vectors are separately copied below. Their raster footprint
    # also overrides any numeric hypothesis in this comparison scenario.
    for tid in masks:
        polylines = [map_to_pixel(tiles[tid], f['geometry']['coordinates']) for f in approved['features'] if f['properties'].get('tile_id') == tid]
        masks[tid]['protected'] |= rasterize_line_cells(images[tid].shape, polylines)
        masks[tid]['glyph'] &= ~(masks[tid]['protected'] | masks[tid]['ambiguous'])
    merged_masks = []
    mask_folder = output/'merged-masks'; mask_folder.mkdir()
    for tid, value in masks.items():
        mask_path = mask_folder/(tid+'.npz')
        np.savez_compressed(mask_path, **value)
        merged_masks.append({'tile_id': tid, 'source_raster_sha256': tiles[tid]['source_raster_sha256'],
            'path': str(mask_path.relative_to(output)), 'sha256': sha256_file(mask_path),
            'pixel_counts': {k: int(v.sum()) for k, v in value.items()},
            'scope': 'Final merged masks after overlap ambiguity and exact approved-cell protection.'})
    collections = {name: {'type': 'FeatureCollection', 'crs': crs, 'features': []} for name in ('retained', 'numeric-candidates', 'all-parts')}
    decisions, changed = [], 0
    original_length = numeric_length = 0.
    for f in vectors['features']:
        props, geometry = f['properties'], f['geometry']
        tid = props['tile_id']
        if tid not in tiles or geometry['type'] != 'LineString':
            raise ValueError("unsupported source vector identity or geometry")
        pixels = map_to_pixel(tiles[tid], geometry['coordinates']); validate_pixel_points(pixels, tiles[tid])
        if props.get('source_raster_sha256') != tiles[tid]['source_raster_sha256']:
            raise ValueError("source vector raster identity differs")
        if tid not in masks or props.get('human_approved') is True or props.get('whole_line_semantics_approved') is True:
            for name in ('retained', 'all-parts'): collections[name]['features'].append(copy.deepcopy(f))
            continue
        evidence = masks[tid]
        array = np.asarray(pixels)
        x0, y0 = np.maximum(0, np.floor(array.min(axis=0)+.5).astype(int))
        x1, y1 = np.minimum([evidence['glyph'].shape[1], evidence['glyph'].shape[0]], np.floor(array.max(axis=0)+.5).astype(int)+1)
        if not evidence['glyph'][y0:y1, x0:x1].any():
            original_length += _length(pixels)
            for name in ('retained', 'all-parts'): collections[name]['features'].append(copy.deepcopy(f))
            continue
        parts = partition_pixel_line(pixels, evidence['glyph'], protected=evidence['protected'], ambiguous=evidence['ambiguous'])
        original_length += parts['original_length_px']
        native_parts = [native_part_coordinates(geometry['coordinates'], part['source_edge_locations']) for part in parts['parts']]
        if any(len(set(map(tuple, points))) < 2 for points in native_parts):
            # Nearly simultaneous cell crossings can differ in pixel arithmetic
            # yet round to the same native coordinate. Preserve the whole parent
            # rather than export a zero-length fragment or silently drop it.
            decisions.append({'tile_id': tid, 'parent_path_id': props.get('path_id') or props.get('line_id'),
                              'status': 'retained_unsplit_native_precision_collapse',
                              'human_approved': False})
            for name in ('retained', 'all-parts'): collections[name]['features'].append(copy.deepcopy(f))
            continue
        numeric_length += parts['numeric_candidate_length_px']
        if parts['numeric_candidate_length_px'] <= 1e-9:
            for name in ('retained', 'all-parts'): collections[name]['features'].append(copy.deepcopy(f))
            continue
        changed += 1
        identity = props.get('path_id') or props.get('line_id')
        digest = geometry_digest(geometry)
        decisions.append({'tile_id': tid, 'parent_path_id': identity, 'parent_geometry_sha256': digest,
                          **{k: v for k, v in parts.items() if k != 'parts'}})
        for index, part in enumerate(parts['parts']):
            item = copy.deepcopy(f)
            item['geometry']['coordinates'] = native_part_coordinates(geometry['coordinates'], part['source_edge_locations'])
            item['properties'].update(path_id=f'{identity}:numeric-part-{index:03d}', parent_path_id=identity,
                parent_geometry_sha256=digest, numeric_partition_role=part['kind'], length_px=part['length_px'],
                source_edge_locations=part['source_edge_locations'], human_approved=False, training_eligible=False,
                whole_line_semantics_approved=False, contour_semantics_assigned=False,
                dataset_role='review_only_not_training', source_numeric_ownership_unapproved=True)
            item['properties']['role'] = 'suspected_numeric_ink_review' if part['kind'] == 'numeric_candidate' else props.get('role', 'observed_linework_review')
            item['properties']['geometry_sha256'] = geometry_digest(item['geometry'])
            collections['all-parts']['features'].append(item)
            collections['numeric-candidates' if part['kind'] == 'numeric_candidate' else 'retained']['features'].append(copy.deepcopy(item))
    for name, collection in collections.items(): write(output/(name+'.geojson'), collection)
    shutil.copyfile(vectors_path, output/'before.geojson')
    shutil.copyfile(approved_path, output/'approved-connections-unchanged.geojson')
    if sha256_file(approved_path) != sha256_file(output/'approved-connections-unchanged.geojson'):
        raise ValueError("approved geometry copy changed")
    # Diagnostic masks and line partitions use the same unscaled source crop.
    for row in region_results:
        tid, box = row['tile_id'], row['box']; l, t, r, b = box
        source = Image.fromarray(images[tid])
        crop = source.crop(box).convert('RGB')
        overlay = np.asarray(crop).copy()
        local = {k: a[t:b, l:r] for k, a in masks[tid].items()}
        for mask, color in ((local['ambiguous'], (230, 160, 40)), (local['protected'], (20, 110, 220)), (local['glyph'], (220, 40, 30))):
            overlay[mask] = np.rint(.25*overlay[mask]+.75*np.array(color)).astype(np.uint8)
        panes = [('Original', render_panel(source, box)), ('Numeric ink hypotheses', Image.fromarray(overlay).resize((crop.width*2, crop.height*2), Image.Resampling.NEAREST))]
        kept = [map_to_pixel(tiles[tid], f['geometry']['coordinates']) for f in collections['retained']['features'] if f['properties']['tile_id'] == tid]
        panes.append(('Retained vector draft', render_panel(source, box, lines=[p for p in kept if intersects(p, box)], colour='#078357')))
        w, h = panes[0][1].size
        board = Image.new('RGB', (3*(w+16)+16, h+90), 'white'); pen = ImageDraw.Draw(board); font = ImageFont.load_default(size=15)
        pen.text((16, 8), row['region_id']+' | unapproved numeric/contour separation', fill='black', font=font)
        for i, (title, pane) in enumerate(panes):
            x = 16+i*(w+16); pen.text((x, 35), title, fill='black', font=font); board.paste(pane, (x, 59))
        pen.text((16, h+66), 'Red: numeric hypothesis. Blue: protected ink. Amber: unresolved. No source pixels erased.', fill='#555555', font=font)
        board.save(output/row['region_id']/'comparison.png')
    if before != {str(p.resolve()): sha256_file(p) for p in paths} or code != {str(p.resolve()): sha256_file(p) for p in code_paths}:
        raise ValueError("source or implementation changed during numeric separation")
    result = {'schema': 'jap-map-numeric-separation-run/1', 'status': 'unapproved_lossless_partition_not_applied',
              'regions': region_results, 'merged_masks': merged_masks, 'decisions': decisions, 'human_approvals': 0, 'holdout_used': False,
              'approved_geometry_modified': False, 'original_pixels_modified': False, 'model_fitted': False,
              'counts': {'parent_lines_partitioned': changed, 'input_lines': len(vectors['features']),
                  **{k: len(v['features']) for k, v in collections.items()},
                  'examined_original_length_px': original_length, 'numeric_hypothesis_length_px': numeric_length},
              'provenance': {'input_sha256': before, 'implementation_sha256': code, 'inputs_and_implementation_unchanged': True,
                  'runtime_versions': {p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'Pillow')},
                  'recognition_producer_snapshot_is_historical': True},
              'output_sha256': {p.name: sha256_file(p) for p in sorted(output.glob('*.geojson'))},
              'wall_seconds': time.perf_counter()-started,
              'limitations': ['Numeric strings and glyph pixels remain machine hypotheses, not human truth.',
                  'Only recognized development regions are affected; no general OCR recall is claimed.',
                  'All suspected numeric parts remain in all-parts and numeric-candidates; original before/approved files are unchanged.',
                  'This does not infer connections beneath numbers or promote the retained draft to contour truth.']}
    write(output/'numeric-separation-report.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('packet', 'manifest', 'recognition', 'vectors', 'approved'): parser.add_argument(name, type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.packet, args.manifest, args.recognition, args.vectors, args.approved, args.output)
    print(result['counts'])


if __name__ == '__main__': main()
