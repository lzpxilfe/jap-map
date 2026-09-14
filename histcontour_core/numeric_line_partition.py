"""Lossless geometric partition of a draft line by pixel-cell hypotheses.

This does not erase pixels, smooth a route, choose a connection, or assign
semantic truth. A numerical glyph mask is an unapproved hypothesis. Every
positive-length part is returned, including suspected text, and source-edge
locations permit exact native-coordinate endpoint reuse by the caller.
"""
from __future__ import annotations

import math


def _mask(value, shape=None):
    import numpy as np
    array = np.asarray(value)
    if array.ndim != 2 or not all(array.shape) or array.dtype.kind != 'b' or (shape is not None and array.shape != shape):
        raise ValueError("ownership masks must be boolean arrays on one nonempty source grid")
    if array.size > 4_000_000:
        raise ValueError("ownership grid exceeds the bounded partition budget")
    return array


def _at(points, index, t):
    if t == 0:
        return list(points[index])
    if t == 1:
        return list(points[index+1])
    return [(1-t)*a+t*b for a, b in zip(points[index], points[index+1])]


def native_part_coordinates(native_points, locations):
    """Reuse original vertices exactly; interpolate only new split positions."""
    output = []
    for index, fraction in locations:
        if type(index) is not int or not 0 <= index < len(native_points)-1 or not 0 <= fraction <= 1 or not math.isfinite(fraction):
            raise ValueError("invalid parent-edge location")
        output.append(_at(native_points, index, fraction))
    return output


def _cell_intervals(p, shape, max_intervals):
    """Shared positive-length cell traversal for splitting AND protection."""
    count = 0
    for index, (first, second) in enumerate(zip(p[:-1], p[1:])):
        length = math.dist(first, second)
        if length <= 1e-12:
            continue
        cuts = {0., 1.}
        for a, b in zip(first, second):
            if abs(b-a) <= 1e-12:
                continue
            minimum, maximum = sorted((a, b))
            for k in range(math.floor(minimum-.5)+1, math.ceil(maximum-.5)):
                t = (k+.5-a)/(b-a)
                if 1e-12 < t < 1-1e-12:
                    cuts.add(float(t))
                    if count+len(cuts)-1 > max_intervals:
                        raise ValueError("pixel-cell partition exceeds its interval budget")
        cuts = sorted(cuts)
        for start, end in zip(cuts[:-1], cuts[1:]):
            count += 1
            if count > max_intervals:
                raise ValueError("pixel-cell partition exceeds its interval budget")
            middle = [(1-(start+end)/2)*a+(start+end)/2*b for a, b in zip(first, second)]
            x = min(shape[1]-1, max(0, math.floor(middle[0]+.5)))
            y = min(shape[0]-1, max(0, math.floor(middle[1]+.5)))
            yield index, start, end, y, x, length*(end-start)


def rasterize_line_cells(shape, polylines, *, max_intervals=100_000):
    """Protect every cell that the partitioner can assign to an input line."""
    import numpy as np
    if len(shape) != 2 or any(type(v) is not int or v <= 0 for v in shape) or math.prod(shape) > 4_000_000:
        raise ValueError("protection needs a bounded positive source-grid shape")
    if type(max_intervals) is not int or not 1 <= max_intervals <= 1_000_000:
        raise ValueError("max_intervals must be bounded and positive")
    mask = np.zeros(shape, bool)
    count = vertices = 0
    for line_index, points in enumerate(polylines):
        if line_index >= 10_000:
            raise ValueError("protection polyline count exceeds its budget")
        p = np.asarray(points, dtype=float)
        if p.ndim != 2 or p.shape[1:] != (2,) or not 2 <= len(p) <= 100_000 or not np.isfinite(p).all():
            raise ValueError("invalid protection polyline")
        if np.any(p < -.500001) or np.any(p[:, 0] > shape[1]-.499999) or np.any(p[:, 1] > shape[0]-.499999):
            raise ValueError("protection polyline leaves its source grid")
        vertices += len(p)
        if vertices > 100_000:
            raise ValueError("protection vertex count exceeds its budget")
        for _i, _a, _b, y, x, _length in _cell_intervals(p, shape, max_intervals-count):
            mask[y, x] = True
            count += 1
        for x, y in p:
            mask[min(shape[0]-1, max(0, math.floor(y+.5))), min(shape[1]-1, max(0, math.floor(x+.5)))] = True
    return mask


def partition_pixel_line(points, glyph_candidate, *, protected=None, ambiguous=None, max_intervals=100_000):
    """Split at exact pixel-cell boundaries, returning ALL original geometry.

    Coordinates use integer pixel centres; pixel cells end at half integers.
    Protected/ambiguous cells override a glyph hypothesis. Consecutive duplicate
    vertices carry no geometric length; the original line must still be stored
    by the caller. Unsplit lines return all their original vertices unchanged.
    """
    import numpy as np
    glyph = _mask(glyph_candidate)
    protection = np.zeros_like(glyph) if protected is None else _mask(protected, glyph.shape)
    uncertainty = np.zeros_like(glyph) if ambiguous is None else _mask(ambiguous, glyph.shape)
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1:] != (2,) or not 2 <= len(p) <= 100_000 or not np.isfinite(p).all():
        raise ValueError("expected a bounded finite 2-D polyline")
    if (np.any(p < -.500001) or np.any(p[:, 0] > glyph.shape[1]-.499999)
            or np.any(p[:, 1] > glyph.shape[0]-.499999)):
        raise ValueError("polyline leaves its ownership source grid")
    if type(max_intervals) is not int or not 1 <= max_intervals <= 1_000_000:
        raise ValueError("max_intervals must be a bounded positive integer")
    lengths = np.linalg.norm(np.diff(p, axis=0), axis=1)
    original_length = float(lengths.sum())
    if original_length <= 1e-10:
        raise ValueError("polyline must have positive length")
    original = p.tolist()
    parts, interval_count = [], 0
    protected_length = ambiguous_length = 0.
    for index, start, end, y, x, span_length in _cell_intervals(p, glyph.shape, max_intervals):
        interval_count += 1
        protected_length += span_length*bool(protection[y, x])
        ambiguous_length += span_length*bool(uncertainty[y, x])
        kind = 'numeric_candidate' if glyph[y, x] and not protection[y, x] and not uncertainty[y, x] else 'retained'
        a, b = _at(original, index, start), _at(original, index, end)
        if parts and parts[-1]['kind'] == kind:
            parts[-1]['points'].append(b)
            parts[-1]['source_edge_locations'].append([index, end])
            parts[-1]['length_px'] += span_length
        else:
            parts.append({'kind': kind, 'points': [a, b], 'source_edge_locations': [[index, start], [index, end]],
                          'length_px': span_length, 'human_approved': False, 'training_eligible': False,
                          'contour_semantics_assigned': False, 'inferred_gap': False})
    if len(parts) == 1:
        parts[0]['points'] = original
        parts[0]['source_edge_locations'] = [[0, 0.]]+[[i, 1.] for i in range(len(p)-1)]
    total = sum(part['length_px'] for part in parts)
    if abs(total-original_length) > max(1e-8, original_length*1e-10):
        raise RuntimeError("partition does not preserve original geometric length")
    return {'schema': 'jap-map-numeric-line-partition/1', 'parts': parts,
            'original_length_px': original_length, 'partitioned_length_px': total,
            'numeric_candidate_length_px': sum(q['length_px'] for q in parts if q['kind'] == 'numeric_candidate'),
            'protected_overlap_length_px': protected_length, 'ambiguous_overlap_length_px': ambiguous_length,
            'intervals_examined': interval_count, 'original_geometry_partitioned_not_simplified': True,
            'original_pixels_modified': False, 'human_approvals': 0,
            'warning': 'All parts remain unapproved; separating a numeric hypothesis does not prove glyph ownership.'}
