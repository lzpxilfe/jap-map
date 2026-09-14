"""Extract genuine terminal pixels around a caller supplied label search box.

Only raw integer-grid observed paths are accepted. Junctions, loops, ROI edges,
and text-search-box endpoints cannot become invented gap endpoints. A box is
context, not an erase mask or proof of an actual missing contour.
"""
from collections import Counter
import math


def terminal_context(paths, box, *, image_shape, radius=32., tangent_length=6., exclude_context_interior=True):
    import numpy as np
    if type(exclude_context_interior) is not bool:
        raise ValueError('exclude_context_interior must be explicit boolean')
    b = np.asarray(box, dtype=float)
    if b.shape != (4,) or not np.isfinite(b).all() or not (b[0] < b[2] and b[1] < b[3]):
        raise ValueError('expected finite ordered context box')
    if len(image_shape) != 2 or any(type(n) is not int or n <= 0 for n in image_shape):
        raise ValueError('expected positive image shape')
    if not math.isfinite(radius) or not 0 < radius <= 96 or not math.isfinite(tangent_length) or not 2 <= tangent_length <= 16:
        raise ValueError('context and tangent lengths must be bounded')
    if not isinstance(paths, (tuple, list)) or len(paths) > 20000:
        raise ValueError('raw path count exceeds budget')
    prepared, degree, identities = [], Counter(), set()
    vertex_count = 0
    for row in paths:
        identity = row['source_path_id']
        if not isinstance(identity, str) or not identity or identity in identities:
            raise ValueError('raw source path IDs must be unique')
        identities.add(identity)
        p = np.asarray(row['points'], dtype=float)
        if p.ndim != 2 or p.shape[1:] != (2,) or len(p) < 2 or not np.isfinite(p).all():
            raise ValueError('invalid raw path')
        vertex_count += len(p)
        if vertex_count > 1000000:
            raise ValueError('raw vertex budget exceeded')
        if np.max(np.abs(p-np.rint(p))) > 1e-5:
            raise ValueError('only unsmoothed integer-grid observed paths are supported')
        p = np.rint(p).astype(int)
        if np.any(p < 0) or np.any(p[:, 0] >= image_shape[1]) or np.any(p[:, 1] >= image_shape[0]):
            raise ValueError('raw path outside image')
        # Count actual incident edges, including a terminal touching the middle
        # of another path. Do not rely solely on a path endpoint count.
        for a, c in zip(p, p[1:]):
            if np.array_equal(a, c):
                continue
            if np.max(np.abs(c-a)) > 1:
                raise ValueError('raw paths must follow adjacent source pixels')
            degree[tuple(a)] += 1; degree[tuple(c)] += 1
        prepared.append((identity, p))
    endpoints, exclusions = [], Counter()
    for identity, p in prepared:
        for end in (0, -1):
            q = p if end == 0 else p[::-1]
            point = q[0]
            outside = np.maximum([b[0]-point[0], b[1]-point[1], point[0]-b[2], point[1]-b[3]], 0)
            if float(np.max(outside)) > radius:
                continue
            if degree[tuple(point)] != 1:
                exclusions['junction_or_loop'] += 1; continue
            if point[0] in (0, image_shape[1]-1) or point[1] in (0, image_shape[0]-1):
                exclusions['image_edge'] += 1; continue
            if exclude_context_interior and b[0] <= point[0] <= b[2] and b[1] <= point[1] <= b[3]:
                exclusions['inside_label_context_not_missing_line_end'] += 1; continue
            walked, inner = 0., q[0].astype(float)
            for a, c in zip(q, q[1:]):
                length = float(np.linalg.norm(c-a))
                if not length: continue
                if walked+length >= tangent_length:
                    inner = a+(c-a)*(tangent_length-walked)/length
                    walked = tangent_length; break
                walked += length; inner = c
            if walked < tangent_length or np.linalg.norm(point-inner) < tangent_length*.5:
                exclusions['short_or_folded_tail'] += 1; continue
            tangent = (point-inner)/np.linalg.norm(point-inner)
            endpoints.append({'source_path_id': identity, 'point': point.tolist(),
                              'outward_tangent': tangent.tolist(), 'source_end': int(end)})
    return {'endpoints': endpoints, 'exclusions': dict(exclusions),
            'source_vertices_examined': vertex_count, 'human_approved': False}


def endpoint_banks(endpoints, box, across_axis, *, maximum_axis_error_degrees=55.):
    """Opposite half-planes; tangent rays must enter the label context box."""
    import numpy as np
    axis = np.asarray(across_axis, dtype=float)
    if axis.shape != (2,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-9:
        raise ValueError('invalid across axis')
    if not 0 < maximum_axis_error_degrees < 90:
        raise ValueError('invalid axis tolerance')
    axis = axis/np.linalg.norm(axis)
    b = np.asarray(box, dtype=float)
    if b.shape != (4,) or not np.isfinite(b).all() or not (b[0] < b[2] and b[1] < b[3]):
        raise ValueError('invalid box')
    center = (b[:2]+b[2:])/2
    banks = [[], []]
    for row in endpoints:
        p, d = np.asarray(row['point'], dtype=float), np.asarray(row['outward_tangent'], dtype=float)
        side = float((p-center) @ axis)
        if abs(side) < .5: continue
        sign = 1 if side < 0 else -1
        if float(d @ axis)*sign < math.cos(math.radians(maximum_axis_error_degrees)): continue
        lo, hi = 0., 96.
        for k in (0, 1):
            if abs(d[k]) < 1e-9:
                if not b[k] <= p[k] <= b[k+2]: hi = -1
            else:
                enter, leave = sorted(((b[k]-p[k])/d[k], (b[k+2]-p[k])/d[k]))
                lo, hi = max(lo, enter), min(hi, leave)
        if hi >= lo:
            banks[0 if side < 0 else 1].append(dict(row))
    return banks
