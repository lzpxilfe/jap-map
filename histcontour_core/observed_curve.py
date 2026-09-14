"""Bounded geometric drafts from grayscale evidence along a supplied route.

The route supplies topology and neighbourhood, not the final stroke centre.
Independent normal profiles locate a *single* local ink lobe. Curvature-
regularized parametric cubic splines fit these centres on supported intervals.
White, clipped, broad, or multi-peak profiles split the fit; no spline crosses
those intervals. The returned sampled geometry is authoritative (a bounded
projection can alter a spline sample). This module supplies neither semantic
contour classification nor human approval.

Coordinates, distances, profile widths, and clearance are in source pixels.
This is a local geometry refiner, not a route finder or a topology repairer.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math


OBSERVED_CURVE_VERSION = "source-supported-continuous-curve/1"


@dataclass(frozen=True)
class ObservedCurveConfig:
    sample_spacing: float = .75
    profile_spacing: float = .25
    profile_radius: float = 4.0
    tangent_reach: float = 3.0
    minimum_contrast: float = 8.0
    maximum_normal_shift: float = 1.75
    support_width_multiplier: float = 1.5
    maximum_stroke_half_width: float = 2.0
    minimum_peak_separation: float = 1.25
    competing_peak_fraction: float = .45
    clearance_fraction: float = .35
    smoothing_strength: float = .6
    minimum_fit_length: float = 3.0
    maximum_samples: int = 100000

    def validate(self):
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
               not math.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError("curve settings must be finite and positive")
        if not .25 <= self.sample_spacing <= 1 or not .1 <= self.profile_spacing <= .5:
            raise ValueError("source support must be sampled at pixel/subpixel spacing")
        if not 2 <= self.profile_radius <= 8 or self.maximum_normal_shift > self.profile_radius / 2:
            raise ValueError("normal movement must remain within a bounded local profile")
        if self.tangent_reach > 8 or self.maximum_stroke_half_width >= self.profile_radius:
            raise ValueError("tangent and stroke support must remain local")
        if self.clearance_fraction > .4 or self.competing_peak_fraction > .7:
            raise ValueError("clearance and competing-peak guards cannot be relaxed this far")
        if type(self.maximum_samples) is not int or self.maximum_samples < 16:
            raise ValueError("maximum_samples must be an integer of at least 16")
        return self


def _geometry(points):
    import numpy as np
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1:] != (2,) or len(p) < 2 or not np.isfinite(p).all():
        raise ValueError("expected at least two finite 2-D route vertices")
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    if distance[-1] <= 1e-9:
        raise ValueError("route must have positive length")
    return p.copy(), distance


def _indices(values, count, name):
    import numpy as np
    result = set()
    for value in values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not 0 <= value < count:
            raise ValueError(f"{name} must contain original-vertex indices")
        result.add(int(value))
    return result


def _at(p, distances, values):
    import numpy as np
    keep = np.r_[True, np.diff(distances) > 1e-12]
    return np.column_stack([np.interp(values, distances[keep], p[keep, axis]) for axis in (0, 1)])


def _frame(p, distances, values, reach):
    import numpy as np
    base = _at(p, distances, values)
    vector = _at(p, distances, np.minimum(values + reach, distances[-1])) - _at(p, distances, np.maximum(values - reach, 0))
    lengths = np.linalg.norm(vector, axis=1)
    # A hairpin may have coincident context points. Its immediate segment is
    # a deterministic fallback, and the profile will still need local evidence.
    bad = lengths < 1e-8
    if bad.any():
        right = _at(p, distances, np.minimum(values[bad] + .1, distances[-1]))
        left = _at(p, distances, np.maximum(values[bad] - .1, 0))
        vector[bad] = right - left
        lengths[bad] = np.linalg.norm(vector[bad], axis=1)
    tangent = vector / np.maximum(lengths[:, None], 1e-12)
    return base, tangent, np.column_stack([-tangent[:, 1], tangent[:, 0]])


def _profiles(image, base, normal, clearance, config):
    import numpy as np
    from scipy.ndimage import gaussian_filter1d, map_coordinates
    from scipy.signal import find_peaks

    offsets = np.linspace(-config.profile_radius, config.profile_radius,
                          1 + math.ceil(2 * config.profile_radius / config.profile_spacing))
    sample = base[:, None, :] + offsets[None, :, None] * normal[:, None, :]
    inside = ((sample[:, :, 0] >= 0) & (sample[:, :, 0] <= image.shape[1] - 1) &
              (sample[:, :, 1] >= 0) & (sample[:, :, 1] <= image.shape[0] - 1)).all(axis=1)
    profiles = map_coordinates(image, [sample[:, :, 1], sample[:, :, 0]], order=1, mode="nearest")
    profiles = gaussian_filter1d(profiles, .28 / (offsets[1] - offsets[0]), axis=1, mode="nearest")
    if clearance is None:
        clear = np.full(len(base), np.inf)
    elif np.ndim(clearance) == 0:
        clear = np.full(len(base), float(clearance))
    else:
        clear = map_coordinates(clearance, [base[:, 1], base[:, 0]], order=1, mode="nearest")
    shifts = np.zeros(len(base)); half_widths = np.zeros(len(base))
    limits = np.zeros(len(base)); contrast_values = np.zeros(len(base))
    supported = np.zeros(len(base), dtype=bool)
    reasons = ["source_border" if not valid else "low_contrast" for valid in inside]
    for i in np.flatnonzero(inside):
        profile = profiles[i]
        background = float(np.quantile(profile, .9))
        contrast = background - float(profile.min())
        contrast_values[i] = contrast
        if contrast < config.minimum_contrast:
            continue
        ink = np.clip((background - profile) / contrast, 0, 1)
        peaks, _ = find_peaks(ink, prominence=.12, plateau_size=True)
        strong = [int(index) for index in peaks if ink[index] >= config.competing_peak_fraction]
        if not strong:
            reasons[i] = "unbounded_ink_profile"; continue
        if any(abs(offsets[a] - offsets[b]) >= config.minimum_peak_separation
               for j, a in enumerate(strong) for b in strong[j + 1:]):
            reasons[i] = "ambiguous_multiple_peaks"; continue
        # find_peaks deliberately ignores endpoints. A separate stroke cut
        # by a profile boundary must not disappear from the ambiguity test,
        # nor set the normalization for a weaker, falsely isolated lobe.
        # Both sides must reach the local paper background before fitting.
        if max(ink[0], ink[-1]) > .2:
            reasons[i] = "ambiguous_truncated_boundary_ink"; continue
        peak = max(strong, key=lambda index: ink[index])
        lobe_cutoff = .2 * ink[peak]
        half_height = .5 * ink[peak]
        left = peak
        while left > 0 and ink[left] > lobe_cutoff:
            left -= 1
        right = peak
        while right < len(ink) - 1 and ink[right] > lobe_cutoff:
            right += 1
        if ink[left] > lobe_cutoff or ink[right] > lobe_cutoff:
            reasons[i] = "unbounded_ink_profile"; continue
        # Use only the selected connected lobe, never an all-profile centroid
        # that can pull a route into the white midpoint of parallel strokes.
        lobe_offsets = offsets[left:right + 1]
        weights = np.maximum(ink[left:right + 1] - lobe_cutoff, 0) ** 2
        centre = float(np.dot(weights, lobe_offsets) / weights.sum())
        above = np.flatnonzero(ink[left:right + 1] >= half_height) + left
        if not len(above) or above[0] == 0 or above[-1] == len(ink) - 1:
            reasons[i] = "unbounded_ink_profile"; continue
        lo, hi = int(above[0]), int(above[-1])
        low = float(np.interp(half_height, ink[lo - 1:lo + 1], offsets[lo - 1:lo + 1]))
        high = float(np.interp(half_height, ink[hi:hi + 2][::-1], offsets[hi:hi + 2][::-1]))
        half_width = (high - low) / 2
        if half_width > config.maximum_stroke_half_width:
            reasons[i] = "broad_ink_profile"; continue
        left_width, right_width = centre - low, high - centre
        if min(left_width, right_width) <= 0 or max(left_width, right_width) > 3 * min(left_width, right_width):
            reasons[i] = "asymmetric_ink_profile"; continue
        limit = min(config.maximum_normal_shift, config.support_width_multiplier * half_width,
                    config.clearance_fraction * clear[i])
        limits[i] = limit; half_widths[i] = half_width; shifts[i] = centre
        if abs(centre) > limit + 1e-10:
            reasons[i] = "centre_outside_movement_bound"; continue
        supported[i] = True; reasons[i] = "single_lobe_source_support"
    return {"supported": supported, "reasons": reasons, "shift": shifts,
            "half_width": half_widths, "limit": limits, "contrast": contrast_values}


def _cubic_fit(s, targets, start, end, strength):
    """Fit a cubic B-spline with exact ends and adaptive curvature penalty.

    No fixed quadratic/bend model is imposed. The knots follow source sample
    distances. The integral of squared second derivative is evaluated with
    two-point Gaussian quadrature (exact per knot interval before adaptation).
    Strong source-supported turns reduce the local penalty to retain valleys.
    """
    import numpy as np
    from scipy.interpolate import BSpline
    from scipy.sparse import bmat, csr_matrix, diags
    from scipy.sparse.linalg import spsolve

    knots = np.r_[np.repeat(s[0], 4), s[1:-1], np.repeat(s[-1], 4)]
    count = len(knots) - 4
    basis = BSpline.design_matrix(s, knots, 3).tocsr()
    first_scale = 3 / (knots[4:count + 3] - knots[1:count])
    first = diags([-first_scale, first_scale], [0, 1], shape=(count - 1, count), format="csr")
    second_scale = 2 / (knots[4:count + 2] - knots[2:count])
    second = diags([-second_scale, second_scale], [0, 1], shape=(count - 2, count - 1), format="csr") @ first
    widths = np.diff(s)
    midpoints = (s[:-1] + s[1:]) / 2
    quad = np.column_stack([midpoints - widths / (2 * np.sqrt(3)), midpoints + widths / (2 * np.sqrt(3))]).ravel()
    derivative = BSpline.design_matrix(quad, knots[2:-2], 1).tocsr() @ second
    reach = min(3., (s[-1] - s[0]) / 4)
    interpolated = lambda positions: np.column_stack([np.interp(positions, s, targets[:, axis]) for axis in (0, 1)])
    centre = interpolated(quad)
    before = centre - interpolated(np.maximum(quad - reach, s[0]))
    after = interpolated(np.minimum(quad + reach, s[-1])) - centre
    denominator = np.linalg.norm(before, axis=1) * np.linalg.norm(after, axis=1)
    cosine = np.sum(before * after, axis=1) / np.maximum(denominator, 1e-12)
    angle = np.arccos(np.clip(cosine, -1, 1))
    adaptive = 1 / (1 + (angle / np.deg2rad(20)) ** 4)
    penalty_weight = np.repeat(widths / 2, 2) * strength * adaptive
    data_weight = np.r_[widths[0] / 2, (widths[:-1] + widths[1:]) / 2, widths[-1] / 2]
    matrix = basis.T @ diags(data_weight) @ basis + derivative.T @ diags(penalty_weight) @ derivative
    constraints = basis[[0, len(s) - 1], :]
    system = bmat([[matrix, constraints.T], [constraints, csr_matrix((2, 2))]], format="csc")
    rhs = np.vstack([basis.T @ (data_weight[:, None] * targets), start, end])
    coefficients = spsolve(system, rhs)[:count]
    return BSpline(knots, coefficients, 3)


def _shape(points):
    import numpy as np
    vectors = np.diff(points, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    vectors = vectors[lengths > 1e-9]; lengths = lengths[lengths > 1e-9]
    if len(vectors) < 2:
        return {"length_pixels": float(lengths.sum()), "total_turn_degrees": 0., "maximum_curvature_per_pixel": 0.}
    angles = np.unwrap(np.arctan2(vectors[:, 1], vectors[:, 0]))
    turns = np.diff(angles)
    curvature = abs(turns) / ((lengths[:-1] + lengths[1:]) / 2)
    return {"length_pixels": float(lengths.sum()), "total_turn_degrees": float(np.degrees(abs(turns)).sum()),
            "maximum_curvature_per_pixel": float(curvature.max())}


def _span_records(s, base, result, edge_supported, edge_reasons, locked_edges):
    import numpy as np
    records = []; first = 0
    keys = [(bool(edge_supported[i]), edge_reasons[i], bool(locked_edges[i])) for i in range(len(s) - 1)]
    for last in range(1, len(keys) + 1):
        if last < len(keys) and keys[last] == keys[first]:
            continue
        observed, reason, locked = keys[first]
        records.append({"start_index": first, "end_index": last,
                        "source_distance_start": float(s[first]), "source_distance_end": float(s[last]),
                        "observed_source_support": observed, "reason": reason, "locked": locked,
                        "changed": bool(np.any(abs(result[first:last + 1] - base[first:last + 1]) > 1e-9))})
        first = last
    return records


def refine_observed_curve(gray, points, *, anchor_indices=(), locked_indices=(), locked_spans=(),
                          fully_approved=False, clearance=None, config=ObservedCurveConfig()):
    """Return copied geometry, evidence spans, and an audit; mutate no inputs.

    ``anchor_indices`` and ``locked_indices`` address original vertices and
    are kept exactly. ``locked_spans`` are inclusive original-index pairs;
    their entire original polyline is retained exactly, including each vertex.
    A ``fully_approved`` route returns its input vertices unchanged, without
    computing source evidence or asserting that approval on the output.

    Optional ``clearance`` is a nonnegative scalar or source-shaped field of
    Euclidean distance to OTHER line centres; the caller must exclude this
    route. It reduces movement to at most ``clearance_fraction`` of that
    distance. It is a movement guard, not a topology/line-identity guarantee.

    Output vertices may be denser than input. ``original_vertex_indices``
    maps each original vertex to its output position; a vertex not locked
    can move. ``spans`` address inclusive output endpoints, and unsupported
    spans retain the original path. Never treat the whole returned route as
    observed ink without checking these span flags.
    """
    import numpy as np
    config.validate()
    original, distances = _geometry(points)
    image = np.asarray(gray, dtype=float)
    if image.ndim != 2 or min(image.shape, default=0) < 2 or not np.isfinite(image).all() or image.min() < 0 or image.max() > 255:
        raise ValueError("expected finite grayscale source pixels in [0,255]")
    if np.any(original < 0) or np.any(original[:, 0] > image.shape[1] - 1) or np.any(original[:, 1] > image.shape[0] - 1):
        raise ValueError("route leaves source image")
    anchors = _indices(anchor_indices, len(original), "anchor_indices")
    locks = _indices(locked_indices, len(original), "locked_indices")
    intervals = []
    for span in locked_spans:
        if not hasattr(span, "__len__") or len(span) != 2:
            raise ValueError("locked_spans must contain inclusive original-index pairs")
        _indices(span, len(original), "locked_spans")
        if span[0] > span[1]:
            raise ValueError("locked span indices must be ordered")
        intervals.append((float(distances[span[0]]), float(distances[span[1]])))
        locks.update(range(span[0], span[1] + 1))
    if type(fully_approved) is not bool:
        raise ValueError("fully_approved must be boolean")
    if clearance is not None:
        clearance = np.asarray(clearance, dtype=float)
        if clearance.ndim not in (0, 2) or (clearance.ndim == 2 and clearance.shape != image.shape) or np.isnan(clearance).any() or np.any(clearance < 0):
            raise ValueError("clearance must be a nonnegative scalar or source-shaped distance field")
    before = _shape(original)
    audit = {"version": OBSERVED_CURVE_VERSION, "changed": False,
             "method": "source_normal_lobes_adaptive_curvature_cubic_spline",
             "before": before, "after": before, "maximum_displacement_pixels": 0.,
             "maximum_normal_displacement_pixels": 0., "maximum_tangent_change_degrees": 0.,
             "endpoint_positions_unchanged": True, "original_coordinates_retained": True,
             "locked_coordinates_unchanged": True, "fitted_interval_count": 0,
             "bounded_sample_count": 0, "abstentions": {}, "source_support_fraction": 0.}
    result = {"version": OBSERVED_CURVE_VERSION, "original_points": original.tolist(),
              "points": original.tolist(), "point_source_distances": distances.tolist(),
              "original_vertex_indices": list(range(len(original))), "spans": [], "audit": audit}
    all_locked = len(locks | anchors | {0, len(original) - 1}) == len(original) and any(a == 0 and b == distances[-1] for a, b in intervals)
    sample_count = math.ceil(distances[-1] / config.sample_spacing) + len(original) + 1
    if fully_approved or all_locked or sample_count > config.maximum_samples:
        reason = "fully_locked_input" if fully_approved or all_locked else "sample_budget_exceeded"
        audit["abstentions"] = {reason: 1}
        result["spans"] = [{"start_index": 0, "end_index": len(original) - 1,
                            "source_distance_start": 0., "source_distance_end": float(distances[-1]),
                            "observed_source_support": False, "reason": reason,
                            "locked": bool(fully_approved or all_locked), "changed": False}]
        return result

    # Retain every input vertex, but avoid near-duplicate uniform knots that
    # would ill-condition a spline. Exact locked coordinates win over a grid.
    grid = np.arange(0, distances[-1], config.sample_spacing)
    nearest = np.searchsorted(distances, grid)
    gap = np.minimum(abs(grid - distances[np.minimum(nearest, len(distances) - 1)]),
                     abs(grid - distances[np.maximum(nearest - 1, 0)]))
    s = np.unique(np.r_[distances, grid[gap > .05]])
    base, tangent, normal = _frame(original, distances, s, config.tangent_reach)
    mapping = np.searchsorted(s, distances)
    base[mapping] = original
    mid = (s[:-1] + s[1:]) / 2
    mid_base, _, mid_normal = _frame(original, distances, mid, config.tangent_reach)
    evidence = _profiles(image, base, normal, clearance, config)
    edge_evidence = _profiles(image, mid_base, mid_normal, clearance, config)
    edge_supported = evidence["supported"][:-1] & evidence["supported"][1:] & edge_evidence["supported"]
    edge_reasons = []
    for index in range(len(mid)):
        failures = [evidence["reasons"][index], edge_evidence["reasons"][index], evidence["reasons"][index + 1]]
        edge_reasons.append(next((reason for reason in failures if reason != "single_lobe_source_support"), "single_lobe_source_support"))
    locked = np.zeros(len(s), dtype=bool); locked_edges = np.zeros(len(mid), dtype=bool)
    for a, b in intervals:
        locked |= (s >= a) & (s <= b)
        locked_edges |= (mid >= a) & (mid <= b)
    locked[mapping[list(anchors | locks | {0, len(original) - 1})]] = True
    fixed = locked.copy()
    # Unsupported edges keep BOTH endpoints, so their complete original
    # polyline cannot shift when a neighbouring supported interval is fitted.
    unsupported = ~edge_supported
    fixed[:-1] |= unsupported; fixed[1:] |= unsupported
    fixed[0] = fixed[-1] = True
    refined = base.copy()
    centre_targets = base + evidence["shift"][:, None] * normal
    boundaries = np.flatnonzero(fixed)
    abstentions = Counter(reason for reason in evidence["reasons"] + edge_evidence["reasons"] if reason != "single_lobe_source_support")
    for first, last in zip(boundaries[:-1], boundaries[1:]):
        if last - first < 3 or s[last] - s[first] < config.minimum_fit_length:
            if last - first > 1:
                abstentions["supported_interval_too_short"] += 1
            continue
        if not edge_supported[first:last].all() or locked_edges[first:last].any():
            continue
        local_s = s[first:last + 1] - s[first]
        targets = centre_targets[first:last + 1]
        spline = _cubic_fit(local_s, targets, base[first], base[last], config.smoothing_strength)
        # Intersect the fitted curve with the original route's normal rays.
        # This changes parameterization, not its locus, and avoids tangential
        # sliding of a bend. A bounded sampled projection is used if needed.
        positions = local_s.copy()
        section = slice(first, last + 1)
        for _ in range(5):
            error = np.sum((spline(positions) - base[section]) * tangent[section], axis=1)
            slope = np.sum(spline(positions, nu=1) * tangent[section], axis=1)
            step = np.divide(error, slope, out=np.zeros_like(error), where=abs(slope) > .2)
            positions = np.clip(positions - np.clip(step, -.5, .5), 0, local_s[-1])
            positions[0], positions[-1] = 0, local_s[-1]
        fit = spline(positions)
        if not np.isfinite(fit).all() or np.any(np.diff(positions) <= 0):
            abstentions["nonmonotone_curve_parameter"] += 1
            continue
        shift = np.sum((fit - base[section]) * normal[section], axis=1)
        limit = evidence["limit"][section]
        bounded = np.clip(shift, -limit, limit)
        # Retain the selected dark lobe if regularization tries to leave it.
        # Fixed anchors can deliberately lie off its centre; taper this guard
        # to zero at exact ends so no approved coordinate is moved.
        taper = np.minimum(1., np.minimum(local_s, local_s[-1] - local_s) / 2)
        centre = evidence["shift"][section] * taper
        tolerance = evidence["half_width"][section] * .65 + abs(evidence["shift"][section]) * (1 - taper)
        lower = np.maximum(-limit, centre - tolerance)
        upper = np.minimum(limit, centre + tolerance)
        bounded = np.clip(bounded, lower, upper)
        bounded[0] = bounded[-1] = 0
        audit["bounded_sample_count"] += int(np.count_nonzero(abs(bounded - shift) > 1e-5))
        refined[section] = base[section] + bounded[:, None] * normal[section]
        refined[first], refined[last] = base[first], base[last]
        audit["fitted_interval_count"] += 1
    refined[locked] = base[locked]
    displacement = np.linalg.norm(refined - base, axis=1)
    old_vectors = np.diff(base, axis=0); new_vectors = np.diff(refined, axis=0)
    cosine = np.sum(old_vectors * new_vectors, axis=1) / np.maximum(np.linalg.norm(old_vectors, axis=1) * np.linalg.norm(new_vectors, axis=1), 1e-12)
    turn_change = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    audit.update(changed=bool(displacement.max() > 1e-9), after=_shape(refined),
                 maximum_displacement_pixels=float(displacement.max()),
                 maximum_normal_displacement_pixels=float(displacement.max()),
                 maximum_tangent_change_degrees=float(turn_change.max()),
                 source_support_fraction=float(np.dot(np.diff(s), edge_supported) / distances[-1]),
                 source_profile_count=int(len(s) + len(mid)),
                 supported_profile_count=int(evidence["supported"].sum() + edge_evidence["supported"].sum()),
                 abstentions=dict(sorted(abstentions.items())),
                 geometry_representation="sampled_piecewise_curve_with_exact_locked_and_unsupported_spans",
                 limitations=["local_ink_support_does_not_establish_contour_identity",
                              "input_route_topology_and_large_missing_bends_are_not_recovered",
                              "sampled_bounds_do_not_prove_global_topology_or_subpixel_clearance"])
    result.update(points=refined.tolist(), point_source_distances=s.tolist(), original_vertex_indices=mapping.tolist(),
                  spans=_span_records(s, base, refined, edge_supported, edge_reasons, locked_edges))
    return result
