"""Conservative, feedback-informed geometry drafts in source-pixel coordinates.

No contour class is inferred here.  Fixed-endpoint smoothing and relocation
of short source tails are distinct results; the latter is never append-only.
Numeral blanks are not observed ink, and weak/ambiguous support causes
abstention rather than an invented continuation.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math

REFINEMENT_VERSION = "connected-ink-gap-refinement/1"


@dataclass(frozen=True)
class GapRefinementConfig:
    maximum_gap_pixels: float = 16.0
    maximum_fixed_displacement: float = .35
    fixed_displacement_fraction: float = .05
    context_reach: float = 12.0
    profile_radius: float = 4.0
    minimum_contrast: float = 8.0
    maximum_centre_shift: float = 2.5
    minimum_samples_per_side: int = 5
    maximum_fit_rms: float = .65
    reanchor_threshold: float = .55
    maximum_tip_shift: float = 2.5
    maximum_tail_trim: float = 8.0
    maximum_model_slope: float = 1.25
    sample_spacing: float = .25

    def validate(self):
        values = asdict(self)
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in values.values()):
            raise ValueError("refinement settings must be finite and positive")
        if self.maximum_gap_pixels > 16 or self.maximum_tail_trim > 16:
            raise ValueError("automatic refinement must remain within local 16-pixel contracts")
        if self.profile_radius > 8 or self.context_reach > 32 or self.maximum_tip_shift > 4:
            raise ValueError("refinement support must remain local")
        if type(self.minimum_samples_per_side) is not int or self.minimum_samples_per_side < 3:
            raise ValueError("at least three independent support samples per side are required")
        if self.maximum_fixed_displacement > .5 or self.fixed_displacement_fraction > .1:
            raise ValueError("fixed-endpoint edits must remain subtle")
        return self


def _points(points):
    import numpy as np
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1:] != (2,) or len(p) < 2 or not np.isfinite(p).all():
        raise ValueError("expected at least two finite 2-D points")
    if float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) <= 1e-9:
        raise ValueError("zero-length geometry")
    return p.copy()


def _resample(points, spacing=.25):
    import numpy as np
    p = _points(points)
    distances = np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    keep = np.r_[True, np.diff(distances) > 1e-9]
    p, distances = p[keep], distances[keep]
    values = np.linspace(0, distances[-1], max(3, math.ceil(distances[-1]/spacing)+1))
    return np.c_[np.interp(values, distances, p[:, 0]), np.interp(values, distances, p[:, 1])]


def shape_audit(points):
    import numpy as np
    p = _points(points)
    vectors = np.diff(p, axis=0)
    vectors = vectors[np.linalg.norm(vectors, axis=1) > 1e-8]
    angles = np.unwrap(np.arctan2(vectors[:, 1], vectors[:, 0]))
    turns = np.degrees(np.diff(angles))
    signs = np.sign(turns[abs(turns) > .05])
    chord = p[-1]-p[0]; gap = float(np.linalg.norm(chord))
    if gap <= 1e-9:
        raise ValueError("closed paths are not gap connectors")
    axis = chord/gap; normal = np.array([-axis[1], axis[0]])
    progress = (p-p[0])@axis
    return {"length_pixels": float(np.linalg.norm(vectors, axis=1).sum()),
            "gap_pixels": gap, "detour_ratio": float(np.linalg.norm(vectors, axis=1).sum()/gap),
            "inflection_count": int(np.count_nonzero(np.diff(signs))) if len(signs) else 0,
            "total_turn_degrees": float(abs(turns).sum()),
            "maximum_chord_deviation_pixels": float(abs((p-p[0])@normal).max()),
            "forward_progress_monotone": bool(np.all(np.diff(progress) > -1e-8))}


def regularize_gap_path(points, config=GapRefinementConfig()):
    """Remove only subpixel inflections while preserving both exact endpoints.

    A genuine, larger S-bend or a long label blank is left unchanged.  This
    function accepts arbitrary polylines, not just evenly sampled cubics.
    """
    import numpy as np
    config.validate(); original = _points(points)
    before = shape_audit(original)
    result = {"version": REFINEMENT_VERSION, "changed": False, "kind": "fixed_endpoints",
              "before": before, "after": before, "maximum_displacement_pixels": 0.,
              "endpoint_positions_unchanged": True}
    if before["gap_pixels"] > config.maximum_gap_pixels or not before["forward_progress_monotone"] or before["detour_ratio"] > 1.2:
        result["reason"] = "not_a_short_forward_gap"
        return original.tolist(), result
    if before["inflection_count"] == 0:
        result["reason"] = "no_unnecessary_inflection"
        return original.tolist(), result
    dense = _resample(original, config.sample_spacing/2)
    a, b = original[0], original[-1]; chord = b-a; length = float(np.linalg.norm(chord))
    axis = chord/length; normal = np.array([-axis[1], axis[0]])
    t = np.clip((dense-a)@axis/length, 0, 1); phi = 4*t*(1-t)
    amplitude = float(np.dot(phi, (dense-a)@normal)/max(np.dot(phi, phi), 1e-12))
    u = np.linspace(0, 1, max(33, math.ceil(length/config.sample_spacing)+1))
    new_v = 4*u*(1-u)*amplitude
    refined = a+u[:, None]*chord+new_v[:, None]*normal
    refined[0], refined[-1] = a, b
    # The saved geometry and input are both polylines. Their lateral
    # difference is linear between the union of their projected vertices,
    # so its exact maximum occurs at one of those knots (not a sample grid).
    original_t = (original-a)@axis/length
    original_v = (original-a)@normal
    error = float(max(abs(np.interp(original_t,u,new_v)-original_v).max(),
                      abs(new_v-np.interp(u,original_t,original_v)).max()))
    limit = min(config.maximum_fixed_displacement, length*config.fixed_displacement_fraction)
    if error > limit:
        result.update(reason="shape_change_exceeds_subpixel_limit", proposed_displacement_pixels=error)
        return original.tolist(), result
    after = shape_audit(refined)
    if not after["forward_progress_monotone"] or after["inflection_count"]:
        result["reason"] = "regularized_curve_not_simple"
        return original.tolist(), result
    result.update(changed=True, reason="subpixel_inflection_removed", after=after,
                  maximum_displacement_pixels=error, signed_bend_pixels=amplitude,
                  method="one_quadratic_bend_at_exact_endpoints")
    return refined.tolist(), result


def _ordered_tail(path, endpoint):
    import numpy as np
    p = _points(path); end = np.asarray(endpoint, dtype=float)
    matches = [i for i in (0, -1) if np.linalg.norm(p[i]-end) < 1e-5]
    if len(matches) != 1:
        raise ValueError("gap tip must be a unique source-line endpoint")
    return (p if matches[0] == 0 else p[::-1]).copy()


def _at_distance(ordered, distance):
    import numpy as np
    lengths = np.linalg.norm(np.diff(ordered, axis=0), axis=1)
    if distance >= lengths.sum()-1e-6:
        return None
    for i, length in enumerate(lengths):
        if length <= 1e-9:
            continue
        if distance <= length:
            direction = (ordered[i+1]-ordered[i])/length
            return ordered[i]+distance*direction, direction
        distance -= length
    return None


def _ink_samples(gray, ordered, config):
    import numpy as np
    from scipy.ndimage import gaussian_filter1d, map_coordinates
    from scipy.signal import find_peaks
    samples, ambiguous = [], 0
    offsets = np.linspace(-config.profile_radius, config.profile_radius, 33)
    for distance in np.arange(2., config.context_reach+.01, 1.):
        located = _at_distance(ordered, distance)
        if located is None:break
        centre, direction = located; normal = np.array([-direction[1], direction[0]])
        points = centre+offsets[:, None]*normal
        if (points[:, 0].min() < 0 or points[:, 1].min() < 0 or
                points[:, 0].max() >= gray.shape[1]-1 or points[:, 1].max() >= gray.shape[0]-1):
            continue
        profile = map_coordinates(gray, [points[:, 1], points[:, 0]], order=1)
        background = float(np.quantile(profile, .9)); contrast = background-float(profile.min())
        if contrast < config.minimum_contrast:continue
        ink = np.clip((background-profile)/contrast, 0, 1)
        smoothed = gaussian_filter1d(ink, 1.)
        peaks, _ = find_peaks(smoothed, prominence=.18, distance=5)
        strong = [p for p in peaks if smoothed[p] >= .65*smoothed.max()]
        if any(abs(offsets[a]-offsets[b]) >= 1.5 for a in strong for b in strong if a < b):
            ambiguous += 1;continue
        weights = ink**2*np.exp(-.5*(offsets/2.5)**2)
        if weights.sum() <= 1e-9:continue
        shift = float(np.dot(weights, offsets)/weights.sum())
        if abs(shift) > config.maximum_centre_shift:continue
        samples.append({"point": centre+shift*normal, "reach": float(distance),
                        "weight": min(1., contrast/40.), "contrast": contrast})
    return samples, ambiguous


def _fit_model(samples, a, axis, normal, gap):
    import numpy as np
    points = np.array([s["point"] for s in samples]); weights = np.array([s["weight"] for s in samples])
    groups = np.array([s["side"] for s in samples])
    for side in (0, 1):weights[groups == side] /= weights[groups == side].sum()
    u = (points-a)@axis; v = (points-a)@normal; scale = max(12., gap)
    z = (u-gap/2)/scale; models = []
    for degree in (1, 2):
        matrix = np.vander(z, degree+1); robust = weights.copy()
        for _ in range(4):
            coeff = np.linalg.lstsq(matrix*np.sqrt(robust[:, None]), v*np.sqrt(robust), rcond=None)[0]
            residual = v-matrix@coeff
            robust = weights*np.minimum(1., .5/np.maximum(abs(residual), 1e-9))
        rms = float(np.sqrt(np.average(residual**2, weights=weights)))
        models.append((rms, coeff))
    selected = models[1] if models[1][0] < .8*models[0][0] and models[0][0] > .2 else models[0]
    rms, coeff = selected
    return {"coefficients": coeff, "rms": rms, "scale": scale, "u": u,
            "v": v, "points": points, "weights": weights,
            "degree": len(coeff)-1, "linear_rms": models[0][0], "quadratic_rms": models[1][0]}


def propose_gap_refinement(gray, source_path, target_path, connector, *, config=GapRefinementConfig()):
    """Propose a conservative local edit, with explicit tail-cut requirements.

    Sampling uses connected source tails and excludes the inferred gap.  A
    caller must additionally screen third-line contacts and other proposals,
    lock every human-reviewed case, and obtain approval for the new geometry.
    """
    import numpy as np
    config.validate(); original = _points(connector)
    image = np.asarray(gray, dtype=float)
    if image.ndim != 2 or min(image.shape, default=0) < 2 or not np.isfinite(image).all() or image.min() < 0 or image.max() > 255:
        raise ValueError("expected finite grayscale source pixels in [0,255]")
    if (original[:, 0].min() < 0 or original[:, 1].min() < 0 or
            original[:, 0].max() >= image.shape[1] or original[:, 1].max() >= image.shape[0]):
        raise ValueError("original connector leaves source pixels")
    a, b = original[0], original[-1]; gap = float(np.linalg.norm(b-a))
    if gap <= 1e-9:raise ValueError("gap is closed")
    tails = [_ordered_tail(source_path, a), _ordered_tail(target_path, b)]
    regularized, smooth_audit = regularize_gap_path(original, config)
    result = {"version": REFINEMENT_VERSION, "status": "unchanged", "kind": "fixed_endpoints",
              "points": original.tolist(), "human_approved": False, "model_fitted": False,
              "requires_source_tail_replacement": False, "reasons": [], "smoothing": smooth_audit}
    if gap > config.maximum_gap_pixels:
        result.update(status="context_review_required", reasons=["long_gap_requires_regional_anchor_review"])
        return result
    if not shape_audit(original)["forward_progress_monotone"]:
        result.update(status="context_review_required", reasons=["original_path_doubles_back"])
        return result
    axis = (b-a)/gap; normal = np.array([-axis[1], axis[0]])
    samples, counts, ambiguities = [], [], []
    for side, tail in enumerate(tails):
        found, ambiguous = _ink_samples(image, tail, config)
        # Exclude the inferred gap. Connected source support is still not a
        # semantic guarantee: a source line can itself be a glyph or river.
        found = [s for s in found if ((s["point"]-a)@axis < -.5 if side == 0 else (s["point"]-a)@axis > gap+.5)]
        counts.append(len(found));ambiguities.append(ambiguous)
        samples.extend({**s, "side": side} for s in found)
    result["support"] = {"samples_per_side": counts, "ambiguous_profiles_per_side": ambiguities,
                         "gap_pixels_sampled_as_ink": 0}
    if min(counts) < config.minimum_samples_per_side or max(ambiguities) > 2:
        result.update(status="context_review_required", reasons=["insufficient_or_ambiguous_connected_ink"])
        return result
    model = _fit_model(samples, a, axis, normal, gap)
    value = lambda u: np.polyval(model["coefficients"], (np.asarray(u)-gap/2)/model["scale"])
    slope = lambda u: np.polyval(np.polyder(model["coefficients"]), (np.asarray(u)-gap/2)/model["scale"])/model["scale"]
    shifts = np.array([value(0.), value(gap)])
    result["support"].update(degree=model["degree"], fit_rms_pixels=model["rms"],
        predicted_tip_normal_shifts=shifts.tolist(), linear_rms=model["linear_rms"], quadratic_rms=model["quadratic_rms"])
    if model["rms"] > config.maximum_fit_rms or max(abs(slope(np.linspace(-4,gap+4,33)))) > config.maximum_model_slope:
        result.update(status="context_review_required", reasons=["source_flow_fit_unstable"])
        return result
    near = [s for s in samples if s["reach"] <= 8]
    if all(sum(s["side"] == side for s in near) >= 3 for side in (0,1)):
        shorter = _fit_model(near, a, axis, normal, gap)
        other = np.polyval(shorter["coefficients"], (np.array([0.,gap])-gap/2)/shorter["scale"])
        stability = float(max(abs(other-shifts)))
        result["support"]["multi_reach_tip_disagreement_pixels"] = stability
        if stability > .75:
            result.update(status="context_review_required", reasons=["ink_direction_changes_with_context_reach"])
            return result
    if max(abs(shifts)) <= config.reanchor_threshold:
        if smooth_audit["changed"]:
            result.update(status="proposed", points=regularized, reasons=["remove_subpixel_inflection"])
        else:result["reasons"] = ["current_tips_and_shape_are_supported"]
        return result
    if max(abs(shifts)) > config.maximum_tip_shift:
        result.update(status="context_review_required", reasons=["large_tip_relocation_requires_human_context"])
        return result
    # Fitting two distant tails can displace the inferred middle to the
    # opposite side of a valley bend. That is a competing shape hypothesis,
    # not merely endpoint cleanup, even when the least-squares RMS is low.
    original_u = (original-a)@axis
    original_v = (original-a)@normal
    middle_u = np.linspace(.2*gap, .8*gap, 25)
    existing_bend = np.interp(middle_u, original_u, original_v)
    predicted_bend = value(middle_u)
    if np.any((abs(existing_bend) > .3) & (existing_bend*predicted_bend < 0)
              & (abs(existing_bend-predicted_bend) > config.maximum_fixed_displacement)):
        result.update(status="context_review_required", reasons=["local_fit_reverses_existing_bend"])
        return result
    cuts, trims, tangents = [], [], []
    for side, tail in enumerate(tails):
        choices = []
        for trim in np.arange(2., config.maximum_tail_trim+.01, 2.):
            located = _at_distance(tail, trim)
            if located is None:continue
            cut, inward = located; direction = -inward if side == 0 else inward
            u = float((cut-a)@axis); v = float((cut-a)@normal); forward = float(direction@axis)
            if forward <= .25 or not (u < -.5 if side == 0 else u > gap+.5):continue
            tangent = float(direction@normal/forward)
            mismatch = abs(math.degrees(math.atan(tangent)-math.atan(float(slope(u)))))
            cost = abs(v-float(value(u)))+.02*mismatch+.01*trim
            choices.append((cost, float(trim), cut, u, v, tangent))
        if not choices:
            result.update(status="context_review_required", reasons=["no_stable_local_tail_join"])
            return result
        _, trim, cut, u, v, tangent = min(choices, key=lambda c:(c[0],c[1]))
        cuts.append((cut,u,v));trims.append(trim);tangents.append(tangent)
    def hermite(u0,u1,v0,v1,m0,m1):
        t = np.linspace(0,1,max(9,math.ceil((u1-u0)/config.sample_spacing)+1));du=u1-u0
        v = (2*t**3-3*t**2+1)*v0+(t**3-2*t**2+t)*du*m0+(-2*t**3+3*t**2)*v1+(t**3-t**2)*du*m1
        return np.c_[u0+t*du,v]
    central_u = np.linspace(0,gap,max(17,math.ceil(gap/config.sample_spacing)+1))
    left = hermite(cuts[0][1],0.,cuts[0][2],float(value(0.)),tangents[0],float(slope(0.)))
    centre = np.c_[central_u,value(central_u)]
    right = hermite(gap,cuts[1][1],float(value(gap)),cuts[1][2],float(slope(gap)),tangents[1])
    uv = np.vstack([left,centre[1:],right[1:]])
    patch = a+uv[:,0,None]*axis+uv[:,1,None]*normal
    patch[0],patch[-1] = cuts[0][0],cuts[1][0]
    audit = shape_audit(patch)
    if (not audit["forward_progress_monotone"] or audit["maximum_chord_deviation_pixels"] > 4 or
            audit["detour_ratio"] > 1.2 or audit["total_turn_degrees"] > 75):
        result.update(status="context_review_required", reasons=["local_curve_not_sufficiently_simple"])
        return result
    if (patch[:,0].min()<0 or patch[:,1].min()<0 or patch[:,0].max()>=image.shape[1] or patch[:,1].max()>=image.shape[0]):
        result.update(status="context_review_required", reasons=["refinement_leaves_source"])
        return result
    result.update(status="proposed",kind="local_tail_replacement",points=patch.tolist(),
        reasons=["stable_connected_ink_supports_relocated_local_joins"],requires_source_tail_replacement=True,
        source_tail_trim_pixels=trims[0],target_tail_trim_pixels=trims[1],
        join_points_pixels=[cuts[0][0].tolist(),cuts[1][0].tolist()],curve_audit=audit,
        join_tangent_continuity=True,append_only_safe=False)
    return result
