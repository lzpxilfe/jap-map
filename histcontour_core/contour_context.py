"""Experimental local line-family descriptors; never edits candidate geometry.

Direction coherence is evidence, not a contour detector: roads can be parallel
too. All distances are native raster pixels. Models trained with AI labels must
remain research-only until an independent human contour benchmark exists.
"""

from __future__ import annotations

import math

from .segment_review import FEATURE_NAMES as BASE_FEATURE_NAMES

CONTEXT_SCHEMA = "ink-contour-neighborhood/1"
RADII = (12, 36, 84)
POSITIONS = (0.1, 0.3, 0.5, 0.7, 0.9)
NEIGHBOR_FEATURE_NAMES = tuple(
    f"{name}_r{radius}" for radius in RADII
    for name in ("coherence", "normal_alignment", "alignment_spread", "ink_fraction")
) + (
    "profile_peaks", "profile_peaks_spread", "profile_near_peaks", "profile_far_peaks",
    "profile_bilateral", "profile_peak_spacing_cv", "profile_central_width",
    "profile_contrast", "profile_valid_fraction", "smooth_turn_mean", "smooth_turn_max",
)
CONTEXT_FEATURE_NAMES = BASE_FEATURE_NAMES + NEIGHBOR_FEATURE_NAMES


def _validated_points(points, width, height):
    import numpy as np
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("candidate must contain at least two finite pixel points")
    if (values[:, 0] < -0.01).any() or (values[:, 0] > width-1+0.01).any() or (values[:, 1] < -0.01).any() or (values[:, 1] > height-1+0.01).any():
        raise ValueError("candidate pixel points lie outside the source raster")
    lengths = np.linalg.norm(np.diff(values, axis=0), axis=1)
    keep = np.r_[True, lengths > 1e-9]
    values = values[keep]
    if len(values) < 2:
        raise ValueError("candidate needs positive length")
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(values, axis=0), axis=1))]
    return values, distance


class NeighborhoodFeatures:
    """Cache raster derivatives once, then describe many existing polylines."""

    def __init__(self, grayscale):
        import numpy as np
        from scipy.ndimage import gaussian_filter, uniform_filter
        values = np.asarray(grayscale)
        if values.ndim != 2 or min(values.shape, default=0) < 2 or not np.isfinite(values).all():
            raise ValueError("context needs a non-empty finite 2-D grayscale raster")
        if values.min() < 0 or values.max() > 255:
            raise ValueError("context grayscale must use the [0, 255] intensity range")
        self.height, self.width = values.shape
        gray = values.astype(np.float32) / 255.0
        # Local background removal preserves faint marks without changing inputs.
        background = gaussian_filter(gray, 12.0, mode="reflect")
        ink = np.clip((background-gray) / np.maximum(background, 0.15), 0, 1)
        self.ink = ink
        smooth = gaussian_filter(ink, 1.0, mode="reflect")
        gy, gx = np.gradient(smooth)
        self.tensors = {}
        for radius in RADII:
            size = 2*radius+1
            self.tensors[radius] = tuple(uniform_filter(channel, size, mode="reflect") for channel in (gx*gx, gx*gy, gy*gy, ink))

    def describe(self, points):
        import numpy as np
        from scipy.ndimage import map_coordinates
        from scipy.signal import find_peaks
        values, distance = _validated_points(points, self.width, self.height)
        total = distance[-1]

        def at(positions):
            return np.stack([np.interp(positions, distance, values[:, axis]) for axis in (0, 1)], axis=-1)

        locations = np.asarray(POSITIONS)*total
        centers = at(locations)
        before, after = at(np.maximum(0, locations-8)), at(np.minimum(total, locations+8))
        tangent = after-before
        norms = np.linalg.norm(tangent, axis=1)
        # A tiny closed loop may return to the same point over a 16-px window.
        bad = norms < 1e-6
        if bad.any():
            tangent[bad] = at(np.minimum(total, locations[bad]+1))-at(np.maximum(0, locations[bad]-1))
            norms = np.linalg.norm(tangent, axis=1)
        tangent /= np.maximum(norms[:, None], 1e-6)
        normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
        coords = [centers[:, 1], centers[:, 0]]
        features = {}
        for radius, channels in self.tensors.items():
            xx, xy, yy, ink = [map_coordinates(channel, coords, order=1, mode="nearest") for channel in channels]
            trace = xx+yy+1e-8
            coherence = np.sqrt((xx-yy)**2+4*xy*xy)/trace
            nx, ny = normal[:, 0], normal[:, 1]
            alignment = ((nx*nx-ny*ny)*(xx-yy)+4*nx*ny*xy)/trace
            features.update({f"coherence_r{radius}": float(np.mean(coherence)),
                             f"normal_alignment_r{radius}": float(np.mean(alignment)),
                             f"alignment_spread_r{radius}": float(np.std(alignment)),
                             f"ink_fraction_r{radius}": float(np.mean(ink))})

        offsets = np.arange(-48, 49, dtype=np.float32)
        along = np.asarray([-6, -3, 0, 3, 6], dtype=np.float32)
        grid = centers[:, None, None, :] + normal[:, None, None, :]*offsets[None, :, None, None] + tangent[:, None, None, :]*along[None, None, :, None]
        valid = (grid[..., 0] >= 0) & (grid[..., 0] <= self.width-1) & (grid[..., 1] >= 0) & (grid[..., 1] <= self.height-1)
        profiles = map_coordinates(self.ink, [grid[..., 1], grid[..., 0]], order=1, mode="constant", cval=0).mean(axis=2)
        profile_rows = []
        for profile, validity in zip(profiles, valid):
            peak_indices, _ = find_peaks(profile, height=0.10, prominence=0.07, distance=3)
            peak_offsets = offsets[peak_indices]
            neighbors = peak_offsets[np.abs(peak_offsets) > 4]
            spacings = np.diff(peak_offsets)
            spacing_cv = float(np.std(spacings)/(np.mean(spacings)+1e-6)) if len(spacings) > 1 else 0.0
            left, right = 48, 48
            cutoff = max(0.08, float(profile[48])*0.5)
            if profile[48] >= cutoff:
                while left > 0 and profile[left-1] >= cutoff:
                    left -= 1
                while right < 96 and profile[right+1] >= cutoff:
                    right += 1
                central_width = right-left+1
            else:
                central_width = 0
            profile_rows.append((len(neighbors), int((np.abs(neighbors) <= 16).sum()), int((np.abs(neighbors) > 16).sum()),
                                 float((neighbors < 0).any() and (neighbors > 0).any()), spacing_cv, central_width,
                                 float(profile.max()-profile.min()), float(validity.mean())))
        rows = np.asarray(profile_rows, dtype=float)
        names = ("profile_peaks", "profile_near_peaks", "profile_far_peaks", "profile_bilateral", "profile_peak_spacing_cv", "profile_central_width", "profile_contrast", "profile_valid_fraction")
        features.update(zip(names, map(float, rows.mean(axis=0))))
        features["profile_peaks_spread"] = float(rows[:, 0].std())
        # Coarsened turn angles avoid confounding one-pixel staircase corners.
        smooth_points = at(np.linspace(0, total, max(3, min(65, int(math.ceil(total/8))+1))))
        vectors = np.diff(smooth_points, axis=0)
        lengths = np.linalg.norm(vectors, axis=1)
        dots = np.sum(vectors[:-1]*vectors[1:], axis=1)/(lengths[:-1]*lengths[1:]+1e-8)
        turns = np.arccos(np.clip(dots, -1, 1))/np.pi
        features["smooth_turn_mean"] = float(turns.mean())
        features["smooth_turn_max"] = float(turns.max())
        if set(features) != set(NEIGHBOR_FEATURE_NAMES) or not all(math.isfinite(value) for value in features.values()):
            raise ValueError("non-finite or incomplete neighborhood descriptor")
        return features


def fit_classifier(records, feature_names=CONTEXT_FEATURE_NAMES, *, l2=0.1):
    """Deterministic, class-balanced ridge logistic model; train-only scaling."""
    import numpy as np
    from scipy.optimize import minimize
    from scipy.special import expit
    names = tuple(feature_names)
    if names not in (BASE_FEATURE_NAMES, CONTEXT_FEATURE_NAMES) or not math.isfinite(l2) or l2 <= 0:
        raise ValueError("unsupported feature set or regularization")
    matrix = np.asarray([[row[name] for name in names] for row in records], dtype=float)
    labels = np.asarray([row["class"] == "contour" for row in records], dtype=float)
    if matrix.ndim != 2 or not np.isfinite(matrix).all() or min(labels.sum(), len(labels)-labels.sum()) < 2:
        raise ValueError("training requires finite descriptors and at least two rows of each class")
    if any(row["class"] not in ("contour", "text", "road_river", "symbol") for row in records):
        raise ValueError("ambiguous or unreviewed labels may not enter training")
    means = matrix.mean(axis=0)
    scales = np.maximum(matrix.std(axis=0), 1e-6)
    z = (matrix-means)/scales
    weights = np.where(labels == 1, len(labels)/(2*labels.sum()), len(labels)/(2*(len(labels)-labels.sum())))

    def objective(parameters):
        intercept, coefficients = parameters[0], parameters[1:]
        logits = intercept+z@coefficients
        loss = np.mean(weights*(np.logaddexp(0, logits)-labels*logits)) + l2*float(coefficients@coefficients)/2
        error = weights*(expit(logits)-labels)
        gradient = np.r_[error.mean(), z.T@error/len(labels)+l2*coefficients]
        return float(loss), gradient

    fit = minimize(objective, np.zeros(len(names)+1), method="L-BFGS-B", jac=True, options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8})
    if not fit.success or not np.isfinite(fit.x).all():
        raise ValueError(f"classifier optimization did not converge: {fit.message}")
    return {"feature_schema": CONTEXT_SCHEMA, "feature_names": list(names), "means": means.tolist(), "scales": scales.tolist(),
            "coefficients": fit.x[1:].tolist(), "intercept": float(fit.x[0]), "l2": l2,
            "training_count": len(records), "training_contours": int(labels.sum()), "review_only": True}


def validate_classifier(model):
    if model.get("feature_schema") != CONTEXT_SCHEMA or tuple(model.get("feature_names", ())) not in (BASE_FEATURE_NAMES, CONTEXT_FEATURE_NAMES):
        raise ValueError("unsupported neighborhood model schema or feature order")
    count = len(model["feature_names"])
    for name in ("means", "scales", "coefficients"):
        values = model.get(name, [])
        if len(values) != count or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
            raise ValueError("model arrays must have matching finite numeric parameters")
    if any(value <= 0 for value in model["scales"]) or isinstance(model.get("intercept"), bool) or not isinstance(model.get("intercept"), (int, float)) or not math.isfinite(model["intercept"]):
        raise ValueError("model scales/intercept are invalid")
    return model


def probability(model, record):
    # Validate once at file load for bulk use, and reject non-finite descriptors.
    logit = float(model["intercept"])
    for name, mean, scale, coefficient in zip(model["feature_names"], model["means"], model["scales"], model["coefficients"]):
        value = float(record[name])
        if not math.isfinite(value):
            raise ValueError("descriptor values must be finite")
        logit += coefficient*(value-mean)/scale
    if not math.isfinite(logit):
        raise ValueError("model produced a non-finite score logit")
    if logit >= 0:
        return 1/(1+math.exp(-logit))
    exponent = math.exp(logit)
    return exponent/(1+exponent)


def recall_first_threshold(rows, *, target_recall=0.95):
    """Choose a threshold using validation rows only, never an outer test fold."""
    if not 0 < target_recall <= 1:
        raise ValueError("target recall must be in (0, 1]")
    scores = []
    for row in rows:
        score = row["score"]
        if not isinstance(score, (float, int)) or isinstance(score, bool) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("validation scores must be finite in [0, 1]")
        if row["class"] == "contour":
            scores.append(float(score))
    if not scores:
        raise ValueError("recall calibration needs positive validation rows")
    scores.sort()
    maximum_misses = len(scores)-math.ceil(target_recall*len(scores))
    return scores[maximum_misses]
