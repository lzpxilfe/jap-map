"""Optional-dependency grayscale line-candidate extraction for review workflows.

The backend deliberately produces *candidates*, not asserted contours.  On
monochrome historical maps, text, roads, and contours share ink and must remain
distinguishable during human review or later supervised learning.
"""

from __future__ import annotations

from dataclasses import dataclass


class GrayscaleBackendUnavailable(RuntimeError):
    """Raised when the optional numerical image stack is unavailable."""


@dataclass(frozen=True)
class GrayscaleCandidateSettings:
    background_sigma_px: float = 18.0
    sauvola_window_px: int = 41
    sauvola_k: float = 0.2
    ridge_scales_px: tuple[float, ...] = (0.7, 1.0, 1.4)
    ridge_quantile: float = 0.55
    minimum_component_pixels: int = 24

    def __post_init__(self) -> None:
        if self.background_sigma_px <= 0 or self.sauvola_window_px < 3 or self.sauvola_window_px % 2 == 0:
            raise ValueError("background_sigma_px must be positive and sauvola_window_px must be odd and at least 3")
        if not 0 < self.sauvola_k < 1 or not self.ridge_scales_px or any(scale <= 0 for scale in self.ridge_scales_px):
            raise ValueError("sauvola_k and ridge_scales_px must be positive")
        if not 0 < self.ridge_quantile < 1 or self.minimum_component_pixels < 1:
            raise ValueError("ridge_quantile must be in (0, 1) and minimum_component_pixels must be positive")


@dataclass(frozen=True)
class GrayscaleCandidateResult:
    """A review mask plus continuous ridge likelihood and descriptive metrics."""

    candidate_mask: object
    likelihood: object
    ink_fraction: float
    candidate_fraction: float
    ridge_threshold: float


def _dependencies():
    try:
        import numpy as np
        from scipy.ndimage import gaussian_filter, gaussian_laplace
        from skimage.filters import threshold_sauvola
        from skimage.morphology import remove_small_objects
    except ImportError as error:
        raise GrayscaleBackendUnavailable(
            "grayscale candidate extraction requires NumPy and scikit-image in the QGIS Python environment"
        ) from error
    return np, gaussian_filter, gaussian_laplace, threshold_sauvola, remove_small_objects


def grayscale_line_candidates(image, settings: GrayscaleCandidateSettings = GrayscaleCandidateSettings()) -> GrayscaleCandidateResult:
    """Create conservative dark-ridge candidates from a 2-D grayscale image.

    The caller is responsible for tiled execution on large scans.  Pixels are
    normalised against a local background, gated by Sauvola dark-ink detection,
    and ranked with a multi-scale black-ridge response.
    """
    np, gaussian_filter, gaussian_laplace, threshold_sauvola, remove_small_objects = _dependencies()
    source = np.asarray(image)
    if source.ndim == 3:
        source = source[..., :3].mean(axis=2)
    if source.ndim != 2 or min(source.shape) < settings.sauvola_window_px:
        raise ValueError("image must be a 2-D grayscale array at least as large as the Sauvola window")
    source = source.astype(np.float32, copy=False)
    background = gaussian_filter(source, sigma=settings.background_sigma_px)
    reference = float(np.median(background))
    normalised = np.clip(source * reference / np.maximum(background, 1.0), 0, 255)
    normalised_unit = normalised / 255.0
    sauvola = threshold_sauvola(normalised_unit, window_size=settings.sauvola_window_px, k=settings.sauvola_k)
    ink = normalised_unit < sauvola
    if not np.any(ink):
        empty = np.zeros(source.shape, dtype=bool)
        return GrayscaleCandidateResult(empty, np.zeros(source.shape, dtype=np.uint8), 0.0, 0.0, 0.0)
    # Scale-normalised Laplacian response is markedly faster than Frangi on
    # 1k tiles and is sufficient for a review-only dark-line proposal.
    ridge = np.maximum.reduce(
        [np.abs(gaussian_laplace(normalised_unit, sigma=scale)) * scale * scale for scale in settings.ridge_scales_px]
    )
    ridge_threshold = float(np.quantile(ridge[ink], settings.ridge_quantile))
    candidates = remove_small_objects(ink & (ridge >= ridge_threshold), min_size=settings.minimum_component_pixels, connectivity=2)
    scale = max(float(np.quantile(ridge[ink], 0.995)), 1e-12)
    likelihood = np.clip(ridge / scale * 255, 0, 255).astype(np.uint8)
    likelihood[~ink] = 0
    return GrayscaleCandidateResult(
        candidates,
        likelihood,
        float(ink.mean()),
        float(candidates.mean()),
        ridge_threshold,
    )
