"""Reusable, QGIS-independent tools for historical map reconstruction."""

from .contours import ContourCandidate, ContourLine, extract_visible_contours, generate_link_candidates
from .completion import (
    COMPLETION_BACKEND_ID,
    CompletionBackendUnavailable,
    ContourCompletionCandidate,
    ContourCompletionResult,
    ContourCompletionSettings,
    ContourEndpointAnchor,
    anchors_from_polyline,
    propose_contour_completions,
    rasterize_polylines,
)
from .grayscale import GrayscaleCandidateResult, GrayscaleCandidateSettings, grayscale_line_candidates
from .ink import (
    ARCHAEOTRACE_UPSTREAM_COMMIT,
    INK_BACKEND_ID,
    InkBackendUnavailable,
    InkCenterlineResult,
    InkCenterlineSettings,
    ink_centerline_candidates,
)
from .models import ControlPoint, MapProfile, MapSheet
from .pilot import BaselineMetrics, PilotManifest, PilotSheet, make_baseline_report
from .registration import SheetRegistration
from .vectorization import (
    PixelLineProposal,
    VectorizationBackendUnavailable,
    mask_to_pixel_line_proposals,
    simplify_polyline,
    skeleton_to_pixel_line_proposals,
)

__all__ = [
    "ControlPoint",
    "ContourCandidate",
    "COMPLETION_BACKEND_ID",
    "CompletionBackendUnavailable",
    "ContourCompletionCandidate",
    "ContourCompletionResult",
    "ContourCompletionSettings",
    "ContourEndpointAnchor",
    "ContourLine",
    "GrayscaleCandidateResult",
    "GrayscaleCandidateSettings",
    "ARCHAEOTRACE_UPSTREAM_COMMIT",
    "INK_BACKEND_ID",
    "InkBackendUnavailable",
    "InkCenterlineResult",
    "InkCenterlineSettings",
    "MapProfile",
    "MapSheet",
    "PixelLineProposal",
    "BaselineMetrics",
    "PilotManifest",
    "PilotSheet",
    "SheetRegistration",
    "extract_visible_contours",
    "generate_link_candidates",
    "anchors_from_polyline",
    "grayscale_line_candidates",
    "ink_centerline_candidates",
    "make_baseline_report",
    "mask_to_pixel_line_proposals",
    "propose_contour_completions",
    "rasterize_polylines",
    "simplify_polyline",
    "skeleton_to_pixel_line_proposals",
]
