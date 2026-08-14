"""Reusable, QGIS-independent tools for historical map reconstruction."""

from .contours import ContourCandidate, ContourLine, extract_visible_contours, generate_link_candidates
from .grayscale import GrayscaleCandidateResult, GrayscaleCandidateSettings, grayscale_line_candidates
from .models import ControlPoint, MapProfile, MapSheet
from .pilot import BaselineMetrics, PilotManifest, PilotSheet, make_baseline_report
from .registration import SheetRegistration

__all__ = [
    "ControlPoint",
    "ContourCandidate",
    "ContourLine",
    "GrayscaleCandidateResult",
    "GrayscaleCandidateSettings",
    "MapProfile",
    "MapSheet",
    "BaselineMetrics",
    "PilotManifest",
    "PilotSheet",
    "SheetRegistration",
    "extract_visible_contours",
    "generate_link_candidates",
    "grayscale_line_candidates",
    "make_baseline_report",
]
