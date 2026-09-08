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
    INK_EVIDENCE_SCHEMA,
    InkBackendUnavailable,
    InkCenterlineResult,
    InkCenterlineSettings,
    ink_centerline_candidates,
)
from .livewire import LiveWireConfig, LiveWireError, LiveWirePath, LiveWireUnavailable, trace_ink_path
from .manual_gap_bridge import ManualGapBridge, ManualGapBridgeConfig, ManualGapBridgeError, build_manual_gap_bridge, sample_evidence_tangent
from .provenance import INK_ADAPTER_VERSION, execution_id, polyline_geometry_id, sha256_file
from .onnx_scorer import OnnxScorer, OnnxScorerUnavailable
from .patch_context import PATCH_CONTEXT_SCHEMA, PatchContextUnavailable, patch_metadata, segment_context_tensor
from .trace_guidance import TraceGuidance, TraceGuidanceUnavailable, guidance_from_boxes
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
    "INK_EVIDENCE_SCHEMA",
    "INK_ADAPTER_VERSION",
    "InkBackendUnavailable",
    "InkCenterlineResult",
    "InkCenterlineSettings",
    "LiveWireConfig",
    "LiveWireError",
    "LiveWirePath",
    "LiveWireUnavailable",
    "ManualGapBridge",
    "ManualGapBridgeConfig",
    "ManualGapBridgeError",
    "OnnxScorer",
    "OnnxScorerUnavailable",
    "PATCH_CONTEXT_SCHEMA",
    "PatchContextUnavailable",
    "MapProfile",
    "MapSheet",
    "PixelLineProposal",
    "BaselineMetrics",
    "PilotManifest",
    "PilotSheet",
    "SheetRegistration",
    "TraceGuidance",
    "TraceGuidanceUnavailable",
    "extract_visible_contours",
    "generate_link_candidates",
    "anchors_from_polyline",
    "build_manual_gap_bridge",
    "execution_id",
    "grayscale_line_candidates",
    "ink_centerline_candidates",
    "guidance_from_boxes",
    "make_baseline_report",
    "mask_to_pixel_line_proposals",
    "propose_contour_completions",
    "polyline_geometry_id",
    "patch_metadata",
    "rasterize_polylines",
    "simplify_polyline",
    "skeleton_to_pixel_line_proposals",
    "sample_evidence_tangent",
    "sha256_file",
    "segment_context_tensor",
    "trace_ink_path",
]
