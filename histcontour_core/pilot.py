"""Pilot-corpus manifests and repeatable classical-baseline measurements."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable

from .contours import ContourCandidate, ContourLine


class PilotManifestError(ValueError):
    """Raised when the initial three-sheet corpus is not reproducible."""


PILOT_SCENARIOS = ("mountain_clear", "label_dense", "degraded_complex")


@dataclass(frozen=True)
class PilotSheet:
    """A source-controlled record of one raw map held outside the repository."""

    sheet_id: str
    scenario: str
    image_path: str
    profile_path: str
    registration_path: str
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.sheet_id or not self.image_path or not self.profile_path or not self.registration_path:
            raise PilotManifestError("sheet_id, image_path, profile_path, and registration_path are required")
        if self.scenario not in PILOT_SCENARIOS:
            raise PilotManifestError("scenario must be one of: " + ", ".join(PILOT_SCENARIOS))


@dataclass(frozen=True)
class PilotManifest:
    corpus_id: str
    sheets: tuple[PilotSheet, ...]
    version: str = "1"

    def __post_init__(self) -> None:
        if not self.corpus_id:
            raise PilotManifestError("corpus_id is required")
        if len(self.sheets) != 3:
            raise PilotManifestError("The first pilot corpus must contain exactly three sheets")
        if {sheet.scenario for sheet in self.sheets} != set(PILOT_SCENARIOS):
            raise PilotManifestError("Pilot corpus must contain one sheet for every target scenario")
        if len({sheet.sheet_id for sheet in self.sheets}) != len(self.sheets):
            raise PilotManifestError("Pilot sheet IDs must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "corpus_id": self.corpus_id, "sheets": [asdict(sheet) for sheet in self.sheets]}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PilotManifest":
        return cls(value["corpus_id"], tuple(PilotSheet(**sheet) for sheet in value["sheets"]), value.get("version", "1"))

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def read_json(cls, path: str | Path) -> "PilotManifest":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class BaselineMetrics:
    image_width_px: int
    image_height_px: int
    contour_line_count: int
    total_visible_length_px: float
    endpoint_count: int
    candidate_count: int
    mean_candidate_confidence: float
    line_density_per_megapixel: float

    @classmethod
    def from_results(cls, image_width_px: int, image_height_px: int, lines: Iterable[ContourLine], candidates: Iterable[ContourCandidate]) -> "BaselineMetrics":
        lines, candidates = tuple(lines), tuple(candidates)
        length = sum(
            math.hypot(line.points[index + 1][0] - line.points[index][0], line.points[index + 1][1] - line.points[index][1])
            for line in lines
            for index in range(len(line.points) - 1)
        )
        megapixels = max((image_width_px * image_height_px) / 1_000_000, 1e-12)
        return cls(
            image_width_px,
            image_height_px,
            len(lines),
            length,
            sum(2 for line in lines if len(line.points) >= 2),
            len(candidates),
            sum(candidate.link_confidence for candidate in candidates) / len(candidates) if candidates else 0.0,
            len(lines) / megapixels,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_baseline_report(sheet_id: str, profile_id: str, metrics: BaselineMetrics, analysis_scale: float) -> dict[str, Any]:
    if analysis_scale <= 0 or analysis_scale > 1:
        raise ValueError("analysis_scale must be in (0, 1]")
    return {
        "report_version": "1",
        "sheet_id": sheet_id,
        "profile_id": profile_id,
        "analysis_scale": analysis_scale,
        "metrics": metrics.to_dict(),
        "interpretation": {
            "endpoint_count": "Lower is generally better only when false joins do not increase.",
            "candidate_count": "Review burden; candidates remain proposed until a human approves them.",
            "limitations": "This is a classical-baseline diagnostic, not precision/recall. Add manual reference lines before making accuracy claims.",
        },
    }
