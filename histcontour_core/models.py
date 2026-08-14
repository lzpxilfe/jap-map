"""Portable data models.  They deliberately contain no country-specific rules."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


class MetadataError(ValueError):
    """Raised when a sheet, profile, or control point is incomplete."""


@dataclass(frozen=True)
class ControlPoint:
    """A point in any declared CRS, represented consistently as x/easting, y/northing."""

    x: float
    y: float
    label: str = ""

    def as_tuple(self) -> tuple[float, float]:
        return self.x, self.y


@dataclass(frozen=True)
class MapSheet:
    """Provenance required to make a reconstructed sheet interpretable later."""

    sheet_id: str
    source_title: str
    display_title: str
    series: str
    edition: str
    producer: str
    survey_purpose: str
    survey_year: str
    publication_year: str
    scale: str
    contour_interval_m: float | None
    source_language: str
    source_script: str
    horizontal_crs: str
    vertical_datum: str
    scan_source: str
    rights: str
    context_note: str = ""

    def __post_init__(self) -> None:
        required = ("sheet_id", "source_title", "display_title", "producer", "horizontal_crs", "scan_source", "rights")
        missing = [name for name in required if not str(getattr(self, name)).strip()]
        if missing:
            raise MetadataError("Missing required sheet metadata: " + ", ".join(missing))
        if self.contour_interval_m is not None and self.contour_interval_m <= 0:
            raise MetadataError("contour_interval_m must be positive when provided")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MapSheet":
        return cls(**value)

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def read_json(cls, path: str | Path) -> "MapSheet":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class MapProfile:
    """Configurable visual and textual conventions for a map series, not a country."""

    profile_id: str
    display_name: str
    contour_rgb: tuple[int, int, int] | None = None
    color_distance: float = 36.0
    min_component_pixels: int = 12
    ocr_charset: str = "0123456789.-"
    contour_rules: dict[str, Any] = field(default_factory=dict)
    crs_presets: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.profile_id or not self.display_name:
            raise MetadataError("profile_id and display_name are required")
        if self.contour_rgb is not None and any(channel < 0 or channel > 255 for channel in self.contour_rgb):
            raise MetadataError("contour_rgb channels must be between 0 and 255")
        if self.color_distance < 0 or self.min_component_pixels < 1:
            raise MetadataError("profile thresholds must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        if self.contour_rgb is not None:
            value["contour_rgb"] = list(self.contour_rgb)
        value["crs_presets"] = list(self.crs_presets)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MapProfile":
        value = dict(value)
        if value.get("contour_rgb") is not None:
            value["contour_rgb"] = tuple(value["contour_rgb"])
        if value.get("crs_presets") is not None:
            value["crs_presets"] = tuple(value["crs_presets"])
        return cls(**value)
