"""CRS-independent four-corner sheet geometry used by the QGIS interface."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from histcontour_core.models import ControlPoint


class FrameValidationError(ValueError):
    """Raised when four corners cannot form a valid map frame."""


class CornerRole(str, Enum):
    NW = "NW"
    NE = "NE"
    SE = "SE"
    SW = "SW"


@dataclass(frozen=True)
class Corner:
    role: CornerRole
    x: float
    y: float

    def point(self) -> ControlPoint:
        return ControlPoint(self.x, self.y, self.role.value)


def _cross(a: Corner, b: Corner, c: Corner) -> float:
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)


def _on_segment(a: Corner, b: Corner, point: Corner) -> bool:
    return min(a.x, b.x) <= point.x <= max(a.x, b.x) and min(a.y, b.y) <= point.y <= max(a.y, b.y)


def _segments_intersect(a: Corner, b: Corner, c: Corner, d: Corner) -> bool:
    ab_c, ab_d, cd_a, cd_b = _cross(a, b, c), _cross(a, b, d), _cross(c, d, a), _cross(c, d, b)
    epsilon = 1e-12
    if abs(ab_c) <= epsilon and _on_segment(a, b, c):
        return True
    if abs(ab_d) <= epsilon and _on_segment(a, b, d):
        return True
    if abs(cd_a) <= epsilon and _on_segment(c, d, a):
        return True
    if abs(cd_b) <= epsilon and _on_segment(c, d, b):
        return True
    return (ab_c > 0) != (ab_d > 0) and (cd_a > 0) != (cd_b > 0)


@dataclass(frozen=True)
class SheetFrame:
    sheet_name: str
    crs_authid: str
    corners: tuple[Corner, Corner, Corner, Corner]

    @classmethod
    def create(cls, sheet_name: str, crs_authid: str, corners: dict[CornerRole, Corner]) -> "SheetFrame":
        if set(corners) != set(CornerRole):
            raise FrameValidationError("Enter all four corners: NW, NE, SE, and SW.")
        ordered = tuple(corners[role] for role in (CornerRole.NW, CornerRole.NE, CornerRole.SE, CornerRole.SW))
        if any(not math.isfinite(value) for corner in ordered for value in (corner.x, corner.y)):
            raise FrameValidationError("All coordinate values must be finite.")
        if len({(corner.x, corner.y) for corner in ordered}) != 4:
            raise FrameValidationError("Corner coordinates must be distinct.")
        if _segments_intersect(ordered[0], ordered[1], ordered[2], ordered[3]) or _segments_intersect(ordered[1], ordered[2], ordered[3], ordered[0]):
            raise FrameValidationError("Frame edges must not cross.")
        area2 = sum(ordered[index].x * ordered[(index + 1) % 4].y - ordered[(index + 1) % 4].x * ordered[index].y for index in range(4))
        if abs(area2) <= 1e-12:
            raise FrameValidationError("The four corners must enclose an area.")
        return cls(sheet_name.strip() or "Sheet frame", crs_authid, ordered)

    def ring_xy(self) -> tuple[tuple[float, float], ...]:
        return tuple((corner.x, corner.y) for corner in self.corners)
