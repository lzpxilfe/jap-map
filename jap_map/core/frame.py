"""Domain model and topology checks for a four-corner map frame."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


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
    lat: float
    lon: float


def _cross(a: Corner, b: Corner, c: Corner) -> float:
    return (b.lon - a.lon) * (c.lat - a.lat) - (b.lat - a.lat) * (c.lon - a.lon)


def _on_segment(a: Corner, b: Corner, p: Corner) -> bool:
    return (
        min(a.lon, b.lon) <= p.lon <= max(a.lon, b.lon)
        and min(a.lat, b.lat) <= p.lat <= max(a.lat, b.lat)
    )


def _segments_intersect(a: Corner, b: Corner, c: Corner, d: Corner) -> bool:
    ab_c = _cross(a, b, c)
    ab_d = _cross(a, b, d)
    cd_a = _cross(c, d, a)
    cd_b = _cross(c, d, b)
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
    def create(
        cls,
        sheet_name: str,
        crs_authid: str,
        corners: dict[CornerRole, Corner],
    ) -> "SheetFrame":
        expected = set(CornerRole)
        if set(corners) != expected:
            raise FrameValidationError("좌상·우상·우하·좌하 네 모서리를 모두 입력해 주세요.")
        ordered = tuple(corners[role] for role in (CornerRole.NW, CornerRole.NE, CornerRole.SE, CornerRole.SW))
        if any(not math.isfinite(value) for corner in ordered for value in (corner.lat, corner.lon)):
            raise FrameValidationError("모든 좌표는 유한한 숫자여야 합니다.")
        if len({(corner.lat, corner.lon) for corner in ordered}) != 4:
            raise FrameValidationError("모서리 좌표가 중복됩니다.")

        longitude_span = max(corner.lon for corner in ordered) - min(corner.lon for corner in ordered)
        if longitude_span > 180:
            raise FrameValidationError("날짜변경선을 가로지르는 도곽은 v0.1에서 지원하지 않습니다.")

        top = (ordered[0].lat + ordered[1].lat) / 2
        bottom = (ordered[2].lat + ordered[3].lat) / 2
        left = (ordered[0].lon + ordered[3].lon) / 2
        right = (ordered[1].lon + ordered[2].lon) / 2
        if top <= bottom:
            raise FrameValidationError("좌상·우상 모서리가 좌하·우하보다 북쪽이어야 합니다.")
        if right <= left:
            raise FrameValidationError("우측 모서리가 좌측 모서리보다 동쪽이어야 합니다.")

        edges = ((ordered[0], ordered[1]), (ordered[1], ordered[2]), (ordered[2], ordered[3]), (ordered[3], ordered[0]))
        if _segments_intersect(*edges[0], *edges[2]) or _segments_intersect(*edges[1], *edges[3]):
            raise FrameValidationError("모서리 선분이 서로 교차합니다.")

        area2 = sum(
            ordered[index].lon * ordered[(index + 1) % 4].lat
            - ordered[(index + 1) % 4].lon * ordered[index].lat
            for index in range(4)
        )
        if abs(area2) <= 1e-12:
            raise FrameValidationError("네 모서리가 면적을 가진 도곽을 만들지 못합니다.")

        return cls(sheet_name.strip(), crs_authid, ordered)

    def ring_xy(self) -> tuple[tuple[float, float], ...]:
        """Return QGIS-style x/y tuples: longitude first, latitude second."""
        return tuple((corner.lon, corner.lat) for corner in self.corners)
