"""Parsing and validation for latitude/longitude text inputs."""

from __future__ import annotations

import math
import re


class CoordinateParseError(ValueError):
    """Raised when a coordinate cannot be parsed or is out of range."""


_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)")
_CARDINAL = re.compile(r"([NSEW])\s*$", re.IGNORECASE)


def _normalise(text: str) -> str:
    value = text.strip()
    replacements = {
        "º": "°",
        "˚": "°",
        "′": "'",
        "’": "'",
        "‘": "'",
        "″": '"',
        "“": '"',
        "”": '"',
        "도": "°",
        "분": "'",
        "초": '"',
        ":": " ",
        ",": " ",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value


def _extract_cardinal(value: str, axis: str) -> tuple[str, str | None]:
    match = _CARDINAL.search(value)
    if not match:
        return value, None

    cardinal = match.group(1).upper()
    if axis == "lat" and cardinal not in {"N", "S"}:
        raise CoordinateParseError("위도에는 N 또는 S만 사용할 수 있습니다.")
    if axis == "lon" and cardinal not in {"E", "W"}:
        raise CoordinateParseError("경도에는 E 또는 W만 사용할 수 있습니다.")
    return value[: match.start()].strip(), cardinal


def parse_angle(text: str, axis: str) -> float:
    """Parse decimal, degree-minute, or degree-minute-second text.

    ``axis`` must be ``"lat"`` or ``"lon"``.  QGIS geometry later stores
    the returned latitude/longitude values as ``y``/``x`` respectively.
    """

    if axis not in {"lat", "lon"}:
        raise ValueError("axis must be 'lat' or 'lon'")
    if not isinstance(text, str) or not text.strip():
        raise CoordinateParseError("좌표를 입력해 주세요.")

    value, cardinal = _extract_cardinal(_normalise(text), axis)
    if not value:
        raise CoordinateParseError("좌표 숫자가 없습니다.")

    # A direction suffix and an explicit sign together are ambiguous.  Requiring
    # one convention makes copied map labels predictable and avoids double signs.
    if cardinal and re.search(r"[+-]", value):
        raise CoordinateParseError("방위 표기와 숫자 부호를 함께 사용할 수 없습니다.")

    cleaned = value.replace("°", " ").replace("'", " ").replace('"', " ")
    parts = cleaned.split()
    if not parts or len(parts) > 3 or any(not _NUMBER.fullmatch(part) for part in parts):
        raise CoordinateParseError("십진도 또는 도·분·초 형식으로 입력해 주세요.")

    numbers = [float(part) for part in parts]
    if not all(math.isfinite(number) for number in numbers):
        raise CoordinateParseError("유효한 숫자를 입력해 주세요.")

    degrees = numbers[0]
    if len(numbers) > 1 and degrees < 0:
        raise CoordinateParseError("도분초 형식에서는 도 값만 음수로 입력할 수 없습니다.")
    minutes = numbers[1] if len(numbers) > 1 else 0.0
    seconds = numbers[2] if len(numbers) > 2 else 0.0
    if not 0 <= minutes < 60 or not 0 <= seconds < 60:
        raise CoordinateParseError("분과 초는 0 이상 60 미만이어야 합니다.")

    magnitude = abs(degrees) + minutes / 60.0 + seconds / 3600.0
    if cardinal in {"S", "W"}:
        result = -magnitude
    elif cardinal in {"N", "E"}:
        result = magnitude
    else:
        result = -magnitude if degrees < 0 else magnitude

    maximum = 90.0 if axis == "lat" else 180.0
    if abs(result) > maximum or (abs(result) == maximum and (minutes or seconds)):
        label = "위도" if axis == "lat" else "경도"
        raise CoordinateParseError(f"{label} 범위를 벗어났습니다.")
    return result
