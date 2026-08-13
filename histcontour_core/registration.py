"""Projective image-to-map registration and portable registration files."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Iterable

from .models import ControlPoint


class RegistrationError(ValueError):
    """Raised when GCPs cannot define a usable projective transformation."""


@dataclass(frozen=True)
class GroundControlPoint:
    pixel_x: float
    pixel_y: float
    map_x: float
    map_y: float
    label: str = ""

    @classmethod
    def from_points(cls, pixel: ControlPoint, map_point: ControlPoint) -> "GroundControlPoint":
        return cls(pixel.x, pixel.y, map_point.x, map_point.y, pixel.label or map_point.label)

    def pixel(self) -> ControlPoint:
        return ControlPoint(self.pixel_x, self.pixel_y, self.label)

    def map_point(self) -> ControlPoint:
        return ControlPoint(self.map_x, self.map_y, self.label)


def _solve_linear(system: list[list[float]], values: list[float]) -> list[float]:
    """Gauss-Jordan solve; small and dependency-free for 8 projective parameters."""
    size = len(values)
    matrix = [row[:] + [values[index]] for index, row in enumerate(system)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(matrix[row][column]))
        if abs(matrix[pivot][column]) < 1e-12:
            raise RegistrationError("GCPs are degenerate and cannot define a projective transform")
        matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
        divisor = matrix[column][column]
        matrix[column] = [entry / divisor for entry in matrix[column]]
        for row in range(size):
            if row == column:
                continue
            factor = matrix[row][column]
            matrix[row] = [entry - factor * pivot_entry for entry, pivot_entry in zip(matrix[row], matrix[column])]
    return [matrix[row][-1] for row in range(size)]


def fit_projective(gcps: Iterable[GroundControlPoint]) -> tuple[float, ...]:
    """Fit a homography; extra GCPs are fitted by ordinary least squares."""
    points = list(gcps)
    if len(points) < 4:
        raise RegistrationError("Projective registration requires at least four GCPs")
    system: list[list[float]] = []
    values: list[float] = []
    for point in points:
        u, v, x, y = point.pixel_x, point.pixel_y, point.map_x, point.map_y
        system.extend(([u, v, 1.0, 0.0, 0.0, 0.0, -u * x, -v * x], [0.0, 0.0, 0.0, u, v, 1.0, -u * y, -v * y]))
        values.extend((x, y))
    if len(points) == 4:
        return tuple(_solve_linear(system, values))

    # Normal equations keep this portable: the core intentionally has no NumPy
    # dependency so it can run inside a standard QGIS Python environment.
    normal = [[0.0] * 8 for _ in range(8)]
    normal_values = [0.0] * 8
    for row, value in zip(system, values):
        for column in range(8):
            normal_values[column] += row[column] * value
            for other in range(8):
                normal[column][other] += row[column] * row[other]
    return tuple(_solve_linear(normal, normal_values))


def apply_projective(coefficients: tuple[float, ...], pixel_x: float, pixel_y: float) -> ControlPoint:
    if len(coefficients) != 8:
        raise RegistrationError("A projective transform has eight coefficients")
    a, b, c, d, e, f, g, h = coefficients
    denominator = g * pixel_x + h * pixel_y + 1.0
    if abs(denominator) < 1e-12:
        raise RegistrationError("Pixel lies on the projective horizon")
    return ControlPoint((a * pixel_x + b * pixel_y + c) / denominator, (d * pixel_x + e * pixel_y + f) / denominator)


@dataclass(frozen=True)
class SheetRegistration:
    sheet_id: str
    image_path: str
    image_width: int
    image_height: int
    crs_authid: str
    method: str
    gcps: tuple[GroundControlPoint, ...]
    coefficients: tuple[float, ...]
    rmse: float

    @classmethod
    def create(cls, sheet_id: str, image_path: str, image_width: int, image_height: int, crs_authid: str, gcps: Iterable[GroundControlPoint]) -> "SheetRegistration":
        points = tuple(gcps)
        if image_width <= 0 or image_height <= 0:
            raise RegistrationError("Image dimensions must be positive")
        if not sheet_id or not image_path or not crs_authid:
            raise RegistrationError("sheet_id, image_path, and crs_authid are required")
        coefficients = fit_projective(points)
        residuals = []
        for point in points:
            actual = apply_projective(coefficients, point.pixel_x, point.pixel_y)
            residuals.append((actual.x - point.map_x) ** 2 + (actual.y - point.map_y) ** 2)
        return cls(sheet_id, image_path, image_width, image_height, crs_authid, "projective", points, coefficients, math.sqrt(sum(residuals) / len(residuals)))

    def transform(self, pixel_x: float, pixel_y: float) -> ControlPoint:
        return apply_projective(self.coefficients, pixel_x, pixel_y)

    def to_dict(self) -> dict:
        value = asdict(self)
        value["gcps"] = [asdict(point) for point in self.gcps]
        value["coefficients"] = list(self.coefficients)
        return value

    @classmethod
    def from_dict(cls, value: dict) -> "SheetRegistration":
        return cls(**{**value, "gcps": tuple(GroundControlPoint(**point) for point in value["gcps"]), "coefficients": tuple(value["coefficients"])})

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def read_json(cls, path: str | Path) -> "SheetRegistration":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
