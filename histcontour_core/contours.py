"""Classical contour extraction and conservative reconnection candidates.

The functions operate on plain Python RGB pixels and polylines.  That makes the
baseline reproducible without a GPU, OpenCV, or a QGIS installation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


Pixel = tuple[int, int, int]
Point = tuple[float, float]


@dataclass(frozen=True)
class ContourLine:
    line_id: str
    points: tuple[Point, ...]
    confidence: float
    segment_kind: str = "visible"


@dataclass(frozen=True)
class ContourCandidate:
    candidate_id: str
    source_line_ids: tuple[str, str]
    points: tuple[Point, Point]
    cost: float
    link_confidence: float
    status: str = "proposed"


def color_mask(image: list[list[Pixel]], target: Pixel, maximum_distance: float) -> list[list[bool]]:
    """Return pixels near a profile's contour colour using Euclidean RGB distance."""
    if maximum_distance < 0:
        raise ValueError("maximum_distance must be non-negative")
    return [
        [sum((channel - target[index]) ** 2 for index, channel in enumerate(pixel)) ** 0.5 <= maximum_distance for pixel in row]
        for row in image
    ]


def remove_small_components(mask: list[list[bool]], minimum_pixels: int) -> list[list[bool]]:
    if not mask or not mask[0]:
        return mask
    height, width = len(mask), len(mask[0])
    result = [[False] * width for _ in range(height)]
    visited: set[tuple[int, int]] = set()
    for y in range(height):
        for x in range(width):
            if not mask[y][x] or (x, y) in visited:
                continue
            stack, component = [(x, y)], []
            visited.add((x, y))
            while stack:
                current_x, current_y = stack.pop()
                component.append((current_x, current_y))
                for next_y in range(max(0, current_y - 1), min(height, current_y + 2)):
                    for next_x in range(max(0, current_x - 1), min(width, current_x + 2)):
                        if mask[next_y][next_x] and (next_x, next_y) not in visited:
                            visited.add((next_x, next_y))
                            stack.append((next_x, next_y))
            if len(component) >= minimum_pixels:
                for current_x, current_y in component:
                    result[current_y][current_x] = True
    return result


def skeletonize(mask: list[list[bool]]) -> list[list[bool]]:
    """Zhang-Suen thinning for a binary contour mask."""
    result = [row[:] for row in mask]
    if len(result) < 3 or len(result[0]) < 3:
        return result
    changed = True
    while changed:
        changed = False
        for phase in (0, 1):
            remove: list[tuple[int, int]] = []
            for y in range(1, len(result) - 1):
                for x in range(1, len(result[0]) - 1):
                    if not result[y][x]:
                        continue
                    # P2..P9 clockwise, starting at north.
                    values = [result[y - 1][x], result[y - 1][x + 1], result[y][x + 1], result[y + 1][x + 1], result[y + 1][x], result[y + 1][x - 1], result[y][x - 1], result[y - 1][x - 1]]
                    count = sum(values)
                    transitions = sum((not values[index] and values[(index + 1) % 8]) for index in range(8))
                    if not 2 <= count <= 6 or transitions != 1:
                        continue
                    north, northeast, east, southeast, south, southwest, west, northwest = values
                    if phase == 0 and (north and east and south or east and south and west):
                        continue
                    if phase == 1 and (north and east and west or north and south and west):
                        continue
                    remove.append((x, y))
            if remove:
                changed = True
                for x, y in remove:
                    result[y][x] = False
    return result


def _neighbors(mask: list[list[bool]], point: tuple[int, int]) -> list[tuple[int, int]]:
    x, y = point
    height, width = len(mask), len(mask[0])
    return [
        (candidate_x, candidate_y)
        for candidate_y in range(max(0, y - 1), min(height, y + 2))
        for candidate_x in range(max(0, x - 1), min(width, x + 2))
        if (candidate_x, candidate_y) != point and mask[candidate_y][candidate_x]
    ]


def trace_polylines(mask: list[list[bool]]) -> list[tuple[Point, ...]]:
    """Trace endpoint-led skeleton segments; branch nodes deliberately split lines."""
    if not mask or not mask[0]:
        return []
    pixels = {(x, y) for y, row in enumerate(mask) for x, value in enumerate(row) if value}
    nodes = {point for point in pixels if len(_neighbors(mask, point)) != 2}
    seen_edges: set[frozenset[tuple[int, int]]] = set()
    lines: list[tuple[Point, ...]] = []
    for node in sorted(nodes):
        for neighbor in _neighbors(mask, node):
            edge = frozenset((node, neighbor))
            if edge in seen_edges:
                continue
            path, previous, current = [node], node, neighbor
            seen_edges.add(edge)
            while True:
                path.append(current)
                if current in nodes:
                    break
                next_points = [candidate for candidate in _neighbors(mask, current) if candidate != previous]
                if not next_points:
                    break
                following = next_points[0]
                seen_edges.add(frozenset((current, following)))
                previous, current = current, following
            if len(path) > 1:
                lines.append(tuple((float(x), float(y)) for x, y in path))
    return lines


def extract_visible_contours(image: list[list[Pixel]], contour_rgb: Pixel, color_distance: float, minimum_pixels: int = 12) -> list[ContourLine]:
    mask = skeletonize(remove_small_components(color_mask(image, contour_rgb, color_distance), minimum_pixels))
    return [ContourLine(f"visible-{index + 1}", line, 1.0) for index, line in enumerate(trace_polylines(mask))]


def _distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _orientation(a: Point, b: Point) -> float:
    return math.atan2(b[1] - a[1], b[0] - a[0])


def _segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    def cross(first: Point, second: Point, third: Point) -> float:
        return (second[0] - first[0]) * (third[1] - first[1]) - (second[1] - first[1]) * (third[0] - first[0])

    first_a, first_b, second_a, second_b = cross(a, b, c), cross(a, b, d), cross(c, d, a), cross(c, d, b)
    return (first_a > 0) != (first_b > 0) and (second_a > 0) != (second_b > 0)


def generate_link_candidates(lines: Iterable[ContourLine], maximum_distance: float = 32.0) -> list[ContourCandidate]:
    """Generate non-crossing, direction-aware proposals; never alters source lines."""
    source = [line for line in lines if len(line.points) >= 2]
    candidates: list[ContourCandidate] = []
    for index, first in enumerate(source):
        for second in source[index + 1 :]:
            pair_candidates: list[tuple[Point, Point, float, float]] = []
            for first_end, first_previous in ((first.points[0], first.points[1]), (first.points[-1], first.points[-2])):
                for second_end, second_previous in ((second.points[0], second.points[1]), (second.points[-1], second.points[-2])):
                    distance = _distance(first_end, second_end)
                    if distance > maximum_distance:
                        continue
                    direction_penalty = abs(math.sin(_orientation(first_previous, first_end) - _orientation(second_end, second_previous)))
                    crossing = any(
                        _segments_intersect(first_end, second_end, line.points[position], line.points[position + 1])
                        for line in source
                        for position in range(len(line.points) - 1)
                        if line.line_id not in {first.line_id, second.line_id}
                    )
                    if crossing:
                        continue
                    cost = distance + maximum_distance * direction_penalty
                    confidence = max(0.0, min(1.0, 1.0 - cost / (maximum_distance * 2)))
                    pair_candidates.append((first_end, second_end, cost, confidence))
            if pair_candidates:
                first_end, second_end, cost, confidence = min(pair_candidates, key=lambda item: item[2])
                candidates.append(ContourCandidate(f"link-{len(candidates) + 1}", (first.line_id, second.line_id), (first_end, second_end), cost, confidence))
    return sorted(candidates, key=lambda candidate: candidate.cost)
