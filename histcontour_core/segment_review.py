"""Ink-segment descriptors, diversity sampling, and a tiny baseline classifier.

The functions in this module are deliberately independent of QGIS.  Ink is
responsible for preserving the visible centreline geometry; this module only
prepares segment-level evidence for human review and later classification.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence


FEATURE_NAMES = (
    "log_length",
    "straightness",
    "mean_turn",
    "max_turn",
    "bbox_fill",
    "bbox_aspect_log",
    "line_darkness",
    "dark_fraction_near",
    "dark_fraction_context",
    "mid_fraction_context",
    "context_std",
)

NEGATIVE_REVIEW_STATUSES = frozenset(("text", "road_river", "symbol"))


def _distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return math.hypot(second[0] - first[0], second[1] - first[1])


def geometry_features(points: Sequence[tuple[float, float]]) -> dict[str, float]:
    """Return scale-aware shape features for a pixel-coordinate polyline."""

    if len(points) < 2:
        raise ValueError("a segment needs at least two points")
    lengths = [_distance(first, second) for first, second in zip(points, points[1:])]
    length = sum(lengths)
    if length <= 0:
        raise ValueError("a segment must have positive length")
    chord = _distance(points[0], points[-1])
    turns = []
    for first, middle, last in zip(points, points[1:], points[2:]):
        before = (middle[0] - first[0], middle[1] - first[1])
        after = (last[0] - middle[0], last[1] - middle[1])
        before_length, after_length = math.hypot(*before), math.hypot(*after)
        if before_length == 0 or after_length == 0:
            continue
        cosine = max(-1.0, min(1.0, (before[0] * after[0] + before[1] * after[1]) / (before_length * after_length)))
        turns.append(math.acos(cosine) / math.pi)
    xs, ys = zip(*points)
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    major, minor = max(width, height), min(width, height)
    bbox_diagonal = math.hypot(width, height)
    return {
        "log_length": math.log1p(length),
        "straightness": min(1.0, chord / length),
        "mean_turn": sum(turns) / len(turns) if turns else 0.0,
        "max_turn": max(turns, default=0.0),
        "bbox_fill": min(1.0, chord / bbox_diagonal) if bbox_diagonal else 0.0,
        "bbox_aspect_log": math.log1p(major / max(1.0, minor)),
    }


def percentile_vectors(records: Sequence[Mapping[str, object]], feature_names: Sequence[str]) -> list[tuple[float, ...]]:
    """Convert features to deterministic percentile ranks, robust to outliers."""

    if not records:
        return []
    ranks: list[list[float]] = [[0.0] * len(feature_names) for _ in records]
    denominator = max(1, len(records) - 1)
    for feature_index, name in enumerate(feature_names):
        ordered = sorted(range(len(records)), key=lambda index: (float(records[index][name]), str(records[index]["segment_uid"])))
        for rank, record_index in enumerate(ordered):
            ranks[record_index][feature_index] = rank / denominator
    return [tuple(values) for values in ranks]


def diverse_sample(
    records: Sequence[Mapping[str, object]],
    count: int,
    feature_names: Sequence[str] = FEATURE_NAMES,
) -> list[Mapping[str, object]]:
    """Select a deterministic max-min sample spanning shape and context.

    This is an annotation-priority sampler, not an automatic contour labeler.
    It intentionally has no knowledge of the scene name or expected class.
    """

    if count < 0:
        raise ValueError("count must be non-negative")
    if count >= len(records):
        return sorted(records, key=lambda record: str(record["segment_uid"]))
    if count == 0:
        return []
    vectors = percentile_vectors(records, feature_names)
    centre = (0.5,) * len(feature_names)

    def squared(first, second):
        return sum((a - b) ** 2 for a, b in zip(first, second))

    first = min(range(len(records)), key=lambda index: (squared(vectors[index], centre), str(records[index]["segment_uid"])))
    selected = [first]
    minimum_distances = [squared(vector, vectors[first]) for vector in vectors]
    while len(selected) < count:
        candidate = max(
            (index for index in range(len(records)) if index not in selected),
            key=lambda index: (minimum_distances[index], str(records[index]["segment_uid"])),
        )
        selected.append(candidate)
        for index, vector in enumerate(vectors):
            minimum_distances[index] = min(minimum_distances[index], squared(vector, vectors[candidate]))
    return [records[index] for index in selected]


@dataclass(frozen=True)
class LogisticModel:
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float

    def probability(self, record: Mapping[str, object]) -> float:
        score = self.intercept
        for name, mean, scale, coefficient in zip(self.feature_names, self.means, self.scales, self.coefficients):
            score += coefficient * (float(record[name]) - mean) / scale
        if score >= 0:
            return 1.0 / (1.0 + math.exp(-score))
        exponential = math.exp(score)
        return exponential / (1.0 + exponential)


def train_logistic_baseline(
    records: Sequence[Mapping[str, object]],
    *,
    feature_names: Sequence[str] = FEATURE_NAMES,
    iterations: int = 1200,
    learning_rate: float = 0.08,
    l2: float = 0.02,
) -> LogisticModel:
    """Fit a deterministic class-balanced logistic baseline without sklearn."""

    if iterations < 1 or learning_rate <= 0 or l2 < 0:
        raise ValueError("invalid training settings")
    labels = [1 if record["review_status"] == "contour" else 0 for record in records]
    positives, negatives = sum(labels), len(labels) - sum(labels)
    if positives < 2 or negatives < 2:
        raise ValueError("need at least two contour and two non-contour labels")
    columns = [[float(record[name]) for record in records] for name in feature_names]
    means = [sum(column) / len(column) for column in columns]
    scales = []
    for column, mean in zip(columns, means):
        variance = sum((value - mean) ** 2 for value in column) / len(column)
        scales.append(max(math.sqrt(variance), 1e-9))
    matrix = [
        [(float(record[name]) - mean) / scale for name, mean, scale in zip(feature_names, means, scales)]
        for record in records
    ]
    coefficients = [0.0] * len(feature_names)
    intercept = 0.0
    class_weights = {1: len(labels) / (2.0 * positives), 0: len(labels) / (2.0 * negatives)}
    for iteration in range(iterations):
        gradient = [0.0] * len(coefficients)
        intercept_gradient = 0.0
        for values, label in zip(matrix, labels):
            score = intercept + sum(coefficient * value for coefficient, value in zip(coefficients, values))
            probability = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, score))))
            error = (probability - label) * class_weights[label]
            intercept_gradient += error
            for index, value in enumerate(values):
                gradient[index] += error * value
        rate = learning_rate / math.sqrt(1.0 + iteration / 200.0)
        intercept -= rate * intercept_gradient / len(records)
        for index in range(len(coefficients)):
            coefficients[index] -= rate * (gradient[index] / len(records) + l2 * coefficients[index])
    return LogisticModel(tuple(feature_names), tuple(means), tuple(scales), tuple(coefficients), intercept)


def binary_metrics(labels: Iterable[int], probabilities: Iterable[float], threshold: float = 0.5) -> dict[str, float | int]:
    pairs = [(int(label), float(probability) >= threshold) for label, probability in zip(labels, probabilities)]
    if not pairs:
        raise ValueError("metrics need at least one item")
    true_positive = sum(label == 1 and predicted for label, predicted in pairs)
    false_positive = sum(label == 0 and predicted for label, predicted in pairs)
    true_negative = sum(label == 0 and not predicted for label, predicted in pairs)
    false_negative = sum(label == 1 and not predicted for label, predicted in pairs)
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    specificity = true_negative / max(1, true_negative + false_positive)
    return {
        "count": len(pairs),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(1e-12, precision + recall),
        "balanced_accuracy": (recall + specificity) / 2.0,
    }


def labelled_training_records(records: Iterable[Mapping[str, object]]) -> list[Mapping[str, object]]:
    """Keep reviewed binary labels; unsure and unreviewed stay out of training."""

    return [
        record
        for record in records
        if record.get("review_status") == "contour" or record.get("review_status") in NEGATIVE_REVIEW_STATUSES
    ]
