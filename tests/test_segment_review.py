import unittest

from histcontour_core.segment_review import (
    FEATURE_NAMES,
    binary_metrics,
    diverse_sample,
    geometry_features,
    labelled_training_records,
    train_logistic_baseline,
)


def _record(uid, value, status="unreviewed", sheet="sheet-a"):
    record = {name: float(value) for name in FEATURE_NAMES}
    record.update(segment_uid=uid, review_status=status, sheet_id=sheet)
    return record


class SegmentReviewTest(unittest.TestCase):
    def test_straight_and_bent_geometry_have_different_descriptors(self):
        straight = geometry_features(((0.0, 0.0), (10.0, 0.0), (20.0, 0.0)))
        bent = geometry_features(((0.0, 0.0), (10.0, 0.0), (10.0, 10.0)))
        self.assertAlmostEqual(straight["straightness"], 1.0)
        self.assertGreater(bent["mean_turn"], straight["mean_turn"])
        self.assertLess(bent["straightness"], straight["straightness"])

    def test_degenerate_geometry_is_rejected(self):
        with self.assertRaises(ValueError):
            geometry_features(((1.0, 1.0),))
        with self.assertRaises(ValueError):
            geometry_features(((1.0, 1.0), (1.0, 1.0)))

    def test_diversity_sample_is_deterministic_and_spans_extremes(self):
        records = [_record(f"segment-{index:02d}", index) for index in range(20)]
        first = diverse_sample(records, 5)
        second = diverse_sample(list(reversed(records)), 5)
        self.assertEqual({row["segment_uid"] for row in first}, {row["segment_uid"] for row in second})
        values = {row["log_length"] for row in first}
        self.assertIn(0.0, values)
        self.assertIn(19.0, values)

    def test_only_explicit_binary_labels_enter_training(self):
        records = [
            _record("a", 0, "contour"),
            _record("b", 0, "text"),
            _record("c", 0, "road_river"),
            _record("d", 0, "symbol"),
            _record("e", 0, "unsure"),
            _record("f", 0, "unreviewed"),
        ]
        self.assertEqual(len(labelled_training_records(records)), 4)

    def test_baseline_learns_a_separable_signal(self):
        records = []
        for index in range(12):
            records.append(_record(f"positive-{index}", 2.0 + index / 20.0, "contour"))
            records.append(_record(f"negative-{index}", -2.0 - index / 20.0, "text"))
        model = train_logistic_baseline(records)
        probabilities = [model.probability(record) for record in records]
        labels = [1 if record["review_status"] == "contour" else 0 for record in records]
        self.assertGreater(binary_metrics(labels, probabilities)["balanced_accuracy"], 0.95)


if __name__ == "__main__":
    unittest.main()
