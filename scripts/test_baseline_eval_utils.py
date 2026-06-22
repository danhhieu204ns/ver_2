#!/usr/bin/env python3
"""Regression tests for shared object- and image-level metrics."""

from __future__ import annotations

import math
import unittest

from baseline_eval_utils import (
    average_precision,
    binary_counts,
    predictions_for_scale_bucket,
    sanitize_predictions,
)


def row(label: int, score: float, detections: int = 1) -> dict[str, int | float]:
    return {"label": label, "score": score, "detections": detections, "group": "test"}


class ImageMetricTests(unittest.TestCase):
    def test_average_precision_is_invariant_to_order_inside_score_ties(self) -> None:
        first = [row(1, 0.9), row(0, 0.9), row(1, 0.1), row(0, 0.1)]
        second = [row(0, 0.9), row(1, 0.9), row(0, 0.1), row(1, 0.1)]
        self.assertAlmostEqual(average_precision(first), 0.5)
        self.assertEqual(average_precision(first), average_precision(second))

    def test_no_detection_is_negative_at_zero_threshold(self) -> None:
        counts = binary_counts([row(1, 0.0, 0), row(0, 0.0, 0)], threshold=0.0)
        self.assertEqual((counts["tp"], counts["fp"], counts["tn"], counts["fn"]), (0, 0, 1, 1))

    def test_prediction_sanitization_rejects_nonfinite_and_out_of_range_scores(self) -> None:
        predictions = [
            {"image_id": 1, "category_id": 1, "bbox": [0, 0, 2, 2], "score": 0.5},
            {"image_id": 1, "category_id": 1, "bbox": [0, 0, 2, 2], "score": math.nan},
            {"image_id": 1, "category_id": 1, "bbox": [0, 0, 2, 2], "score": 1.1},
            {"image_id": 1, "category_id": 1, "bbox": [0, 0, math.inf, 2], "score": 0.5},
        ]
        self.assertEqual(len(sanitize_predictions(predictions, [1])), 1)

    def test_relative_scale_filter_does_not_mix_detection_buckets(self) -> None:
        records = [{"width": 100, "height": 100, "objects": []}]
        predictions = [
            {"image_id": 1, "category_id": 1, "bbox": [0, 0, 5, 5], "score": 0.9},
            {"image_id": 1, "category_id": 1, "bbox": [0, 0, 20, 20], "score": 0.8},
        ]
        tiny = predictions_for_scale_bucket(records, [1], predictions, "tiny")
        small = predictions_for_scale_bucket(records, [1], predictions, "small")
        self.assertEqual([item["score"] for item in tiny], [0.9])
        self.assertEqual([item["score"] for item in small], [0.8])


if __name__ == "__main__":
    unittest.main()
