#!/usr/bin/env python3
"""Regression tests for shared object- and image-level metrics."""

from __future__ import annotations

import math
import unittest

from baseline_eval_utils import (
    auroc_score,
    average_precision,
    binary_counts,
    operation_at_target_tpr,
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




class AurocTests(unittest.TestCase):
    def test_perfect_separation_gives_auroc_one(self) -> None:
        # All positives rank above all negatives -> AUROC = 1.0
        rows = [row(1, 0.9), row(1, 0.8), row(0, 0.2), row(0, 0.1)]
        self.assertAlmostEqual(auroc_score(rows), 1.0)

    def test_worst_separation_gives_auroc_zero(self) -> None:
        # All negatives rank above all positives -> AUROC = 0.0
        rows = [row(0, 0.9), row(0, 0.8), row(1, 0.2), row(1, 0.1)]
        self.assertAlmostEqual(auroc_score(rows), 0.0)

    def test_all_tied_scores_give_auroc_half(self) -> None:
        # All samples share the same score -> average rank shared -> AUROC = 0.5
        rows = [row(1, 0.5), row(1, 0.5), row(0, 0.5), row(0, 0.5)]
        self.assertAlmostEqual(auroc_score(rows), 0.5)

    def test_known_value(self) -> None:
        # Sorted ascending: (0.3,neg), (0.7,pos), (0.9,neg)
        # rank_sum_pos = 2 -> AUROC = (2 - 1*2/2) / (1*2) = 0.5
        rows = [row(1, 0.7), row(0, 0.9), row(0, 0.3)]
        self.assertAlmostEqual(auroc_score(rows), 0.5)

    def test_returns_none_when_no_positives(self) -> None:
        rows = [row(0, 0.9), row(0, 0.5)]
        self.assertIsNone(auroc_score(rows))

    def test_returns_none_when_no_negatives(self) -> None:
        rows = [row(1, 0.9), row(1, 0.5)]
        self.assertIsNone(auroc_score(rows))


class OperationAtTargetTprTests(unittest.TestCase):
    def _rows_2pos_2neg(self):
        # pos at 0.9 and 0.7, neg at 0.5 and 0.3
        return [row(1, 0.9), row(1, 0.7), row(0, 0.5), row(0, 0.3)]

    def test_achieves_target_tpr_with_minimum_fpr(self) -> None:
        result = operation_at_target_tpr(self._rows_2pos_2neg(), target_tpr=0.95)
        self.assertIsNotNone(result["tpr"])
        self.assertGreaterEqual(result["tpr"], 0.95)
        # At threshold 0.7 both positives are caught; negatives (0.5, 0.3) below -> fpr=0
        self.assertEqual(result["fpr"], 0.0)

    def test_achieves_perfect_tpr_at_lowest_useful_threshold(self) -> None:
        result = operation_at_target_tpr(self._rows_2pos_2neg(), target_tpr=1.0)
        self.assertEqual(result["tpr"], 1.0)

    def test_returns_none_fields_when_infeasible(self) -> None:
        # Only negative images; TPR is undefined -> infeasible
        rows = [row(0, 0.9, 1), row(0, 0.5, 1)]
        result = operation_at_target_tpr(rows, target_tpr=0.95)
        self.assertIsNone(result["tpr"])
        self.assertIsNone(result["fpr"])

    def test_target_tpr_stored_in_result(self) -> None:
        result = operation_at_target_tpr(self._rows_2pos_2neg(), target_tpr=0.95)
        self.assertEqual(result["target_tpr"], 0.95)

if __name__ == "__main__":
    unittest.main()
