import csv
import json
import math
import unittest
from collections import defaultdict
from pathlib import Path

from ocr_trust import analyze, defects, match, metrics

ROOT = Path(__file__).resolve().parents[1]


def W(text, x0, y0, x1, y1, conf=None, line=0):
    return match.Word(text, (x0, y0, x1, y1), conf, line)


class Geometry(unittest.TestCase):
    def test_overlap_uses_smaller_box(self):
        self.assertAlmostEqual(match.overlap((0, 0, 10, 10), (2, 2, 4, 4)), 1.0)
        self.assertAlmostEqual(match.overlap((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)

    def test_iou(self):
        self.assertAlmostEqual(match.iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertAlmostEqual(match.iou((0, 0, 10, 10), (5, 0, 15, 10)), 50 / 150)

    def test_reading_order_rows_then_columns(self):
        words = [W("b", 50, 0, 60, 10), W("c", 0, 20, 10, 30), W("a", 0, 1, 10, 11)]
        self.assertEqual(match._reading_order(words, [0, 1, 2]), [2, 0, 1])


class Labelling(unittest.TestCase):
    def test_different_tokenisation_still_matches(self):
        pred = [W("Fax:614-466-5087", 0, 0, 100, 10, 0.99)]
        gt = [W("Fax:", 0, 0, 25, 10), W("614-466-5087", 28, 0, 100, 10)]
        rows, counts = match.label_document(pred, gt)
        self.assertEqual(rows[0]["status"], "correct")
        self.assertEqual(counts["missed_gt_words"], 0)

    def test_split_prediction_matches_one_label(self):
        pred = [W("Cover", 0, 0, 30, 10, 0.9), W("Sheet", 32, 0, 60, 10, 0.8)]
        gt = [W("CoverSheet", 0, 0, 60, 10)]
        rows, _ = match.label_document(pred, gt)
        self.assertEqual([r["status"] for r in rows], ["correct", "correct"])

    def test_wrong_and_lenient(self):
        pred = [W("York,", 0, 0, 30, 10, 0.99), W("lowa", 0, 20, 30, 30, 0.99)]
        gt = [W("York", 0, 0, 28, 10), W("Iowa", 0, 20, 30, 30)]
        rows, _ = match.label_document(pred, gt)
        self.assertEqual(rows[0]["status"], "wrong")
        self.assertTrue(rows[0]["lenient_correct"])
        self.assertFalse(rows[1]["lenient_correct"])  # l vs I is a real error, not punctuation

    def test_unmatched_and_missed(self):
        pred = [W("stray", 200, 200, 230, 210, 0.5)]
        gt = [W("label", 0, 0, 30, 10)]
        rows, counts = match.label_document(pred, gt)
        self.assertEqual(rows[0]["status"], "unmatched")
        self.assertEqual(counts["missed_gt_words"], 1)

    def test_checkboxes_are_set_aside(self):
        pred = [W("[x]", 0, 0, 10, 10, 1.0)]
        gt = [W("☑", 0, 0, 10, 10)]
        rows, _ = match.label_document(pred, gt)
        self.assertEqual(rows[0]["status"], "checkbox")

    def test_normalisation(self):
        self.assertEqual(match.norm_strict("“JJ”"), '"JJ"')
        self.assertEqual(match.norm_lenient("_X_"), "x")
        self.assertEqual(match.norm_lenient("No."), match.norm_lenient("NO"))


class Metrics(unittest.TestCase):
    def test_ece_zero_when_calibrated(self):
        conf = [0.75] * 4 + [0.25] * 4
        correct = [True, True, True, False, True, False, False, False]
        self.assertAlmostEqual(metrics.ece(conf, correct), 0.0)

    def test_ece_overconfident(self):
        self.assertAlmostEqual(metrics.ece([0.9] * 10, [True] * 5 + [False] * 5), 0.4)

    def test_auroc(self):
        self.assertAlmostEqual(metrics.auroc_error_detection([0.9, 0.8, 0.2, 0.1], [True, True, False, False]), 1.0)
        self.assertAlmostEqual(metrics.auroc_error_detection([0.5] * 4, [True, False, True, False]), 0.5)
        self.assertAlmostEqual(metrics.auroc_error_detection([0.1, 0.2, 0.8, 0.9], [True, True, False, False]), 0.0)

    def test_review_needed(self):
        conf = [0.99] * 8 + [0.5, 0.4]
        correct = [True] * 8 + [False, False]
        r = metrics.review_needed_for(conf, correct, 0.99)
        self.assertAlmostEqual(r["review_rate"], 0.2)
        self.assertAlmostEqual(r["errors_caught"], 1.0)
        self.assertAlmostEqual(metrics.oracle_review_rate(correct, 0.99), 0.2)

    def test_unreachable_target(self):
        self.assertIsNone(metrics.review_needed_for([0.9, 0.9], [True, False], 0.99))

    def test_bootstrap_is_deterministic(self):
        docs = {str(i): [(0.9, i % 3 != 0), (0.6, i % 2 == 0)] for i in range(20)}
        a = metrics.bootstrap_ci(docs, metrics.ece, n_boot=200)
        self.assertEqual(a, metrics.bootstrap_ci(docs, metrics.ece, n_boot=200))
        self.assertLessEqual(a[0], a[1])


class Defects(unittest.TestCase):
    def test_duplicated_split_found(self):
        pred = [W("NO", 168, 468, 186, 484, 0.99, 35), W("NO.:", 188, 469, 199, 484, 0.83, 35)]
        self.assertEqual(len(defects.duplicated_splits(pred)), 1)

    def test_normal_neighbours_not_flagged(self):
        pred = [W("New", 0, 0, 30, 10, 0.99), W("York", 34, 0, 70, 10, 0.99), W("Yorkshire", 74, 0, 150, 10, 0.99)]
        self.assertEqual(defects.duplicated_splits(pred), [])


class PublishedNumbers(unittest.TestCase):
    """The committed CSVs reproduce the committed summary without the API or the raw responses."""

    def test_rebuild_from_csv_matches_summary(self):
        results = ROOT / "results"
        rows = analyze.read_csv(results / "word_scores.csv")
        with (results / "adjudication.csv").open() as f:
            adj = list(csv.DictReader(f))
        defect_scan = json.loads((results / "defects.json").read_text())
        rebuilt = json.loads(json.dumps(analyze.summarise(rows, adj, defect_scan)))
        committed = json.loads((results / "summary.json").read_text())
        for path in (("lenient", "accuracy"), ("lenient", "ece"), ("lenient", "auroc_error_detection"),
                     ("strict", "accuracy"), ("corrected", "high", "est_accuracy"), ("corrected", "low", "est_accuracy"),
                     ("corrected", "low_combined", "est_accuracy"),
                     ("scored_words",), ("defects", "duplicated_splits")):
            a, b = rebuilt, committed
            for k in path:
                a, b = a[k], b[k]
            self.assertTrue(math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9), f"{path}: {a} != {b}")  # published floats are rounded to 10 places

    def test_word_scores_contain_no_ocr_text(self):
        with (ROOT / "results" / "word_scores.csv").open() as f:
            header = next(csv.reader(f))
        self.assertEqual(header, analyze.CSV_FIELDS)
        self.assertFalse({"text", "pred", "gt", "group_pred_text"} & set(header))


if __name__ == "__main__":
    unittest.main()
