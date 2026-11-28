import csv
import json
import math
import unittest
from collections import defaultdict
from pathlib import Path

from ocr_trust import analyze, calibrate, defects, match, metrics

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

    def test_word_without_bounds_is_not_scored(self):
        # Seen on very degraded pages: words after some point in a line come back with text and confidence only.
        resp = {"precontext": [{"name": "ocr", "result": {"sections": [{"lines": [{"words": [
            {"text": "Total", "confidence": 0.9, "bounds": {k: {"x": x, "y": y} for k, (x, y) in
                                                            {"top_left": (0, 0), "top_right": (30, 0),
                                                             "bottom_right": (30, 10), "bottom_left": (0, 10)}.items()}},
            {"text": "$4.00", "confidence": 0.4}]}]}]}}]}
        pred = match.load_pred(resp)
        self.assertEqual([w.box for w in pred], [(0, 0, 30, 10), None])
        rows, counts = match.label_document(pred, [W("Total", 0, 0, 30, 10), W("$4.00", 40, 0, 70, 10)])
        self.assertEqual([r["status"] for r in rows], ["correct", "no_box"])
        self.assertEqual((counts["no_box_pred_words"], counts["missed_gt_words"]), (1, 1))
        self.assertEqual(defects.duplicated_splits(pred), [])

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
    """word_scores.csv and word_features.csv reproduce summary.json and routing.json without the API or the raw
    responses. (confident_errors.*, consistency.json and defects.json need the raw responses; see the README.)"""

    def assertSameTree(self, rebuilt, committed, path=()):
        """Equal structure and values; floats within the 1e-9 that rounding to 10 places allows."""
        if isinstance(committed, dict):
            self.assertEqual(sorted(rebuilt), sorted(committed), path)
            for k in committed:
                self.assertSameTree(rebuilt[k], committed[k], path + (k,))
        elif isinstance(committed, list):
            self.assertEqual(len(rebuilt), len(committed), path)
            for k, (a, b) in enumerate(zip(rebuilt, committed)):
                self.assertSameTree(a, b, path + (k,))
        elif isinstance(committed, float) and isinstance(rebuilt, (int, float)):
            self.assertTrue(math.isclose(rebuilt, committed, rel_tol=1e-9, abs_tol=1e-9), f"{path}: {rebuilt} != {committed}")
        else:
            self.assertEqual(rebuilt, committed, path)

    def rebuild_summary(self, folder: str) -> tuple[dict, dict]:
        results = ROOT / "results" / folder
        committed = json.loads((results / "summary.json").read_text())
        adj_path = results / "adjudication.csv"
        adj = list(csv.DictReader(adj_path.open())) if adj_path.exists() else None
        defect_scan = json.loads((results / "defects.json").read_text())
        exclude = committed.get("excluded", {}).get("pattern")
        rebuilt = analyze.summarise(analyze.read_csv(results / "word_scores.csv"), adj, defect_scan, exclude=exclude)
        return json.loads(json.dumps(analyze.stable({"dataset": committed["dataset"], **rebuilt}))), committed

    def test_funsd_summary_rebuilds_from_csv(self):
        self.assertSameTree(*self.rebuild_summary("funsd"))

    def test_synthetic_summary_rebuilds_from_csv(self):
        self.assertSameTree(*self.rebuild_summary("synth"))

    def test_routing_comparison_rebuilds_from_feature_csvs(self):
        rebuilt = calibrate.evaluate(calibrate.read_words(ROOT / "results" / "synth" / "word_features.csv"),
                                     calibrate.read_words(ROOT / "results" / "funsd" / "word_features.csv"))
        committed = json.loads((ROOT / "results" / "synth" / "routing.json").read_text())
        self.assertSameTree(json.loads(json.dumps(analyze.stable(rebuilt))), committed)

    def test_score_and_feature_tables_contain_no_ocr_text(self):
        for folder in ("funsd", "synth"):
            with (ROOT / "results" / folder / "word_scores.csv").open() as f:
                self.assertEqual(next(csv.reader(f)), analyze.CSV_FIELDS)
            with (ROOT / "results" / folder / "word_features.csv").open() as f:
                self.assertEqual(next(csv.reader(f)), calibrate.CSV_FIELDS)
        self.assertFalse({"text", "pred", "gt", "group_pred_text"} & set(analyze.CSV_FIELDS + calibrate.CSV_FIELDS))


if __name__ == "__main__":
    unittest.main()
