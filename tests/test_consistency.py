import copy
import json
import unittest
from pathlib import Path

from ocr_trust import consistency

ROOT = Path(__file__).resolve().parents[1]


def response(words):
    """A minimal Interfaze OCR response: one line per call, words as (text, x0, conf)."""
    def bounds(x0):
        x1 = x0 + 40
        return {"top_left": {"x": x0, "y": 10}, "top_right": {"x": x1, "y": 10},
                "bottom_right": {"x": x1, "y": 30}, "bottom_left": {"x": x0, "y": 30}}
    return {"precontext": [{"name": "ocr", "result": {
        "extracted_text": " ".join(t for t, _, _ in words),
        "sections": [{"lines": [{"words": [{"text": t, "bounds": bounds(x), "confidence": c} for t, x, c in words]}]}]}}]}


class Consistency(unittest.TestCase):
    def test_identical_runs(self):
        r = response([("Total", 0, 0.99), ("$12.00", 50, 0.91)])
        c = consistency.compare(r, copy.deepcopy(r))
        self.assertTrue(c["text_identical"])
        self.assertEqual((c["paired"], c["paired_same_text"], c["conf_abs_diff_max"]), (2, 2, 0.0))
        self.assertEqual(c["flips"]["0.9"], 0)

    def test_changed_text_and_flipped_decision(self):
        a = response([("Total", 0, 0.99), ("$12.00", 50, 0.91)])
        b = response([("Total", 0, 0.99), ("$12.80", 50, 0.88)])
        c = consistency.compare(a, b)
        self.assertFalse(c["text_identical"])
        self.assertEqual(c["paired_same_text"], 1)
        self.assertEqual(c["flips"]["0.9"], 1)
        self.assertEqual(c["changed"][0]["b"], "$12.80")

    def test_pairing_is_one_to_one(self):
        a = response([("a", 0, 0.9), ("b", 45, 0.9)])
        b = response([("a", 2, 0.9)])
        self.assertEqual(len(consistency.compare(a, b)["changed"]), 0)
        self.assertEqual(consistency.compare(a, b)["paired"], 1)

    def test_words_without_bounds_are_unpaired(self):
        a = response([("a", 0, 0.9), ("b", 45, 0.9)])
        b = copy.deepcopy(a)
        del b["precontext"][0]["result"]["sections"][0]["lines"][0]["words"][1]["bounds"]
        c = consistency.compare(a, b)
        self.assertEqual((c["paired"], c["words_a"], c["words_b"]), (1, 2, 2))


if __name__ == "__main__":
    unittest.main()
