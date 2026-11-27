import math
import random
import unittest

from ocr_trust import calibrate, match
from tests.test_consistency import response


class Solver(unittest.TestCase):
    def test_solve(self):
        x = calibrate._solve([[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 4.0]], [3.0, 5.0, 5.0])
        for got, want in zip(x, [1.0, 1.0, 1.0]):
            self.assertAlmostEqual(got, want)


class Fit(unittest.TestCase):
    def test_recovers_a_known_relationship(self):
        """Labels drawn from p = sigmoid(-1 + 3 * logit(conf)); the fit should find that slope."""
        rng = random.Random(0)
        feats, y = [], []
        for _ in range(4000):
            c = rng.uniform(0.05, 0.95)
            feats.append({"logit_conf": calibrate._logit(c)})
            y.append(rng.random() < calibrate._sigmoid(-1 + 3 * calibrate._logit(c)))
        m = calibrate.Calibrator(["logit_conf"], l2=0.0).fit(feats, y)
        slope = m.weights[1] / m.scale[0]
        intercept = m.weights[0] - m.weights[1] * m.mean[0] / m.scale[0]
        self.assertAlmostEqual(slope, 3.0, delta=0.25)
        self.assertAlmostEqual(intercept, -1.0, delta=0.25)

    def test_json_round_trip(self):
        feats = [{"logit_conf": v} for v in (-2.0, 0.0, 2.0, 3.0)]
        m = calibrate.Calibrator(["logit_conf"]).fit(feats, [False, True, True, True])
        m2 = calibrate.Calibrator.from_json(m.to_json())
        self.assertEqual(m.predict(feats), m2.predict(feats))


class Features(unittest.TestCase):
    def test_features_from_a_response(self):
        r = response([("INV-8O1l", 0, 0.97), ("Total", 50, 0.60)])
        f = calibrate.word_features(match.load_pred(r))
        self.assertEqual(f[0]["digit_and_alpha"], 1.0)
        self.assertAlmostEqual(f[0]["lookalike_share"], 4 / 8)
        self.assertEqual(f[0]["neighbour_min_conf"], 0.60)
        self.assertAlmostEqual(f[1]["line_mean_conf"], (0.97 + 0.60) / 2)
        self.assertTrue(all(set(x) == set(calibrate.FEATURES) for x in f))
        self.assertTrue(all(math.isfinite(v) for x in f for v in x.values()))

    def test_annotate(self):
        r = response([("a", 0, 0.5), ("b", 50, 0.99)])
        m = calibrate.Calibrator(["logit_conf"], mean=[0.0], scale=[1.0], weights=[0.0, 1.0])
        out = m.annotate(r)
        self.assertEqual([w["text"] for w in out], ["a", "b"])
        self.assertLess(out[0]["p_correct"], out[1]["p_correct"])


if __name__ == "__main__":
    unittest.main()
