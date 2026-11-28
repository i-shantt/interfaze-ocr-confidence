"""Turn Interfaze's word confidence into a calibrated probability that the word is right.

Uses only what is already in an OCR response: the word's confidence, its neighbours' and line's
confidences, the word's shape (digits, letters, look-alike characters, length) and its box height
(relative to the page's median; a word returned without bounds gets the median, 1.0).
The model is an L2-regularised logistic regression fitted with Newton's method; two feature sets are
compared so the extra features have to earn their place:

    platt   logit(confidence) only (classic Platt scaling)
    full    every feature in FEATURES

    python -m ocr_trust.calibrate fit --raw RAW --labels LABELS --features full --out calibrator.json
    python -m ocr_trust.calibrate apply calibrator.json response.json     # prints word, confidence, p_correct

Standard library only.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from . import match

LOOKALIKE = set("0O1lI5S8B2Z6G")
FEATURES = ["logit_conf", "line_mean_conf", "line_min_conf", "neighbour_min_conf", "has_digit", "has_alpha",
            "digit_and_alpha", "short", "log_length", "lookalike_share", "has_punct", "rel_height"]
FEATURE_SETS = {"platt": ["logit_conf"], "full": FEATURES}


def _logit(p: float, eps: float = 1e-3) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def word_features(words: list[match.Word]) -> list[dict[str, float]]:
    """One feature dict per word, computed from the response alone."""
    if not words:
        return []
    heights = [w.box[3] - w.box[1] for w in words if w.box is not None]
    median_h = (statistics.median(heights) if heights else 0.0) or 1.0
    by_line: dict[int, list[float]] = {}
    for w in words:
        by_line.setdefault(w.line, []).append(w.conf)
    out = []
    for k, w in enumerate(words):
        t = w.text
        prev = words[k - 1].conf if k > 0 and words[k - 1].line == w.line else 1.0
        nxt = words[k + 1].conf if k + 1 < len(words) and words[k + 1].line == w.line else 1.0
        digit, alpha = any(c.isdigit() for c in t), any(c.isalpha() for c in t)
        out.append({
            "logit_conf": _logit(w.conf),
            "line_mean_conf": sum(by_line[w.line]) / len(by_line[w.line]),
            "line_min_conf": min(by_line[w.line]),
            "neighbour_min_conf": min(prev, nxt),
            "has_digit": float(digit), "has_alpha": float(alpha), "digit_and_alpha": float(digit and alpha),
            "short": float(len(t) <= 2), "log_length": math.log(len(t)),
            "lookalike_share": sum(c in LOOKALIKE for c in t) / len(t),
            "has_punct": float(bool(re.search(r"[^\w\s]", t))),
            "rel_height": (w.box[3] - w.box[1]) / median_h if w.box is not None else 1.0,
        })
    return out


def _solve(a: list[list[float]], b: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting (the systems here are 2x2 to 13x13)."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        m[col], m[piv] = m[piv], m[col]
        for r in range(col + 1, n):
            f = m[r][col] / m[col][col]
            for c in range(col, n + 1):
                m[r][c] -= f * m[col][c]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (m[r][n] - sum(m[r][c] * x[c] for c in range(r + 1, n))) / m[r][r]
    return x


def _sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))


@dataclass
class Calibrator:
    features: list[str]
    mean: list[float] = field(default_factory=list)
    scale: list[float] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)  # intercept first
    l2: float = 1.0

    def _row(self, f: dict[str, float]) -> list[float]:
        return [1.0] + [(f[name] - m) / s for name, m, s in zip(self.features, self.mean, self.scale)]

    def fit(self, feats: list[dict[str, float]], correct: list[bool], iterations: int = 30) -> "Calibrator":
        cols = [[f[name] for f in feats] for name in self.features]
        self.mean = [sum(c) / len(c) for c in cols]
        self.scale = [statistics.pstdev(c) or 1.0 for c in cols]
        x = [self._row(f) for f in feats]
        y = [float(v) for v in correct]
        d = len(self.features) + 1
        w = [0.0] * d
        for _ in range(iterations):
            p = [_sigmoid(sum(wi * xi for wi, xi in zip(w, row))) for row in x]
            grad = [sum((pi - yi) * row[j] for pi, yi, row in zip(p, y, x)) + (self.l2 * w[j] if j else 0.0) for j in range(d)]
            hess = [[sum(pi * (1 - pi) * row[i] * row[j] for pi, row in zip(p, x)) + (self.l2 if i == j and i else 0.0)
                     for j in range(d)] for i in range(d)]
            step = _solve(hess, grad)
            w = [wi - si for wi, si in zip(w, step)]
            if max(abs(s) for s in step) < 1e-8:
                break
        self.weights = w
        return self

    def predict(self, feats: list[dict[str, float]]) -> list[float]:
        return [_sigmoid(sum(wi * xi for wi, xi in zip(self.weights, self._row(f)))) for f in feats]

    def annotate(self, response: dict) -> list[dict]:
        words = match.load_pred(response)
        return [{"text": w.text, "confidence": w.conf, "p_correct": round(p, 4), "bounds": w.box}
                for w, p in zip(words, self.predict(word_features(words)))]

    def to_json(self) -> str:
        return json.dumps({"features": self.features, "mean": self.mean, "scale": self.scale,
                           "weights": self.weights, "l2": self.l2}, indent=1)

    @classmethod
    def from_json(cls, text: str) -> "Calibrator":
        return cls(**json.loads(text))


def labelled_words(raw_dir: Path, labels_dir: Path, run: int = 1) -> list[dict]:
    """Scored words with features, strict and lenient correctness, image id and label kind."""
    out = []
    for path in sorted(raw_dir.glob(f"*__r{run}.json")):
        image = path.name.split("__")[0]
        lab = labels_dir / f"{image}.json"
        if not lab.exists():
            continue
        pred = match.load_pred(json.loads(path.read_text()))
        rows, _ = match.label_document(pred, match.load_gt(lab))
        for f, r, w in zip(word_features(pred), rows, pred):
            if r["status"] in ("correct", "wrong"):
                out.append({"image": image, "conf": w.conf, "features": f, "strict": r["status"] == "correct",
                            "lenient": bool(r["lenient_correct"]), "kind": r.get("gt_kind")})
    return out


def cross_fitted(words: list[dict], feature_set: str, target: str = "strict", folds: int = 5, l2: float = 1.0) -> list[float]:
    """Out-of-fold probabilities, folding by image so no page is scored by a model that saw it."""
    images = sorted({w["image"] for w in words})
    fold_of = {im: k % folds for k, im in enumerate(images)}
    probs = [0.0] * len(words)
    for k in range(folds):
        train = [w for w in words if fold_of[w["image"]] != k]
        model = Calibrator(FEATURE_SETS[feature_set], l2=l2).fit([w["features"] for w in train], [w[target] for w in train])
        test_idx = [i for i, w in enumerate(words) if fold_of[w["image"]] == k]
        for i, p in zip(test_idx, model.predict([words[i]["features"] for i in test_idx])):
            probs[i] = p
    return probs


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--raw", type=Path, required=True)
    f.add_argument("--labels", type=Path, required=True)
    f.add_argument("--features", choices=sorted(FEATURE_SETS), default="full")
    f.add_argument("--target", choices=["strict", "lenient"], default="strict")
    f.add_argument("--out", type=Path, required=True)
    a_ = sub.add_parser("apply")
    a_.add_argument("model", type=Path)
    a_.add_argument("response", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "fit":
        words = labelled_words(a.raw, a.labels)
        model = Calibrator(FEATURE_SETS[a.features]).fit([w["features"] for w in words], [w[a.target] for w in words])
        a.out.write_text(model.to_json())
        print(f"fitted on {len(words)} words from {len({w['image'] for w in words})} images -> {a.out}")
    else:
        model = Calibrator.from_json(a.model.read_text())
        for w in model.annotate(json.loads(a.response.read_text())):
            print(f"{w['confidence']:.2f}  {w['p_correct']:.3f}  {w['text']}")


if __name__ == "__main__":
    main()
