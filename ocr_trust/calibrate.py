"""Turn Interfaze's word confidence into a calibrated probability that the word is right.

Uses only what is already in an OCR response: the word's confidence, its neighbours' and line's
confidences, the word's shape (digits, letters, look-alike characters, length) and its box height
(relative to the page's median; a word returned without bounds gets the median, 1.0). The model is an
L2-regularised logistic regression fitted with Newton's method. Four feature sets are compared, and all
four are reported, so extra features have to earn their place:

    platt      logit(confidence) only (classic Platt scaling; rescales, cannot reorder words)
    shape      + whether the word mixes letters and digits
    lookalike  + the share of its characters that have look-alikes (0/O, 1/l/I, 5/S, 8/B, 2/Z, 6/G)
    full       every feature in FEATURES

    python -m ocr_trust.calibrate features --raw RAW --labels LABELS [--exclude GLOB] --out words.csv
    python -m ocr_trust.calibrate evaluate --train synth.csv [--transfer funsd.csv] --out routing.json
    python -m ocr_trust.calibrate fit --train synth.csv --set lookalike --out calibrator.json
    python -m ocr_trust.calibrate apply calibrator.json response.json    # confidence, p_correct, word

The feature CSVs hold no OCR text, so they are committed and every routing number can be rebuilt from
them. Standard library only.
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from . import match, metrics

LOOKALIKE = set("0O1lI5S8B2Z6G")
FEATURES = ["logit_conf", "line_mean_conf", "line_min_conf", "neighbour_min_conf", "has_digit", "has_alpha",
            "digit_and_alpha", "short", "log_length", "lookalike_share", "has_punct", "rel_height"]
FEATURE_SETS = {
    "platt": ["logit_conf"],
    "shape": ["logit_conf", "digit_and_alpha"],
    "lookalike": ["logit_conf", "digit_and_alpha", "lookalike_share"],
    "full": FEATURES,
}
BUDGET_THRESHOLDS = (0.9, 0.95)  # each policy reviews as many words as a plain confidence threshold here would
CSV_FIELDS = ["image", "kind", "conf", "strict", "lenient", *FEATURES]


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
    weights: list[float] = field(default_factory=list)  # intercept first, on standardised features
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

    def coefficients(self) -> dict[str, float]:
        """The fitted model in the features' own units: p = sigmoid(intercept + sum(coef * feature))."""
        coef = {name: w / s for name, w, s in zip(self.features, self.weights[1:], self.scale)}
        return {"intercept": self.weights[0] - sum(coef[n] * m for n, m in zip(self.features, self.mean)), **coef}

    def annotate(self, response: dict) -> list[dict]:
        words = match.load_pred(response)
        return [{"text": w.text, "confidence": w.conf, "p_correct": round(p, 4), "bounds": w.box}
                for w, p in zip(words, self.predict(word_features(words)))]

    def to_dict(self) -> dict:
        return {"features": self.features, "mean": self.mean, "scale": self.scale, "weights": self.weights, "l2": self.l2}

    @classmethod
    def from_dict(cls, d: dict) -> "Calibrator":
        return cls(**{k: d[k] for k in ("features", "mean", "scale", "weights", "l2")})


# ---------- labelled words ----------

def labelled_words(raw_dir: Path, labels_dir: Path, run: int = 1, exclude: str | None = None) -> list[dict]:
    """Scored words with features, strict and lenient correctness, image id and label kind."""
    out = []
    for path in sorted(raw_dir.glob(f"*__r{run}.json")):
        image = path.name.split("__")[0]
        lab = labels_dir / f"{image}.json"
        if not lab.exists() or (exclude and fnmatch.fnmatch(image, exclude)):
            continue
        pred = match.load_pred(json.loads(path.read_text()))
        rows, _ = match.label_document(pred, match.load_gt(lab))
        for f, r, w in zip(word_features(pred), rows, pred):
            if r["status"] in ("correct", "wrong"):
                out.append({"image": image, "conf": w.conf, "features": f, "strict": r["status"] == "correct",
                            "lenient": bool(r["lenient_correct"]), "kind": r.get("gt_kind")})
    return out


def write_words(words: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        out = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        out.writeheader()
        for w in words:
            out.writerow({"image": w["image"], "kind": w["kind"] or "", "conf": w["conf"], "strict": int(w["strict"]),
                          "lenient": int(w["lenient"]), **{k: round(w["features"][k], 6) for k in FEATURES}})


def read_words(path: Path) -> list[dict]:
    with path.open() as f:
        return [{"image": r["image"], "kind": r["kind"] or None, "conf": float(r["conf"]), "strict": r["strict"] == "1",
                 "lenient": r["lenient"] == "1", "features": {k: float(r[k]) for k in FEATURES}} for r in csv.DictReader(f)]


# ---------- evaluation ----------

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


def route(words: list[dict], scores: list[float], n_review: int, target: str) -> dict:
    """Send the n_review lowest-scoring words to review and accept the rest. Ties are broken by position."""
    review = set(sorted(range(len(words)), key=lambda i: (scores[i], i))[:n_review])
    accepted = [w for i, w in enumerate(words) if i not in review]
    wrong = [w for w in accepted if not w[target]]
    return {"review_rate": n_review / len(words), "accepted": len(accepted), "silent_errors": len(wrong),
            "silent_errors_by_kind": dict(sorted(Counter(w["kind"] for w in wrong if w["kind"]).items())),
            "silent_error_rate": len(wrong) / len(accepted)}


def auroc_gain_ci(words: list[dict], scores: list[float], target: str, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% interval for AUROC(scores) - AUROC(raw confidence), resampling whole images."""
    by_image: dict[str, list[int]] = defaultdict(list)
    for i, w in enumerate(words):
        by_image[w["image"]].append(i)
    images = sorted(by_image)
    rng = random.Random(seed)
    gains = []
    for _ in range(n_boot):
        idx = [i for im in (rng.choice(images) for _ in images) for i in by_image[im]]
        y = [words[i][target] for i in idx]
        gain = (metrics.auroc_error_detection([scores[i] for i in idx], y)
                - metrics.auroc_error_detection([words[i]["conf"] for i in idx], y))
        if not math.isnan(gain):
            gains.append(gain)
    gains.sort()
    return gains[int(0.025 * len(gains))], gains[min(len(gains) - 1, int(0.975 * len(gains)))]


def policy(words: list[dict], scores: list[float], target: str, compare_to_raw: bool = True) -> dict:
    """AUROC and the outcome of reviewing as many words as each threshold in BUDGET_THRESHOLDS would."""
    y = [w[target] for w in words]
    out = {"auroc_error_detection": metrics.auroc_error_detection(scores, y),
           "at_budget_of_threshold": {str(t): route(words, scores, sum(w["conf"] < t for w in words), target)
                                      for t in BUDGET_THRESHOLDS}}
    if compare_to_raw:
        out["auroc_gain_ci95"] = auroc_gain_ci(words, scores, target)
    return out


def evaluate(train: list[dict], transfer: list[dict] | None = None, folds: int = 5) -> dict:
    """Compare review policies at equal review budgets.

    In-domain: out-of-fold by page on `train` (exact labels, strict target). Transfer: fit on all of `train`
    and score `transfer` (FUNSD, lenient target as in its main analysis); no labels from it are used to fit.
    """
    out = {"in_domain": {"words": len(train), "images": len({w["image"] for w in train}), "folds": folds, "target": "strict",
                         "raw": policy(train, [w["conf"] for w in train], "strict", compare_to_raw=False)}}
    for name in FEATURE_SETS:
        out["in_domain"][name] = policy(train, cross_fitted(train, name, "strict", folds), "strict")
    if transfer:
        out["transfer"] = {"words": len(transfer), "images": len({w["image"] for w in transfer}), "target": "lenient",
                           "raw": policy(transfer, [w["conf"] for w in transfer], "lenient", compare_to_raw=False)}
        for name, feats in FEATURE_SETS.items():
            model = Calibrator(feats).fit([w["features"] for w in train], [w["strict"] for w in train])
            out["transfer"][name] = policy(transfer, model.predict([w["features"] for w in transfer]), "lenient")
    return out


def main(argv=None) -> None:
    from .analyze import stable
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("features", help="write the scored words' features and labels (no OCR text) to a CSV")
    f.add_argument("--raw", type=Path, required=True)
    f.add_argument("--labels", type=Path, required=True)
    f.add_argument("--exclude", metavar="GLOB")
    f.add_argument("--out", type=Path, required=True)
    e = sub.add_parser("evaluate", help="compare review policies at equal budgets")
    e.add_argument("--train", type=Path, required=True)
    e.add_argument("--transfer", type=Path)
    e.add_argument("--out", type=Path, required=True)
    t = sub.add_parser("fit", help="fit one feature set on all words in a feature CSV")
    t.add_argument("--train", type=Path, required=True)
    t.add_argument("--set", choices=list(FEATURE_SETS), default="lookalike")
    t.add_argument("--target", choices=["strict", "lenient"], default="strict")
    t.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("apply", help="print confidence, calibrated probability and text for each word")
    p.add_argument("model", type=Path)
    p.add_argument("response", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "features":
        words = labelled_words(a.raw, a.labels, exclude=a.exclude)
        write_words(words, a.out)
        print(f"{len(words)} words from {len({w['image'] for w in words})} images -> {a.out}")
    elif a.cmd == "evaluate":
        result = stable(evaluate(read_words(a.train), read_words(a.transfer) if a.transfer else None))
        a.out.write_text(json.dumps(result, indent=1))
        print(f"-> {a.out}")
    elif a.cmd == "fit":
        words = read_words(a.train)
        model = Calibrator(FEATURE_SETS[a.set]).fit([w["features"] for w in words], [w[a.target] for w in words])
        a.out.write_text(json.dumps(stable({**model.to_dict(), "coefficients": model.coefficients()}), indent=1))
        print(f"fitted {a.set} on {len(words)} words -> {a.out}")
    else:
        model = Calibrator.from_dict(json.loads(a.model.read_text()))
        for w in model.annotate(json.loads(a.response.read_text())):
            print(f"{w['confidence']:.2f}  {w['p_correct']:.3f}  {w['text']}")


if __name__ == "__main__":
    main()
