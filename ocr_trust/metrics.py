"""Calibration and selective-prediction metrics for per-word confidence scores. Standard library only."""
from __future__ import annotations

import math
import random
from typing import Callable, Sequence

DEFAULT_BINS = (0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0000001)


def accuracy(correct: Sequence[bool]) -> float:
    return sum(correct) / len(correct) if correct else float("nan")


def reliability_table(conf: Sequence[float], correct: Sequence[bool], edges=DEFAULT_BINS) -> list[dict]:
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = [i for i, c in enumerate(conf) if lo <= c < hi]
        if not idx:
            rows.append({"lo": lo, "hi": min(hi, 1.0), "n": 0, "mean_conf": None, "accuracy": None, "gap": None})
            continue
        mc = math.fsum(conf[i] for i in idx) / len(idx)
        acc = sum(correct[i] for i in idx) / len(idx)
        rows.append({"lo": lo, "hi": min(hi, 1.0), "n": len(idx), "mean_conf": mc, "accuracy": acc, "gap": mc - acc})
    return rows


def ece(conf: Sequence[float], correct: Sequence[bool], n_bins: int = 10) -> float:
    """Expected calibration error with equal-width bins; the top bin includes 1.0."""
    n = len(conf)
    if n == 0:
        return float("nan")
    total = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [i for i, c in enumerate(conf) if lo <= c < hi or (b == n_bins - 1 and c == hi)]
        if idx:
            mc = math.fsum(conf[i] for i in idx) / len(idx)
            acc = sum(correct[i] for i in idx) / len(idx)
            total += len(idx) / n * abs(mc - acc)
    return total


def brier(conf: Sequence[float], correct: Sequence[bool]) -> float:
    return math.fsum((c - float(y)) ** 2 for c, y in zip(conf, correct)) / len(conf) if conf else float("nan")


def auroc_error_detection(conf: Sequence[float], correct: Sequence[bool]) -> float:
    """Probability that a random wrong word has lower confidence than a random correct word (ties count half).

    0.5 = confidence carries no information about errors, 1.0 = it separates them perfectly.
    """
    pos = [c for c, y in zip(conf, correct) if y]
    neg = [c for c, y in zip(conf, correct) if not y]
    if not pos or not neg:
        return float("nan")
    ranked = sorted((c, y) for c, y in zip(conf, correct))
    # Mann-Whitney U with average ranks for ties.
    i = 0
    rank_sum_pos = 0.0
    while i < len(ranked):
        j = i
        while j + 1 < len(ranked) and ranked[j + 1][0] == ranked[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        rank_sum_pos += avg * sum(1 for k in range(i, j + 1) if ranked[k][1])
        i = j + 1
    u = rank_sum_pos - len(pos) * (len(pos) + 1) / 2
    return u / (len(pos) * len(neg))


def risk_coverage(conf: Sequence[float], correct: Sequence[bool]) -> list[dict]:
    """For every distinct threshold t: accept words with conf >= t, send the rest to review."""
    n = len(conf)
    total_wrong = n - sum(correct)
    out = []
    for t in sorted(set(conf)):
        acc_idx = [i for i in range(n) if conf[i] >= t]
        accepted_wrong = sum(1 for i in acc_idx if not correct[i])
        out.append({
            "threshold": t,
            "review_rate": 1 - len(acc_idx) / n,
            "accepted_accuracy": 1 - accepted_wrong / len(acc_idx) if acc_idx else float("nan"),
            "errors_caught": 1 - accepted_wrong / total_wrong if total_wrong else float("nan"),
        })
    return out


def review_needed_for(conf: Sequence[float], correct: Sequence[bool], target: float) -> dict | None:
    """Lowest review rate whose accepted words reach `target` accuracy, or None if no threshold does."""
    ok = [r for r in risk_coverage(conf, correct) if r["accepted_accuracy"] >= target]
    return min(ok, key=lambda r: r["review_rate"]) if ok else None


def oracle_review_rate(correct: Sequence[bool], target: float) -> float:
    """Review rate a perfect error detector would need: send only wrong words until the rest reach target."""
    n, wrong = len(correct), len(correct) - sum(correct)
    # Remove k wrong words: accepted accuracy = (n - wrong) / (n - k) >= target.
    k = max(0, math.ceil(n - (n - wrong) / target)) if target > 0 else 0
    return min(k, wrong) / n


def silent_error_rate(conf: Sequence[float], correct: Sequence[bool], threshold: float) -> float:
    """Share of words accepted at a threshold (conf >= threshold) that are wrong: errors nobody will look at."""
    accepted = [y for c, y in zip(conf, correct) if c >= threshold]
    return (len(accepted) - sum(accepted)) / len(accepted) if accepted else float("nan")


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    mx, my = math.fsum(rx) / len(rx), math.fsum(ry) / len(ry)
    cov = math.fsum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sx = math.sqrt(math.fsum((a - mx) ** 2 for a in rx))
    sy = math.sqrt(math.fsum((b - my) ** 2 for b in ry))
    return cov / (sx * sy) if sx and sy else float("nan")


def percentile_interval(values, alpha: float = 0.05) -> tuple[float, float]:
    """Central (1 - alpha) interval of bootstrap values, NaNs dropped. With n values, as many lie below the
    lower end as above the upper one (for n = 1000: values[25] and values[974])."""
    vals = sorted(v for v in values if not math.isnan(v))
    if not vals:
        return (float("nan"), float("nan"))
    k = math.floor(alpha / 2 * len(vals))
    return vals[k], vals[len(vals) - 1 - k]


def bootstrap_ci(docs: dict[str, list[tuple[float, bool]]], stat: Callable[[list[float], list[bool]], float],
                 n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """95% interval by resampling whole documents (words within a document are not independent)."""
    rng = random.Random(seed)
    keys = sorted(docs)
    vals = []
    for _ in range(n_boot):
        conf, corr = [], []
        for k in (rng.choice(keys) for _ in keys):
            for c, y in docs[k]:
                conf.append(c)
                corr.append(y)
        vals.append(stat(conf, corr))
    return percentile_interval(vals, alpha)

