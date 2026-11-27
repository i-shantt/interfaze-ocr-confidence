"""Does the same image give the same OCR twice?

Interfaze markets itself for deterministic tasks and reports #1 on consistency benchmarks. For a review
threshold the question is narrower: if the same page is sent twice, do the words, and the side of the
threshold each word falls on, stay the same?

Words from two runs are paired when their boxes overlap (IoU >= 0.5, greedy by IoU). Reported per image
set: identical full text, share of paired words with identical text, words present in only one run,
confidence differences, and the share of words whose review decision at a threshold flips between runs.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import match

THRESHOLDS = (0.8, 0.9, 0.95)


def pair_words(a: list[match.Word], b: list[match.Word], min_iou: float = 0.5) -> list[tuple[int, int]]:
    candidates = []
    for i, wa in enumerate(a):
        for j, wb in enumerate(b):
            v = match.iou(wa.box, wb.box)
            if v >= min_iou:
                candidates.append((v, i, j))
    used_a, used_b, pairs = set(), set(), []
    for _, i, j in sorted(candidates, reverse=True):
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            pairs.append((i, j))
    return pairs


def compare(resp_a: dict, resp_b: dict) -> dict:
    a, b = match.load_pred(resp_a), match.load_pred(resp_b)
    pairs = pair_words(a, b)
    deltas = [abs(a[i].conf - b[j].conf) for i, j in pairs]
    out = {
        "text_identical": match.ocr_result(resp_a).get("extracted_text") == match.ocr_result(resp_b).get("extracted_text"),
        "words_a": len(a), "words_b": len(b), "paired": len(pairs),
        "paired_same_text": sum(a[i].text == b[j].text for i, j in pairs),
        "paired_same_conf": sum(a[i].conf == b[j].conf for i, j in pairs),
        "conf_abs_diff_sum": sum(deltas),
        "conf_abs_diff_max": max(deltas, default=0.0),
        "flips": {str(t): sum((a[i].conf >= t) != (b[j].conf >= t) for i, j in pairs) for t in THRESHOLDS},
        "changed": [{"a": a[i].text, "b": b[j].text, "conf_a": a[i].conf, "conf_b": b[j].conf}
                    for i, j in pairs if a[i].text != b[j].text],
    }
    return out


def run(raw_dir: Path, run_a: int = 1, run_b: int = 2) -> dict:
    per_image = {}
    for path_b in sorted(raw_dir.glob(f"*__r{run_b}.json")):
        image = path_b.name.split("__")[0]
        path_a = raw_dir / f"{image}__r{run_a}.json"
        if path_a.exists():
            per_image[image] = compare(json.loads(path_a.read_text()), json.loads(path_b.read_text()))
    if not per_image:
        return {"images": 0}
    tot = lambda k: sum(v[k] for v in per_image.values())  # noqa: E731
    paired = tot("paired")
    return {
        "images": len(per_image),
        "images_text_identical": sum(v["text_identical"] for v in per_image.values()),
        "paired_words": paired,
        "unpaired_words": tot("words_a") + tot("words_b") - 2 * paired,
        "same_text_share": tot("paired_same_text") / paired if paired else None,
        "same_conf_share": tot("paired_same_conf") / paired if paired else None,
        "mean_abs_conf_diff": tot("conf_abs_diff_sum") / paired if paired else None,
        "max_abs_conf_diff": max(v["conf_abs_diff_max"] for v in per_image.values()),
        "flip_share": {t: sum(v["flips"][t] for v in per_image.values()) / paired for t in map(str, THRESHOLDS)} if paired else None,
        "per_image": {k: {kk: vv for kk, vv in v.items() if kk != "changed"} for k, v in per_image.items()},
        "changed_examples": [dict(image=k, **c) for k, v in per_image.items() for c in v["changed"]][:25],
    }
