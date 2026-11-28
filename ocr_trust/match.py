"""Align Interfaze's OCR words with FUNSD ground-truth words and label each predicted word.

Tokenisation differs between the two (Interfaze may return "Fax:614-466-5087" as one word
where FUNSD has two), so words are not matched one-to-one. Instead, predicted and
ground-truth boxes that overlap form connected groups, and a group is correct when the
concatenated predicted text equals the concatenated ground-truth text. Every predicted word
in the group shares that label.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

Box = tuple[float, float, float, float]  # x0, y0, x1, y1

OVERLAP_THRESHOLD = 0.5  # intersection / smaller box area


@dataclass
class Word:
    text: str
    box: Box | None  # None when the response gives the word no bounds (seen on very degraded pages)
    conf: float | None = None
    line: int | None = None
    kind: str | None = None  # synthetic labels only: word, rare_word, amount, quantity, date, id, phone, email, label
    index: int | None = None  # labels only: position in the label file's word list


@dataclass
class Group:
    pred: list[int] = field(default_factory=list)
    gt: list[int] = field(default_factory=list)


def load_gt(annotation: Path) -> list[Word]:
    """Labelled words from a FUNSD annotation or a synthetic label file (ocr_trust.synth)."""
    data = json.loads(annotation.read_text())
    if "form" in data:
        raw = [(w["text"], w["box"], None) for entity in data["form"] for w in entity["words"]]
    else:
        raw = [(w["text"], w["box"], w["kind"]) for w in data["words"]]
    words = []
    for k, (text, (x0, y0, x1, y1), kind) in enumerate(raw):
        if text.strip():
            words.append(Word(text, (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)), kind=kind, index=k))
    return words


def _corner_box(b: dict) -> Box:
    xs = [b[k]["x"] for k in ("top_left", "top_right", "bottom_right", "bottom_left")]
    ys = [b[k]["y"] for k in ("top_left", "top_right", "bottom_right", "bottom_left")]
    return (min(xs), min(ys), max(xs), max(ys))


def ocr_result(response: dict) -> dict:
    for item in response.get("precontext") or []:
        if item.get("name") == "ocr":
            return item["result"]
    raise ValueError("response has no OCR precontext")


def load_pred(response: dict) -> list[Word]:
    words, line_no = [], 0
    for section in ocr_result(response).get("sections", []):
        for line in section.get("lines", []):
            for w in line.get("words", []):
                if w.get("text", "").strip():
                    box = _corner_box(w["bounds"]) if "bounds" in w else None
                    words.append(Word(w["text"], box, float(w["confidence"]), line_no))
            line_no += 1
    return words


def _area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def overlap(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min(_area(a), _area(b))
    return (ix * iy) / smaller if smaller > 0 else 0.0


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def union_box(boxes: list[Box]) -> Box:
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def groups(pred: list[Word], gt: list[Word], threshold: float = OVERLAP_THRESHOLD) -> list[Group]:
    """Connected components of the pred/gt overlap graph. Unmatched and boxless words get their own group."""
    n = len(pred)
    parent = list(range(n + len(gt)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, p in enumerate(pred):
        if p.box is None:
            continue
        for j, g in enumerate(gt):
            if overlap(p.box, g.box) >= threshold:
                parent[find(i)] = find(n + j)

    by_root: dict[int, Group] = {}
    for k in range(n + len(gt)):
        grp = by_root.setdefault(find(k), Group())
        (grp.pred if k < n else grp.gt).append(k if k < n else k - n)
    return list(by_root.values())


QUOTES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-"})
CHECKBOX_GT = set("\u2610\u2611\u2612\u25a1\u25a0")  # ballot boxes as FUNSD writes them
CHECKBOX_PRED = re.compile(r"^\[\s*[xX\u2713\u2714]?\s*\]?[A-Za-z]?$")  # "[ ]", "[x]", "[", "[ ]I"


def norm_strict(s: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", s).translate(QUOTES))


def norm_lenient(s: str) -> str:
    """Case, punctuation and underscores ignored. FUNSD labels routinely drop trailing punctuation."""
    return re.sub(r"[\W_]", "", norm_strict(s).casefold())


def is_checkbox(gt_text: str, pred_text: str) -> bool:
    g = norm_strict(gt_text)
    if g and set(g) <= CHECKBOX_GT:
        return True
    return bool(CHECKBOX_PRED.match(norm_strict(pred_text))) and len(g) <= 1


def _reading_order(words: list[Word], idx: list[int]) -> list[int]:
    """Rows first (words whose vertical extents overlap by half the smaller height), then left to right."""
    rows: list[list[int]] = []
    for i in sorted(idx, key=lambda i: (words[i].box[1] + words[i].box[3]) / 2):
        b = words[i].box
        for row in rows:
            r = words[row[0]].box
            shared = min(b[3], r[3]) - max(b[1], r[1])
            if shared >= 0.5 * min(b[3] - b[1], r[3] - r[1]):
                row.append(i)
                break
        else:
            rows.append([i])
    return [i for row in rows for i in sorted(row, key=lambda i: words[i].box[0])]


def _group_kind(kinds: list[str | None]) -> str | None:
    distinct = set(kinds)
    if distinct == {None}:
        return None
    return kinds[0] if len(distinct) == 1 else "mixed"


def label_document(pred: list[Word], gt: list[Word]) -> tuple[list[dict], dict]:
    """Per predicted word: status ('correct' | 'wrong' | 'unmatched' | 'checkbox' | 'no_box'), lenient match, box IoU.

    A word returned without bounds cannot be placed on the page, so it is 'no_box' and is not scored.

    Returns (rows, counts). counts includes ground-truth words no prediction touched ('missed').
    """
    rows: list[dict | None] = [None] * len(pred)
    missed = 0
    for grp in groups(pred, gt):
        if not grp.pred:
            missed += len(grp.gt)
            continue
        if pred[grp.pred[0]].box is None:  # boxless words are never joined, so the group is just this word
            rows[grp.pred[0]] = {"status": "no_box", "lenient_correct": None, "iou": None, "group_size": 1, "group_box": None}
            continue
        p_idx = _reading_order(pred, grp.pred)
        if not grp.gt:
            for i in p_idx:
                rows[i] = {"status": "unmatched", "lenient_correct": None, "iou": None, "group_size": len(p_idx),
                           "group_box": pred[i].box}
            continue
        g_idx = _reading_order(gt, grp.gt)
        p_text = "".join(pred[i].text for i in p_idx)
        g_text = "".join(gt[j].text for j in g_idx)
        if is_checkbox(g_text, p_text):
            for i in p_idx:
                rows[i] = {"status": "checkbox", "lenient_correct": None, "iou": None, "group_size": len(p_idx) + len(g_idx)}
            continue
        strict = norm_strict(p_text) == norm_strict(g_text)
        lenient = norm_lenient(p_text) == norm_lenient(g_text)
        box_iou = iou(union_box([pred[i].box for i in p_idx]), union_box([gt[j].box for j in g_idx]))
        for i in p_idx:
            rows[i] = {"status": "correct" if strict else "wrong", "lenient_correct": lenient,
                       "iou": round(box_iou, 4), "group_size": len(p_idx) + len(g_idx),
                       "group_pred_text": p_text, "group_gt_text": g_text, "gt_index": g_idx,
                       "group_box": union_box([pred[i].box for i in p_idx] + [gt[j].box for j in g_idx]),
                       "gt_kind": _group_kind([gt[j].kind for j in g_idx])}
    out = []
    for i, r in enumerate(rows):
        r = dict(r)
        r["conf"] = pred[i].conf
        r["line"] = pred[i].line
        out.append(r)
    counts = {"pred_words": len(pred), "gt_words": len(gt), "missed_gt_words": missed,
              "unmatched_pred_words": sum(r["status"] == "unmatched" for r in out),
              "checkbox_pred_words": sum(r["status"] == "checkbox" for r in out),
              "no_box_pred_words": sum(r["status"] == "no_box" for r in out)}
    return out, counts
