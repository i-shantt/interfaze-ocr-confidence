"""Count specific, checkable OCR output defects across all cached responses. No labels needed.

1. Duplicated text across a split word: one printed word comes back as two adjacent boxes on the
   same line, and the second box repeats the first box's text even though it is far too narrow to
   hold it ("NO" + "NO.:" for a single printed "NO.:"). Flagged when the second word's text starts
   with the first word's, its width per character is under 60% of the first word's, and the gap
   between them is under 60% of the line height.
2. Checkbox renderings: how checkboxes come back ("[ ]", "[x]", ...), which matters to anyone
   parsing forms.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from . import match


def duplicated_splits(pred: list[match.Word]) -> list[dict]:
    found = []
    for a, b in zip(pred, pred[1:]):
        if a.line != b.line or a.box is None or b.box is None:
            continue
        ta, tb = match.norm_strict(a.text), match.norm_strict(b.text)
        if len(match.norm_lenient(ta)) < 2 or not tb.startswith(ta):
            continue
        per_char_a = (a.box[2] - a.box[0]) / len(ta)
        per_char_b = (b.box[2] - b.box[0]) / len(tb)
        gap, height = b.box[0] - a.box[2], a.box[3] - a.box[1]
        if per_char_b < 0.6 * per_char_a and gap < 0.6 * height:
            found.append({"first": a.text, "second": b.text, "first_conf": a.conf, "second_conf": b.conf})
    return found


def scan(raw_dir: Path, ann_dir: Path | None = None, run: int = 1) -> dict:
    dups, dup_docs, checkbox, total = [], set(), Counter(), 0
    for path in sorted(raw_dir.glob(f"*__r{run}.json")):
        doc = path.name.split("__")[0]
        pred = match.load_pred(json.loads(path.read_text()))
        total += len(pred)
        d = duplicated_splits(pred)
        if d:
            dup_docs.add(doc)
            dups += [{**x, "doc": doc} for x in d]
        if ann_dir and (ann_dir / f"{doc}.json").exists():
            rows, _ = match.label_document(pred, match.load_gt(ann_dir / f"{doc}.json"))
            for w, r in zip(pred, rows):
                if r["status"] == "checkbox":
                    checkbox[w.text.strip()] += 1
    return {
        "predicted_words": total,
        "duplicated_splits": len(dups),
        "duplicated_split_documents": len(dup_docs),
        "duplicated_split_examples": [{"first": x["first"], "second": x["second"], "first_conf": x["first_conf"],
                                       "second_conf": x["second_conf"]} for x in dups],
        "checkbox_renderings": dict(checkbox.most_common()),
    }
