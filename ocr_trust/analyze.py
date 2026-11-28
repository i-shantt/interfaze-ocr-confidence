"""Turn cached Interfaze responses + labels (FUNSD or ocr_trust.synth) into word scores, summary numbers and a report.

    python -m ocr_trust.analyze --raw RAW_DIR --annotations FUNSD/testing_data/annotations --out results/funsd

Writes OUT/word_scores.csv (one row per predicted word: confidence and whether it matched the label;
no OCR text), OUT/summary.json, OUT/REPORT.md and, with matplotlib, OUT/reliability.png. If
OUT/adjudication.csv exists (hand-checked disagreements), the report adds label-noise-corrected estimates.
`--from-csv` rebuilds everything from the committed CSVs, without the API or the raw responses.
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from . import defects, match, metrics

CSV_FIELDS = ["doc", "word", "line", "conf", "status", "lenient_correct", "iou", "group_size", "group_box", "gt_kind"]
SCORED = ("correct", "wrong")
TARGETS = (0.98, 0.99, 0.995)
HIGH, LOW = 0.99, 0.9  # adjudication strata: conf >= HIGH, conf < LOW


# ---------- scoring ----------

def score_documents(raw_dir: Path, ann_dir: Path, run: int = 1) -> tuple[list[dict], list[dict]]:
    rows, examples = [], []
    for path in sorted(raw_dir.glob(f"*__r{run}.json")):
        doc = path.name.split("__")[0]
        ann = ann_dir / f"{doc}.json"
        if not ann.exists():
            continue
        pred, gt = match.load_pred(json.loads(path.read_text())), match.load_gt(ann)
        labelled, counts = match.label_document(pred, gt)
        for k, r in enumerate(labelled):
            rows.append({"doc": doc, "word": k, **{f: r.get(f) for f in CSV_FIELDS[2:]}})
            if r["status"] == "wrong":
                examples.append({"doc": doc, "conf": r["conf"], "pred": r["group_pred_text"], "gt": r["group_gt_text"],
                                 "lenient_correct": r["lenient_correct"], "box": list(r["group_box"]), "status": "wrong"})
            elif r["status"] == "unmatched":
                examples.append({"doc": doc, "conf": r["conf"], "pred": pred[k].text, "gt": "(no label)",
                                 "lenient_correct": None, "box": list(pred[k].box), "status": "unmatched"})
        rows.append({"doc": doc, "word": -1, "line": None, "conf": None, "status": "gt_missed", "lenient_correct": None,
                     "iou": None, "group_size": counts["missed_gt_words"], "group_box": None})
    return rows, examples


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            out = {k: ("" if r.get(k) is None else r.get(k)) for k in CSV_FIELDS}
            if r.get("group_box") is not None:
                out["group_box"] = json.dumps([round(v, 1) for v in r["group_box"]])
            w.writerow(out)


def read_csv(path: Path) -> list[dict]:
    out = []
    with path.open() as f:
        for r in csv.DictReader(f):
            out.append({
                "doc": r["doc"], "word": int(r["word"]), "line": int(r["line"]) if r["line"] else None,
                "conf": float(r["conf"]) if r["conf"] else None, "status": r["status"],
                "lenient_correct": None if r["lenient_correct"] == "" else r["lenient_correct"] == "True",
                "iou": float(r["iou"]) if r["iou"] else None, "group_size": int(r["group_size"]),
                "group_box": json.loads(r["group_box"]) if r["group_box"] else None,
                "gt_kind": r.get("gt_kind") or None,
            })
    return out


# ---------- adjudication ----------

def _key(doc: str, box) -> tuple:
    return (doc, tuple(round(float(v), 1) for v in box))


def _ratio_estimate(sample: list[dict], weight: dict, pool_size: int, bad: set) -> float:
    """Estimated number of words in the pool with a 'bad' verdict: pool groups / checked groups x bad words checked."""
    if not sample:
        return float("nan")
    return pool_size / len(sample) * sum(weight[a["_key"]] for a in sample if a["verdict"] in bad)


def _doc_bootstrap(sample: list[dict], fn, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% percentile interval, resampling whole documents (checked items from one form are not independent)."""
    import random
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for a in sample:
        by_doc[a["doc"]].append(a)
    docs = sorted(by_doc)
    rng = random.Random(seed)
    vals = sorted(fn([a for d in (rng.choice(docs) for _ in docs) for a in by_doc[d]]) for _ in range(n_boot))
    return vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot) - 1]


def corrected(rows: list[dict], adjudication: list[dict]) -> dict:
    """Label-noise-corrected accuracy per confidence band, counted in words.

    The hand check samples disagreement *groups*; a group can hold several words. The number of real OCR
    error words in a band is estimated as (groups in band / groups checked) x (words in checked groups judged
    OCR errors). Ambiguous cases count as correct for the point estimate and as errors for the low end of the
    interval. For the low band, predicted words with no label under them are also estimated from their own
    hand-checked sample and reported both separately and combined.
    """
    scored = [r for r in rows if r["status"] in SCORED]
    for a in adjudication:
        a["_key"] = _key(a["doc"], json.loads(a["box"]))
    out, dropped = {}, 0
    for name, in_band in (("high", lambda c: c >= HIGH), ("low", lambda c: c < LOW)):
        band = [r for r in scored if in_band(r["conf"])]
        weight: dict[tuple, int] = Counter(_key(r["doc"], r["group_box"]) for r in band if not r["lenient_correct"])
        observed_wrong = sum(weight.values())
        all_adj = [a for a in adjudication if a["stratum"] == f"{name}_conf"]
        sample = [a for a in all_adj if a["_key"] in weight]
        dropped += len(all_adj) - len(sample)
        n_words = len(band)

        def acc(smp, bad, n_words=n_words, weight=weight):
            return 1 - _ratio_estimate(smp, weight, len(weight), bad) / n_words

        point = acc(sample, {"ocr_wrong"})
        lo, _ = _doc_bootstrap(sample, lambda smp: acc(smp, {"ocr_wrong", "ambiguous"}))
        _, hi = _doc_bootstrap(sample, lambda smp: acc(smp, {"ocr_wrong"}))
        out[name] = {
            "band": f"conf ≥ {HIGH}" if name == "high" else f"conf < {LOW}",
            "words": n_words, "observed_errors": observed_wrong, "observed_accuracy": 1 - observed_wrong / n_words,
            "error_groups": len(weight), "checked": len(sample), "checked_docs": len({a["doc"] for a in sample}),
            "verdicts": dict(Counter(a["verdict"] for a in sample)),
            "est_accuracy": point, "est_accuracy_range": (lo, hi),
        }
    # Low band, words with no label under them.
    unl = [r for r in rows if r["status"] == "unmatched" and r["conf"] < LOW]
    ukeys = {_key(r["doc"], r["group_box"]) for r in unl}
    usample = [a for a in adjudication if a["stratum"] == "unmatched_low" and a["_key"] in ukeys]
    if usample:
        def u_acc(smp, ok):
            return sum(a["verdict"] in ok for a in smp) / len(smp)
        u_point = u_acc(usample, {"ocr_right"})
        u_lo, _ = _doc_bootstrap(usample, lambda smp: u_acc(smp, {"ocr_right"}))
        _, u_hi = _doc_bootstrap(usample, lambda smp: u_acc(smp, {"ocr_right", "ambiguous"}))
        low = out["low"]
        comb = lambda m, u: (low["words"] * m + len(unl) * u) / (low["words"] + len(unl))
        out["low_unlabelled"] = {
            "words": len(unl), "checked": len(usample), "verdicts": dict(Counter(a["verdict"] for a in usample)),
            "mean_conf": sum(r["conf"] for r in unl) / len(unl),
            "est_accuracy": u_point, "est_accuracy_range": (u_lo, u_hi)}
        out["low_combined"] = {
            "words": low["words"] + len(unl),
            "mean_conf": (sum(r["conf"] for r in scored if r["conf"] < LOW) + sum(r["conf"] for r in unl)) / (low["words"] + len(unl)),
            "est_accuracy": comb(low["est_accuracy"], u_point),
            "est_accuracy_range": (comb(low["est_accuracy_range"][0], u_lo), comb(low["est_accuracy_range"][1], u_hi))}
    # Case/punctuation-only disagreements that are still disagreements under the current matching rules.
    fmt_keys = {_key(r["doc"], r["group_box"]) for r in scored if r["status"] == "wrong" and r["lenient_correct"]}
    fmt_all = [a for a in adjudication if a["stratum"] == "format_only"]
    fmt = [a for a in fmt_all if a["_key"] in fmt_keys]
    dropped += len(fmt_all) - len(fmt)
    out["format_only"] = {"checked": len(fmt), "verdicts": dict(Counter(a["verdict"] for a in fmt))}
    out["checked_total"] = len(adjudication)
    out["dropped_no_longer_disagreements"] = dropped
    for a in adjudication:
        a.pop("_key", None)
    return out


# ---------- summary ----------

def set_aside(rows: list[dict], pattern: str) -> tuple[list[dict], dict]:
    """Split off documents whose name matches a glob pattern, with a short summary of what was left out."""
    out = [r for r in rows if fnmatch.fnmatch(r["doc"], pattern)]
    scored = [r for r in out if r["status"] in SCORED]
    status = Counter(r["status"] for r in out if r["status"] != "gt_missed")
    return [r for r in rows if not fnmatch.fnmatch(r["doc"], pattern)], {
        "pattern": pattern, "documents": len({r["doc"] for r in out}), "predicted_words": sum(status.values()),
        "scored_words": len(scored), "no_box_words": status["no_box"], "unmatched_words": status["unmatched"],
        "strict_accuracy": metrics.accuracy([r["status"] == "correct" for r in scored]),
        "mean_confidence": sum(r["conf"] for r in scored) / len(scored) if scored else None,
        "share_at_or_above_0.9": sum(r["conf"] >= 0.9 for r in scored) / len(scored) if scored else None,
    }


def summarise(rows: list[dict], adjudication: list[dict] | None = None, defect_scan: dict | None = None,
              exclude: str | None = None) -> dict:
    excluded = None
    if exclude:
        rows, excluded = set_aside(rows, exclude)
    scored = [r for r in rows if r["status"] in SCORED]
    conf = [r["conf"] for r in scored]

    def block(correct: list[bool]) -> dict:
        docs = defaultdict(list)
        for r, y in zip(scored, correct):
            docs[r["doc"]].append((r["conf"], y))
        out = {
            "accuracy": metrics.accuracy(correct),
            "mean_confidence": sum(conf) / len(conf),
            "ece": metrics.ece(conf, correct),
            "ece_ci95": metrics.bootstrap_ci(docs, metrics.ece),
            "brier": metrics.brier(conf, correct),
            "auroc_error_detection": metrics.auroc_error_detection(conf, correct),
            "auroc_ci95": metrics.bootstrap_ci(docs, metrics.auroc_error_detection),
            "reliability": metrics.reliability_table(conf, correct),
            "review_for_target": {},
        }
        for t in TARGETS:
            r = metrics.review_needed_for(conf, correct, t)
            out["review_for_target"][str(t)] = {
                "threshold": r["threshold"] if r else None, "review_rate": r["review_rate"] if r else None,
                "errors_caught": r["errors_caught"] if r else None,
                "oracle_review_rate": metrics.oracle_review_rate(correct, t)}
        at_top = [i for i, c in enumerate(conf) if c >= 0.995]
        wrong_total = max(1, len(correct) - sum(correct))
        out["at_confidence_0995"] = {
            "share_of_words": len(at_top) / len(conf),
            "accuracy": metrics.accuracy([correct[i] for i in at_top]),
            "share_of_all_errors": sum(1 for i in at_top if not correct[i]) / wrong_total}
        doc_conf = [sum(c for c, _ in v) / len(v) for v in docs.values()]
        doc_err = [1 - sum(y for _, y in v) / len(v) for v in docs.values()]
        out["doc_spearman_meanconf_vs_error"] = metrics.spearman(doc_conf, doc_err)
        return out

    ious = sorted(r["iou"] for r in scored if r["iou"] is not None)
    status = Counter(r["status"] for r in rows if r["status"] != "gt_missed")
    summary = {
        "documents": len({r["doc"] for r in rows}),
        "predicted_words": sum(status.values()),
        "scored_words": len(scored),
        "unmatched_words": status["unmatched"],
        "checkbox_words": status["checkbox"],
        "no_box_words": status["no_box"],
        "gt_words_missed": sum(r["group_size"] for r in rows if r["status"] == "gt_missed"),
        "lenient": block([bool(r["lenient_correct"]) for r in scored]),
        "strict": block([r["status"] == "correct" for r in scored]),
        "box_iou_median": ious[len(ious) // 2] if ious else None,
        "box_iou_share_ge_0.5": sum(i >= 0.5 for i in ious) / len(ious) if ious else None,
    }
    if excluded:
        summary["excluded"] = excluded
    if adjudication:
        summary["corrected"] = corrected(rows, adjudication)
    if defect_scan:
        summary["defects"] = defect_scan
    if any(r.get("gt_kind") for r in scored):
        summary["by_kind"] = breakdown(scored, lambda r: r["gt_kind"])
        summary["by_level"] = breakdown(scored, lambda r: r["doc"].split("_")[0])
    return summary


SILENT_AT = 0.9


def breakdown(scored: list[dict], key) -> dict:
    """Strict accuracy, calibration and silent errors per group of words (exact labels only).

    A silent error is a wrong word at or above SILENT_AT: a review threshold there would accept it.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in scored:
        groups[key(r)].append(r)
    out = {}
    for name, rs in sorted(groups.items()):
        conf = [r["conf"] for r in rs]
        correct = [r["status"] == "correct" for r in rs]
        accepted = [y for c, y in zip(conf, correct) if c >= SILENT_AT]
        wrong = len(correct) - sum(correct)
        docs = defaultdict(list)
        for r, c, y in zip(rs, conf, correct):
            docs[r["doc"]].append((c, y))
        out[name] = {
            "words": len(rs), "accuracy": metrics.accuracy(correct), "mean_confidence": sum(conf) / len(conf),
            "ece": metrics.ece(conf, correct),
            "auroc_error_detection": metrics.auroc_error_detection(conf, correct),
            "accepted": len(accepted), "accepted_share": len(accepted) / len(rs),
            "silent_errors": len(accepted) - sum(accepted),
            "silent_error_rate": metrics.silent_error_rate(conf, correct, SILENT_AT),
            "silent_error_rate_ci95": metrics.bootstrap_ci(docs, lambda c, y: metrics.silent_error_rate(c, y, SILENT_AT)),
            "silent_share_of_errors": (len(accepted) - sum(accepted)) / wrong if wrong else None,
        }
    return out


def stable(obj, digits: int = 10):
    """Prepare numbers for publishing as JSON. Floats are rounded: Python 3.12 made sum() of floats
    compensated, so 3.10 and 3.12+ differ in the last digits of the same sums. NaN (a metric undefined for a
    group, e.g. AUROC with no errors) and infinity become null, since JSON has no such values."""
    if isinstance(obj, float):
        return round(obj, digits) if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: stable(v, digits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [stable(v, digits) for v in obj]
    return obj


# ---------- report ----------

def _p(x, digits=1):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.{digits}f}%"


def _n(x, digits=3):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}f}"


def _excluded_note(summary: dict) -> list[str]:
    x = summary.get("excluded")
    if not x:
        return []
    return [f"Left out of every number below: {x['documents']} documents matching `{x['pattern']}`. On them, "
            f"{x['scored_words']} words could be scored ({_p(x['strict_accuracy'])} strictly correct, mean confidence "
            f"{_p(x['mean_confidence'])}, {_p(x['share_at_or_above_0.9'])} at or above 0.9) and {x['no_box_words']} of "
            f"{x['predicted_words']} predicted words came back without bounds.", ""]


def report(summary: dict, out: Path) -> str:
    L, S = summary["lenient"], summary["strict"]

    def ci(b, k):
        return f"[{_n(b[k][0])}, {_n(b[k][1])}]"

    lines = [
        "# Results", "",
        f"Interfaze-1, OCR task, {summary['documents']} {summary['dataset']}. {summary['scored_words']} predicted words were scored "
        f"against the labels; {summary['unmatched_words']} had no labelled box under them and {summary['checkbox_words']} were checkboxes, "
        f"both left out. {summary['gt_words_missed']} labelled words had no predicted word over them."
        + (f" {summary['no_box_words']} predicted words came back without bounds and could not be placed or scored."
           if summary.get("no_box_words") else ""), "", *_excluded_note(summary),
        "**Lenient** ignores case and punctuation. "
        "**Strict** compares exact text after Unicode normalisation.", "",
        "| | lenient | strict |", "|---|---|---|",
        f"| word accuracy vs. labels | {_p(L['accuracy'])} | {_p(S['accuracy'])} |",
        f"| mean stated confidence | {_p(L['mean_confidence'])} | {_p(S['mean_confidence'])} |",
        f"| ECE, 10 bins (lower is better) | {_n(L['ece'])} {ci(L, 'ece_ci95')} | {_n(S['ece'])} {ci(S, 'ece_ci95')} |",
        f"| Brier score | {_n(L['brier'])} | {_n(S['brier'])} |",
        f"| AUROC, confidence separates right from wrong | {_n(L['auroc_error_detection'])} {ci(L, 'auroc_ci95')} | "
        f"{_n(S['auroc_error_detection'])} {ci(S, 'auroc_ci95')} |",
        f"| words with confidence ≥ 0.995 | {_p(L['at_confidence_0995']['share_of_words'])} | {_p(S['at_confidence_0995']['share_of_words'])} |",
        f"| their accuracy vs. labels | {_p(L['at_confidence_0995']['accuracy'])} | {_p(S['at_confidence_0995']['accuracy'])} |",
        f"| document mean confidence vs. document error rate (Spearman) | {_n(L['doc_spearman_meanconf_vs_error'], 2)} | "
        f"{_n(S['doc_spearman_meanconf_vs_error'], 2)} |",
        "", "Brackets: 95% intervals from resampling whole documents (1,000 resamples).", "",
        "## Reliability (lenient)", "", "| stated confidence | words | mean confidence | accuracy vs. labels |", "|---|---|---|---|",
    ]
    for r in L["reliability"]:
        lines.append(f"| {r['lo']:.2f}–{r['hi']:.2f} | {r['n']} | {_p(r['mean_conf'])} | {_p(r['accuracy'])} |")
    if "corrected" in summary:
        c = summary["corrected"]
        lines += ["", "## After checking disagreements by hand", "",
                  f"{c['checked_total']} randomly sampled items were checked against the image crop (`adjudication.csv`); "
                  f"{c['dropped_no_longer_disagreements']} of them (checkboxes, `_X_`) are no longer disagreements under the current matching "
                  "rules and are not counted.", "",
                  "| band | words | words wrong vs. labels | groups checked (forms) | OCR wrong | label wrong | ambiguous | artifact | estimated real accuracy (95% range) |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for k in ("high", "low"):
            b = c[k]
            v = b["verdicts"]
            rng = b["est_accuracy_range"]
            lines.append(f"| {b['band']} | {b['words']} | {b['observed_errors']} ({_p(1 - b['observed_accuracy'])}) | {b['checked']} ({b['checked_docs']}) | "
                         f"{v.get('ocr_wrong', 0)} | {v.get('label_wrong', 0)} | {v.get('ambiguous', 0)} | {v.get('matching_artifact', 0)} | "
                         f"{_p(b['est_accuracy'])} ({_p(rng[0])}–{_p(rng[1])}) |")
        if "low_unlabelled" in c:
            u, cb = c["low_unlabelled"], c["low_combined"]
            v = u["verdicts"]
            lines += ["", f"Predicted words below {LOW} with **no label under them** ({u['words']} words, mean confidence {_n(u['mean_conf'], 2)}) are left out "
                      f"of the scores above. {u['checked']} were checked: {v.get('ocr_right', 0)} real text read correctly, {v.get('ocr_wrong', 0)} misread, "
                      f"{v.get('spurious', 0)} specks or lines read as characters, {v.get('ambiguous', 0)} ambiguous: estimated "
                      f"{_p(u['est_accuracy'])} correct ({_p(u['est_accuracy_range'][0])}–{_p(u['est_accuracy_range'][1])}). "
                      f"Counting them, the whole band below {LOW} ({cb['words']} words, mean stated confidence {_n(cb['mean_conf'], 2)}) is an estimated "
                      f"**{_p(cb['est_accuracy'])}** correct ({_p(cb['est_accuracy_range'][0])}–{_p(cb['est_accuracy_range'][1])})."]
        f = c["format_only"]["verdicts"]
        lines += ["", "Estimated real accuracy counts words: (disagreement groups in the band / groups checked) × (words in checked groups judged OCR "
                  "errors), divided by the band's words. Ambiguous cases count as correct for the estimate and as errors for the low end of the "
                  "range. Ranges resample whole forms (2,000 resamples), because several checked items can come from one form. "
                  "Words between 0.90 and 0.99 were not hand-checked.", "",
                  f"Disagreements in case or punctuation only: {c['format_only']['checked']} checked, "
                  f"{f.get('label_wrong', 0)} were label conventions or label errors, {f.get('ocr_wrong', 0)} were OCR errors, "
                  f"{f.get('ambiguous', 0)} ambiguous. That is why lenient matching is the headline number."]
    lines += ["", "## Using confidence to route words to review (lenient, vs. labels)", "",
              "Accept words at or above a threshold; send the rest to a person. Measured against the labels, so it overstates the review needed.", "",
              "| target accuracy | threshold | words sent to review | errors caught | a perfect detector would send |", "|---|---|---|---|---|"]
    for t, r in L["review_for_target"].items():
        lines.append(f"| {float(t) * 100:g}% | {_n(r['threshold'], 2)} | {_p(r['review_rate'])} | {_p(r['errors_caught'])} | "
                     f"{_p(r['oracle_review_rate'])} |")
    for title, k in (("By field type", "by_kind"), ("By degradation level", "by_level")):
        if k in summary:
            lines += ["", f"## {title} (strict, exact labels)", "",
                      f"Silent errors: wrong words with confidence ≥ {SILENT_AT}, which a review threshold at {SILENT_AT} would accept.", "",
                      "| | words | accuracy | mean confidence | ECE | AUROC | accepted at ≥ 0.9 | silent errors | "
                      "silent error rate among accepted (95% range) | share of errors that are silent |",
                      "|---|---|---|---|---|---|---|---|---|---|"]
            for name, b in summary[k].items():
                lo, hi = b["silent_error_rate_ci95"]
                rate = "–" if b["silent_error_rate"] is None else f"{_p(b['silent_error_rate'], 2)} ({_p(lo, 2)}–{_p(hi, 2)})"
                lines.append(f"| {name} | {b['words']} | {_p(b['accuracy'])} | {_p(b['mean_confidence'])} | {_n(b['ece'])} | "
                             f"{_n(b['auroc_error_detection'])} | {b['accepted']} ({_p(b['accepted_share'])}) | {b['silent_errors']} | "
                             f"{rate} | {_p(b['silent_share_of_errors'])} |")
            lines += ["", "Ranges resample whole pages (1,000 resamples)."]
    if "defects" in summary:
        d = summary["defects"]
        ex = d["duplicated_split_examples"]
        sample = ", ".join(f"\"{e['first']}\" ({e['first_conf']:.2f}) + \"{e['second']}\" ({e['second_conf']:.2f})" for e in ex[:3])
        lines += ["", "## Output defects found", "",
                  f"- **Duplicated text when a word is split in two:** flagged automatically {d['duplicated_splits']} times in "
                  f"{d['duplicated_split_documents']} of {summary['documents']} forms. The detector is conservative: the hand check found one "
                  f"more it misses (`37` read as `3737`). One printed word comes back as two boxes and the second, narrow box repeats the "
                  f"first box's text, so joining the words gives e.g. \"NONO.:\" for \"NO.:\". Examples: {sample}.",
                  f"- **Checkboxes:** returned as {', '.join(f'`{k}` ×{v}' for k, v in d['checkbox_renderings'].items())}."]
    lines += ["", f"Box quality: median IoU between matched predicted and labelled word groups is {_n(summary['box_iou_median'], 2)}; "
              f"{_p(summary['box_iou_share_ge_0.5'])} are at IoU ≥ 0.5.", ""]
    text = "\n".join(lines)
    (out / "REPORT.md").write_text(text)
    return text


def figure(rows: list[dict], out: Path, summary: dict | None = None) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    scored = [r for r in rows if r["status"] in SCORED]
    conf = [r["conf"] for r in scored]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    ax1.plot([0, 1], [0, 1], ls="--", c="grey", lw=1, label="perfectly calibrated")
    for name, correct, color in (("lenient", [bool(r["lenient_correct"]) for r in scored], "#2563eb"),
                                 ("strict", [r["status"] == "correct" for r in scored], "#94a3b8")):
        t = [r for r in metrics.reliability_table(conf, correct) if r["n"]]
        ax1.plot([r["mean_conf"] for r in t], [r["accuracy"] for r in t], marker="o", c=color, label=f"Interfaze-1 ({name})")
        if name == "lenient":
            rc = metrics.risk_coverage(conf, correct)
            ax2.plot([100 * r["review_rate"] for r in rc], [100 * r["accepted_accuracy"] for r in rc], c=color)
    corr = (summary or {}).get("corrected")
    if corr:
        for k, band in (("high", lambda c: c >= HIGH), ("low_combined" if "low_combined" in corr else "low", lambda c: c < LOW)):
            b = corr[k]
            mc = b.get("mean_conf") or sum(c for c in conf if band(c)) / max(1, sum(1 for c in conf if band(c)))
            lo, hi = b["est_accuracy_range"]
            ax1.errorbar([mc], [b["est_accuracy"]], yerr=[[b["est_accuracy"] - lo], [hi - b["est_accuracy"]]], fmt="*", ms=12,
                         c="#dc2626", capsize=3, label="after hand-checking (incl. unlabelled words)" if k == "high" else None)
    ax1.set_xlabel("stated word confidence")
    ax1.set_ylabel("accuracy vs. labels")
    ax1.set_title(f"Reliability ({(summary or {}).get('documents', '')} {(summary or {}).get('dataset', '')})")
    ax1.set_xlim(0, 1.02)
    ax1.set_ylim(0, 1.02)
    ax1.legend(loc="upper left", fontsize=8)
    ax2.set_xlabel("% of words sent to review, lowest confidence first")
    ax2.set_ylabel("% of accepted words matching labels (lenient)")
    ax2.set_title("Review vs. accuracy")
    ax2.set_xlim(0, 60)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "reliability.png", dpi=150)
    plt.close(fig)
    return True


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, help="folder of cached Interfaze responses")
    ap.add_argument("--annotations", type=Path, help="labels folder: FUNSD annotations or synth/labels")
    ap.add_argument("--name", default="FUNSD test forms", help="dataset name used in the report")
    ap.add_argument("--from-csv", action="store_true", help="rebuild from OUT/word_scores.csv (+ defects.json)")
    ap.add_argument("--out", type=Path, required=True, help="results folder, e.g. results/funsd")
    ap.add_argument("--examples", type=Path, help="write disagreements (contains OCR text) here for hand checking; keep private")
    ap.add_argument("--exclude", metavar="GLOB", help="leave documents matching this pattern out of the scores and "
                                                      "summarise them separately, e.g. 'L3_*'")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.from_csv:
        rows = read_csv(a.out / "word_scores.csv")
        defect_scan = json.loads((a.out / "defects.json").read_text()) if (a.out / "defects.json").exists() else None
    else:
        if not (a.raw and a.annotations):
            ap.error("give --raw and --annotations, or --from-csv")
        rows, examples = score_documents(a.raw, a.annotations)
        write_csv(rows, a.out / "word_scores.csv")
        defect_scan = defects.scan(a.raw, a.annotations)
        defect_scan = stable(defect_scan)
        (a.out / "defects.json").write_text(json.dumps(defect_scan, indent=1, ensure_ascii=False))
        if a.examples:
            a.examples.write_text(json.dumps(sorted(examples, key=lambda e: -e["conf"]), indent=1, ensure_ascii=False))
    adj_path = a.out / "adjudication.csv"
    adjudication = None
    if adj_path.exists():
        with adj_path.open() as f:
            adjudication = list(csv.DictReader(f))
    summary = {"dataset": a.name, **summarise(rows, adjudication, defect_scan, exclude=a.exclude)}
    summary = stable(summary)
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(report(summary, a.out))
    if figure(rows, a.out, summary):
        print(f"figure: {a.out / 'reliability.png'}")


if __name__ == "__main__":
    main()
