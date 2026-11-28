"""The confident errors on the synthetic pages: what they are, whether the page showed the printed text,
and whether asking again would have caught them.

For every word Interfaze got wrong at or above a confidence threshold (default 0.9) on exact-label pages:

    edits       character edits from the printed text to the returned text, e.g. "0>O"
    category    lookalike    every edit swaps a character for one in the same CLASSES group (0/O/o, 1/I/l/i, ...)
                case         every edit changes only capitalisation (in > In)
                punctuation  every edit swaps punctuation ($84.07 > $84,07)
                other        anything else, including insertions and deletions
    read_right  for each swapped pair, how many times Interfaze returned each character inside other words on
                the same page (same font) that it read exactly right, e.g. "0:41 O:12". If both counts are
                high, the model tells these two characters apart on this very page.
    d_prime     how far apart the two characters are for a template matcher that knows the font: each pair is
                drawn alone in the word's font and size through the page's blur, downscale and contrast, the
                returned character may slide 3 px sideways (so position earns no credit; width still counts),
                and the summed squared difference S gives d' = sqrt(S) / sigma. sigma is the noise on that
                word's ink pixels: the real crop against the same page re-rendered without noise
                (synth.make_page(clean=True)), over pixels at least 16 levels darker than the page background.
                JPEG loss is correlated rather than white noise, so d' is descriptive, not a probability.
                Infinite on noise-free pages, where the crop is exactly the printed text.
    repeated    the second run returned the same text with the same confidence

    python -m ocr_trust.lookalike --raw RAW --labels synth/labels --images synth/images --exclude 'L3_*' --out results/synth

writes OUT/confident_errors.csv, OUT/confident_errors.json (totals, and the ID silent-error rate without
codes that contain lower-case letters, counted per predicted word as in the field table) and
OUT/confident_errors.png (each crop next to the printed and returned text). Needs Pillow and the fonts in
synth.FONTS, so it is not part of `make check`; the outputs are committed.
"""
from __future__ import annotations

import argparse
import csv
import difflib
import fnmatch
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from . import consistency, match, synth

CLASSES = ["0Oo", "1Il|i", "5Ss", "8B", "2Zz", "6G"]  # characters that print alike in common fonts
INK_MARGIN = 16  # grey levels below the noise-free background that count as ink
CSV_FIELDS = ["doc", "level", "font", "kind", "conf", "printed", "returned", "edits", "category", "read_right",
              "d_prime", "repeated"]


def spans(printed: str, returned: str) -> list[tuple[bool, str, str]]:
    """The two strings cut into (same?, printed part, returned part) runs: character by character when they
    have the same length (so "0O0O" vs "0000" is two swaps), otherwise by difflib's alignment."""
    if len(printed) == len(returned):
        return [(a == b, a, b) for a, b in zip(printed, returned)]
    sm = difflib.SequenceMatcher(a=printed, b=returned, autojunk=False)
    return [(op == "equal", printed[i1:i2], returned[j1:j2]) for op, i1, i2, j1, j2 in sm.get_opcodes()]


def edits(printed: str, returned: str) -> list[tuple[str, str]]:
    """(printed part, returned part) for every place the two strings differ."""
    return [(a, b) for same, a, b in spans(printed, returned) if not same]


def swaps(printed: str, returned: str) -> list[tuple[str, str]] | None:
    """Character-for-character swaps, or None if the returned text has a different length."""
    if len(printed) != len(returned):
        return None
    return [(a, b) for a, b in zip(printed, returned) if a != b]


def category(printed: str, returned: str) -> str:
    pairs = swaps(printed, returned)
    if not pairs:
        return "other"
    if all(a.lower() == b.lower() for a, b in pairs):
        return "case"
    if all(not a.isalnum() and not b.isalnum() for a, b in pairs):
        return "punctuation"
    if all(any(a in cls and b in cls for cls in CLASSES) for a, b in pairs):
        return "lookalike"
    return "other"


def _crop(img, box, pad: int = 2) -> list[int]:
    x0, y0, x1, y1 = box
    region = (max(0, math.floor(x0) - pad), max(0, math.floor(y0) - pad),
              min(img.width, math.ceil(x1) + pad), min(img.height, math.ceil(y1) + pad))
    return list(img.crop(region).tobytes())


def glyph_separation(a: str, b: str, word: dict, level: int, slack: int = 3) -> int:
    """Squared pixel distance between characters a and b as printed in this word's font and size on a page of
    this level (rotation, at most 1.5 degrees, is left out), with b slid sideways by up to slack px before
    downscaling, to where it fits a best."""
    from PIL import Image, ImageDraw
    face, size = synth.font(word["font"], word["size"], word["bold"]), word["size"]

    def draw(ch: str, dx: int) -> bytes:
        img = Image.new("L", (3 * size, 2 * size), 255)
        ImageDraw.Draw(img).text((size + dx, size // 3), ch, font=face, fill=0)
        return synth.blur_scale_fade(img, synth.LEVELS[level]).tobytes()

    ref = draw(a, 0)
    return min(sum((x - y) ** 2 for x, y in zip(ref, draw(b, dx))) for dx in range(-slack, slack + 1))


def ink_noise(real, clean, box) -> float:
    """Mean squared difference between the real crop and the noise-free render, over the word's ink pixels."""
    r, c = _crop(real, box), _crop(clean, box)
    background = max(c)
    ink = [(x - y) ** 2 for x, y in zip(r, c) if y <= background - INK_MARGIN]
    return sum(ink) / len(ink)


def discriminability(printed: str, returned: str, word: dict, level: int, noise_var: float) -> float | None:
    """d' for one error (see the module docstring); None when the returned text has a different length."""
    pairs = swaps(printed, returned)
    if pairs is None:
        return None
    separation = sum(glyph_separation(a, b, word, level) for a, b in pairs)
    if separation == 0:
        return 0.0
    return math.inf if noise_var == 0 else math.sqrt(separation / noise_var)


def confident_errors(raw_dir: Path, labels_dir: Path, images_dir: Path, threshold: float = 0.9,
                     exclude: str | None = None) -> tuple[list[dict], dict]:
    from PIL import Image
    dictionary = synth.load_dictionary()
    errors, ids_upper = [], {"accepted": 0, "silent_errors": 0}
    for path in sorted(raw_dir.glob("*__r1.json")):
        doc = path.name.split("__")[0]
        if exclude and fnmatch.fnmatch(doc, exclude):
            continue
        label = json.loads((labels_dir / f"{doc}.json").read_text())
        gt = match.load_gt(labels_dir / f"{doc}.json")
        pred = match.load_pred(json.loads(path.read_text()))
        rows, _ = match.label_document(pred, gt)
        run2 = path.with_name(f"{doc}__r2.json")
        pred2 = match.load_pred(json.loads(run2.read_text())) if run2.exists() else None
        paired = dict(consistency.pair_words(pred, pred2)) if pred2 else {}
        read_right = Counter(c for w, r in zip(pred, rows) if r["status"] == "correct" for c in w.text)

        for w, r in zip(pred, rows):  # robustness: IDs written in capitals and digits only, as real codes are
            if r["status"] in ("correct", "wrong") and r.get("gt_kind") == "id" and w.conf >= threshold \
                    and not any(c.islower() for c in r["group_gt_text"]):
                ids_upper["accepted"] += 1
                ids_upper["silent_errors"] += r["status"] == "wrong"

        by_group: dict[tuple, list[int]] = {}
        for k, r in enumerate(rows):
            if r["status"] == "wrong":
                by_group.setdefault(tuple(r["gt_index"]), []).append(k)
        real, clean = None, None
        for gt_index, ks in by_group.items():
            conf = max(pred[k].conf for k in ks)
            if conf < threshold:
                continue
            r = rows[ks[0]]
            printed, returned = r["group_gt_text"], r["group_pred_text"]
            word = label["words"][gt[gt_index[0]].index]
            pairs = swaps(printed, returned)
            d_prime = None
            if len(gt_index) == 1 and pairs is not None:
                if real is None:
                    real = Image.open(images_dir / f"{doc}.png").convert("L")
                    clean, _ = synth.make_page(label["page"], label["level"], label["seed"], dictionary, clean=True)
                d_prime = discriminability(printed, returned, word, label["level"], ink_noise(real, clean, word["box"]))
            repeated = pred2 is not None and all(
                k in paired and pred2[paired[k]].text == pred[k].text and pred2[paired[k]].conf == pred[k].conf for k in ks)
            errors.append({
                "doc": doc, "level": label["level"], "font": word["font"], "kind": r["gt_kind"], "conf": conf,
                "printed": printed, "returned": returned,
                "edits": " ".join(f"{a or '∅'}>{b or '∅'}" for a, b in edits(printed, returned)),
                "category": category(printed, returned),
                "read_right_counts": None if pairs is None else [(read_right[a], read_right[b]) for a, b in pairs],
                "read_right": None if pairs is None else " ".join(f"{a}:{read_right[a]} {b}:{read_right[b]}" for a, b in pairs),
                "d_prime": d_prime, "repeated": repeated, "box": word["box"]})
    errors.sort(key=lambda e: (-e["conf"], e["doc"], e["box"][1], e["box"][0]))
    finite = [e["d_prime"] for e in errors if e["d_prime"] is not None and math.isfinite(e["d_prime"])]
    counts = [n for e in errors if e["read_right_counts"] for pair in e["read_right_counts"] for n in pair]
    summary = {
        "threshold": threshold, "errors": len(errors),
        "by_category": dict(sorted(Counter(e["category"] for e in errors).items())),
        "by_kind": dict(sorted(Counter(e["kind"] for e in errors).items())),
        "repeated_in_run2": sum(e["repeated"] for e in errors),
        "both_characters_read_right_elsewhere_on_page": sum(
            bool(e["read_right_counts"]) and all(min(pair) > 0 for pair in e["read_right_counts"]) for e in errors),
        "read_right_count_min": min(counts, default=None),
        "read_right_count_median": statistics.median(counts) if counts else None,
        "noise_free_pages": sum(e["d_prime"] == math.inf for e in errors),
        "d_prime_min": min(finite, default=None), "d_prime_median": statistics.median(finite) if finite else None,
        "ids_without_lowercase": {**ids_upper, "silent_error_rate":
                                  ids_upper["silent_errors"] / ids_upper["accepted"] if ids_upper["accepted"] else None},
    }
    return errors, summary


def write_csv(errors: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        out = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        out.writeheader()
        for e in errors:
            row = {k: e[k] for k in CSV_FIELDS}
            if e["d_prime"] is not None:
                row["d_prime"] = "inf" if math.isinf(e["d_prime"]) else round(e["d_prime"], 1)
            out.writerow(row)


def figure(errors: list[dict], images_dir: Path, path: Path, row_h: int = 46) -> None:
    """One row per error: the page crop as it was scanned, then the printed and returned text with the
    differing characters in red."""
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(str(synth.FONT_DIR / "Verdana.ttf"), 17)
    small = ImageFont.truetype(str(synth.FONT_DIR / "Verdana.ttf"), 12)
    crop_w, text_x, width, pad = 330, 350, 940, 10
    canvas = Image.new("RGB", (width, pad + len(errors) * (row_h + pad)), "white")
    draw = ImageDraw.Draw(canvas)
    pages: dict[str, Image.Image] = {}
    for n, e in enumerate(errors):
        y = pad + n * (row_h + pad)
        page = pages.setdefault(e["doc"], Image.open(images_dir / f"{e['doc']}.png").convert("L"))
        x0, y0, x1, y1 = e["box"]
        crop = page.crop((math.floor(x0) - 3, math.floor(y0) - 3, math.ceil(x1) + 3, math.ceil(y1) + 3))
        scale = min(row_h / crop.height, crop_w / crop.width)
        crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.NEAREST)
        canvas.paste(crop.convert("RGB"), (pad, y + (row_h - crop.height) // 2))
        runs = spans(e["printed"], e["returned"])
        for line, caption in enumerate(("printed", "returned")):
            ty = y + line * (row_h // 2)
            draw.text((text_x, ty + 3), caption, font=small, fill=(110, 110, 110))
            x = text_x + 70
            for same, a, b in runs:
                seg = a if line == 0 else b
                draw.text((x, ty), seg, font=font, fill=(0, 0, 0) if same else (200, 30, 30))
                x += draw.textlength(seg, font=font)
        draw.text((text_x + 330, y + 3), f"confidence {e['conf']:.2f}", font=small, fill=(0, 0, 0))
        draw.text((text_x + 330, y + row_h // 2 + 3), f"level {e['level']} · {e['font']}", font=small, fill=(110, 110, 110))
    canvas.save(path, optimize=True)


def main(argv=None) -> None:
    from .analyze import stable
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--exclude", metavar="GLOB")
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    errors, summary = confident_errors(a.raw, a.labels, a.images, a.threshold, a.exclude)
    write_csv(errors, a.out / "confident_errors.csv")
    (a.out / "confident_errors.json").write_text(json.dumps(stable(summary), indent=1))
    figure(errors, a.images, a.out / "confident_errors.png")
    print(json.dumps(stable(summary, 4), indent=1))


if __name__ == "__main__":
    main()
