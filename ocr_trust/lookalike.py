"""The confident errors on the synthetic pages: what they are, whether the printed text was visible, and
whether asking again would have caught them.

For every word Interfaze got wrong at or above a confidence threshold (default 0.9) on exact-label pages:

    edits      character edits from the printed text to the returned text, e.g. "0>O"
    lookalike  every edit swaps one character for a look-alike (LOOKALIKE_CLASSES, or the same letter in
               the other case)
    d_prime    how different the swapped characters look on that page, in units of the page's noise. Each
               printed character and the character returned in its place are drawn alone in the word's font
               and size and put through the page's blur, downscale and contrast steps; the returned one may
               slide up to 3 px sideways to fit best, so position and width earn no credit. With S the summed
               squared difference over the swapped characters, d' = sqrt(S) / sigma, where sigma^2 is the
               mean squared difference between the real word crop and the same page re-rendered without noise
               (synth.make_page(clean=True)): the noise actually on the page, JPEG loss included, so if
               anything sigma is too large and d' too small. An observer who knew the font and had to choose
               between the two characters would pick wrongly with probability Phi(-d'/2). Only for errors
               that keep the word's length; infinite on noise-free pages, where the crop is the printed text.
    repeated   the second run returned the same text with the same confidence

    python -m ocr_trust.lookalike --raw RAW --labels synth/labels --images synth/images --exclude 'L3_*' --out results/synth

writes OUT/confident_errors.csv, OUT/confident_errors.json (totals, and the ID silent-error rate without
codes that contain lower-case letters) and OUT/confident_errors.png (each crop next to the printed and
returned text). Needs Pillow and the fonts in synth.FONTS; the outputs are committed.
"""
from __future__ import annotations

import argparse
import csv
import difflib
import fnmatch
import json
import math
import statistics
from pathlib import Path

from . import consistency, match, synth

LOOKALIKE_CLASSES = ["0OoQD", "1Il|i!", "5Ss", "8B", "2Zz", "6Gb", ".,"]
CSV_FIELDS = ["doc", "level", "font", "kind", "conf", "printed", "returned", "edits", "lookalike", "d_prime",
              "ideal_observer_error", "repeated"]


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


def _swap_is_lookalike(a: str, b: str) -> bool:
    return a.lower() == b.lower() or any(a in cls and b in cls for cls in LOOKALIKE_CLASSES)


def is_lookalike_error(printed: str, returned: str) -> bool:
    """True when the strings have the same length and every differing character is swapped for a look-alike."""
    if len(printed) != len(returned) or printed == returned:
        return False
    return all(_swap_is_lookalike(a, b) for a, b in zip(printed, returned) if a != b)


def _crop(img, box, pad: int = 2) -> list[int]:
    x0, y0, x1, y1 = box
    region = (max(0, math.floor(x0) - pad), max(0, math.floor(y0) - pad),
              min(img.width, math.ceil(x1) + pad), min(img.height, math.ceil(y1) + pad))
    return list(img.crop(region).tobytes())


def _blur_scale_fade(img, level: int):
    """synth.degrade's deterministic steps after rotation (at most 1.5 degrees, which moves a single glyph's
    pixels by well under one)."""
    from PIL import Image, ImageEnhance, ImageFilter
    p = synth.LEVELS[level]
    if p["blur"]:
        img = img.filter(ImageFilter.GaussianBlur(p["blur"]))
    if p["scale"] != 1.0:
        img = img.resize((round(img.width * p["scale"]), round(img.height * p["scale"])), Image.LANCZOS)
    if p["fade"] != 1.0:
        img = ImageEnhance.Contrast(img).enhance(p["fade"])
    return img


def glyph_separation(a: str, b: str, word: dict, level: int, slack: int = 3) -> int:
    """Squared pixel distance between characters a and b as printed in this word's font and size on a page of
    this level, with b slid sideways (|shift| <= slack px, before downscaling) to where it fits a best."""
    from PIL import Image, ImageDraw
    face, size = synth.font(word["font"], word["size"], word["bold"]), word["size"]

    def draw(ch: str, dx: int) -> bytes:
        img = Image.new("L", (3 * size, 2 * size), 255)
        ImageDraw.Draw(img).text((size + dx, size // 3), ch, font=face, fill=0)
        return _blur_scale_fade(img, level).tobytes()

    ref = draw(a, 0)
    return min(sum((x - y) ** 2 for x, y in zip(ref, draw(b, dx))) for dx in range(-slack, slack + 1))


def page_noise(real, clean, box) -> float:
    """Mean squared difference between the real crop and the noise-free render of the same page."""
    r, c = _crop(real, box), _crop(clean, box)
    return sum((x - y) ** 2 for x, y in zip(r, c)) / len(r)


def discriminability(printed: str, returned: str, word: dict, level: int, noise_var: float) -> float | None:
    """d' for one error (see the module docstring); None when the returned text has a different length."""
    if len(printed) != len(returned):
        return None
    separation = sum(glyph_separation(a, b, word, level) for a, b in zip(printed, returned) if a != b)
    return math.inf if noise_var == 0 else math.sqrt(separation / noise_var)


def ideal_observer_error(d_prime: float) -> float:
    return 0.5 * math.erfc(d_prime / (2 * math.sqrt(2)))


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
        kept = [k for k, w in enumerate(label["words"]) if w["text"].strip()]  # load_gt drops blank words
        pred = match.load_pred(json.loads(path.read_text()))
        rows, _ = match.label_document(pred, match.load_gt(labels_dir / f"{doc}.json"))
        run2 = path.with_name(f"{doc}__r2.json")
        pred2 = match.load_pred(json.loads(run2.read_text())) if run2.exists() else None
        paired = dict(consistency.pair_words(pred, pred2)) if pred2 else {}

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
            word = label["words"][kept[gt_index[0]]]
            d_prime = None
            if len(gt_index) == 1:
                if real is None:
                    real = Image.open(images_dir / f"{doc}.png").convert("L")
                    clean, _ = synth.make_page(label["page"], label["level"], label["seed"], dictionary, clean=True)
                d_prime = discriminability(printed, returned, word, label["level"], page_noise(real, clean, word["box"]))
            repeated = pred2 is not None and all(
                k in paired and pred2[paired[k]].text == pred[k].text and pred2[paired[k]].conf == pred[k].conf for k in ks)
            errors.append({
                "doc": doc, "level": label["level"], "font": word["font"], "kind": r["gt_kind"], "conf": conf,
                "printed": printed, "returned": returned,
                "edits": " ".join(f"{a or '∅'}>{b or '∅'}" for a, b in edits(printed, returned)),
                "lookalike": is_lookalike_error(printed, returned), "d_prime": d_prime,
                "ideal_observer_error": None if d_prime is None else ideal_observer_error(d_prime),
                "repeated": repeated, "box": word["box"]})
    errors.sort(key=lambda e: (-e["conf"], e["doc"], e["box"][1], e["box"][0]))
    finite = [e["d_prime"] for e in errors if e["d_prime"] is not None and math.isfinite(e["d_prime"])]
    summary = {
        "threshold": threshold, "errors": len(errors),
        "lookalike": sum(e["lookalike"] for e in errors),
        "by_kind": {k: sum(e["kind"] == k for e in errors) for k in sorted({e["kind"] for e in errors})},
        "repeated_in_run2": sum(e["repeated"] for e in errors),
        "noise_free_pages": sum(e["d_prime"] == math.inf for e in errors),
        "d_prime_min": min(finite, default=None), "d_prime_median": statistics.median(finite) if finite else None,
        "ideal_observer_error_max": ideal_observer_error(min(finite)) if finite else None,
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
                row["d_prime"] = "inf" if math.isinf(e["d_prime"]) else round(e["d_prime"], 2)
                row["ideal_observer_error"] = f"{e['ideal_observer_error']:.2g}"
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
        d = e["d_prime"]
        d_txt = "no noise" if d == math.inf else ("" if d is None else f"d′ {d:.0f}")
        draw.text((text_x + 330, y + 3), f"confidence {e['conf']:.2f}", font=small, fill=(0, 0, 0))
        draw.text((text_x + 330, y + row_h // 2 + 3), f"L{e['level']} · {e['font']} · {d_txt}", font=small, fill=(110, 110, 110))
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
