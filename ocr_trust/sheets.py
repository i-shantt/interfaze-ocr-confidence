"""Draw a random sample of disagreements as labelled image crops, for hand adjudication.

    python -m ocr_trust.sheets --examples sheets/disagreements.json --images data/dataset/testing_data/images --out sheets

Each tile shows the crop with the OCR text and the FUNSD label under it. Sheets contain dataset
crops and OCR text, so they are for local review only. Record verdicts in results/adjudication.csv
(columns: sample_id, stratum, doc, box, conf, verdict, note). Verdicts for disagreements: ocr_wrong,
label_wrong, ambiguous, matching_artifact. For unlabelled words (--unmatched): ocr_right, ocr_wrong,
spurious (no text there), ambiguous.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

STRATA = {
    "high_conf": lambda e: e.get("status", "wrong") == "wrong" and e["conf"] >= 0.99 and not e["lenient_correct"],
    "low_conf": lambda e: e.get("status", "wrong") == "wrong" and e["conf"] < 0.9 and not e["lenient_correct"],
    "format_only": lambda e: e.get("status", "wrong") == "wrong" and bool(e["lenient_correct"]),  # case or punctuation only
    "unmatched_low": lambda e: e.get("status") == "unmatched" and e["conf"] < 0.9,  # no label under the word
}


def unique_groups(examples: list[dict]) -> list[dict]:
    seen, out = set(), []
    for e in examples:
        key = (e["doc"], tuple(e["box"]))
        if key not in seen:
            seen.add(key)
            out.append(e)
    return out


def sample(examples: list[dict], sizes: dict[str, int], seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for name, n in sizes.items():
        pool = [e for e in unique_groups(examples) if STRATA[name](e)]
        for k, e in enumerate(rng.sample(pool, min(n, len(pool)))):
            out.append({**e, "stratum": name, "sample_id": f"{name}-{k:02d}", "pool_size": len(pool)})
    return out


def render(samples: list[dict], images: Path, out: Path, per_sheet: int = 10) -> list[Path]:
    from PIL import Image, ImageDraw, ImageFont
    out.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 16)
    except OSError:
        font = ImageFont.load_default()
    tile_w, crop_h, text_h = 900, 110, 64
    paths = []
    for s in range(0, len(samples), per_sheet):
        chunk = samples[s:s + per_sheet]
        sheet = Image.new("RGB", (tile_w, (crop_h + text_h) * len(chunk)), "white")
        draw = ImageDraw.Draw(sheet)
        for k, e in enumerate(chunk):
            img = Image.open(images / f"{e['doc']}.png").convert("RGB")
            x0, y0, x1, y1 = e["box"]
            pad = 30
            crop = img.crop((max(0, x0 - pad), max(0, y0 - 12), min(img.width, x1 + pad), min(img.height, y1 + 12)))
            draw_c = ImageDraw.Draw(crop)
            draw_c.rectangle((x0 - max(0, x0 - pad), y0 - max(0, y0 - 12), x1 - max(0, x0 - pad), y1 - max(0, y0 - 12)),
                             outline=(255, 0, 0), width=1)
            scale = min((tile_w - 20) / crop.width, crop_h / crop.height, 4.0)
            crop = crop.resize((max(1, int(crop.width * scale)), max(1, int(crop.height * scale))))
            top = k * (crop_h + text_h)
            sheet.paste(crop, (10, top))
            draw.text((10, top + crop_h + 4), f"{e['sample_id']}  conf={e['conf']:.2f}", fill=(90, 90, 90), font=font)
            draw.text((10, top + crop_h + 22), f"OCR  : {e['pred']}", fill=(0, 0, 160), font=font)
            draw.text((10, top + crop_h + 40), f"FUNSD: {e['gt']}", fill=(0, 110, 0), font=font)
            draw.line((0, top + crop_h + text_h - 1, tile_w, top + crop_h + text_h - 1), fill=(200, 200, 200))
        p = out / f"sheet_{s // per_sheet:02d}.png"
        sheet.save(p)
        paths.append(p)
    return paths


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--examples", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--high", type=int, default=40)
    ap.add_argument("--low", type=int, default=40)
    ap.add_argument("--format", type=int, default=20)
    ap.add_argument("--unmatched", type=int, default=0, help="also sample predicted words with no label under them")
    ap.add_argument("--only", nargs="*", choices=sorted(STRATA), help="sample only these strata")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    sizes = {"high_conf": a.high, "low_conf": a.low, "format_only": a.format, "unmatched_low": a.unmatched}
    sizes = {k: v for k, v in sizes.items() if v and (not a.only or k in a.only)}
    samples = sample(json.loads(a.examples.read_text()), sizes, a.seed)
    (a.out).mkdir(parents=True, exist_ok=True)
    with (a.out / "samples.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "stratum", "pool_size", "doc", "conf", "box"])
        w.writeheader()
        for s in samples:
            w.writerow({k: s[k] for k in w.fieldnames})
    for p in render(samples, a.images, a.out):
        print(p)


if __name__ == "__main__":
    main()
