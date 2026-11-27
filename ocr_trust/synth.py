"""Synthetic form pages with exact word labels, rendered at controlled levels of degradation.

FUNSD's labels contain errors (see results/adjudication.csv), so any calibration measured on it has to be
corrected by hand. Rendered pages remove that problem: every word's text and box is known exactly. Each
token is also tagged with what kind of value it is, so errors can be broken down by the fields that matter
in document extraction (IDs, amounts, dates) rather than by prose words.

    python -m ocr_trust.synth --out synth --pages-per-level 5

writes synth/images/<page>.png and synth/labels/<page>.json. Requires Pillow and the macOS system fonts
listed in FONTS; the rendered pages are committed, so analysis needs neither.

Degradation levels (applied in this order: rotate, blur, downscale, fade, noise, speckle, JPEG):

    0  clean 150 dpi render
    1  light scan:  blur 0.6, 75% scale, noise 6, JPEG 70
    2  office scan: blur 0.9, 55% scale, rotate up to 1 deg, fade, noise 10, JPEG 50
    3  bad fax:     blur 1.2, 45% scale, rotate up to 1.5 deg, fade, noise 16, speckle, JPEG 35
"""
from __future__ import annotations

import argparse
import io
import json
import math
import random
import string
from dataclasses import dataclass
from pathlib import Path

FONT_DIR = Path("/System/Library/Fonts/Supplemental")
FONTS = ["Arial", "Times New Roman", "Courier New", "Georgia", "Verdana", "Tahoma", "Trebuchet MS"]
PAGE_W, PAGE_H = 1275, 1650  # US letter at 150 dpi
MARGIN = 90

LEVELS = {
    0: dict(blur=0.0, scale=1.0, rotate=0.0, fade=1.0, noise=0, speckle=0, jpeg=None),
    1: dict(blur=0.6, scale=0.75, rotate=0.0, fade=1.0, noise=6, speckle=0, jpeg=70),
    2: dict(blur=0.9, scale=0.55, rotate=1.0, fade=0.75, noise=10, speckle=0, jpeg=50),
    3: dict(blur=1.2, scale=0.45, rotate=1.5, fade=0.6, noise=16, speckle=400, jpeg=35),
}

COMMON = """account address after amount approved balance billing business by call cash change check city
claim client code company contact cost credit customer date day delivery department description details
due each estimate for from full general goods hours in insurance invoice item labor last name net new
note number of office on order other paid part payment per period phone please policy price received
reference remit report return sales service ship shipping signature state statement subtotal supply
tax terms the to total transfer unit vendor week with work year your""".split()

CONFUSABLE = "0O1lI5S8B2Z6G"
ID_PREFIXES = ["INV", "PO", "ACCT", "REF", "SKU", "CLM", "ORD", "TX"]


@dataclass
class Token:
    text: str
    kind: str  # word | rare_word | amount | quantity | date | id | phone | email | label


class TokenSource:
    """Seeded generator for each kind of token found on business forms."""

    def __init__(self, rng: random.Random, dictionary: list[str]):
        self.rng = rng
        self.dictionary = dictionary

    def word(self) -> Token:
        return Token(self.rng.choice(COMMON), "word")

    def rare_word(self) -> Token:
        return Token(self.rng.choice(self.dictionary), "rare_word")

    def amount(self) -> Token:
        v = self.rng.choice([self.rng.uniform(1, 99), self.rng.uniform(100, 9999), self.rng.uniform(10000, 250000)])
        return Token(f"${v:,.2f}", "amount")

    def quantity(self) -> Token:
        return Token(str(self.rng.randint(1, 480)), "quantity")

    def date(self) -> Token:
        y, m, d = self.rng.randint(1985, 2026), self.rng.randint(1, 12), self.rng.randint(1, 28)
        fmt = self.rng.choice(["{m:02d}/{d:02d}/{y}", "{y}-{m:02d}-{d:02d}", "{d:02d}.{m:02d}.{y}"])
        return Token(fmt.format(y=y, m=m, d=d), "date")

    def ident(self) -> Token:
        """Alphanumeric codes, deliberately rich in look-alike characters (0/O, 1/l/I, 5/S, 8/B)."""
        body = "".join(self.rng.choice(CONFUSABLE + string.ascii_uppercase + string.digits) for _ in range(self.rng.randint(5, 9)))
        prefix = self.rng.choice(ID_PREFIXES)
        return Token(f"{prefix}-{body}" if self.rng.random() < 0.7 else body, "id")

    def phone(self) -> Token:
        return Token(f"({self.rng.randint(201, 989)}){self.rng.randint(200, 999)}-{self.rng.randint(0, 9999):04d}", "phone")

    def email(self) -> Token:
        user = self.rng.choice(self.dictionary)[:8] + str(self.rng.randint(1, 99))
        return Token(f"{user}@{self.rng.choice(self.dictionary)[:10]}.com", "email")

    def value(self, kind: str) -> Token:
        return getattr(self, {"id": "ident"}.get(kind, kind))()


def load_dictionary(path: Path = Path("/usr/share/dict/words")) -> list[str]:
    words = [w.strip() for w in path.read_text().splitlines()]
    return sorted({w.lower() for w in words if 4 <= len(w) <= 11 and w.isalpha() and w.isascii()})


def _font(name: str, size: int, bold: bool = False):
    from PIL import ImageFont
    return ImageFont.truetype(str(FONT_DIR / f"{name}{' Bold' if bold else ''}.ttf"), size)


def compose(rng: random.Random, src: TokenSource) -> list[tuple[list[Token], str, int, bool]]:
    """Lines of a form page: (tokens, font, size, bold). Header, key-value block, line-item table, notes."""
    font = rng.choice(FONTS)
    size = rng.randint(17, 24)
    lines: list[tuple[list[Token], str, int, bool]] = []
    lines.append(([Token(w.upper(), "label") for w in rng.sample(["statement", "invoice", "claim", "order", "remittance"], 1)]
                  + [src.ident()], font, size + 12, True))
    for kind_pair in rng.sample([("id", "date"), ("phone", "email"), ("date", "amount"), ("id", "phone"),
                                 ("amount", "date"), ("email", "id")], 5):
        row: list[Token] = []
        for kind in kind_pair:
            row += [Token(rng.choice(COMMON).capitalize() + ":", "label"), src.value(kind)]
        lines.append((row, font, size, False))
    lines.append(([Token(h, "label") for h in ("Item", "Description", "Qty", "Amount")], font, size, True))
    for _ in range(rng.randint(12, 15)):
        desc = [src.word() if rng.random() < 0.5 else src.rare_word() for _ in range(rng.randint(2, 3))]
        lines.append(([src.ident(), *desc, src.quantity(), src.amount()], font, size, False))
    for _ in range(14):  # notes fill the rest of the page; render() stops at the bottom margin
        lines.append(([src.word() if rng.random() < 0.6 else src.rare_word() for _ in range(rng.randint(6, 9))], font, size, False))
    return lines


def render(lines) -> tuple["Image.Image", list[dict]]:
    from PIL import Image, ImageDraw
    img = Image.new("L", (PAGE_W, PAGE_H), 255)
    draw = ImageDraw.Draw(img)
    y, words = MARGIN, []
    for tokens, font_name, size, bold in lines:
        font = _font(font_name, size, bold)
        x = MARGIN
        space = draw.textlength(" ", font=font)
        for t in tokens:
            width = draw.textlength(t.text, font=font)
            if x + width > PAGE_W - MARGIN:  # wrap
                x, y = MARGIN, y + int(size * 1.6)
            draw.text((x, y), t.text, font=font, fill=0)
            words.append({"text": t.text, "kind": t.kind, "box": list(draw.textbbox((x, y), t.text, font=font))})
            x += width + space * 2
        y += int(size * (2.0 if bold else 1.7))
        if y > PAGE_H - MARGIN - size:
            break
    return img, words


def rotate_point(x: float, y: float, cx: float, cy: float, degrees: float) -> tuple[float, float]:
    """Where a pixel lands after PIL's Image.rotate(degrees) about (cx, cy): counter-clockwise on screen, y down."""
    t = math.radians(degrees)
    dx, dy = x - cx, y - cy
    return cx + dx * math.cos(t) + dy * math.sin(t), cy - dx * math.sin(t) + dy * math.cos(t)


def degrade(img, words: list[dict], level: int, rng: random.Random):
    from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter
    p = LEVELS[level]
    if p["rotate"]:
        angle = rng.uniform(-p["rotate"], p["rotate"])
        cx, cy = img.width / 2, img.height / 2
        img = img.rotate(angle, resample=Image.BICUBIC, fillcolor=255)
        for w in words:
            x0, y0, x1, y1 = w["box"]
            pts = [rotate_point(x, y, cx, cy, angle) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
            w["box"] = [min(p_[0] for p_ in pts), min(p_[1] for p_ in pts), max(p_[0] for p_ in pts), max(p_[1] for p_ in pts)]
    if p["blur"]:
        img = img.filter(ImageFilter.GaussianBlur(p["blur"]))
    if p["scale"] != 1.0:
        img = img.resize((round(img.width * p["scale"]), round(img.height * p["scale"])), Image.LANCZOS)
        for w in words:
            w["box"] = [v * p["scale"] for v in w["box"]]
    if p["fade"] != 1.0:
        img = ImageEnhance.Contrast(img).enhance(p["fade"])
    if p["noise"]:
        # Seeded Gaussian noise (Image.effect_noise is not seeded, so pages would not regenerate identically).
        sigma = p["noise"]
        noise = bytes(min(255, max(0, round(128 + rng.gauss(0, sigma)))) for _ in range(img.width * img.height))
        img = ImageChops.add(img, Image.frombytes("L", img.size, noise), 1.0, -128)
    if p["speckle"]:
        d = ImageDraw.Draw(img)
        for _ in range(p["speckle"]):
            x, y, r = rng.uniform(0, img.width), rng.uniform(0, img.height), rng.uniform(0.4, 1.4)
            d.ellipse((x - r, y - r, x + r, y + r), fill=rng.randint(0, 90))
    if p["jpeg"]:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=p["jpeg"])
        img = Image.open(io.BytesIO(buf.getvalue())).convert("L")
    for w in words:
        w["box"] = [round(v, 1) for v in w["box"]]
    return img


def make_page(page_id: str, level: int, seed: int, dictionary: list[str]):
    rng = random.Random(seed)
    img, words = render(compose(rng, TokenSource(rng, dictionary)))
    img = degrade(img, words, level, rng)
    label = {"page": page_id, "level": level, "seed": seed, "width": img.width, "height": img.height, "words": words}
    return img, label


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("synth"))
    ap.add_argument("--pages-per-level", type=int, default=5)
    a = ap.parse_args(argv)
    (a.out / "images").mkdir(parents=True, exist_ok=True)
    (a.out / "labels").mkdir(parents=True, exist_ok=True)
    dictionary = load_dictionary()
    for level in LEVELS:
        for i in range(a.pages_per_level):
            page_id = f"L{level}_p{i:02d}"
            img, label = make_page(page_id, level, seed=1000 * level + i, dictionary=dictionary)
            img.save(a.out / "images" / f"{page_id}.png", optimize=True)
            (a.out / "labels" / f"{page_id}.json").write_text(json.dumps(label, indent=0))
            print(f"{page_id}: {len(label['words'])} words, {img.width}x{img.height}")


if __name__ == "__main__":
    main()
