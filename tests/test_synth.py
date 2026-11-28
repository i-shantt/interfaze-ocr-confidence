import json
import random
import unittest
from pathlib import Path

from ocr_trust import match, synth

ROOT = Path(__file__).resolve().parents[1]

try:
    import PIL  # noqa: F401
    HAVE_PIL = True
except ImportError:
    HAVE_PIL = False


class Rotation(unittest.TestCase):
    @unittest.skipUnless(HAVE_PIL, "Pillow not installed")
    def test_rotate_point_matches_pil(self):
        """A single dark pixel lands where rotate_point says it will."""
        from PIL import Image
        for angle in (-1.5, 0.7, 10.0):
            img = Image.new("L", (400, 300), 255)
            img.putpixel((310, 70), 0)
            out = img.rotate(angle, resample=Image.NEAREST, fillcolor=255)
            dark = [(x, y) for y in range(out.height) for x in range(out.width) if out.getpixel((x, y)) < 128]
            ex, ey = synth.rotate_point(310.5, 70.5, 200, 150, angle)
            self.assertTrue(any(abs(x + 0.5 - ex) <= 1 and abs(y + 0.5 - ey) <= 1 for x, y in dark), (angle, dark, ex, ey))


class Pages(unittest.TestCase):
    def test_committed_labels_are_inside_their_images(self):
        labels = sorted((ROOT / "synth" / "labels").glob("*.json"))
        self.assertEqual(len(labels), 20)
        for path in labels:
            lab = json.loads(path.read_text())
            for w in lab["words"]:
                x0, y0, x1, y1 = w["box"]
                self.assertTrue(0 <= x0 < x1 <= lab["width"] and 0 <= y0 < y1 <= lab["height"], (path.name, w))

    def test_labels_load_with_kinds(self):
        words = match.load_gt(ROOT / "synth" / "labels" / "L0_p00.json")
        self.assertTrue(words and all(w.kind for w in words))
        self.assertIn("id", {w.kind for w in words})

    @unittest.skipUnless(HAVE_PIL and synth.FONT_DIR.exists(), "needs Pillow and the macOS fonts")
    def test_generation_is_deterministic(self):
        dictionary = synth.load_dictionary()
        img_a, a = synth.make_page("t", 3, seed=3001, dictionary=dictionary)
        img_b, b = synth.make_page("t", 3, seed=3001, dictionary=dictionary)
        self.assertEqual(a, b)
        self.assertEqual(img_a.tobytes(), img_b.tobytes())

    @unittest.skipUnless(HAVE_PIL and synth.FONT_DIR.exists(), "needs Pillow and the macOS fonts")
    def test_clean_render_is_the_page_without_noise(self):
        """The noise-free render used to measure page noise lines up with the committed page: identical on a
        level-0 page (no noise at all), and within the added noise on a rotated level-2 page."""
        from PIL import Image
        dictionary = synth.load_dictionary()
        for page, sigma_max in (("L0_p01", 0.0), ("L2_p03", 2.5 * synth.LEVELS[2]["noise"])):
            label = json.loads((ROOT / "synth" / "labels" / f"{page}.json").read_text())
            clean, clean_label = synth.make_page(page, label["level"], label["seed"], dictionary, clean=True)
            real = Image.open(ROOT / "synth" / "images" / f"{page}.png").convert("L")
            self.assertEqual(clean_label["words"], label["words"])
            r, c = real.tobytes(), clean.tobytes()
            rms = (sum((x - y) ** 2 for x, y in zip(r, c)) / len(r)) ** 0.5
            self.assertLessEqual(rms, sigma_max, page)

    def test_token_kinds(self):
        src = synth.TokenSource(random.Random(0), ["alpha", "bravo"])
        self.assertRegex(src.amount().text, r"^\$[\d,]+\.\d\d$")
        self.assertRegex(src.phone().text, r"^\(\d{3}\)\d{3}-\d{4}$")
        self.assertEqual(src.value("id").kind, "id")


if __name__ == "__main__":
    unittest.main()
