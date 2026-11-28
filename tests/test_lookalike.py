import math
import unittest

from ocr_trust import lookalike, synth

try:
    import PIL  # noqa: F401
    HAVE_FONTS = synth.FONT_DIR.exists()
except ImportError:
    HAVE_FONTS = False


class Edits(unittest.TestCase):
    def test_same_length_strings_are_compared_position_by_position(self):
        self.assertEqual(lookalike.edits("TX-0O0OC92", "TX-0000C92"), [("O", "0"), ("O", "0")])
        self.assertEqual(lookalike.swaps("L63BI", "L6381"), [("B", "8"), ("I", "1")])

    def test_different_lengths_use_an_alignment(self):
        self.assertEqual(lookalike.edits("$212,424.12", "$212,$212,424.12"), [("", "$212,")])
        self.assertIsNone(lookalike.swaps("37", "3737"))

    def test_categories(self):
        self.assertEqual(lookalike.category("H0WY6ZSS", "HOWY6ZSS"), "lookalike")   # 0 for O
        self.assertEqual(lookalike.category("ZNIXW0", "ZNIXWo"), "lookalike")       # 0 for o
        self.assertEqual(lookalike.category("L63BI", "L6381"), "lookalike")         # B for 8 and I for 1
        self.assertEqual(lookalike.category("insurance", "Insurance"), "case")
        self.assertEqual(lookalike.category("$84.07", "$84,07"), "punctuation")
        self.assertEqual(lookalike.category("labor", "tabor"), "other")
        self.assertEqual(lookalike.category("37", "3737"), "other")


class Discriminability(unittest.TestCase):
    def test_undefined_and_degenerate_cases(self):
        word = {"font": "Verdana", "size": 20, "bold": False}
        self.assertIsNone(lookalike.discriminability("37", "3737", word, 1, 50.0))
        if HAVE_FONTS:
            self.assertEqual(lookalike.discriminability("0", "O", word, 0, 0.0), math.inf)  # noise-free page

    @unittest.skipUnless(HAVE_FONTS, "needs Pillow and the macOS fonts")
    def test_glyph_separation(self):
        word = {"font": "Verdana", "size": 20, "bold": False}
        self.assertEqual(lookalike.glyph_separation("O", "O", word, 2), 0)
        # Blurring and downscaling (level 2) brings two different characters closer than on a clean page.
        clean, scanned = lookalike.glyph_separation("0", "O", word, 0), lookalike.glyph_separation("0", "O", word, 2)
        self.assertGreater(scanned, 0)
        self.assertGreater(clean, scanned)


if __name__ == "__main__":
    unittest.main()
