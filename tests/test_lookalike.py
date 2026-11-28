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

    def test_different_lengths_use_an_alignment(self):
        self.assertEqual(lookalike.edits("$212,424.12", "$212,$212,424.12"), [("", "$212,")])

    def test_lookalike_classification(self):
        self.assertTrue(lookalike.is_lookalike_error("H0WY6ZSS", "HOWY6ZSS"))    # 0 for O
        self.assertTrue(lookalike.is_lookalike_error("L63BI", "L6381"))          # B for 8 and I for 1
        self.assertTrue(lookalike.is_lookalike_error("insurance", "Insurance"))  # case
        self.assertTrue(lookalike.is_lookalike_error("$84.07", "$84,07"))
        self.assertFalse(lookalike.is_lookalike_error("labor", "tabor"))
        self.assertFalse(lookalike.is_lookalike_error("37", "3737"))
        self.assertFalse(lookalike.is_lookalike_error("same", "same"))


class Discriminability(unittest.TestCase):
    def test_ideal_observer_error(self):
        self.assertAlmostEqual(lookalike.ideal_observer_error(0.0), 0.5)
        self.assertAlmostEqual(lookalike.ideal_observer_error(2.0), 0.158655, places=5)  # Phi(-1)

    def test_undefined_and_noise_free_cases(self):
        word = {"font": "Verdana", "size": 20, "bold": False}
        self.assertIsNone(lookalike.discriminability("37", "3737", word, 1, 50.0))
        if HAVE_FONTS:
            self.assertEqual(lookalike.discriminability("0", "O", word, 0, 0.0), math.inf)

    @unittest.skipUnless(HAVE_FONTS, "needs Pillow and the macOS fonts")
    def test_glyph_separation(self):
        word = {"font": "Verdana", "size": 20, "bold": False}
        self.assertEqual(lookalike.glyph_separation("O", "O", word, 2), 0)
        self.assertGreater(lookalike.glyph_separation("0", "O", word, 2), 0)
        # Sliding sideways can only make two glyphs look more alike, never less.
        self.assertLessEqual(lookalike.glyph_separation("l", "I", word, 2), lookalike.glyph_separation("l", "I", word, 2, slack=0))


if __name__ == "__main__":
    unittest.main()
