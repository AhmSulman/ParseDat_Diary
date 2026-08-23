"""
Regression tests for the text pipeline: normalisation and the quality gate.

Stdlib unittest only — no pytest dependency.

    _venv\\Scripts\\python.exe -m unittest discover -s tests -v

The cases here are not hypothetical. Every one of them is a defect measured in
the real corpus, or a false positive that a naive implementation produced and
that would have corrupted real books.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.normalize import normalize, build_word_freq, page_offsets  # noqa: E402
from core.quality import score, homoglyph_count                      # noqa: E402

_FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
_POISONED = os.path.join(_FIXTURES, "poisoned_homoglyph_sample.txt")


def _read_poisoned() -> str:
    with open(_POISONED, encoding="utf-8") as f:
        return f.read()


class TestNormaliseRepairs(unittest.TestCase):
    """Defects that must be fixed."""

    def test_ligatures_are_expanded(self):
        # 3,411 of these in the corpus. The embedder never saw "efficient".
        self.assertEqual(normalize("e\ufb03cient \ufb01le"), "efficient file")

    def test_ligature_fi_becomes_two_chars(self):
        self.assertNotIn("\ufb01", normalize("speci\ufb01c"))
        self.assertIn("specific", normalize("speci\ufb01c"))

    def test_nbsp_and_soft_hyphen_removed(self):
        out = normalize("a\u00a0b\u00adc")
        self.assertNotIn("\u00a0", out)
        self.assertNotIn("\u00ad", out)

    def test_control_chars_stripped_but_newline_tab_kept(self):
        out = normalize("a\x00b\x07c\nd\te")
        self.assertNotIn("\x00", out)
        self.assertNotIn("\x07", out)
        self.assertIn("\n", out)
        self.assertIn("\t", out)

    def test_smart_quotes_folded_to_ascii(self):
        out = normalize("\u2018a\u2019 \u201cb\u201d \u2013 \u2026")
        for ch in "\u2018\u2019\u201c\u201d\u2013\u2026":
            self.assertNotIn(ch, out)


class TestDehyphenation(unittest.TestCase):
    """
    The hard case: 'en-\\ncapsulated' must fuse, 'beginning-\\nlevel' must not.
    Decided from corpus word frequency, so both directions need testing.
    """

    def test_split_word_is_fused_when_fused_form_exists(self):
        text = "encapsulated data\nen-\ncapsulated more"
        self.assertIn("encapsulated", normalize(text))
        self.assertNotIn("en-\ncapsulated", normalize(text))

    def test_real_compound_keeps_its_hyphen(self):
        # Both halves are common words and the fused form never appears,
        # so this is a compound, not a line-break split.
        text = "the beginning of a level\nbeginning-\nlevel course"
        out = normalize(text)
        self.assertIn("beginning-level", out)
        self.assertNotIn("beginninglevel", out)

    def test_fragment_prefix_is_fused(self):
        # 'experi' is not a word, so this must fuse even without corpus evidence.
        out = normalize("experi-\nmentally")
        self.assertIn("experimentally", out)

    def test_newline_always_removed_so_pass_is_idempotent(self):
        for text in ["en-\ncapsulated", "beginning-\nlevel", "experi-\nmentally"]:
            self.assertNotIn("-\n", normalize(text), text)

    def test_trailing_space_before_break_still_handled(self):
        # Regression: stripping trailing whitespace AFTER de-hyphenating left
        # 'word-   \n' unmatched on pass 1 and matched on pass 2, which made
        # normalize() non-idempotent on a real book (772 breaks, 21 left).
        once = normalize("objects here\nob-   \njects")
        self.assertEqual(normalize(once), once)
        self.assertNotIn("-\n", once)


class TestNormaliseDoesNotCorrupt(unittest.TestCase):
    """
    False positives that would destroy real data. These looked like defects
    during analysis and are not.
    """

    def test_camelcase_identifiers_are_never_split(self):
        # 6,119 [a-z][A-Z] hits in the corpus are code, not missing spaces.
        for ident in ["MutableMapping", "ValueError", "CreditCard", "trainPredictPlot"]:
            self.assertIn(ident, normalize(f"use {ident} here"))

    def test_no_space_injected_at_case_boundaries(self):
        src = "getValue setName IndexError"
        self.assertEqual(normalize(src), src)

    def test_leading_indentation_preserved(self):
        # Python listings are indentation-significant.
        src = "def f():\n    if x:\n        return 1\n"
        self.assertIn("\n    if x:", normalize(src))
        self.assertIn("\n        return 1", normalize(src))

    def test_page_markers_survive_verbatim(self):
        src = "text\n--- Page 12 ---\nmore\n--- Page 13 (OCR) ---\nend"
        out = normalize(src)
        self.assertIn("--- Page 12 ---", out)
        self.assertIn("--- Page 13 (OCR) ---", out)

    def test_page_offsets_reports_each_marker(self):
        src = "a\n--- Page 4 ---\nb\n--- Page 5 ---\nc"
        self.assertEqual([p for _, p in page_offsets(src)], [4, 5])

    def test_idempotent_on_empty_and_plain(self):
        for src in ["", "plain text", "a\nb\n"]:
            self.assertEqual(normalize(normalize(src)), normalize(src))


class TestHomoglyphDetection(unittest.TestCase):
    """
    Script-mixing awareness: a watermark hides inside Latin words, real maths
    notation stands alone. Getting this wrong discards mml-book.pdf.
    """

    def test_confusable_inside_latin_word_is_counted(self):
        # 'The' with a Cyrillic 'e' (U+0435)
        self.assertEqual(homoglyph_count("Th\u0435 class"), 1)

    def test_standalone_greek_is_not_counted(self):
        # Real maths: alpha, rho, omicron as their own tokens.
        self.assertEqual(homoglyph_count("let \u03b1 = \u03c1 + \u03bf"), 0)

    def test_greek_formula_among_latin_prose_is_not_counted(self):
        self.assertEqual(homoglyph_count("where \u03b1 denotes the learning rate"), 0)


class TestQualityGate(unittest.TestCase):

    def test_clean_english_passes(self):
        text = ("The quick brown fox jumps over the lazy dog. " * 60)
        self.assertTrue(score(text).passed, score(text).summary())

    def test_too_short_fails(self):
        self.assertFalse(score("tiny").passed)

    @unittest.skipUnless(os.path.exists(_POISONED), "fixture not present")
    def test_homoglyph_watermarked_book_is_rejected(self):
        """
        The real failure this gate exists for: a book whose Latin vowels were
        replaced with Cyrillic lookalikes. It contributed 791 chunks of noise.
        """
        r = score(_read_poisoned())
        self.assertFalse(r.passed, "watermarked book must be rejected")
        joined = " ".join(r.reasons)
        self.assertIn("homoglyph", joined)
        self.assertLess(r.metrics["mean_word_len"], 3.0)

    @unittest.skipUnless(os.path.exists(_POISONED), "fixture not present")
    def test_rejection_survives_normalisation(self):
        # NFKC does not fold Cyrillic to Latin, so normalising must not hide it.
        self.assertFalse(score(normalize(_read_poisoned())).passed)

    @unittest.skipUnless(os.path.exists(_POISONED), "fixture not present")
    def test_alpha_ratio_alone_would_not_have_caught_it(self):
        """
        Documents why the gate is multi-metric: the poisoned book's alpha ratio
        sits inside the healthy range (0.713-0.781) observed across 16 books.
        """
        r = score(_read_poisoned())
        self.assertGreater(r.metrics["alpha_ratio"], 0.70)


if __name__ == "__main__":
    unittest.main(verbosity=2)
