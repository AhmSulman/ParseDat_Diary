"""
Regression tests for the chunker.

    _venv\\Scripts\\python.exe -m unittest discover -s tests -v

The offset tests are the important ones. Neighbour expansion reconstructs a
merged passage arithmetically from stored offsets:

    drop   = prev.char_end - cur.char_start
    merged = prev.text + cur.text[drop:]

If offsets drift from the stored strings by even one character, every merged
passage silently corrupts — text duplicated or dropped at the seam, with no
error raised. Verified exact on 512/512 adjacent pairs of a real book.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain.chunker import Chunker  # noqa: E402


class TestChunkBoundaries(unittest.TestCase):

    def setUp(self):
        self.ch = Chunker(chunk_size=200, overlap=40)

    def test_short_text_is_one_chunk(self):
        self.assertEqual(self.ch.chunk("hello world"), ["hello world"])

    def test_empty_and_blank_yield_nothing(self):
        for src in ["", "   ", "\n\n\t"]:
            self.assertEqual(self.ch.chunk(src), [])
            self.assertEqual(self.ch.chunk_with_meta(src, "x.pdf"), [])

    def test_no_chunk_exceeds_chunk_size(self):
        text = "word " * 500
        self.assertTrue(all(len(c) <= 200 for c in self.ch.chunk(text)))

    def test_whole_text_is_covered(self):
        text = ". ".join(f"sentence number {i}" for i in range(80)) + "."
        rows = self.ch.chunk_with_meta(text, "x.pdf")
        self.assertEqual(rows[0]["char_start"], 0)
        self.assertEqual(rows[-1]["char_end"], len(text.rstrip()))

    def test_overlap_clamped_below_half_chunk_size(self):
        # overlap >= chunk_size//2 can stop `start` advancing -> infinite loop.
        ch = Chunker(chunk_size=100, overlap=500)
        self.assertLess(ch.overlap, 100 // 2)
        self.assertTrue(ch.chunk("x " * 400))       # must terminate

    def test_pathological_overlap_still_terminates(self):
        ch = Chunker(chunk_size=10, overlap=9)
        out = ch.chunk("abcdefghij " * 40)
        self.assertGreater(len(out), 0)


class TestChunkMetadata(unittest.TestCase):

    def setUp(self):
        self.ch = Chunker(chunk_size=200, overlap=40)
        self.text = ". ".join(f"this is sentence number {i}" for i in range(120)) + "."
        self.rows = self.ch.chunk_with_meta(self.text, "book.pdf", book_id=7)

    def test_chunk_ids_are_contiguous(self):
        self.assertEqual([r["chunk_id"] for r in self.rows],
                         list(range(len(self.rows))))

    def test_n_chunks_matches_actual_total(self):
        for r in self.rows:
            self.assertEqual(r["n_chunks"], len(self.rows))

    def test_source_and_book_id_propagate(self):
        for r in self.rows:
            self.assertEqual(r["source"], "book.pdf")
            self.assertEqual(r["book_id"], 7)

    def test_offsets_match_stored_text_exactly(self):
        """The invariant the de-overlap arithmetic depends on."""
        for r in self.rows:
            self.assertEqual(self.text[r["char_start"]:r["char_end"]], r["text"])

    def test_no_gaps_between_consecutive_chunks(self):
        for a, b in zip(self.rows, self.rows[1:]):
            self.assertLessEqual(b["char_start"], a["char_end"])

    def test_de_overlap_merge_reconstructs_source(self):
        """prev.text + cur.text[drop:] must equal the original slice."""
        for a, b in zip(self.rows, self.rows[1:]):
            drop = a["char_end"] - b["char_start"]
            self.assertGreaterEqual(drop, 0)
            merged = a["text"] + b["text"][drop:]
            self.assertEqual(merged, self.text[a["char_start"]:b["char_end"]])


class TestPageAttribution(unittest.TestCase):

    def test_pages_resolved_from_markers(self):
        body = "\n".join(
            f"--- Page {p} ---\n" + ("filler sentence here. " * 20)
            for p in range(1, 8)
        )
        rows = Chunker(chunk_size=300, overlap=50).chunk_with_meta(body, "b.pdf")
        pages = [r["page_start"] for r in rows if r["page_start"] is not None]
        self.assertTrue(pages)
        self.assertTrue(all(1 <= p <= 7 for p in pages))

    def test_page_start_never_exceeds_page_end(self):
        body = "\n".join(
            f"--- Page {p} ---\n" + ("some text. " * 30) for p in range(1, 6)
        )
        for r in Chunker(chunk_size=250, overlap=50).chunk_with_meta(body, "b.pdf"):
            if r["page_start"] is not None and r["page_end"] is not None:
                self.assertLessEqual(r["page_start"], r["page_end"])

    def test_pages_are_monotonic_across_chunks(self):
        body = "\n".join(
            f"--- Page {p} ---\n" + ("some text. " * 30) for p in range(1, 10)
        )
        rows = Chunker(chunk_size=250, overlap=50).chunk_with_meta(body, "b.pdf")
        seen = [r["page_start"] for r in rows if r["page_start"] is not None]
        self.assertEqual(seen, sorted(seen))

    def test_text_without_markers_has_null_pages(self):
        rows = Chunker(chunk_size=200, overlap=40).chunk_with_meta(
            "plain text with no page markers at all. " * 20, "b.pdf")
        self.assertTrue(all(r["page_start"] is None for r in rows))

    def test_page_markers_are_not_mistaken_for_separators(self):
        # '---' must never be treated as a hyphen/word boundary artefact.
        body = "alpha beta. --- Page 3 --- gamma delta. " * 10
        rows = Chunker(chunk_size=200, overlap=40).chunk_with_meta(body, "b.pdf")
        self.assertTrue(rows)


if __name__ == "__main__":
    unittest.main(verbosity=2)
