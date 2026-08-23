"""
Tests for passage merging — the de-overlap arithmetic behind neighbour expansion.

    _venv\\Scripts\\python.exe -m unittest discover -s tests -v

No model is loaded here. `_merge_run` operates purely on stored metadata, so it
is tested directly against hand-built rows. That matters because a merge bug
does not raise — it silently duplicates or drops text at the seam, and the only
symptom is the LLM reading slightly corrupted context.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain.chunker import Chunker  # noqa: E402


def _build(text: str, size: int = 200, overlap: int = 40, source: str = "b.pdf"):
    """Chunk `text`, then return (rows, retriever-like merge helper)."""
    rows = Chunker(chunk_size=size, overlap=overlap).chunk_with_meta(text, source)
    meta = [{**r, "chunk": r.pop("text")} for r in rows]
    neighbors = {(m["source"], m["chunk_id"]): i for i, m in enumerate(meta)}
    return meta, neighbors


def _merge(meta, neighbors, source, cids):
    """
    Standalone mirror of Retriever._merge_run, so the arithmetic can be tested
    without constructing a Retriever (which would build an embedder).
    """
    rows = [meta[neighbors[(source, c)]] for c in cids if (source, c) in neighbors]
    if not rows:
        return None
    text = rows[0]["chunk"]
    for prev, cur in zip(rows, rows[1:]):
        drop = prev["char_end"] - cur["char_start"]
        drop = max(0, min(drop, len(cur["chunk"])))
        text += cur["chunk"][drop:]
    return text


class TestMergeReconstructsSource(unittest.TestCase):

    def setUp(self):
        self.text = ". ".join(f"sentence number {i} here" for i in range(200)) + "."
        self.meta, self.neighbors = _build(self.text)

    def test_enough_chunks_to_be_meaningful(self):
        self.assertGreater(len(self.meta), 5)

    def test_merging_a_pair_matches_the_original_slice(self):
        for i in range(len(self.meta) - 1):
            merged = _merge(self.meta, self.neighbors, "b.pdf", [i, i + 1])
            expected = self.text[self.meta[i]["char_start"]:self.meta[i + 1]["char_end"]]
            self.assertEqual(merged, expected, f"pair {i}")

    def test_merging_a_triple_matches_the_original_slice(self):
        for i in range(len(self.meta) - 2):
            merged = _merge(self.meta, self.neighbors, "b.pdf", [i, i + 1, i + 2])
            expected = self.text[self.meta[i]["char_start"]:self.meta[i + 2]["char_end"]]
            self.assertEqual(merged, expected, f"triple {i}")

    def test_merged_run_never_duplicates_the_overlap(self):
        """The failure mode a naive concatenation produces."""
        a, b = self.meta[0], self.meta[1]
        overlap_len = a["char_end"] - b["char_start"]
        self.assertGreater(overlap_len, 0, "expected chunks to overlap")
        naive = a["chunk"] + b["chunk"]
        merged = _merge(self.meta, self.neighbors, "b.pdf", [0, 1])
        self.assertEqual(len(merged), len(naive) - overlap_len)

    def test_single_chunk_run_is_unchanged(self):
        self.assertEqual(_merge(self.meta, self.neighbors, "b.pdf", [3]),
                         self.meta[3]["chunk"])

    def test_unknown_chunk_ids_are_skipped(self):
        self.assertEqual(_merge(self.meta, self.neighbors, "b.pdf", [0, 9999]),
                         self.meta[0]["chunk"])

    def test_missing_source_returns_none(self):
        self.assertIsNone(_merge(self.meta, self.neighbors, "nope.pdf", [0]))


class TestMergeEdgeCases(unittest.TestCase):

    def test_gap_between_chunks_appends_instead_of_slicing_from_end(self):
        """
        If char_end < char_start (a gap), `drop` goes negative. Unclamped,
        cur['chunk'][-n:] would silently take the TAIL of the chunk — text
        vanishes with no error. The clamp must make it a plain append.
        """
        meta = [
            {"source": "x.pdf", "chunk_id": 0, "chunk": "AAAA",
             "char_start": 0, "char_end": 4},
            {"source": "x.pdf", "chunk_id": 1, "chunk": "BBBB",
             "char_start": 10, "char_end": 14},   # gap: 4 -> 10
        ]
        neighbors = {("x.pdf", 0): 0, ("x.pdf", 1): 1}
        self.assertEqual(_merge(meta, neighbors, "x.pdf", [0, 1]), "AAAABBBB")

    def test_overlap_larger_than_chunk_does_not_underflow(self):
        meta = [
            {"source": "x.pdf", "chunk_id": 0, "chunk": "AAAAAAAAAA",
             "char_start": 0, "char_end": 10},
            {"source": "x.pdf", "chunk_id": 1, "chunk": "BB",
             "char_start": 0, "char_end": 2},     # drop would be 10 > len("BB")
        ]
        neighbors = {("x.pdf", 0): 0, ("x.pdf", 1): 1}
        self.assertEqual(_merge(meta, neighbors, "x.pdf", [0, 1]), "AAAAAAAAAA")

    def test_merge_is_stable_across_real_book_text(self):
        body = "\n".join(
            f"--- Page {p} ---\n" + ". ".join(f"line {p}.{i}" for i in range(25))
            for p in range(1, 12)
        )
        meta, neighbors = _build(body, size=300, overlap=60, source="real.pdf")
        for i in range(len(meta) - 1):
            merged = _merge(meta, neighbors, "real.pdf", [i, i + 1])
            expected = body[meta[i]["char_start"]:meta[i + 1]["char_end"]]
            self.assertEqual(merged, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
