"""
Tests for page-marker suppression.

    _venv\Scripts\python.exe -m unittest discover -s tests -v

`--- Page N ---` markers must survive into the STORED chunk text, because
chunk offsets are recorded against it and the retriever's de-overlap
arithmetic depends on those offsets matching character for character. They
must NOT reach the embedder or the prompt: measured on the previous corpus,
3,928 markers spanned 38.8% of chunks and cost ~14,480 tokens of furniture in
every context window, and they carry no meaning to match a query against.

No model is loaded here — both paths under test are pure text transforms.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain.chunker import Chunker            # noqa: E402
from core.normalize import strip_furniture   # noqa: E402
from brain.llm import LocalLLM               # noqa: E402

SAMPLE = ("--- Page 12 ---\nReplication means keeping a copy of the same data\n"
          "--- Page 13 ---\non several machines.")


class TestStripFurniture(unittest.TestCase):
    def test_removes_markers(self):
        out = strip_furniture(SAMPLE)
        self.assertNotIn("Page 12", out)
        self.assertNotIn("Page 13", out)
        self.assertIn("Replication means", out)
        self.assertIn("on several machines", out)

    def test_idempotent(self):
        once = strip_furniture(SAMPLE)
        self.assertEqual(once, strip_furniture(once))

    def test_leaves_clean_text_alone(self):
        clean = "Nothing to strip here."
        self.assertEqual(strip_furniture(clean), clean)

    def test_leaves_a_boundary_where_the_marker_was(self):
        # The marker sat on its own line. The regex is anchored to whole lines,
        # so the surrounding newlines survive and the words either side of it
        # do not run together.
        self.assertRegex(strip_furniture(SAMPLE), r"data\s+on several machines")

    def test_matches_chunker_regex(self):
        self.assertEqual(strip_furniture(SAMPLE),
                         Chunker.strip_page_markers(SAMPLE).strip())


class TestPromptExcludesFurniture(unittest.TestCase):
    def test_build_rag_messages_strips_markers(self):
        llm = LocalLLM.__new__(LocalLLM)          # no config read, no model
        passages = [{"source": "ddia.pdf", "chunk": SAMPLE,
                     "page_start": 12, "page_end": 13}]
        msgs = LocalLLM.build_rag_messages(llm, "how does replication work?",
                                           passages, library_titles=["ddia.pdf"],
                                           char_budget=20000)
        system = msgs[0]["content"]
        self.assertNotIn("--- Page", system)
        self.assertIn("Replication means", system)
        # The page span is still cited — it comes from metadata, not the text.
        self.assertIn("p.12-13", system)


if __name__ == "__main__":
    unittest.main()
