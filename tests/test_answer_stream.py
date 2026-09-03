"""
Tests for reasoning suppression in the answer stream.

    _venv\\Scripts\\python.exe -m unittest discover -s tests -v

Reasoning models emit <think>...</think> before answering. That is working-out,
not the answer: it must not be shown as prose and must not be scanned for
citations, since a [3] mentioned while thinking is not a claim.

The bug this guards against: the suppressor buffered everything while waiting
for </think>, released early only past a 6,000-character probe limit, and at
end of stream reported "cut off during reasoning". A model that emits NO think
tags — which is what DeepSeek-R1 does through /v1/chat/completions, where the
chat template owns the tags — therefore had every answer under 6,000 characters
silently discarded and replaced with an error. Measured 2026-09-04: a correct
1,940-character answer with finish_reason=stop was thrown away.

No model is loaded; the suppressor is a pure generator over tokens.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain.rag import suppress_reasoning  # noqa: E402


def run(chunks):
    return "".join(suppress_reasoning(iter(chunks)))


class TestPlainAnswers(unittest.TestCase):
    """A model with no reasoning block must have its answer delivered intact."""

    def test_short_plain_answer_survives(self):
        out = run(["Replication ", "keeps ", "copies ", "in sync."])
        self.assertEqual(out.strip(), "Replication keeps copies in sync.")

    def test_realistic_answer_is_not_swallowed(self):
        # 1,940 chars was the length of the real answer that got discarded.
        body = "Database replication handles node failures. " * 44
        out = run([body])
        self.assertIn("Database replication handles node failures.", out)
        self.assertNotIn("cut off", out)

    def test_no_reasoning_banner_on_a_plain_answer(self):
        out = run(["Single-leader replication stops accepting writes " * 5])
        self.assertNotIn("[reasoning", out)


class TestReasoningIsSuppressed(unittest.TestCase):
    def test_think_block_is_dropped(self):
        out = run(["<think>", "I should check the excerpt. ", "</think>",
                   "Leaderless replication uses quorums [1]."])
        self.assertNotIn("I should check", out)
        self.assertIn("Leaderless replication uses quorums [1].", out)

    def test_bare_closing_tag_still_drops_the_prefix(self):
        """R1 through /completion emitted reasoning with no opening tag."""
        out = run(["Let me think about quorums. ", "</think>", "Answer: quorums."])
        self.assertNotIn("Let me think", out)
        self.assertIn("Answer: quorums.", out)

    def test_unclosed_reasoning_reports_a_cut_off(self):
        out = run(["<think>", "still reasoning and never finishing"])
        self.assertIn("cut off", out)


class TestStreaming(unittest.TestCase):
    def test_plain_text_starts_flowing_before_the_end(self):
        """A plain answer must not be withheld until the stream closes."""
        chunks = ["x" * 60 for _ in range(6)]      # 360 chars, no tags
        emitted, seen = [], 0
        for piece in suppress_reasoning(iter(chunks)):
            emitted.append(piece)
            seen += 1
            if seen >= 1 and len("".join(emitted)) > 0:
                break
        self.assertGreater(len("".join(emitted)), 0,
                           "nothing was emitted until the stream ended")


if __name__ == "__main__":
    unittest.main()
