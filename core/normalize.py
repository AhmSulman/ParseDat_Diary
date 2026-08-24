"""
Text Normaliser — repairs PDF text-layer damage before chunking
================================================================
PDF character extraction is not reliable. `page.get_text("text")` returns what
the PDF's font tables claim, and for real books that means, measured across this
corpus:

  ligatures            3,411   'ef<U+FB01>cient'  -> the embedder never sees "efficient"
  hyphen + linebreak   3,369   'en-\\ncapsulated' -> one word indexed as two
  smart quotes        10,914
  nbsp / control       1,046   invisible, breaks tokenisation

Every one of those makes a word embed as a *different token* than the same word
written normally, which is why retrieval degrades on affected books.

WHAT THIS DELIBERATELY DOES NOT DO
----------------------------------
Two "obvious" cleanups would corrupt this corpus, verified against real samples:

  1. Inserting spaces at [a-z][A-Z] boundaries. 6,119 hits looked like missing
     spaces; they are almost all legitimate identifiers from programming books —
     MutableMapping, ValueError, CreditCard. Splitting those destroys code.
  2. Collapsing leading whitespace. Python listings are indentation-significant.

Normalisation is IDEMPOTENT: running it twice yields a byte-identical result.
That matters because it runs both on fresh extraction and on re-index.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

# Page markers written by the OCR stage — the chunker maps chunks to pages via
# these, so they must survive normalisation untouched.
_PAGE_MARKER = re.compile(r"^-{2,}\s*Page\s+\d+[^\n]*?-{2,}$", re.MULTILINE)

# Control characters, keeping \t (09) and \n (0A).
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# A word, hyphen, newline, then a word: the line-break hyphenation case.
# Requires a word character immediately before the hyphen, so the '---' in a
# page marker can never match.
_HYPHEN_BREAK = re.compile(r"(\w+)-\n[ \t]*(\w+)")

# Lowercase alphabetic words, for the frequency table.
_WORD = re.compile(r"[A-Za-z]{2,}")

# Punctuation that NFKC leaves alone but that hurts exact matching.
_PUNCT_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "′": "'", "″": '"',
    "–": "-", "—": "-", "―": "-", "−": "-",
    "…": "...",
    "­": "",          # soft hyphen: invisible, always noise
    "​": "", "‌": "", "‍": "", "﻿": "",
}
_PUNCT_RE = re.compile("|".join(map(re.escape, _PUNCT_MAP)))


def build_word_freq(text: str) -> Counter:
    """
    Word frequency over the document, used to decide de-hyphenation.

    Line-break hyphenations are REMOVED before counting. Counting them would be
    circular: 'experi-\\nmentally' would register 'experi' as a real word, and
    the "both halves are words" rule would then preserve a hyphen that never
    belonged there. Only clean, unbroken occurrences are evidence.
    """
    return Counter(w.lower() for w in _WORD.findall(_HYPHEN_BREAK.sub(" ", text)))


def _dehyphenate(text: str, freq: Counter) -> str:
    """
    Join words broken across a line break, without destroying real compounds.

    A naive rule is wrong in both directions:
        'en-\\ncapsulated'    -> must become  'encapsulated'
        'beginning-\\nlevel'  -> must stay    'beginning-level'

    Decided from the corpus itself, so there is no dictionary dependency:

      fused form seen elsewhere        -> join          (encapsulated exists)
      both halves are real words       -> keep hyphen   (beginning + level)
      otherwise                        -> join          ('experi' is a fragment)

    The newline is always removed, whichever branch runs. That is what makes the
    pass idempotent: a second run finds no '-\\n' left to reconsider.
    """
    def repl(m: re.Match) -> str:
        a, b = m.group(1), m.group(2)
        fused = (a + b).lower()
        if freq.get(fused, 0) > 0:
            return a + b
        if freq.get(a.lower(), 0) > 0 and freq.get(b.lower(), 0) > 0:
            return f"{a}-{b}"
        return a + b

    return _HYPHEN_BREAK.sub(repl, text)


def normalize(text: str, freq: Counter | None = None) -> str:
    """
    Repair extracted text. Safe to call repeatedly.

    Args:
        text: extracted page/document text, page markers included
        freq: word frequency for de-hyphenation. Defaults to a table built from
              `text` itself, which is right for whole-document normalisation.
              Pass a corpus-wide table when normalising a single page, where the
              fused word may only appear elsewhere in the book.

    Returns:
        Normalised text, with `--- Page N ---` markers preserved verbatim.
    """
    if not text:
        return ""

    # NFKC folds the compatibility characters: ligatures (U+FB01 -> "fi") and
    # nbsp (U+00A0 -> " ") both go here. It also folds superscripts (U+00B2 ->
    # "2"), a small loss in maths-heavy books, accepted as the price of fixing
    # 3,411 broken words.
    text = unicodedata.normalize("NFKC", text)

    text = _PUNCT_RE.sub(lambda m: _PUNCT_MAP[m.group()], text)
    text = _CONTROL.sub("", text)

    # Strip trailing whitespace BEFORE de-hyphenating — order is load-bearing.
    # '_HYPHEN_BREAK' needs the hyphen immediately before the newline, so a line
    # ending 'word-   \n' would survive the first pass, get its spaces stripped,
    # and then match on the second — making the pass non-idempotent. Measured on
    # a real book: 772 breaks, 21 left behind, second run differed.
    # Leading whitespace is load-bearing (code indentation) and is never touched.
    text = "\n".join(line.rstrip() for line in text.split("\n"))

    if freq is None:
        freq = build_word_freq(text)
    text = _dehyphenate(text, freq)

    return text


def page_offsets(text: str) -> list[tuple[int, int]]:
    """
    Map character offset -> page number, from the `--- Page N ---` markers.

    Returns a list of (char_offset, page_number) sorted by offset, so a chunk
    spanning [start, end) can resolve the pages it covers. Used by the chunker
    to attach real page citations to each chunk.
    """
    out: list[tuple[int, int]] = []
    for m in _PAGE_MARKER.finditer(text):
        digits = re.search(r"Page\s+(\d+)", m.group())
        if digits:
            out.append((m.start(), int(digits.group(1))))
    return out

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def heading_offsets(text: str) -> list[tuple[int, str]]:
    """
    Map character offset -> markdown heading, for locating a chunk in a .md file.

    Markdown has no pages, so a citation cannot say "p.187". Its structure is
    headings, so a chunk is located by the nearest heading above it — giving
    "notes.md - Installation" instead of a bare filename.
    """
    out: list[tuple[int, str]] = []
    for m in _MD_HEADING.finditer(text):
        level = len(m.group(1))
        title = m.group(2).strip()
        if title:
            out.append((m.start(), ("#" * level) + " " + title))
    return out
