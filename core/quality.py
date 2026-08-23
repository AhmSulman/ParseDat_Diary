"""
PDF Quality Gate — keep unreadable books out of the index
==========================================================
Nothing previously stopped a badly-extracted PDF from being chunked and
embedded, and one book in this corpus was actively poisoning retrieval.

`C# and .NET QuickStart Guide` had its Latin vowels replaced with Cyrillic
homoglyphs — 'The class Keyword' is stored as 'Th<U+0435> cl<U+0430>ss
K<U+0435>yw<U+043E>rd'. That is a piracy watermark, not bad OCR. Every word
containing a/e/o embeds as a different token, so its 791 chunks were noise that
the retriever could still surface and Mistral could still cite.

Measured separation across the 18-book corpus:

    metric              16 healthy books   mml-book (maths)   poisoned book
    mean word length       3.76 - 5.45          4.54              1.97
    homoglyphs /10k          <= 2.4              2.4            1608.8
    alpha ratio           0.713 - 0.781         0.713            0.585
    non-ASCII ratio         <= 0.053            0.017            0.163

Thresholds sit in the wide gap between those columns, so the gate is decisive
rather than borderline.

SCRIPT-MIXING AWARENESS — the reason this is not a one-line check
-----------------------------------------------------------------
`mml-book.pdf` legitimately contains Greek letters as mathematical notation.
A naive "count Cyrillic/Greek characters" rule would flag it and discard a
perfectly good book. A confusable is therefore only counted when it appears
*inside a word that also contains Latin letters* — real notation stands alone,
a watermark hides inside Latin words.

Failing books are QUARANTINED, never repaired: the user's call is that quality
matters more than coverage. Repair is possible (mapping the confusables back
restores mean word length 1.97 -> 5.15) and is recorded in OKF/NEXT.md.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field

# Cyrillic and Greek characters that render identically (or near-identically)
# to Latin letters. These are the ones actually used for watermarking.
CONFUSABLES = {
    # Cyrillic lowercase
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c",
    "у": "y", "х": "x", "і": "i", "ѕ": "s", "ј": "j",
    # Cyrillic uppercase
    "А": "A", "Е": "E", "О": "O", "Р": "P", "С": "C",
    "У": "Y", "Х": "X", "М": "M", "Н": "H", "К": "K",
    "В": "B", "Т": "T",
    # Greek
    "ο": "o", "α": "a", "ρ": "p",
    "Α": "A", "Β": "B", "Ε": "E", "Ο": "O",
}

_LATIN = re.compile(r"[A-Za-z]")
_LATIN_WORD = re.compile(r"[A-Za-z]+")
_TOKEN = re.compile(r"\w+", re.UNICODE)
_PAGE_MARKER = re.compile(r"-{2,}\s*Page\s+\d+[^\n]*?-{2,}")
_LONG_RUN = re.compile(r"[A-Za-z]{25,}")

# Thresholds, set from the measured corpus above.
MIN_MEAN_WORD_LEN = 3.0
MAX_HOMOGLYPH_PER_10K = 50.0
MIN_ALPHA_RATIO = 0.60
MAX_NON_ASCII_RATIO = 0.10
MIN_SPACE_RATIO = 0.08
MAX_SPACE_RATIO = 0.35
MAX_REPLACEMENT_PER_10K = 1.0
MAX_LONG_RUN_RATIO = 0.005
MIN_CHARS = 500          # below this there is nothing meaningful to score


@dataclass
class QualityReport:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        head = "PASS" if self.passed else "FAIL"
        if self.reasons:
            return f"{head}: " + "; ".join(self.reasons)
        return head


def homoglyph_count(text: str) -> int:
    """
    Confusables appearing inside words that also contain Latin letters.

    Standalone Greek (real maths notation) is not counted — that is what keeps
    `mml-book.pdf` out of quarantine while still catching a watermarked book.
    """
    n = 0
    for tok in _TOKEN.findall(text):
        if not _LATIN.search(tok):
            continue                      # pure Greek/Cyrillic token: notation
        n += sum(1 for ch in tok if ch in CONFUSABLES)
    return n


def score(text: str) -> QualityReport:
    """Score extracted book text. Returns PASS/FAIL with the reasons and metrics."""
    body = _PAGE_MARKER.sub("", text or "").strip()
    n = len(body)

    if n < MIN_CHARS:
        return QualityReport(
            passed=False,
            reasons=[f"only {n} chars of text extracted (min {MIN_CHARS})"],
            metrics={"chars": float(n)},
        )

    words = _LATIN_WORD.findall(body)
    mean_wl = statistics.mean(len(w) for w in words) if words else 0.0
    homo = homoglyph_count(body)
    homo_10k = 10000.0 * homo / n
    alpha = sum(c.isalpha() for c in body) / n
    space = sum(c.isspace() for c in body) / n
    non_ascii = sum(ord(c) > 127 for c in body) / n
    repl_10k = 10000.0 * body.count("�") / n
    long_runs = len(_LONG_RUN.findall(body)) / max(len(words), 1)

    m = {
        "chars": float(n),
        "mean_word_len": round(mean_wl, 2),
        "homoglyphs_per_10k": round(homo_10k, 1),
        "alpha_ratio": round(alpha, 3),
        "space_ratio": round(space, 3),
        "non_ascii_ratio": round(non_ascii, 4),
        "replacement_per_10k": round(repl_10k, 2),
        "long_run_ratio": round(long_runs, 5),
    }

    reasons: list[str] = []
    if mean_wl < MIN_MEAN_WORD_LEN:
        reasons.append(
            f"mean word length {mean_wl:.2f} < {MIN_MEAN_WORD_LEN} "
            f"(text is fragmented into 1-2 char pieces)"
        )
    if homo_10k > MAX_HOMOGLYPH_PER_10K:
        reasons.append(
            f"{homo_10k:.0f} homoglyphs/10k > {MAX_HOMOGLYPH_PER_10K:.0f} "
            f"(Latin letters replaced with Cyrillic/Greek lookalikes)"
        )
    if alpha < MIN_ALPHA_RATIO:
        reasons.append(f"alpha ratio {alpha:.3f} < {MIN_ALPHA_RATIO}")
    if non_ascii > MAX_NON_ASCII_RATIO:
        reasons.append(f"non-ASCII ratio {non_ascii:.3f} > {MAX_NON_ASCII_RATIO}")
    if not (MIN_SPACE_RATIO <= space <= MAX_SPACE_RATIO):
        reasons.append(
            f"whitespace ratio {space:.3f} outside "
            f"{MIN_SPACE_RATIO}-{MAX_SPACE_RATIO} (word boundaries lost)"
        )
    if repl_10k > MAX_REPLACEMENT_PER_10K:
        reasons.append(f"{repl_10k:.1f} U+FFFD/10k > {MAX_REPLACEMENT_PER_10K}")
    if long_runs > MAX_LONG_RUN_RATIO:
        reasons.append(f"long unbroken runs {long_runs:.4f} > {MAX_LONG_RUN_RATIO}")

    return QualityReport(passed=not reasons, reasons=reasons, metrics=m)
