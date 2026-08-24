"""
Source discovery — what counts as a document
=============================================
One place that decides which files in data/input/ are ingestable, so the
pipeline, the reindexer and the library reconciler cannot disagree about what
the library contains. They previously each hardcoded `.pdf`, which meant adding
a new format required finding every one of them.

TEXT FORMATS SKIP OCR ENTIRELY
------------------------------
A .md or .txt file is already text. Routing it through PyMuPDF and Tesseract
would be pointless and lossy, so the reader branches on extension: PDFs go
through extraction, text formats are read straight off disk.
"""

from __future__ import annotations

import os

# Extensions treated as documents. PDFs need extraction; the rest are text.
PDF_EXTS = (".pdf",)
TEXT_EXTS = (".md", ".markdown", ".txt")
SUPPORTED_EXTS = PDF_EXTS + TEXT_EXTS


def is_supported(name: str) -> bool:
    return name.lower().endswith(SUPPORTED_EXTS)


def is_pdf(name: str) -> bool:
    return name.lower().endswith(PDF_EXTS)


def is_text(name: str) -> bool:
    return name.lower().endswith(TEXT_EXTS)


def is_markdown(name: str) -> bool:
    return name.lower().endswith((".md", ".markdown"))


def list_sources(input_dir: str) -> list[str]:
    """
    Every ingestable file in `input_dir`, sorted.

    README files are skipped: data/input/README.md explains how to use the
    folder and is not part of anyone's library.
    """
    os.makedirs(input_dir, exist_ok=True)
    return sorted(
        f for f in os.listdir(input_dir)
        if is_supported(f) and os.path.splitext(f)[0].lower() != "readme"
    )


def read_text_source(path: str) -> str:
    """
    Read a text document off disk.

    errors="replace" rather than raising: one bad byte in a large note should
    cost that character, not the whole file.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()
