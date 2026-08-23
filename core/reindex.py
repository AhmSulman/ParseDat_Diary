"""
Reindex — rebuild vectors from already-extracted text
======================================================
Re-embedding the whole library normally means re-reading every PDF and
re-running OCR. It does not have to: `data/txt/` already holds the extracted
text, and normalisation is idempotent, so a rebuild can skip straight to
chunking and embedding.

That is what makes an embedder change affordable. Swapping to a 768-dim model
invalidates every existing vector, but 12 of 13 books rebuild from cached text
in the time it takes to embed them — no OCR, no PDF parsing.

ENUMERATION ORDER IS THE WHOLE POINT
------------------------------------
This walks `data/input/*.pdf` and looks up each book's `.txt`. It must NEVER
walk `data/txt/` instead: orphaned text files from deleted books would be
picked up and re-indexed, resurrecting exactly the books a purge just removed.
Books with no cached text are reported, not silently skipped — `ingest` handles
those.
"""

import os
import time

from brain.chunker import Chunker
from brain.retriever import Retriever
from config.config import Config
from core.normalize import normalize
from core.quality import score
from logs.logger import log
from storage.checkpoint import Checkpoint
from storage.exporter import Exporter
from storage.manifest import Manifest, file_sha256

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TXT = os.path.join(_ROOT, "data", "txt")


def reindex(device: str = "cuda") -> dict:
    """
    Rebuild the index from cached text.

    Returns a summary dict. `needs_ingest` lists books with no cached text —
    those require the full PDF pipeline.
    """
    cfg = Config()
    chunker = Chunker()
    exporter = Exporter()
    checkpoint = Checkpoint()

    # Embedding is the entire workload here and no LLM is loaded, so the
    # embedder gets the GPU.
    retriever = Retriever(device=device)
    retriever.reset()

    manifest = Manifest()
    manifest.books = {}
    manifest.set_settings(
        embed_model=cfg.EMBED_MODEL,
        embed_dim=cfg.EMBED_DIM,
        chunk_size=cfg.CHUNK_SIZE,
        chunk_overlap=cfg.CHUNK_OVERLAP,
    )

    os.makedirs(cfg.INPUT_DIR, exist_ok=True)
    # data/input, never data/txt — see module docstring.
    pdfs = sorted(f for f in os.listdir(cfg.INPUT_DIR) if f.lower().endswith(".pdf"))

    stats = {
        "indexed": 0, "chunks": 0, "quarantined": 0,
        "needs_ingest": [], "failed": [],
    }
    t0 = time.perf_counter()
    log.info(f"Reindex: {len(pdfs)} PDFs on disk")

    for book_id, pdf in enumerate(pdfs):
        txt_path = os.path.join(_TXT, f"{os.path.splitext(pdf)[0]}.txt")
        if not os.path.exists(txt_path):
            stats["needs_ingest"].append(pdf)
            log.info(f"  no cached text, needs ingest: {pdf}")
            continue

        try:
            with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
                raw = f.read()

            text = normalize(raw)          # idempotent: no-op if already clean

            report = score(text)
            if not report.passed:
                exporter.quarantine(pdf, text, report)
                checkpoint.mark_failed(pdf)
                stats["quarantined"] += 1
                continue

            rows = chunker.chunk_with_meta(text, pdf, book_id=book_id)
            if not rows:
                stats["failed"].append(pdf)
                continue

            texts = [r.pop("text") for r in rows]
            retriever.add_batch(texts, rows)

            try:
                sha = file_sha256(os.path.join(cfg.INPUT_DIR, pdf))
            except OSError:
                sha = ""

            manifest.add_book(
                pdf,
                book_id=book_id,
                n_chunks=len(rows),
                n_pages=max((r.get("page_end") or 0) for r in rows),
                char_count=len(text),
                sha256=sha,
                quality=report.metrics,
            )
            checkpoint.mark_done(pdf)

            stats["indexed"] += 1
            stats["chunks"] += len(rows)
            log.info(f"  {pdf[:50]}: {len(rows)} chunks")

        except Exception as e:
            log.error(f"  failed {pdf}: {e}")
            stats["failed"].append(pdf)

    retriever.save()
    manifest.save()

    stats["elapsed"] = round(time.perf_counter() - t0, 1)
    stats["books"] = manifest.book_count
    stats["embed_model"] = retriever.embedder.model_name
    stats["device"] = retriever.embedder.device
    return stats
