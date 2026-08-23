"""
Library Service — reconciliation, in one place
===============================================
All library state questions and mutations live here. The CLI, the FastAPI
dashboard and the KivyMD screen all call this module, so `doctor` and
`/library` cannot drift apart: any disagreement would be a bug in one caller,
not a difference of opinion between two implementations.

THE MEMBERSHIP INVARIANT
------------------------
    data/input/*.pdf is the sole authority for what is in the library.
    Everything else is derived and must be reconcilable to it.

Derived state lives in six places, and deleting a PDF used to prune none of
them:

    data/cache/maan.index         vectors
    data/cache/maan_meta.json     chunk metadata
    data/cache/maan_manifest.json book records
    data/checkpoints/state.json   what ingest thinks is done
    data/txt/*.txt                extracted text
    data/categories.json          GUI categories

THE RESURRECTION TRAP
---------------------
`reindex` rebuilds from extracted text. If it enumerated data/txt/ directly, a
deleted book whose .txt still existed would come straight back — undoing a
purge in the very next command. It therefore enumerates data/input/*.pdf and
treats data/txt/ purely as a cache. Same rule for every consumer here.
"""

from __future__ import annotations

import os
import shutil

from config.config import Config
from logs.logger import log
from storage.categories import CategoryManager
from storage.checkpoint import Checkpoint
from storage.manifest import Manifest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CACHE = os.path.join(_ROOT, "data", "cache")
_TXT = os.path.join(_ROOT, "data", "txt")
_QUAR = os.path.join(_ROOT, "data", "quarantine")
_CATEGORIES = os.path.join(_ROOT, "data", "categories.json")

INDEX_FILE = os.path.join(_CACHE, "maan.index")
META_FILE = os.path.join(_CACHE, "maan_meta.json")
MANIFEST_FILE = os.path.join(_CACHE, "maan_manifest.json")

# Per-book status
INDEXED = "indexed"        # PDF present and in the index — the healthy state
HOLE = "hole"              # PDF present, marked done, but NOT indexed
PENDING = "pending"        # PDF present, not yet processed
ORPHAN = "orphan"          # in the index or on disk, but the PDF is gone
QUARANTINED = "quarantined"


def _stem(name: str) -> str:
    return os.path.splitext(name)[0]


class LibraryService:
    def __init__(self):
        self.cfg = Config()
        self.manifest = Manifest()
        self.checkpoint = Checkpoint()

    # ── inspection ────────────────────────────────────────────────────────────
    def pdfs_on_disk(self) -> set[str]:
        os.makedirs(self.cfg.INPUT_DIR, exist_ok=True)
        return {f for f in os.listdir(self.cfg.INPUT_DIR) if f.lower().endswith(".pdf")}

    def txt_stems(self) -> set[str]:
        if not os.path.isdir(_TXT):
            return set()
        return {_stem(f) for f in os.listdir(_TXT) if f.endswith(".txt")}

    def quarantined(self) -> list[dict]:
        if not os.path.isdir(_QUAR):
            return []
        import json
        out = []
        for f in sorted(os.listdir(_QUAR)):
            if not f.endswith(".report.json"):
                continue
            try:
                with open(os.path.join(_QUAR, f), encoding="utf-8") as fh:
                    out.append(json.load(fh))
            except (OSError, json.JSONDecodeError):
                out.append({"source": f, "reasons": ["unreadable report"]})
        return out

    def report(self) -> dict:
        """
        Full reconciliation. Read-only — this never changes anything.

        `drift` is the number of things that disagree with data/input/. Zero
        means every store is consistent.
        """
        pdfs = self.pdfs_on_disk()
        indexed = self.manifest.sources()
        done = set(self.checkpoint.state.get("done", []))
        txts = self.txt_stems()
        quar = self.quarantined()
        quar_names = {q.get("source", "") for q in quar}

        books = []
        for pdf in sorted(pdfs):
            if pdf in quar_names:
                status = QUARANTINED
            elif pdf in indexed:
                status = INDEXED
            elif pdf in done:
                status = HOLE          # claimed done, never actually indexed
            else:
                status = PENDING
            rec = self.manifest.books.get(pdf, {})
            books.append({
                "source": pdf,
                "status": status,
                "n_chunks": rec.get("n_chunks", 0),
                "n_pages": rec.get("n_pages", 0),
                "has_text": _stem(pdf) in txts,
                "ingested_at": rec.get("ingested_at"),
            })

        # Derived state referring to books that no longer exist.
        orphan_indexed = sorted(indexed - pdfs)
        orphan_txt = sorted(txts - {_stem(p) for p in pdfs})
        orphan_done = sorted(done - pdfs)
        for src in orphan_indexed:
            books.append({
                "source": src, "status": ORPHAN,
                "n_chunks": self.manifest.books.get(src, {}).get("n_chunks", 0),
                "n_pages": 0, "has_text": _stem(src) in txts, "ingested_at": None,
            })

        holes = [b["source"] for b in books if b["status"] == HOLE]
        pending = [b["source"] for b in books if b["status"] == PENDING]
        drift = len(orphan_indexed) + len(orphan_txt) + len(orphan_done) + len(holes)

        return {
            "counts": {
                "pdfs_on_disk": len(pdfs),
                "books_indexed": len(indexed),
                "chunks_indexed": self.manifest.chunk_count,
                "checkpoint_done": len(done),
                "txt_files": len(txts),
                "quarantined": len(quar),
            },
            "books": books,
            "orphans": {
                "index": orphan_indexed,
                "txt": orphan_txt,
                "checkpoint": orphan_done,
            },
            "holes": holes,
            "pending": pending,
            "quarantine": quar,
            "settings": self.manifest.settings,
            "drift": drift,
            "healthy": drift == 0 and not pending,
        }

    # ── mutation ──────────────────────────────────────────────────────────────
    def clean(self, *, index: bool = False, checkpoint: bool = False,
              orphans: bool = False, text: bool = False,
              quarantine: bool = False) -> dict:
        """Remove derived state. Never touches data/input/."""
        removed: dict[str, list[str]] = {
            "index": [], "checkpoint": [], "txt": [], "quarantine": []
        }

        if index:
            for p in (INDEX_FILE, META_FILE, MANIFEST_FILE):
                if os.path.exists(p):
                    os.remove(p)
                    removed["index"].append(os.path.basename(p))
            self.manifest = Manifest()

        if checkpoint:
            self.checkpoint.reset()
            removed["checkpoint"].append("state.json")

        if quarantine and os.path.isdir(_QUAR):
            shutil.rmtree(_QUAR, ignore_errors=True)
            removed["quarantine"].append("data/quarantine/")

        if text and os.path.isdir(_TXT):
            for f in sorted(os.listdir(_TXT)):
                if f.endswith(".txt"):
                    os.remove(os.path.join(_TXT, f))
                    removed["txt"].append(f)

        if orphans:
            removed = self._prune_orphans(removed)

        return removed

    def _prune_orphans(self, removed: dict) -> dict:
        """
        Reconcile every derived store to data/input/.

        This is what makes a deletion actually stick. Leaving orphaned .txt in
        place is what would let `reindex` resurrect a deleted book.
        """
        pdfs = self.pdfs_on_disk()
        stems = {_stem(p) for p in pdfs}

        # 1. manifest entries whose PDF is gone
        for src in sorted(self.manifest.sources() - pdfs):
            self.manifest.remove_book(src)
            removed["index"].append(f"manifest:{src}")
        self.manifest.save()

        # 2. checkpoint entries whose PDF is gone
        done = self.checkpoint.state.get("done", [])
        keep = [d for d in done if d in pdfs]
        if len(keep) != len(done):
            for d in done:
                if d not in pdfs:
                    removed["checkpoint"].append(d)
            self.checkpoint.state["done"] = keep
            failed = self.checkpoint.state.get("failed", [])
            self.checkpoint.state["failed"] = [f for f in failed if f in pdfs]
            self.checkpoint._write(self.checkpoint.state)

        # 3. extracted text whose PDF is gone — the resurrection trap
        if os.path.isdir(_TXT):
            for f in sorted(os.listdir(_TXT)):
                if f.endswith(".txt") and _stem(f) not in stems:
                    os.remove(os.path.join(_TXT, f))
                    removed["txt"].append(f)

        # 4. GUI categories referencing deleted PDFs
        try:
            cats = CategoryManager()
            for src in {p for pdfs_ in cats._data.values() for p in pdfs_} - pdfs:
                cats.remove_pdf_everywhere(src)
        except Exception as e:
            log.warning(f"Category prune skipped: {e}")

        return removed

    def purge_book(self, source: str) -> bool:
        """Remove one book from every derived store. Leaves the PDF alone."""
        changed = self.manifest.remove_book(source)
        if changed:
            self.manifest.save()

        done = self.checkpoint.state.get("done", [])
        if source in done:
            self.checkpoint.state["done"] = [d for d in done if d != source]
            self.checkpoint._write(self.checkpoint.state)
            changed = True

        txt = os.path.join(_TXT, f"{_stem(source)}.txt")
        if os.path.exists(txt):
            os.remove(txt)
            changed = True
        return changed
