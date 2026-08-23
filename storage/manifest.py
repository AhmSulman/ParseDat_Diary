"""
Library Manifest — the one place that knows what is in the library
===================================================================
Before this existed, "how many books do you have?" had four different answers:

    books marked done in checkpoint    18
    books actually in the vector index  7
    PDFs in data/input/               13
    .txt extracted / .json written    18 / 11

Nothing reconciled them, and nothing reported a BOOK count at all —
`Retriever.doc_count` returned `index.ntotal`, a chunk count. So the LLM was
never told how many books existed and simply invented the number.

The manifest records, per book, what was actually indexed and from what, plus
the embedding settings the index was built with.

WHY embed_model / embed_dim LIVE HERE
-------------------------------------
Opening a 384-dim index with a 768-dim embedder crashes inside FAISS with an
assertion, not a useful message. Recording the dimensions the index was built
with lets the retriever detect the mismatch on load and say "run reindex".

Writes are atomic (tmp + os.replace): a crash mid-write must not leave a
truncated manifest, because a corrupt manifest looks exactly like an empty
library.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_FILE = os.path.join(_ROOT, "data", "cache", "maan_manifest.json")

_SCHEMA = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def file_sha256(path: str, _chunk: int = 1 << 20) -> str:
    """Content hash of a PDF, so a re-added file is recognised under any name."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_chunk), b""):
            h.update(block)
    return h.hexdigest()


class Manifest:
    """Book-level record of the indexed library. Total on every operation."""

    def __init__(self, path: str | None = None):
        # Resolved at call time, not bound as a default argument. A default of
        # `path=MANIFEST_FILE` captures the value at import, which makes the
        # location impossible to redirect afterwards — tests pointing at a temp
        # directory silently wrote to the real cache instead.
        self.path = path or MANIFEST_FILE
        self.books: dict[str, dict] = {}
        self.settings: dict = {}
        self.load()

    # ── persistence ───────────────────────────────────────────────────────────
    def load(self) -> None:
        """Never raises. A missing or corrupt manifest reads as an empty one."""
        self.books, self.settings = {}, {}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return
            books = data.get("books")
            if isinstance(books, dict):
                self.books = {k: v for k, v in books.items() if isinstance(v, dict)}
            settings = data.get("settings")
            if isinstance(settings, dict):
                self.settings = settings
        except (FileNotFoundError, json.JSONDecodeError, OSError, UnicodeDecodeError):
            pass

    def save(self) -> None:
        """Atomic write — a torn manifest is indistinguishable from an empty library."""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        payload = {
            "schema": _SCHEMA,
            "updated_at": utc_now(),
            "settings": self.settings,
            "totals": {
                "n_books": self.book_count,
                "n_chunks": self.chunk_count,
            },
            "books": self.books,
        }
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    # ── settings ──────────────────────────────────────────────────────────────
    def set_settings(self, embed_model: str, embed_dim: int,
                     chunk_size: int, chunk_overlap: int) -> None:
        self.settings = {
            "embed_model": embed_model,
            "embed_dim": int(embed_dim),
            "chunk_size": int(chunk_size),
            "chunk_overlap": int(chunk_overlap),
        }

    def is_compatible(self, embed_model: str, embed_dim: int) -> tuple[bool, str]:
        """
        Whether the stored index can be used with the current embedder.

        An empty manifest is compatible: there is nothing to conflict with.
        """
        if not self.settings:
            return True, ""
        got_dim = self.settings.get("embed_dim")
        if got_dim is not None and int(got_dim) != int(embed_dim):
            return False, (
                f"index was built at {got_dim} dimensions, embedder produces "
                f"{embed_dim}. Run: python main.py reindex"
            )
        got_model = self.settings.get("embed_model")
        if got_model and got_model != embed_model:
            return False, (
                f"index was built with '{got_model}', configured embedder is "
                f"'{embed_model}'. Run: python main.py reindex"
            )
        return True, ""

    # ── books ─────────────────────────────────────────────────────────────────
    def add_book(self, source: str, *, book_id: int, n_chunks: int,
                 n_pages: int = 0, char_count: int = 0,
                 sha256: str = "", quality: dict | None = None) -> None:
        self.books[source] = {
            "source": source,
            "book_id": book_id,
            "sha256": sha256,
            "n_pages": n_pages,
            "n_chunks": n_chunks,
            "char_count": char_count,
            "quality": quality or {},
            "ingested_at": utc_now(),
        }

    def remove_book(self, source: str) -> bool:
        return self.books.pop(source, None) is not None

    def has(self, source: str) -> bool:
        return source in self.books

    def sources(self) -> set[str]:
        return set(self.books)

    def sha_index(self) -> dict[str, str]:
        """{sha256: source} for content-hash dedup of re-added files."""
        return {b["sha256"]: s for s, b in self.books.items() if b.get("sha256")}

    @property
    def book_count(self) -> int:
        return len(self.books)

    @property
    def chunk_count(self) -> int:
        return sum(int(b.get("n_chunks", 0)) for b in self.books.values())

    def titles(self) -> list[str]:
        """Book names, sorted — used to ground the LLM's library header."""
        return sorted(self.books)
