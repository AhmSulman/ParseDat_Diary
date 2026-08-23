"""
Smart Text Chunker
==================
Splits book text into overlapping, sentence-aligned chunks with enough metadata
to cite and to reassemble.

WHY THE METADATA MATTERS
------------------------
The old pipeline attached only {"source": name} to every chunk. That made three
things impossible:

  - citing a page, so no answer could be checked against the book
  - finding a chunk's neighbours, so a hit was always an isolated fragment
  - knowing how many chunks a book has, so nothing could verify completeness

Each chunk now carries `chunk_id`, `n_chunks`, exact `char_start`/`char_end`
and the `page_start`/`page_end` it covers.

OFFSETS ARE EXACT, NOT APPROXIMATE
----------------------------------
`char_start`/`char_end` describe the STORED text after stripping, not the raw
slice. The retriever de-overlaps adjacent chunks arithmetically:

    drop   = prev.char_end - cur.char_start
    merged = prev.text + cur.text[drop:]

which is only correct if the offsets match the stored strings character for
character. Recording pre-strip offsets would silently corrupt every merge.
"""

import re

from config.config import Config
from core.normalize import page_offsets

# Break points, best first: paragraph, line, then sentence enders.
_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? "]

_PAGE_MARKER = re.compile(r"-{2,}\s*Page\s+\d+[^\n]*?-{2,}")


class Chunker:
    def __init__(self, chunk_size: int | None = None, overlap: int | None = None):
        cfg = Config()
        self.chunk_size = chunk_size if chunk_size is not None else cfg.CHUNK_SIZE
        self.overlap = overlap if overlap is not None else cfg.CHUNK_OVERLAP

        # A stalled loop is the failure mode here: if overlap reaches half the
        # chunk size, `start` can stop advancing and chunk() never terminates.
        # Clamp rather than trust the config.
        max_overlap = self.chunk_size // 2 - 1
        if self.overlap > max_overlap:
            self.overlap = max(0, max_overlap)

    # ── internal ──────────────────────────────────────────────────────────────
    def _spans(self, text: str) -> list[tuple[int, int]]:
        """
        Chunk boundaries as (start, end) offsets into `text`.

        Offsets are computed on the text as given, then tightened to the
        stripped content in `chunk_with_meta` so stored text and offsets agree.
        """
        spans: list[tuple[int, int]] = []
        n = len(text)
        start = 0

        while start < n:
            end = min(start + self.chunk_size, n)

            # Prefer a natural boundary in the second half of the window, so a
            # chunk ends at a paragraph or sentence rather than mid-word.
            if end < n:
                floor = start + self.chunk_size // 2
                for sep in _SEPARATORS:
                    pos = text.rfind(sep, floor, end)
                    if pos != -1:
                        end = pos + len(sep)
                        break

            spans.append((start, end))

            if end >= n:
                break

            nxt = end - self.overlap
            # Guarantee forward progress even if a pathological boundary lands
            # inside the overlap window.
            start = nxt if nxt > start else start + 1

        return spans

    # ── public API ────────────────────────────────────────────────────────────
    def chunk(self, text: str) -> list[str]:
        """Split text into overlapping chunks. Returns the chunk strings only."""
        if not text or not text.strip():
            return []
        out = []
        for s, e in self._spans(text):
            piece = text[s:e].strip()
            if piece:
                out.append(piece)
        return out

    def chunk_with_meta(self, text: str, source: str, book_id: int = 0) -> list[dict]:
        """
        Chunk text and attach the metadata the retriever and prompt builder need.

        Returns a list of dicts:
            text, source, book_id, chunk_id, n_chunks,
            char_start, char_end, page_start, page_end

        `chunk_id` is contiguous 0..n_chunks-1 for the source, which is what
        makes neighbour lookup a simple +/-1.
        """
        if not text or not text.strip():
            return []

        pages = page_offsets(text)
        page_pos = [p[0] for p in pages]
        page_num = [p[1] for p in pages]

        def page_at(offset: int) -> int | None:
            """Page whose marker most recently preceded `offset`."""
            if not page_pos:
                return None
            lo, hi = 0, len(page_pos) - 1
            if offset < page_pos[0]:
                return None
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if page_pos[mid] <= offset:
                    lo = mid
                else:
                    hi = mid - 1
            return page_num[lo]

        rows: list[dict] = []
        for s, e in self._spans(text):
            raw = text[s:e]
            stripped = raw.strip()
            if not stripped:
                continue

            # Tighten offsets onto the stored text: this is what keeps the
            # retriever's de-overlap arithmetic exact.
            lead = len(raw) - len(raw.lstrip())
            c_start = s + lead
            c_end = c_start + len(stripped)

            rows.append({
                "text": stripped,
                "source": source,
                "book_id": book_id,
                "char_start": c_start,
                "char_end": c_end,
                "page_start": page_at(c_start),
                "page_end": page_at(max(c_start, c_end - 1)),
            })

        total = len(rows)
        for i, r in enumerate(rows):
            r["chunk_id"] = i
            r["n_chunks"] = total
        return rows

    @staticmethod
    def strip_page_markers(text: str) -> str:
        """Remove `--- Page N ---` lines. For display, never before chunking."""
        return _PAGE_MARKER.sub("", text)
