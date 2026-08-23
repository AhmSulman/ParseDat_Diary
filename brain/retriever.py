"""
Retriever — FAISS vector store with neighbour expansion
========================================================
Stores chunk vectors and finds them by meaning, then rebuilds readable passages
from the hits.

WHY NEIGHBOUR EXPANSION
-----------------------
A single 1200-char chunk usually starts or ends mid-argument. Retrieving 6 such
fragments from 6 different books gives the LLM disconnected snippets and invites
it to bridge the gaps by inventing. So for every hit we also pull the chunks
immediately before and after it from the SAME book, and merge the run back into
one continuous passage.

The merge is arithmetic, not fuzzy string matching, using the exact offsets the
chunker recorded:

    drop   = prev.char_end - cur.char_start      # the overlap, in characters
    merged = prev.text + cur.text[drop:]

Correct only because chunker offsets describe the STORED (stripped) text exactly.

INDEX / METADATA MUST STAY IN LOCKSTEP
--------------------------------------
FAISS returns row numbers; `self.meta[row]` must be the chunk that produced that
row. Vectors and metadata are therefore appended together under one lock, and
saved together atomically. `save()` writes tmp files and renames, so a crash
cannot leave an 8 MB metadata file half-written and silently desynchronised.
"""

import json
import os
import threading

import numpy as np

from brain.embedder import Embedder
from config.config import Config
from logs.logger import log
from storage.manifest import Manifest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_FILE = os.path.join(_ROOT, "data", "cache", "maan.index")
META_FILE = os.path.join(_ROOT, "data", "cache", "maan_meta.json")

_HNSW_M = 32


class Retriever:
    def __init__(self, device: str | None = None):
        cfg = Config()
        self.cfg = cfg
        self.dim = cfg.EMBED_DIM
        self.embedder = Embedder(device=device)
        self.index = None
        self.meta: list[dict] = []
        self._faiss = self._load_faiss_lib()
        self._lock = threading.Lock()
        self._neighbors: dict[tuple[str, int], int] = {}

        # Absolute path. The old code called os.makedirs("data/cache"), a bare
        # relative path that created a stray folder whenever the working
        # directory was not the project root.
        os.makedirs(os.path.dirname(INDEX_FILE), exist_ok=True)

    # ── Library detection ─────────────────────────────────────────────────────
    def _load_faiss_lib(self):
        try:
            import faiss
            return faiss
        except ImportError:
            log.warning("faiss not installed - pip install faiss-cpu")
            return None

    def _ensure_index(self):
        """HNSW flat: fast approximate NN on CPU, no training step."""
        if self.index is not None or self._faiss is None:
            return
        idx = self._faiss.IndexHNSWFlat(self.dim, _HNSW_M)
        idx.hnsw.efConstruction = 200
        idx.hnsw.efSearch = 64
        self.index = idx
        log.info(f"FAISS: HNSWFlat index, dim={self.dim}")

    def _rebuild_neighbor_map(self):
        """(source, chunk_id) -> row, for O(1) neighbour lookup."""
        self._neighbors = {
            (m["source"], m["chunk_id"]): i
            for i, m in enumerate(self.meta)
            if "chunk_id" in m and "source" in m
        }

    # ── Writing ───────────────────────────────────────────────────────────────
    def add_batch(self, texts: list[str], metas: list[dict]):
        """Embed and append. Vectors and metadata move together, under one lock."""
        if not self._faiss or not texts:
            return
        vecs = self.embedder.embed_batch(texts)
        valid = [(v, m, t) for v, m, t in zip(vecs, metas, texts) if v is not None]
        if not valid:
            return

        arr = np.array([v for v, _, _ in valid], dtype=np.float32)
        with self._lock:
            self._ensure_index()
            self.index.add(arr)
            base = len(self.meta)
            for off, (_, m, t) in enumerate(valid):
                row = {**m, "chunk": t}
                self.meta.append(row)
                key = (row.get("source"), row.get("chunk_id"))
                if key[0] is not None and key[1] is not None:
                    self._neighbors[key] = base + off

    # ── Reading ───────────────────────────────────────────────────────────────
    def search(self, query: str, k: int | None = None) -> list[dict]:
        """Raw top-k chunks, no expansion."""
        k = k or self.cfg.SEARCH_TOP_K
        if not self._faiss or self.index is None or self.index.ntotal == 0:
            return []
        vec = self.embedder.embed_query(query)
        if vec is None:
            return []

        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(np.array([vec], dtype=np.float32), k)

        out = []
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self.meta):
                row = dict(self.meta[idx])
                row["score"] = float(dist)
                out.append(row)
        return out

    def search_with_context(self, query: str, k: int | None = None,
                            radius: int | None = None,
                            char_budget: int | None = None) -> list[dict]:
        """
        Top-k hits expanded into merged passages.

        1. retrieve top-k
        2. for each hit, take chunk_id +/- radius from the same book
        3. merge contiguous runs, de-overlapping via stored offsets
        4. order by (source, chunk_id) so passages read forwards
        5. stop at char_budget
        """
        k = k or self.cfg.SEARCH_TOP_K
        radius = self.cfg.NEIGHBOR_RADIUS if radius is None else radius
        budget = char_budget or self.cfg.CONTEXT_CHAR_BUDGET

        hits = self.search(query, k)
        if not hits:
            return []
        if not self._neighbors:
            self._rebuild_neighbor_map()

        # Collect hit chunks plus their neighbours, best score per chunk.
        wanted: dict[tuple[str, int], float] = {}
        for h in hits:
            src, cid = h.get("source"), h.get("chunk_id")
            if src is None or cid is None:
                continue
            for c in range(cid - radius, cid + radius + 1):
                key = (src, c)
                if key in self._neighbors:
                    prev = wanted.get(key)
                    wanted[key] = h["score"] if prev is None else min(prev, h["score"])
        if not wanted:
            return hits

        # Group into contiguous runs per source.
        by_source: dict[str, list[int]] = {}
        for src, cid in wanted:
            by_source.setdefault(src, []).append(cid)

        passages = []
        for src, cids in by_source.items():
            cids.sort()
            run = [cids[0]]
            for cid in cids[1:]:
                if cid == run[-1] + 1:
                    run.append(cid)
                else:
                    passages.append(self._merge_run(src, run, wanted))
                    run = [cid]
            passages.append(self._merge_run(src, run, wanted))

        passages = [p for p in passages if p]
        passages.sort(key=lambda p: p["score"])

        kept, used = [], 0
        for p in passages:
            if used + len(p["chunk"]) > budget and kept:
                break
            kept.append(p)
            used += len(p["chunk"])
        return kept

    def _merge_run(self, source: str, cids: list[int],
                   scored: dict[tuple[str, int], float]) -> dict | None:
        """
        Stitch consecutive chunks into one passage, removing the overlap exactly.

        `drop` is how many characters of `cur` were already emitted as the tail
        of `prev`. Clamped to [0, len(cur)] so a gap (drop < 0) appends cleanly
        instead of slicing from the end, which negative indexing would do
        silently.
        """
        rows = [self.meta[self._neighbors[(source, c)]] for c in cids
                if (source, c) in self._neighbors]
        if not rows:
            return None

        text = rows[0].get("chunk", "")
        for prev, cur in zip(rows, rows[1:]):
            drop = prev.get("char_end", 0) - cur.get("char_start", 0)
            drop = max(0, min(drop, len(cur.get("chunk", ""))))
            text += cur.get("chunk", "")[drop:]

        pages = [r.get("page_start") for r in rows if r.get("page_start") is not None]
        pages += [r.get("page_end") for r in rows if r.get("page_end") is not None]

        return {
            "source": source,
            "chunk": text,
            "chunk_id": rows[0].get("chunk_id"),
            "chunk_ids": cids,
            "page_start": min(pages) if pages else None,
            "page_end": max(pages) if pages else None,
            "score": min(scored.get((source, c), 1e9) for c in cids),
        }

    # ── Persistence ───────────────────────────────────────────────────────────
    def save(self):
        """Atomic: index and metadata must never disagree after a crash."""
        if not self._faiss or self.index is None:
            return
        with self._lock:
            tmp_idx, tmp_meta = INDEX_FILE + ".tmp", META_FILE + ".tmp"
            self._faiss.write_index(self.index, tmp_idx)
            with open(tmp_meta, "w", encoding="utf-8") as f:
                json.dump(self.meta, f, ensure_ascii=False)
            os.replace(tmp_idx, INDEX_FILE)
            os.replace(tmp_meta, META_FILE)
        log.info(f"Index saved: {self.index.ntotal} vectors, {len(self.meta)} meta")

    def load(self) -> bool:
        """
        Load index + metadata, refusing an index built with a different embedder.

        Opening a 384-dim index with a 768-dim embedder otherwise fails inside
        FAISS with a bare assertion.
        """
        if not self._faiss:
            return False
        if not (os.path.exists(INDEX_FILE) and os.path.exists(META_FILE)):
            log.info("No index found. Run: python main.py ingest")
            return False

        ok, why = Manifest().is_compatible(self.cfg.EMBED_MODEL, self.cfg.EMBED_DIM)
        if not ok:
            log.warning(f"Index incompatible: {why}")
            return False

        try:
            idx = self._faiss.read_index(INDEX_FILE)
            with open(META_FILE, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            log.error(f"Index load failed ({e}). Run: python main.py reindex")
            return False

        if idx.d != self.dim:
            log.warning(f"Index dim {idx.d} != embedder dim {self.dim}. Run reindex.")
            return False
        if idx.ntotal != len(meta):
            log.warning(
                f"Index/metadata desync: {idx.ntotal} vectors vs {len(meta)} rows. "
                "Run: python main.py reindex"
            )
            return False

        self.index, self.meta = idx, meta
        self._rebuild_neighbor_map()
        log.info(f"Index loaded: {idx.ntotal} chunks from {self.book_count} books")
        return True

    def reset(self):
        """Drop the in-memory index. Callers remove the files."""
        with self._lock:
            self.index = None
            self.meta = []
            self._neighbors = {}

    # ── Counts ────────────────────────────────────────────────────────────────
    @property
    def chunk_count(self) -> int:
        return self.index.ntotal if self.index is not None else 0

    @property
    def book_count(self) -> int:
        """Distinct books. `doc_count` used to return this as a CHUNK count."""
        return len({m.get("source") for m in self.meta if m.get("source")})

    @property
    def doc_count(self) -> int:
        """Deprecated alias for chunk_count, kept for existing GUI call sites."""
        return self.chunk_count
