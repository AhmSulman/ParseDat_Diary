"""
Retriever -- Vector Database with CUDA acceleration
=====================================================
Stores and searches document chunks by meaning (not keywords).

GPU path  (Windows + CUDA):
  - faiss-gpu moves the index onto the RTX 4050 VRAM
  - Batch embedding via sentence-transformers on CUDA device
  - ~10-50x faster search than CPU flat scan

CPU path  (Android / no CUDA):
  - faiss-cpu with IndexHNSWFlat -- fast ANN on CPU
  - Transparent fallback, same API

At search time: question -> vector -> GPU/CPU nearest neighbours -> chunks.
"""

import os
import json
import threading
import numpy as np
from logs.logger import log
from brain.embedder import Embedder

INDEX_FILE = "data/cache/maan.index"
META_FILE  = "data/cache/maan_meta.json"
DIM = 384
_HNSW_M = 32


class Retriever:
    def __init__(self):
        self.embedder   = Embedder()
        self.index      = None           # CPU index (always kept for save/load)
        self._gpu_index = None           # GPU-side index (None on CPU-only path)
        self.meta: list[dict] = []
        self._faiss   = self._load_faiss_lib()
        self._gpu_res = None             # faiss GPU resources handle
        self._lock    = threading.Lock() # thread-safe add() during ingest
        os.makedirs("data/cache", exist_ok=True)

        self._init_gpu()

    # ── Library detection ──────────────────────────────────────────────────────
    def _load_faiss_lib(self):
        try:
            import faiss
            return faiss
        except ImportError:
            log.warning("faiss not installed -- pip install faiss-gpu (CUDA) or faiss-cpu")
            return None

    def _init_gpu(self):
        """Try to initialise a CUDA GPU resource for faiss."""
        if self._faiss is None:
            return
        try:
            n_gpus = self._faiss.get_num_gpus()
            if n_gpus > 0:
                self._gpu_res = self._faiss.StandardGpuResources()
                log.info(f"FAISS CUDA: {n_gpus} GPU(s) available -- vector search on GPU")
            else:
                log.info("FAISS: no CUDA GPU -- using CPU HNSW index")
        except AttributeError:
            # faiss-cpu build has no get_num_gpus()
            log.info("FAISS CPU build -- using HNSW index")

    # ── Index management ───────────────────────────────────────────────────────
    def _ensure_index(self):
        """Create CPU index (and optionally move to GPU) on first add."""
        if self.index is not None:
            return

        if self._gpu_res is not None:
            # GPU path: IVF flat is GPU-acceleratable; nlist=1 works for any size
            quantizer = self._faiss.IndexFlatL2(DIM)
            cpu_ivf   = self._faiss.IndexIVFFlat(quantizer, DIM, 1)
            cpu_ivf.train(np.zeros((1, DIM), dtype=np.float32))  # minimal train
            self.index    = cpu_ivf
            self._gpu_index = self._faiss.index_cpu_to_gpu(self._gpu_res, 0, self.index)
            log.info("FAISS: IVFFlat index on GPU")
        else:
            # CPU path: HNSW -- fast approximate NN, no training needed
            idx = self._faiss.IndexHNSWFlat(DIM, _HNSW_M)
            idx.hnsw.efConstruction = 200
            idx.hnsw.efSearch       = 64
            self.index = idx
            log.info("FAISS: HNSWFlat index on CPU")

    # ── Public API ─────────────────────────────────────────────────────────────
    def add(self, text: str, meta: dict):
        """Embed a text chunk and add it to the index (thread-safe)."""
        if not self._faiss:
            return
        vec = self.embedder.embed(text)
        if vec is None:
            return
        arr = np.array([vec], dtype=np.float32)

        with self._lock:
            self._ensure_index()
            if self._gpu_index is not None:
                self._gpu_index.add(arr)
                # Keep CPU index in sync so save() works
                self.index.add(arr)
            else:
                self.index.add(arr)
            self.meta.append({**meta, "chunk": text})

    def add_batch(self, texts: list[str], metas: list[dict]):
        """Batch-add multiple chunks -- uses GPU batch embedding for speed."""
        if not self._faiss or not texts:
            return
        vecs = self.embedder.embed_batch(texts)
        valid = [(v, m, t) for v, m, t in zip(vecs, metas, texts) if v is not None]
        if not valid:
            return
        arr = np.array([v for v, _, _ in valid], dtype=np.float32)

        with self._lock:
            self._ensure_index()
            if self._gpu_index is not None:
                self._gpu_index.add(arr)
                self.index.add(arr)
            else:
                self.index.add(arr)
            for _, m, t in valid:
                self.meta.append({**m, "chunk": t})

    def search(self, query: str, k: int = 12) -> list[dict]:
        """Find the k most relevant chunks for a query (GPU-accelerated if available)."""
        active = self._gpu_index if self._gpu_index is not None else self.index
        if not self._faiss or active is None or active.ntotal == 0:
            return []

        vec = self.embedder.embed(query)
        if vec is None:
            return []

        k = min(k, active.ntotal)
        distances, indices = active.search(
            np.array([vec], dtype=np.float32), k
        )

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self.meta):
                entry = dict(self.meta[idx])
                entry["score"] = float(dist)
                results.append(entry)
        return results

    def save(self):
        """Save the CPU-side index (GPU index is derived from it on load)."""
        if not self._faiss or self.index is None:
            return
        # Always write the CPU index
        cpu_idx = (
            self._faiss.index_gpu_to_cpu(self.index)
            if self._gpu_res is not None and hasattr(self._faiss, "index_gpu_to_cpu")
            else self.index
        )
        self._faiss.write_index(cpu_idx, INDEX_FILE)
        with open(META_FILE, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)
        log.info(f"Index saved: {self.index.ntotal} chunks")

    def load(self):
        if not self._faiss:
            return
        if os.path.exists(INDEX_FILE) and os.path.exists(META_FILE):
            cpu_idx = self._faiss.read_index(INDEX_FILE)
            if self._gpu_res is not None:
                try:
                    self._gpu_index = self._faiss.index_cpu_to_gpu(self._gpu_res, 0, cpu_idx)
                    log.info("FAISS: loaded index onto GPU")
                except Exception as e:
                    log.warning(f"GPU index load failed ({e}), using CPU")
                    self._gpu_index = None
            self.index = cpu_idx
            with open(META_FILE, "r", encoding="utf-8") as f:
                self.meta = json.load(f)
            n_docs = len(set(m["source"] for m in self.meta))
            log.info(f"Index loaded: {self.index.ntotal} chunks from {n_docs} docs")
        else:
            log.info("No index found. Run: python main.py ingest")

    @property
    def doc_count(self) -> int:
        active = self._gpu_index if self._gpu_index is not None else self.index
        return active.ntotal if active else 0
