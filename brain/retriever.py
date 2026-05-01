"""
Retriever — Vector Database
============================
Stores and searches document chunks by meaning (not keywords).

Uses FAISS (Facebook AI Similarity Search) as the vector database.
Each chunk of text gets converted to a 384-dim vector and stored.

At search time: your question → vector → nearest neighbors → chunks
"""

import os
import json
import numpy as np
from logs.logger import log
from brain.embedder import Embedder

INDEX_FILE = "data/cache/maan.index"
META_FILE = "data/cache/maan_meta.json"
DIM = 384


class Retriever:
    def __init__(self):
        self.embedder = Embedder()
        self.index = None
        self.meta: list[dict] = []
        self._faiss = self._load_faiss_lib()
        os.makedirs("data/cache", exist_ok=True)

    def _load_faiss_lib(self):
        try:
            import faiss
            return faiss
        except ImportError:
            log.warning("⚠️  faiss-cpu not installed → pip install faiss-cpu")
            return None

    def _ensure_index(self):
        if self.index is None and self._faiss:
            self.index = self._faiss.IndexFlatL2(DIM)

    def add(self, text: str, meta: dict):
        """Embed a text chunk and add it to the FAISS index."""
        if not self._faiss:
            return

        vec = self.embedder.embed(text)
        if vec is None:
            return

        self._ensure_index()
        self.index.add(np.array([vec], dtype=np.float32))
        self.meta.append({**meta, "chunk": text[:500]})

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Find the k most relevant chunks for a query."""
        if not self._faiss or self.index is None or self.index.ntotal == 0:
            return []

        vec = self.embedder.embed(query)
        if vec is None:
            return []

        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(
            np.array([vec], dtype=np.float32), k
        )

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0:
                entry = dict(self.meta[idx])
                entry["score"] = float(dist)
                results.append(entry)

        return results

    def save(self):
        if not self._faiss or self.index is None:
            return
        self._faiss.write_index(self.index, INDEX_FILE)
        with open(META_FILE, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)
        log.info(f"💾 Index saved: {self.index.ntotal} chunks")

    def load(self):
        if not self._faiss:
            return
        if os.path.exists(INDEX_FILE) and os.path.exists(META_FILE):
            self.index = self._faiss.read_index(INDEX_FILE)
            with open(META_FILE, "r", encoding="utf-8") as f:
                self.meta = json.load(f)
            log.info(f"📂 Index loaded: {self.index.ntotal} chunks from {len(set(m['source'] for m in self.meta))} docs")
        else:
            log.info("ℹ️  No index found. Run: python main.py ingest")

    @property
    def doc_count(self) -> int:
        return self.index.ntotal if self.index else 0
