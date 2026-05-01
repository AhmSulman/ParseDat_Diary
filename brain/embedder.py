"""
Embedder — Text → Vector
=========================
Converts text into a list of numbers (384 floats) using a small
local model (all-MiniLM-L6-v2, ~22MB).

Runs entirely on your machine. No API. No internet required.
"""

import numpy as np
from logs.logger import log

MODEL_NAME = "all-MiniLM-L6-v2"
_model = None


class Embedder:
    def __init__(self):
        self._model = None

    def _get_model(self):
        global _model
        if _model is None:
            try:
                from sentence_transformers import SentenceTransformer
                log.info(f"🧠 Loading embedder: {MODEL_NAME}...")
                _model = SentenceTransformer(MODEL_NAME)
                log.info("✅ Embedder ready")
            except ImportError:
                log.warning("⚠️  sentence-transformers not installed → pip install sentence-transformers")
                return None
        return _model

    def embed(self, text: str) -> np.ndarray | None:
        """Convert text to vector. Returns None if model unavailable."""
        model = self._get_model()
        if model is None:
            return None
        if not text or not text.strip():
            return None
        text = text[:5000]  # cap at 5000 chars
        return model.encode(text, normalize_embeddings=True)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed multiple texts at once (faster than one-by-one)."""
        model = self._get_model()
        if model is None:
            return [None] * len(texts)
        texts = [t[:5000] for t in texts]
        return model.encode(texts, normalize_embeddings=True, batch_size=32)
