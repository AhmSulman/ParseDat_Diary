"""
Embedder -- Text -> Vector (GPU-accelerated)
=============================================
Converts text into 384-dim vectors using all-MiniLM-L6-v2 (~22 MB).

- On Windows with CUDA: runs on GPU via sentence-transformers + torch-cuda
- On Android / CPU-only: runs on CPU (same API, same results)
- Batch embedding uses CUDA streams for maximum throughput

Runs entirely on your machine. No API. No internet required.
"""

import os
import numpy as np
from logs.logger import log

MODEL_NAME = "all-MiniLM-L6-v2"
_CHAR_CAP  = 1500   # MiniLM max is 256 tokens ~= 1500 chars
_model     = None
_device    = None   # "cuda", "mps", or "cpu"


def _detect_device() -> str:
    """Pick the best available compute device."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            log.info(f"Embedder: CUDA device -- {name}")
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            log.info("Embedder: Apple MPS device")
            return "mps"
    except ImportError:
        pass
    log.info("Embedder: CPU device (install torch + CUDA for GPU acceleration)")
    return "cpu"


class Embedder:
    def __init__(self):
        pass

    def _get_model(self):
        global _model, _device
        if _model is None:
            _device = _detect_device()
            try:
                from sentence_transformers import SentenceTransformer
                log.info(f"Loading embedder: {MODEL_NAME} on {_device} ...")
                _model = SentenceTransformer(MODEL_NAME, device=_device)
                log.info("Embedder ready")
            except ImportError:
                log.warning("sentence-transformers not installed -- pip install sentence-transformers")
                return None
        return _model

    def embed(self, text: str) -> np.ndarray | None:
        """Convert text to a normalised 384-dim vector."""
        model = self._get_model()
        if model is None:
            return None
        if not text or not text.strip():
            return None
        return model.encode(
            text[:_CHAR_CAP],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """
        Embed multiple texts in one GPU pass -- much faster than one-by-one.
        Uses a larger batch_size on CUDA to saturate GPU memory bandwidth.
        """
        model = self._get_model()
        if model is None:
            return [None] * len(texts)
        capped      = [t[:_CHAR_CAP] for t in texts]
        batch_size  = 128 if _device == "cuda" else 32
        return model.encode(
            capped,
            normalize_embeddings=True,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
