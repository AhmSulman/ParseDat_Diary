"""
Embedder — text -> vector
==========================
Default model: jinaai/jina-embeddings-v2-base-en
  Apache 2.0, 768-dim, 8192-token context, 137M params.

WHY NOT all-MiniLM-L6-v2
------------------------
MiniLM truncates at 256 tokens. With 1200-char chunks (~300 tokens) it would
silently discard the tail of every chunk — the text would be indexed, just not
all of it, with no error. Jina's 8192-token window embeds a whole chunk, and its
retrieval quality is materially better.

FALLBACK IS DELIBERATE
----------------------
Jina v2 needs `trust_remote_code=True`, which executes modelling code written
against transformers 4.x while this venv runs transformers 5.x. That may break.
Rather than fail the whole ingest, the loader smoke-tests the model (load, embed
one string, check the shape) and falls back to BAAI/bge-base-en-v1.5 — also
768-dim, so the index dimension is unaffected either way.

DEVICE POLICY: A VRAM CONSTRAINT, NOT AN OPTIMISATION
-----------------------------------------------------
The RTX 4050 has 6141 MiB. Mistral-7B Q4_K_M at 35 layers is ~3.6-4.0 GB, plus
~1 GB of KV cache at 8192 context: about 5 GB. There is no room for an embedder
alongside it.

    ingest -> device="cuda"   (the LLM is not loaded; embedding is the workload)
    chat   -> device="cpu"    (one short query per turn, ~15 ms)

So the device is an explicit constructor argument. Guessing globally is what
would put both models on the GPU at once and OOM mid-answer.
"""

import numpy as np

from config.config import Config
from logs.logger import log

# Must exceed CHUNK_SIZE, or chunks are silently truncated before embedding.
# Jina handles 8192 tokens; 8000 chars is comfortably inside that.
_CHAR_CAP = 8000

# bge-* models are trained asymmetrically: queries need this prefix, passages do
# not. Jina v2 is symmetric and needs no prefix. Getting this wrong costs
# retrieval quality silently.
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_model = None
_model_name: str | None = None
_device: str | None = None


def _detect_device(preferred: str | None = None) -> str:
    if preferred:
        return preferred
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _smoke_test(model, expected_dim: int) -> bool:
    """Load-time proof the model actually produces vectors of the right shape."""
    try:
        v = model.encode("smoke test", normalize_embeddings=True, convert_to_numpy=True)
        return getattr(v, "shape", (0,))[0] == expected_dim
    except Exception as e:
        log.warning(f"Embedder smoke test failed: {e}")
        return False


class Embedder:
    def __init__(self, device: str | None = None):
        cfg = Config()
        self.cfg = cfg
        self.requested_device = device

    # ── model loading ─────────────────────────────────────────────────────────
    def _get_model(self):
        global _model, _model_name, _device
        if _model is not None:
            return _model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            log.error("sentence-transformers not installed")
            return None

        _device = _detect_device(self.requested_device)
        cfg = self.cfg

        for name, kwargs in (
            (cfg.EMBED_MODEL, {"trust_remote_code": True}),
            (cfg.EMBED_FALLBACK, {}),
        ):
            try:
                log.info(f"Loading embedder: {name} on {_device}")
                m = SentenceTransformer(name, device=_device, **kwargs)
                if not _smoke_test(m, cfg.EMBED_DIM):
                    log.warning(f"{name} produced an unexpected vector shape")
                    continue
                _model, _model_name = m, name
                log.info(f"Embedder ready: {name} ({cfg.EMBED_DIM}-dim, {_device})")
                return _model
            except Exception as e:
                log.warning(f"Could not load {name}: {str(e)[:200]}")

        log.error("No embedder could be loaded")
        return None

    @property
    def model_name(self) -> str | None:
        return _model_name

    @property
    def device(self) -> str | None:
        return _device

    def _needs_query_prefix(self) -> bool:
        return bool(_model_name and "bge" in _model_name.lower())

    # ── encoding ──────────────────────────────────────────────────────────────
    def embed(self, text: str) -> np.ndarray | None:
        """Embed a passage (no query prefix)."""
        model = self._get_model()
        if model is None or not text or not text.strip():
            return None
        return model.encode(
            text[:_CHAR_CAP],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    def embed_query(self, text: str) -> np.ndarray | None:
        """
        Embed a search query.

        Separate from embed() because bge-* expects a query prefix that must NOT
        be applied to passages. Same call as embed() when the model is symmetric.
        """
        model = self._get_model()
        if model is None or not text or not text.strip():
            return None
        q = text[:_CHAR_CAP]
        if self._needs_query_prefix():
            q = _BGE_QUERY_PREFIX + q
        return model.encode(q, normalize_embeddings=True, convert_to_numpy=True)

    def embed_batch(self, texts: list[str]) -> list:
        """Embed many passages in one pass."""
        model = self._get_model()
        if model is None:
            return [None] * len(texts)
        if not texts:
            return []
        capped = [t[:_CHAR_CAP] for t in texts]
        batch_size = 64 if _device == "cuda" else 16
        return model.encode(
            capped,
            normalize_embeddings=True,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
