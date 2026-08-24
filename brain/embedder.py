"""
Embedder — text -> vector
==========================
Model: BAAI/bge-base-en-v1.5 (MIT, 768-dim, 512-token, 109M params).

OFFLINE IS THE POINT
--------------------
The weights are VENDORED into data/models/embedder/ and loaded from that
folder. A filesystem path is not a repo id, so sentence-transformers treats it
as a plain directory and never consults the hub: nothing to look up, nothing to
time out, and no dependence on ~/.cache surviving.

That matters because the hub is consulted even for fully cached models.
Without local_files_only, sentence-transformers issues a HEAD request to check
for updates, and with no network it retries and then fails — a 419 MB model on
disk still would not load. Vendoring removes the question entirely; the cache
and hub remain as fallbacks if the folder is missing.

Jina was rejected for the same reason: trust_remote_code downloads and EXECUTES
Python from the hub at load time. With 5.48 GB of v3 weights already cached it
still fetched mlp.py, stochastic_depth.py and rotary.py.

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

import os

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


def _free_cuda() -> None:
    """Release cached VRAM after moving off the GPU, so the LLM can claim it."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


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
            # The cache is process-wide, and the GUI does ingest AND chat in one
            # process. Returning the cached model without checking its device
            # meant chat inherited the CUDA-resident embedder loaded by ingest,
            # then sat in VRAM competing with Mistral — the exact OOM the device
            # split exists to prevent. Move it instead; moving is far cheaper
            # than reloading.
            want = _detect_device(self.requested_device)
            if self.requested_device and _device != want:
                try:
                    log.info(f"Moving embedder {_device} -> {want}")
                    _model = _model.to(want)
                    _device = want
                    if want == "cpu":
                        _free_cuda()
                except Exception as e:
                    log.warning(f"Could not move embedder to {want}: {e}")
            return _model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            log.error("sentence-transformers not installed")
            return None

        _device = _detect_device(self.requested_device)
        cfg = self.cfg

        # trust_remote_code is OFF by default and must stay that way unless the
        # offline-first guarantee is being deliberately traded away: it
        # downloads and executes Python from the hub at load time, so the app
        # stops working unplugged even when the weights are cached locally.
        trust = bool(getattr(cfg, "EMBED_TRUST_REMOTE_CODE", False))
        extra = {"trust_remote_code": True} if trust else {}

        def kwargs_for(_name: str) -> dict:
            return dict(extra)

        # Try the LOCAL CACHE for every candidate before touching the network.
        #
        # This is what actually makes the app offline, and it is not optional.
        # Without local_files_only, sentence-transformers issues a HEAD request
        # to the hub to check for updates even when the model is fully cached —
        # so with no network it retries, then fails, and a 419 MB model sitting
        # on disk still will not load. Verified by pointing HF_ENDPOINT at a
        # dead host: cached bge failed to load entirely and fell through to a
        # fallback that was not cached either.
        # The vendored folder comes first. A filesystem path is not a repo id,
        # so sentence-transformers loads it as a plain directory and never
        # consults the hub at all — no lookup to fail, no cache to lose.
        vendored = getattr(cfg, "EMBED_LOCAL_DIR", "")
        candidates = []
        if vendored and os.path.isdir(vendored):
            candidates.append((vendored, True))
        candidates += [(cfg.EMBED_MODEL, True), (cfg.EMBED_FALLBACK, True),
                       (cfg.EMBED_MODEL, False), (cfg.EMBED_FALLBACK, False)]

        for name, local_only in candidates:
            if not name:
                continue
            try:
                log.info(f"Loading embedder: {name} on {_device}"
                         f" ({'cache' if local_only else 'hub'})")
                m = SentenceTransformer(name, device=_device,
                                        local_files_only=local_only, **kwargs_for(name))
                if not _smoke_test(m, cfg.EMBED_DIM):
                    log.warning(f"{name} produced an unexpected vector shape")
                    continue
                # Report the CANONICAL model id, never the folder path. The
                # manifest records which embedder built the index, and the
                # retriever refuses to search on a mismatch — loading the same
                # weights from a vendored directory must not look like a
                # different model.
                _model = m
                _model_name = cfg.EMBED_MODEL if name == vendored else name
                where = "vendored" if name == vendored else name
                log.info(f"Embedder ready: {_model_name} ({cfg.EMBED_DIM}-dim, "
                         f"{_device}, from {where})")
                return _model
            except Exception as e:
                # A local cache miss is expected and routine; only a failed
                # network attempt is worth warning about.
                if local_only:
                    log.info(f"Not in local cache: {name}")
                else:
                    log.warning(f"Could not fetch {name}: {str(e)[:160]}")

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
