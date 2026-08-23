"""
MAAN Configuration -- All Settings in One Place
================================================
Edit this file to customize MAAN's behaviour.
"""

import os

# Project root is two levels up from this file (config/config.py -> root)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*parts) -> str:
    """Absolute path relative to the project root — works from any working dir."""
    return os.path.join(_ROOT, *parts)


class Config:

    # Paths (all absolute so the app works from any working directory)
    INPUT_DIR: str         = _p("data", "input")
    OUTPUT_TXT: str        = _p("data", "txt")
    OUTPUT_JSON: str       = _p("data", "json")
    CACHE_DIR: str         = _p("data", "cache")
    CHECKPOINT_DIR: str    = _p("data", "checkpoints")
    MODELS_DIR: str        = _p("data", "models")
    LOG_FILE: str          = _p("logs", "app.log")

    # LLM (Local Language Model)
    LLM_MODEL_PATH: str    = _p("data", "models", "mistral-7b-instruct-v0.2.Q4_K_M.gguf")
    LLM_GPU_LAYERS: int    = 35        # Layers on RTX 4050 VRAM (0 = CPU only)
    LLM_CONTEXT_SIZE: int  = 8192      # Extended context window
    LLM_MAX_TOKENS: int    = -1        # -1 = no limit (model stops on its own)
    LLM_TEMPERATURE: float = 0.7
    LLM_N_BATCH: int       = 1024      # Tokens per prompt-eval batch (higher = faster)
    LLM_N_THREADS: int     = max(1, (os.cpu_count() or 4))

    # OCR & GPU
    OCR_DPI: int           = 200
    OCR_BATCH_SIZE: int    = 8
    ONNX_MODEL_PATH: str   = ""

    # Async Pipeline
    ASYNC_WORKERS: int     = 4

    # Text Chunking
    # 400 chars x top_k 12 filled only ~1.2K of the 8192-token window with 12
    # disconnected fragments, most cut mid-sentence. At 1200/200 the corpus
    # (8,303,639 chars) yields ~8,300 chunks on a 1000-char stride, down from
    # ~25,900, and each chunk holds a whole thought.
    CHUNK_SIZE: int        = 1200      # ~300 words per chunk
    CHUNK_OVERLAP: int     = 200       # 16.7% -- must stay < CHUNK_SIZE // 2

    # Search / Embeddings
    # jina-embeddings-v2-base-en: Apache 2.0, 768-dim, 8192-token context, 137M
    # params. The long context is what makes 1200-char chunks embeddable whole;
    # all-MiniLM-L6-v2 truncated at 256 tokens. Falls back to bge-base-en-v1.5
    # if the trust_remote_code path breaks against transformers 5.x.
    EMBED_MODEL: str       = "jinaai/jina-embeddings-v2-base-en"
    EMBED_FALLBACK: str    = "BAAI/bge-base-en-v1.5"
    EMBED_DIM: int         = 768       # was hardcoded as DIM=384 in retriever.py

    # Retrieval
    # Budget: 6 windows x ~3200 chars = ~19,200 chars = ~4,800 tokens, leaving
    # ~3,392 of the 8192-token window for the prompt and the answer.
    SEARCH_TOP_K: int      = 6
    NEIGHBOR_RADIUS: int   = 1         # also pull chunk_id +/- 1 from same book
    CONTEXT_CHAR_BUDGET: int = 20000   # hard cap on assembled context

    # Web Server
    SERVER_HOST: str       = "0.0.0.0"
    SERVER_PORT: int       = 8000

    # CPU
    CPU_WORKERS: int       = max(1, (os.cpu_count() or 2) // 2)
