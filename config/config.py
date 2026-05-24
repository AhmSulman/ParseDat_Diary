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

    # Text Chunking -- smaller = more precise retrieval, better forward/back tracing
    CHUNK_SIZE: int        = 400       # ~100 words per chunk
    CHUNK_OVERLAP: int     = 80        # 20% overlap keeps cross-boundary sentences

    # Search / Embeddings
    EMBED_MODEL: str       = "all-MiniLM-L6-v2"
    SEARCH_TOP_K: int      = 12        # Retrieve more small chunks = same text coverage

    # Web Server
    SERVER_HOST: str       = "0.0.0.0"
    SERVER_PORT: int       = 8000

    # CPU
    CPU_WORKERS: int       = max(1, (os.cpu_count() or 2) // 2)
