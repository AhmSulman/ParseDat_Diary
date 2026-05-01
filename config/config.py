"""
MAAN Configuration — All Settings in One Place
================================================
Edit this file to customize MAAN's behaviour.
"""

import os


class Config:

    # ── Paths ─────────────────────────────────────────────────────────────────
    INPUT_DIR: str         = "data/input"
    OUTPUT_TXT: str        = "data/txt"
    OUTPUT_JSON: str       = "data/json"
    CACHE_DIR: str         = "data/cache"
    CHECKPOINT_DIR: str    = "data/checkpoints"
    MODELS_DIR: str        = "data/models"
    LOG_FILE: str          = "logs/app.log"

    # ── LLM (Local Language Model) ────────────────────────────────────────────
    # Download a .gguf model and put it in data/models/
    # Then set the filename here:
    LLM_MODEL_PATH: str    = os.path.join("data", "models", "mistral-7b-instruct-v0.2.Q4_K_M.gguf")
    LLM_GPU_LAYERS: int    = 35       # Layers to offload to RTX 4050 VRAM (0 = CPU only)
    LLM_CONTEXT_SIZE: int  = 4096     # Max tokens the model can "remember" per chat
    LLM_MAX_TOKENS: int    = 1024     # Max tokens to generate per answer
    LLM_TEMPERATURE: float = 0.7      # 0.0 = deterministic, 1.0 = creative

    # ── OCR & GPU ─────────────────────────────────────────────────────────────
    OCR_DPI: int           = 200      # Higher = better quality, slower
    OCR_BATCH_SIZE: int    = 8        # Pages per GPU batch
    ONNX_MODEL_PATH: str   = ""       # Optional: path to your .onnx OCR model

    # ── Async Pipeline ────────────────────────────────────────────────────────
    ASYNC_WORKERS: int     = 4        # Parallel OCR workers

    # ── Text Chunking ─────────────────────────────────────────────────────────
    CHUNK_SIZE: int        = 1000     # Characters per chunk (~250 words)
    CHUNK_OVERLAP: int     = 200      # Overlap between chunks

    # ── Search / Embeddings ───────────────────────────────────────────────────
    EMBED_MODEL: str       = "all-MiniLM-L6-v2"
    SEARCH_TOP_K: int      = 5

    # ── Web Server ────────────────────────────────────────────────────────────
    SERVER_HOST: str       = "0.0.0.0"
    SERVER_PORT: int       = 8000

    # ── CPU ───────────────────────────────────────────────────────────────────
    CPU_WORKERS: int       = max(1, (os.cpu_count() or 2) // 2)
