"""
Settings overlay — GUI-editable config, kept out of config.py
================================================================
data/settings.json holds only values the user changed from a Settings
screen. Config layers this on top of its hardcoded class defaults at
construction time (see Config.__init__). Deleting the file — or this whole
module — drops every override; nothing else reads data/settings.json
directly, so the app falls straight back to the defaults in config.py.

SETTINGS_SCHEMA is the single source of truth the GUI renders its form from
and validates against. Adding a new tunable is one entry here, not a GUI
edit.

RESTART SEMANTICS ARE PART OF THE SCHEMA
-----------------------------------------
Most of these values are read once, at object construction time, by code
that then holds onto them (RAGPipeline.top_k, LocalLLM.temperature,
ServerManager's launch command...). Saving a new value to the overlay does
NOT retroactively change an object that already read the old one. Each
field is tagged with what it actually takes to apply:

    None      -- hot-applied immediately (the GUI patches the live
                 RAGPipeline/LLM objects on Save).
    "server"  -- also needs llama-server relaunched (ServerManager).
    "reindex" -- also needs `main.py reindex` (changes the vectors).
    "app"     -- needs the running process (gui/console/server) relaunched
                 -- nothing hot-patches a value read only at startup.

Guessing "no restart needed" for a field that is actually cached elsewhere
would be a worse bug than being honest about the relaunch.
"""

from __future__ import annotations

import json
import os
import threading

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS_FILE = os.path.join(_ROOT, "data", "settings.json")

_lock = threading.Lock()
_cache: dict | None = None     # None = not loaded yet this process


def _read_raw() -> dict:
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError, UnicodeDecodeError):
        return {}


def load_overlay() -> dict:
    """
    Every user-set override, held in a plain in-memory cache for the life of
    this process -- no per-call disk stat.

    Config() is constructed constantly: once per RAG query
    (brain/llm.py's build_rag_messages), once per Chunker/Retriever/Embedder,
    every GUI callback that reads a setting. An mtime-recheck-on-every-call
    version of this function turned every one of those into a real disk
    stat, which is measurable next to what used to be a free class-attribute
    lookup. It bought safety against a DIFFERENT process editing the file
    mid-run, which barely happens: every `main.py ...` invocation is its own
    fresh process and already reads the file once, cold, at startup. Not
    worth paying for on every call from every process, always.

    save_overlay()/update_overlay() -- the only way *this* process changes
    the file -- update the cache directly, so a GUI save is visible
    immediately without a re-read. Call invalidate_cache() explicitly if you
    ever need to pick up an edit made by something else.
    """
    global _cache
    if _cache is None:
        with _lock:
            if _cache is None:
                _cache = _read_raw()
    return dict(_cache)


def invalidate_cache() -> None:
    """Force the next load_overlay() to re-read from disk."""
    global _cache
    with _lock:
        _cache = None


def save_overlay(data: dict) -> None:
    """Atomic write -- a crash mid-write must not corrupt existing overrides."""
    global _cache
    os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
    tmp = SETTINGS_FILE + ".tmp"
    with _lock:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp, SETTINGS_FILE)
        _cache = dict(data)


def update_overlay(patch: dict) -> dict:
    """Merge `patch` into the existing overlay and persist. Returns the result."""
    current = load_overlay()
    current.update(patch)
    save_overlay(current)
    return current


def clear_overlay() -> None:
    """Drop every override -- back to config.py defaults."""
    save_overlay({})


# ── schema ───────────────────────────────────────────────────────────────
# Excluded on purpose: EMBED_TRUST_REMOTE_CODE and EMBED_FALLBACK are
# deliberate lockdown decisions recorded in CLAUDE.md ("No
# trust_remote_code", "NO FALLBACK, deliberately") -- not something a
# settings screen should let slip by accident. LLM_MODEL_PATH, EMBED_MODEL
# and EMBED_DIM keep their existing dedicated pickers in the GUI rather
# than a duplicate text field here. Filesystem path constants (INPUT_DIR,
# CACHE_DIR, ...) are structural, derived from _ROOT, and not user tunables.
SETTINGS_SCHEMA = [
    # -- LLM generation ------------------------------------------------------
    dict(key="LLM_GPU_LAYERS", group="LLM Generation", label="GPU layers (-ngl)",
         type="int", min=0, max=99, restart="server",
         help="Layers offloaded to VRAM by llama-server. 0 = CPU only."),
    dict(key="LLM_CONTEXT_SIZE", group="LLM Generation", label="Context size (-c)",
         type="int", min=512, max=131072, restart="server",
         help="Must stay >= the retrieval budget or llama.cpp silently drops "
              "the front of the prompt."),
    dict(key="LLM_N_THREADS", group="LLM Generation", label="CPU threads (-t)",
         type="int", min=1, max=128, restart="server"),
    dict(key="LLM_N_BATCH", group="LLM Generation", label="Prompt batch size (-b)",
         type="int", min=1, max=8192, restart="server"),
    dict(key="LLM_TEMPERATURE", group="LLM Generation", label="Temperature",
         type="float", min=0.0, max=2.0, restart=None),
    dict(key="LLM_MAX_TOKENS", group="LLM Generation", label="Max answer tokens",
         type="int", min=-1, max=32768, restart=None,
         help="-1 = no limit (model stops on its own)."),
    dict(key="LLM_ANSWER_RESERVE", group="LLM Generation", label="Answer token reserve",
         type="int", min=0, max=8192, restart="app"),

    # -- LLM backend -----------------------------------------------------------
    dict(key="LLM_BACKEND", group="LLM Backend", label="Backend",
         type="choice", choices=["auto", "server", "local"], restart="app"),
    dict(key="LLM_SERVER_URL", group="LLM Backend", label="llama-server URL",
         type="str", restart="app"),
    dict(key="LLAMA_SERVER_BIN", group="LLM Backend", label="llama-server.exe path",
         type="str", restart="server",
         help="Blank = auto-detect. Set this instead of hardcoding a path "
              "in server_manager.py."),

    # -- Retrieval ---------------------------------------------------------------
    dict(key="SEARCH_TOP_K", group="Retrieval", label="Top-K chunks",
         type="int", min=1, max=50, restart=None),
    dict(key="NEIGHBOR_RADIUS", group="Retrieval", label="Neighbour radius",
         type="int", min=0, max=5, restart=None),
    dict(key="CONTEXT_CHAR_BUDGET", group="Retrieval", label="Context char budget",
         type="int", min=1000, max=200000, restart=None),

    # -- Chunking (ingestion-time; needs reindex) --------------------------------
    dict(key="CHUNK_SIZE", group="Chunking", label="Chunk size (chars)",
         type="int", min=200, max=20000, restart="reindex"),
    dict(key="CHUNK_OVERLAP", group="Chunking", label="Chunk overlap (chars)",
         type="int", min=0, max=10000, restart="reindex",
         help="Must stay under half of chunk size."),

    # -- OCR / ingestion -----------------------------------------------------------
    dict(key="OCR_DPI", group="OCR & Ingestion", label="OCR DPI",
         type="int", min=72, max=600, restart="app"),
    dict(key="OCR_BATCH_SIZE", group="OCR & Ingestion", label="OCR batch size",
         type="int", min=1, max=64, restart="app"),
    dict(key="ASYNC_WORKERS", group="OCR & Ingestion", label="Ingest workers",
         type="int", min=1, max=16, restart="app"),
    dict(key="ONNX_MODEL_PATH", group="OCR & Ingestion", label="ONNX OCR model path",
         type="str", restart="app",
         help="Currently unused: the ONNX inference path is disabled and "
              "Tesseract CPU runs regardless of this value."),

    # -- Web server ------------------------------------------------------------------
    dict(key="SERVER_HOST", group="Web Server", label="Bind host",
         type="str", restart="app",
         help="127.0.0.1 keeps the admin routes (clean/sync -- can delete "
              "files) local-only. 0.0.0.0 exposes them to your whole network."),
    dict(key="SERVER_PORT", group="Web Server", label="Port",
         type="int", min=1, max=65535, restart="app"),

    # -- System -----------------------------------------------------------------------
    dict(key="CPU_WORKERS", group="System", label="CPU workers",
         type="int", min=1, max=64, restart="app"),
]

_BY_KEY = {f["key"]: f for f in SETTINGS_SCHEMA}


def validate(key: str, raw) -> tuple[bool, str, object]:
    """
    Coerce and bounds-check one field's raw input against its schema entry.

    Returns (ok, error_message, coerced_value). On failure the coerced
    value is None and the caller must not write it.
    """
    field = _BY_KEY.get(key)
    if field is None:
        return False, f"unknown setting: {key}", None

    t = field["type"]
    try:
        if t == "int":
            value = int(str(raw).strip())
        elif t == "float":
            value = float(str(raw).strip())
        elif t == "choice":
            value = str(raw).strip()
            if value not in field["choices"]:
                return False, f"must be one of {field['choices']}", None
        else:
            value = str(raw).strip()
    except (TypeError, ValueError):
        return False, f"'{raw}' is not a valid {t}", None

    if t in ("int", "float"):
        lo, hi = field.get("min"), field.get("max")
        if lo is not None and value < lo:
            return False, f"must be >= {lo}", None
        if hi is not None and value > hi:
            return False, f"must be <= {hi}", None

    return True, "", value


def validate_patch(patch: dict, current: dict) -> list[str]:
    """
    Cross-field invariants that a single-field validate() cannot see.

    `current` is the effective config (defaults + existing overlay) so a
    partial patch -- e.g. only CHUNK_SIZE changing -- is checked against the
    CHUNK_OVERLAP already in effect, not against a missing value.
    """
    errors = []
    size = patch.get("CHUNK_SIZE", current.get("CHUNK_SIZE"))
    overlap = patch.get("CHUNK_OVERLAP", current.get("CHUNK_OVERLAP"))
    if size is not None and overlap is not None and overlap >= size // 2:
        errors.append(
            f"CHUNK_OVERLAP ({overlap}) must be less than half of "
            f"CHUNK_SIZE ({size}) -- the chunker's own loop cannot "
            f"terminate otherwise."
        )
    return errors


def restart_labels(keys) -> dict[str, list[str]]:
    """Group a set of changed keys by what applying them requires."""
    out: dict[str, list[str]] = {}
    for k in keys:
        field = _BY_KEY.get(k)
        tag = field["restart"] if field else "app"
        out.setdefault(tag or "none", []).append(k)
    return out
