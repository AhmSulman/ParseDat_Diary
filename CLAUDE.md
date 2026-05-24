# MAAN — Chat with Books

## Project Overview
Local RAG (Retrieval-Augmented Generation) system. PDFs in `data/input/` → FAISS vector index → LLM answers questions from the books. No cloud, no censorship, runs entirely on the user's RTX 4050 GPU.

## Virtual Environment
Always use `_venv` — **not** `.venv`:
```
_venv\Scripts\python.exe   # run scripts
_venv\Scripts\pip.exe      # install packages
```

## Running the App
```
# Launch GUI (primary interface)
_venv\Scripts\python.exe main.py gui

# Ingest PDFs (rebuild FAISS index)
_venv\Scripts\python.exe main.py ingest

# CLI chat (no GUI)
_venv\Scripts\python.exe main.py chat

# API server (for Android client)
_venv\Scripts\python.exe main.py server
```

## Architecture

```
data/input/*.pdf
    → core/async_pipeline.py  (4 async workers)
        → core/gpu_ocr.py       (ONNX CUDA + Tesseract fallback)
        → brain/chunker.py      (400 chars / 80 overlap)
        → brain/embedder.py     (all-MiniLM-L6-v2, CUDA if available)
        → brain/retriever.py    (FAISS: GPU IVFFlat or CPU HNSWFlat)
    → data/cache/maan.index + maan_meta.json

chat: question → brain/retriever.py → top 12 chunks → brain/llm.py → streamed answer
```

## Key Files
| File | Purpose |
|------|---------|
| `config/config.py` | All paths — all are ABSOLUTE via `_ROOT = dirname(dirname(abspath(__file__)))` |
| `brain/llm.py` | Llama.cpp wrapper; max_tokens=-1, GPU layers, auto-downloads model |
| `brain/embedder.py` | sentence-transformers on CUDA/CPU |
| `brain/retriever.py` | FAISS index; GPU path needs `faiss-gpu`, falls back to CPU HNSWFlat |
| `brain/rag_pipeline.py` | RAG pipeline with `query_stream()` for live token streaming |
| `gui/material_app.py` | KivyMD dark-theme GUI (MaanMaterialApp class) |
| `gui/material_app.kv` | KV layout: Library / Read / Ask AI / Settings screens |
| `storage/categories.py` | JSON-backed PDF category manager |
| `core/gpu_ocr.py` | ONNX CUDA OCR + Tesseract; auto-configures TESSDATA_PREFIX on Windows |
| `android_main.py` | Standalone Kivy client for Android (connects to FastAPI server) |

## Model
- File: `data/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf` (~4.1 GB)
- Auto-downloaded on first run via `data/models/auto_download.py`
- GPU layers: 35 (RTX 4050, 6 GB VRAM)

## External Dependencies on Windows
- **Tesseract 5.5**: `C:\Program Files\Tesseract-OCR\` — auto-configured via registry
- **CUDA 12.1**: for torch, onnxruntime-gpu, llama-cpp-python GPU offload
- **Java 17**: required for Android APK build (buildozer)

## Absolute Paths — Important Rule
ALL paths must be constructed with `_ROOT`:
```python
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOME_FILE = os.path.join(_ROOT, "data", "subdir", "file.ext")
```
Never use bare relative strings like `"data/cache/..."` — they break when CWD differs.

## FAISS Index
- Created by: `python main.py ingest`
- Location: `data/cache/maan.index` + `data/cache/maan_meta.json`
- Reset: delete both files OR `Checkpoint().reset()` then re-ingest
- GPU FAISS needs `faiss-gpu` (hard to install on Windows — CPU HNSWFlat is the default)

## Android APK Build
- GitHub Actions: `.github/workflows/build-android.yml` auto-triggers on push to master
- Android app: `android_main.py` — a thin client that calls the FastAPI server
- Artifact: downloaded from GitHub Actions → Artifacts after successful build

## Common Issues
| Symptom | Fix |
|---------|-----|
| Model not found | Check `config/config.py` LLM_MODEL_PATH is absolute; model must be ~4.1 GB |
| Cache empty / ingest skip | `Checkpoint().reset()` then `python main.py ingest` |
| Tesseract errors | `TESSDATA_PREFIX` auto-set from registry; or set manually to `C:\Program Files\Tesseract-OCR\tessdata` |
| GUI not launching | Run `python main.py gui`, not `python gui/material_app.py` directly |
| Push timeout (HTTP 408) | Create a new clean branch from remote HEAD with only source files; push that |
