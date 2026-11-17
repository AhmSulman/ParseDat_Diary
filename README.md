# ParseDat Diary: chat with your books

**A fully local RAG system. Drop in PDFs and ask questions; answers come back with page-level citations. No cloud and no API keys. It runs on a 6 GB laptop GPU.**

![Python](https://img.shields.io/badge/Python_3.12-3776AB?style=flat-square&logo=python&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-HNSW-0467DF?style=flat-square)
![llama.cpp](https://img.shields.io/badge/llama.cpp-Mistral_7B-000?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![Kivy](https://img.shields.io/badge/KivyMD-GUI_+_Android-191A1B?style=flat-square)

---

## How it works

```
data/input/*.pdf
  → core/async_pipeline.py   4 async workers
  → core/gpu_ocr.py          text extraction + Tesseract OCR fallback
  → core/normalize.py        ligature fixes, de-hyphenation (idempotent)
  → core/quality.py          PASS/FAIL gate; failures go to quarantine
  → brain/chunker.py         1200-char chunks, 200 overlap, page metadata
  → brain/embedder.py        BAAI/bge-base-en-v1.5 (768-dim)
  → brain/retriever.py       FAISS HNSWFlat index

question
  → retriever                top-6 chunks, expanded ±1 neighbour and merged
  → brain/llm.py             Mistral-7B (Q4_K_M) via llama-cpp-python
  → streamed answer with inline [N] citations, each validated against its source
```

### Engineering notes

- **VRAM budgeting:** Mistral-7B at 35 GPU layers plus an 8k KV cache uses about 5 GB of 6 GB. So the embedder runs on CUDA during ingest (no LLM loaded) and on CPU during chat, where a single query embedding costs about 15 ms.
- **Pinned, verified dependencies:** loose `>=` pins once let pip resolve a CPU-only torch, which silently disabled every GPU path. `requirements-gpu.txt` pins the exact CUDA builds and includes assert-based checks.
- **Self-healing library:** `doctor`, `reindex`, `sync` and `clean` commands reconcile the index, metadata and manifest against what's actually in `data/input/`.

## Interfaces

| Command | What it does |
|---|---|
| `python main.py gui` | KivyMD desktop app (primary interface) |
| `python main.py chat` | Terminal chat |
| `python main.py server` | FastAPI server, used by the Android client |
| `python main.py ingest` | Extract and index any new PDFs |
| `python main.py doctor` | Read-only health report; run it first when something looks wrong |
| `python main.py reindex` | Rebuild vectors from extracted text (no OCR) |
| `python main.py sync` | Reconcile all stores after deleting PDFs |

It can also run as a Windows service (`service/`) and package for Android with Buildozer.

## Setup

```bash
python -m venv _venv
_venv\Scripts\python.exe -m pip install -r requirements.txt
# optional, NVIDIA only:
_venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
```

Put a GGUF model where `config/config.py` points, drop PDFs into `data/input/`, then run `ingest` and `gui`.

Tests: `python -m unittest discover -s tests -v`

---

Built by [Ahmed Salman (Maan)](https://github.com/AhmSulman)
