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

# Ingest PDFs (extract + index anything new)
_venv\Scripts\python.exe main.py ingest

# CLI chat (no GUI)
_venv\Scripts\python.exe main.py chat

# API server (for Android client)
_venv\Scripts\python.exe main.py server
```

### Library management
```
# Read-only health report. Run this FIRST whenever anything looks wrong.
_venv\Scripts\python.exe main.py doctor

# Rebuild vectors from extracted text -- no OCR, no PDF parsing
_venv\Scripts\python.exe main.py reindex

# After deleting PDFs: reconcile every derived store to data/input/
_venv\Scripts\python.exe main.py sync

# Targeted removal
_venv\Scripts\python.exe main.py clean --index
_venv\Scripts\python.exe main.py clean --orphans
```

## Tests
```
_venv\Scripts\python.exe -m unittest discover -s tests -v
```
Stdlib `unittest`, no pytest.

## Architecture

```
data/input/*.pdf
    → core/async_pipeline.py  (4 async workers)
        → core/gpu_ocr.py       (Tesseract CPU -- see OCR note below)
        → core/normalize.py     (ligatures, de-hyphenation; idempotent)
        → core/quality.py       (PASS/FAIL -- failures go to data/quarantine/)
        → brain/chunker.py      (1200 chars / 200 overlap + page metadata)
        → brain/embedder.py     (jina-embeddings-v2-base-en, 768-dim)
        → brain/retriever.py    (FAISS CPU HNSWFlat)
    → data/cache/maan.index + maan_meta.json + maan_manifest.json

chat: question
    → brain/retriever.py  top-6 chunks, expanded by ±1 and merged
    → brain/llm.py        numbered excerpts with page spans
    → streamed answer with inline [N] citations, each validated
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
| `storage/manifest.py` | Book-level record of the library — the only thing that knows the BOOK count |
| `storage/library.py` | Reconciliation logic shared by CLI, web and GUI |
| `core/normalize.py` | Repairs PDF text-layer damage; idempotent |
| `core/quality.py` | Quality gate — quarantines unreadable/watermarked books |
| `core/reindex.py` | Rebuild vectors from cached text, no OCR |
| `core/gpu_ocr.py` | Tesseract OCR; auto-configures TESSDATA_PREFIX on Windows |
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

## Load-bearing rules — breaking these causes silent data loss

### Checkpoint ordering
A book is marked done **only after its vectors are in the index**, from the
embed stage, and `retriever.save()` runs **after every book**.

The old pipeline marked done in the *writer* stage (as soon as text hit disk)
and saved once at the very end. A crash mid-run threw away every vector from
that session while the checkpoint already said "done", so the next ingest
skipped those books permanently. That is how the library reached 18 marked done,
7 actually indexed, 11 unreachable holes. A book that fails to embed is marked
**failed**, not done, so it is retried.

### Membership invariant
`data/input/*.pdf` is the **sole authority** for what is in the library.
Six stores derive from it: index, metadata, manifest, checkpoint, `data/txt/`,
`data/categories.json`. `sync` reconciles all six.

**`reindex` enumerates `data/input/`, never `data/txt/`.** Walking the text
directory would pick up orphaned `.txt` from deleted books and resurrect them,
undoing a purge in the very next command.

### Embedder device is explicit
Mistral-7B at 35 layers plus an 8192 KV cache is ~5 GB of 6141 MiB. There is no
room for a second model.

    ingest → Embedder(device="cuda")   (no LLM loaded)
    chat   → Embedder(device="cpu")    (one query, ~15 ms)

Auto-detecting globally puts both on the GPU and OOMs mid-answer.

### Chunk offsets are recorded after stripping
Neighbour expansion de-overlaps arithmetically:
`drop = prev.char_end - cur.char_start`. This is correct only if offsets match
the stored text exactly. Pre-strip offsets corrupt every merged passage
silently — duplicated or dropped text at the seam, no error raised.

### Normalisation must stay idempotent
It runs on fresh extraction *and* on reindex. Trailing whitespace is stripped
**before** de-hyphenation; the reverse order leaves `word-   \n` unmatched on
pass one and matched on pass two.

### Never "fix" these
- Spaces at `[a-z][A-Z]` — 6,119 corpus hits are identifiers (`MutableMapping`).
- Leading whitespace — Python listings are indent-significant.

## OCR — what actually runs
`core/gpu_ocr.py` advertises ONNX CUDA OCR, but the session load is commented
out and `_infer_onnx` returns a placeholder, so **Tesseract CPU is the only path
that ever executes**. Either wire a real ONNX model or delete the branch; do not
trust the log line claiming "RTX mode".

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
**Run `python main.py doctor` first.** It names the problem and prints the fix.

| Symptom | Fix |
|---------|-----|
| "How many books" is wrong / AI cites a book you deleted | `python main.py sync` — derived stores still reference it |
| A book exists but the AI never uses it | `doctor` will show it as a HOLE. `python main.py reindex` |
| Ingest skips everything | Checkpoint says done. `doctor`, then `reindex` (holes) or `clean --checkpoint` |
| "index was built at N dimensions" | Embedder changed. `python main.py reindex` |
| Index/metadata desync warning | `python main.py reindex` — never edit `maan_meta.json` by hand |
| A book was quarantined | `data/quarantine/<name>.report.json` names the failed metrics |
| GPU not used | Verify: `torch.cuda.is_available()` and `llama_cpp.llama_supports_gpu_offload()`. See `requirements-gpu.txt` |
| `pip.exe` exits 1 with no output | Use `_venv\Scripts\python.exe -m pip` instead |
| Model not found | `LLM_MODEL_PATH` must be absolute; model is ~4.1 GB |
| Tesseract errors | `TESSDATA_PREFIX` auto-set from registry, or `C:\Program Files\Tesseract-OCR\tessdata` |
| GUI not launching | `python main.py gui`, not `python gui/material_app.py` |
| Push timeout (HTTP 408) | Was caused by a committed `.venv` in history. Fixed by rewrite: 972 MB → 75 KB |
