# Status — what's actually built (2026-08-24)

ParseDat_Diary is a local, offline RAG system: documents in `data/input/` → FAISS index →
a local LLM answers from them with page-level citations. Windows, Python 3.12.0,
venv is `_venv`.

Read `NEXT.md` for the punch list and the decisions not to re-argue.

## Start it

```
_venv\Scripts\python.exe main.py console     # terminal: ask, tune, manage
_venv\Scripts\python.exe main.py gui         # KivyMD desktop app
_venv\Scripts\python.exe main.py doctor      # health check — run this first
```

Answers need a model server running:

```
E:\llama-b10034-bin-win-cpu-x64\llama-server.exe ^
  -m C:\Users\ahmed\Desktop\ParseDat_Diary\data\models\DeepSeek-R1-Distill-Qwen-7B-Q4_K_M_2.gguf ^
  -ngl 0 -c 8192 -t 12 --port 8084
```

Killswitch: `taskkill /IM llama-server.exe /F`

## Working now

**Library** — 13 documents, 9,209 chunks, zero drift. `doctor` reconciles six
derived stores against `data/input/`, which is the sole authority for
membership. `sync` prunes everything belonging to a deleted document; `forget`
removes one.

**Ingestion** — PDFs (PyMuPDF + Tesseract OCR) and `.md`/`.txt` (read directly,
no OCR). Text is normalised (ligatures, de-hyphenation), quality-gated, chunked
at 1200/200 with exact offsets and page or heading locators.

**Retrieval** — bge-base-en-v1.5, 768-dim, vendored into
`data/models/embedder/`. Top-6 hits expand to their neighbours and merge into
continuous passages, de-overlapped arithmetically from stored offsets.

**Answers** — DeepSeek-R1 via `llama-server` over HTTP. Inline `[N]` citations
resolve to book + page (or `notes.md - ## Section`), and every marker is
validated against the sources actually supplied — an out-of-range citation is a
mechanically detectable hallucination.

**History** — conversations persist to `data/sessions/` as JSON plus a readable
`.md`. New chat, session list, replay, export.

**Offline** — verified with the HuggingFace endpoint pointed at a dead host and
no offline flags set: the embedder loads from the vendored folder, the index
loads, search returns passages, Tesseract resolves.

**Tests** — 69, stdlib `unittest`:
`_venv\Scripts\python.exe -m unittest discover -s tests -v`

## Known-not-done

- **GPU generation.** `torch` CUDA works (verified, 8.1x on a matmul), but
  `llama_cpp.llama_supports_gpu_offload()` is False and the CPU-only llama.cpp
  binaries are what is installed. Answers take **~4 minutes**; most of that is
  the model reading ~5,000 tokens of passages, which is exactly what a GPU
  accelerates. Downloading the CUDA build of llama.cpp is the single biggest
  remaining win.
- **Retrieval precision.** Measured: a query about database replication returned
  a bibliography entry. A reranker is the fix and is not built.
- **Answer quality is barely measured.** One end-to-end answer was verified
  correct with citations resolving to the right pages. That is one data point.
- **ONNX GPU OCR is fiction** — `core/gpu_ocr.py` advertises it, the session
  load is commented out, Tesseract CPU is the only path that runs.
- **origin has diverged.** History was rewritten (972 MB → 75 KB, a committed
  `.venv` was in it). Needs `git push --force`. `.git.backup/` is the safety net
  until then; delete it after.
