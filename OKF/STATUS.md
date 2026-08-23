# Status — what's actually built (2026-08-23)

MAAN is a local RAG system: PDFs in `data/input/` → FAISS index → local Mistral
answers questions from the books. Windows, Python 3.12.0, venv is `_venv`.

Read `NEXT.md` for the punch list and the decisions not to re-litigate.

## Working now

**Text pipeline** — `core/normalize.py`, `core/quality.py`, `brain/chunker.py`
- Normalisation repairs PDF text-layer damage before anything else sees it:
  NFKC folds ligatures (3,411 in the corpus — `efﬁcient` was never embedded as
  "efficient"), and de-hyphenation rejoins words split across line breaks
  (3,369). De-hyphenation is decided from corpus word frequency, so
  `en-\ncapsulated` fuses while `scikit-learn` keeps its hyphen. Idempotent.
- Quality gate scores extracted text and quarantines books that fail. It caught
  a book whose Latin vowels were replaced with Cyrillic homoglyphs (a piracy
  watermark) — 791 chunks of noise. Detection is script-mixing aware, so
  `mml-book.pdf`'s real Greek maths passes.
- Chunker at 1200/200 emits `chunk_id`, `n_chunks`, exact `char_start/char_end`
  and `page_start/page_end`. 9,045 chunks across 12 books, 100% with pages.

**Library coherence** — `storage/manifest.py`, `storage/library.py`
- The manifest is the book-level record; it is what can answer "how many books
  do you have?".
- `data/input/*.pdf` is the sole authority for library membership. Everything
  else is derived and reconciles to it.
- CLI: `doctor` (read-only report), `clean`, `sync`, `reindex`.

**Ingest ordering** — `core/async_pipeline.py`
- A book is marked done only after its vectors are in the index, and the index
  is saved after every book. This was the bug behind the permanent holes.

**Retrieval and answers** — `brain/retriever.py`, `brain/llm.py`, `brain/rag.py`
- Neighbour expansion merges chunk ±1 into continuous passages, de-overlapped
  arithmetically from stored offsets.
- Prompts state the library contents explicitly and number every excerpt with
  its page span. Answers cite `[N]` inline; every marker is validated against
  the sources actually supplied, so an out-of-range citation is caught.

**Tests** — 41, stdlib `unittest`:
`_venv\Scripts\python.exe -m unittest discover -s tests -v`
Includes a fixture of the watermarked book recovered from pre-rewrite git
history, so that regression stays testable after the book was purged.

**Repo** — git history rewritten to drop a committed `.venv` (torch_cpu.dll
alone was 253 MB), large PDFs and a model file: **972 MB → 75 KB**. This is the
real fix for the HTTP 408 push timeouts.

## Known-not-done

- **The index is currently EMPTY.** The old 384-dim index was deleted (a 768-dim
  embedder invalidates it). Run `reindex`, then `ingest`. 12 of 13 books rebuild
  from cached text; only `Alfred's Essentials` needs OCR.
- **GPU is not working yet.** `torch` was still installing as of this writing,
  and `llama-cpp-python` reports `llama_supports_gpu_offload() == False`, so
  Mistral runs entirely on CPU and `LLM_GPU_LAYERS=35` is ignored. See
  `requirements-gpu.txt`.
- **Jina embedder is untested against transformers 5.9.** It needs
  `trust_remote_code`, and that code targets transformers 4.x. The loader
  smoke-tests it and falls back to `bge-base-en-v1.5` (also 768-dim), but which
  one actually loads here is not yet known.
- **No dashboard yet.** `storage/library.py` is ready for it; the FastAPI
  endpoints and the KivyMD wiring are not written.
- **ONNX GPU OCR is fiction.** `core/gpu_ocr.py` advertises it, but the session
  load is commented out, so Tesseract CPU is the only path that ever runs.
- **Answer quality is unmeasured.** Nothing has been asked of the rebuilt index
  yet, because there is no rebuilt index.
