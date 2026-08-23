# Next — punch list, roughly in order

## Immediate

1. **Finish the GPU install and verify it.** Both gates must pass; an install
   that "succeeds" while leaving a CPU build in place is the exact failure being
   guarded against.
   ```
   _venv\Scripts\python.exe -c "import torch; assert torch.cuda.is_available()"
   _venv\Scripts\python.exe -c "import llama_cpp; assert llama_cpp.llama_supports_gpu_offload()"
   ```
   llama.cpp is the risky one: PyPI wheels are CPU-only, so it needs the
   prebuilt CUDA wheel index or a source build. **If it cannot be made to work,
   set `LLM_GPU_LAYERS = 0`** rather than leaving the config claiming offload it
   does not have.

2. **Confirm which embedder actually loads.** Jina needs `trust_remote_code`
   against transformers 5.9. Check the log line and the manifest's
   `embed_model`; if it fell back to bge, that is fine — both are 768-dim — but
   it must be *known*, not assumed.

3. **Rebuild the index.** `reindex` (12 books from cached text, no OCR), then
   `ingest` for `Alfred's Essentials`, then `doctor` — every store must agree at
   13 with zero drift.

4. **Measure answer quality.** Nothing has been asked of the new pipeline yet.
   - "How many books do you have?" must answer **13** and list them. This was
     previously unanswerable.
   - A question answerable only from one of the 11 previously-unindexed books
     must now work.
   - Take citations and **open the cited page** to confirm the claim is there.
   - Time a query before/after GPU offload.

5. **Dashboard** — FastAPI `/library` + `/ui` over `storage.library.LibraryService`,
   and point the KivyMD "Library statistics" item at the same `report()`.
   Admin routes (`clean`, `sync`) must bind to 127.0.0.1 only: `run_server`
   defaults to `0.0.0.0`, which would expose file deletion to the whole network.

6. **Fix the two GUI call sites** using `doc_count` (a chunk count) —
   `gui/material_app.py:202` and `:761` — now that `book_count` exists.

## Then

7. **Multi-model layer.** Agreed design: a ~1 GB router model resident, heavier
   models swapped via mmap. Deliberately sequenced after retrieval is measured,
   so improvements can be attributed.

   The constraint is VRAM, not disk: 5,430 MiB free, and Mistral-7B Q4 needs
   ~5 GB, so **only one 7B can be resident**. What makes swapping cheap is
   **11.7 GB of free RAM** — two or three GGUFs stay warm in page cache, making
   a switch a page-table remap rather than a disk read. Both drives are NVMe.
   The Windows pagefile is irrelevant here and should not be enlarged.

   Worth testing first: whether one stronger model (Qwen2.5-7B-Instruct) simply
   beats an ensemble of weaker ones.

## Don't re-litigate (locked decisions)

- **Bad PDFs are quarantined, not repaired.** Quality over coverage. Homoglyph
  repair is *proven to work* (mean word length 1.97 → 5.15, `Thе clаss Kеywоrd`
  → `The class Keyword`) and is a ready future win — but the call was to discard
  for now.
- **No DOCX conversion.** The defects originate in the PDF text layer, so
  `pdf2docx` (which wraps PyMuPDF) inherits them exactly. MAAN already converts
  PDF→text; what was missing was cleanup, not a different container.
- **Never insert spaces at `[a-z][A-Z]`.** 6,119 corpus hits are identifiers —
  `MutableMapping`, `ValueError`. This looks like a defect and is not.
- **Never collapse leading whitespace.** Python listings are indent-significant.
- **`reindex` enumerates `data/input/`, never `data/txt/`.** Walking the text
  directory resurrects deleted books.
- **Chunk offsets are recorded after stripping.** De-overlap is arithmetic;
  pre-strip offsets corrupt every merged passage silently.
- **FastAPI, not Flask.** FastAPI is already a dependency and already serves the
  Android client.
- **Embedder device is explicit, not auto.** Mistral fills the GPU during chat.
- **`data/json/` is not written.** Nothing ever read it; 7.6 MB duplicating
  `data/txt/`.
- **No reranker, no score threshold** — both offered and declined. Still good
  ideas if precision needs another push.

## Deferred ideas

- Cross-encoder reranking (`bge-reranker-base`) — biggest remaining retrieval
  win, ~0.3 s/query on GPU.
- Relevance score threshold, so weak matches yield "not in your books".
- Per-page quality gating, to salvage good pages from mixed-quality books.
- Homoglyph repair (see above).
- A real ONNX OCR model, or delete the ONNX path and say Tesseract plainly.
- DOCX export as a user-facing feature (not for retrieval quality).
