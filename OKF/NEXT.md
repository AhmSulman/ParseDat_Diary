# Next — punch list, roughly in order

Rewritten 2026-09-04. The previous version was stale in ways that actively
misled: it listed jina-v5 as a locked decision while the code ran bge, claimed
5,430 MiB of free VRAM, and said the pagefile was irrelevant. All three were
wrong and are corrected below.

## Immediate

1. **Enlarge the pagefile. This is blocking.** Available commit is ~6 GB against
   a 27.8 GB limit, so `can_load()` correctly refuses both Qwen3-4B and the 7B.
   Three models resident cost ~9.9 GB of commit.

   `Get-PhysicalDisk` shows C:, D: and E: are partitions of **one** 954 GB NVMe;
   only F: is a separate device. The current 3-way, 3,560 MB split therefore
   buys nothing. Consolidate to a single fixed 24 GB pagefile on C: (initial =
   maximum, so Windows never stalls growing it) and remove D:/E:. Needs a
   reboot. Windows setting, not code.

2. **Make the retrieval budget a token budget.** `CONTEXT_CHAR_BUDGET = 20000`
   is in characters. Measured on this corpus the tokenizer averages 4.32
   chars/token but falls to **1.82** on code-dense text, so 20,000 chars can be
   ~11,000 tokens against `-c 8192`. llama.cpp then drops the front of the
   prompt — where the citation rules and numbered excerpts live — and logs
   nothing. All three models share the Qwen2 BPE tokenizer, so one budget works
   everywhere; `llama_cpp.Llama(vocab_only=True)` counts cheaply.

3. **Fix the "maths behind machine learning" failure.** Measured 2026-09-04: the
   top 5 hits are mml-book pp. 7, 17, 8, 7, 1–3 — the preface, never the maths
   chapters. The corpus reset fixed the cross-book case and left this one. Two
   candidates, neither built:
   - **Boilerplate stripping** — detect text repeated near-verbatim across many
     pages during `core/normalize.py` and drop it before chunking. Cheap, no new
     dependency, needs a reindex. Narrow: it will not touch a genuine preface.
   - **Reranker** (`bge-reranker-base`, ~90 MB) — retrieve ~40, re-score with a
     cross-encoder, keep 6. Handles the general case. ~0.3 s/query on GPU.

4. **Scope retrieval to an active set** (1–3 documents → sections → evidence
   units) before FAISS scoring. This is the structural fix for contamination and
   may make the reranker unnecessary.

## Then

5. **Dual index** (bge + jina-v5) so retrieval quality can be A/B'd instead of
   argued about. Blast radius is small: one `Embedder()` construction site
   (`brain/retriever.py:54`) and four `Retriever()` sites. Requires moving
   `_model`/`_model_name`/`_device` off module globals in `brain/embedder.py`,
   making index paths per-instance, and a `Manifest` schema bump.

6. **Grader calibration, as a gate.** Before any mastery model is built, collect
   ~50 answers, hand-grade them, and compare against Qwen3-4B. Under ~80%
   agreement the scores are decoration and the design changes.

7. **Three-model roles**: Qwen2.5-1.5B controller (every turn), Qwen3-4B examiner
   (every turn), DeepSeek-R1-7B critical examiner (selective). Needs item 1 first,
   and `ServerManager` becoming a registry of role → (port, process, model) with
   `can_load()` pricing the aggregate rather than one model.

## Housekeeping

- Tests write to the **real** index: `storage.library`'s path constants are
  monkeypatched but `brain.retriever.INDEX_FILE` / `META_FILE` never are
  (`tests/test_library_service.py:41-50`).
- Three independent definitions of the artifact paths — `brain/retriever.py:43`,
  `storage/library.py:49-51`, `storage/manifest.py:37` — all bypassing
  `Config.CACHE_DIR`, which is defined and never used.
- `gui/material_app.py:1016-1021` nulls the embedder's `_model`/`_model_name` but
  not `_device`, leaving a stale device against a `None` model.
- `main.py:181` still defaults `--host` to `0.0.0.0` despite `Config.SERVER_HOST`
  being fixed to `127.0.0.1`; the CLI never reads the config value. The
  destructive `clean`/`sync` routes are separately protected by a loopback client
  check (`chat/dashboard.py:28-34`), so this exposes `/chat` and `/search` only.
- `Manifest.is_compatible()` has no production caller.
- ONNX GPU OCR is advertised in `core/gpu_ocr.py` but never runs. Wire a real
  model or delete the branch.

## Corrected decisions — do not restore the old versions

- **The embedder is `BAAI/bge-base-en-v1.5`**, vendored in
  `data/models/embedder/`. The old NEXT.md called jina-v5 a locked decision; it
  was reverted by commit `e0c418b` and the weights were deleted. jina-v5 did
  measure better in isolation (relevance gap 0.50 vs 0.41, 8192-token context vs
  512) and remains a legitimate upgrade — but that measurement was never made
  against real queries, which is what item 5 is for.
- **The pagefile is NOT irrelevant.** The old text said so explicitly. It is the
  single binding constraint on running more than one model — see item 1. It is
  irrelevant to *VRAM*, which is where the original claim came from.
- **Free VRAM is ~4.1–4.5 GB, not 5,430 MiB**, and it moves during a session as
  the desktop grows. Budget it at load time; never read it from config.
- **Reranking and a score threshold** were declined when there was no evidence.
  There is now — see item 3.

## Still locked

- **Bad PDFs are quarantined, not repaired.** Quality over coverage.
- **Never insert spaces at `[a-z][A-Z]`** — corpus hits are identifiers
  (`MutableMapping`, `ValueError`). Looks like a defect, is not.
- **Never collapse leading whitespace** — Python listings are indent-significant.
- **`reindex` enumerates `data/input/`, never `data/txt/`.** Walking the text
  directory resurrects deleted books. (Verified still correct: `core/reindex.py:63`.)
- **Chunk offsets are recorded after stripping.** De-overlap is arithmetic;
  pre-strip offsets corrupt every merged passage silently.
- **Page markers stay in stored text** and are stripped only at the point of use,
  for the same reason.
- **Mark a book done only after its vectors land.** (Verified still correct:
  `core/async_pipeline.py:286-302`.)
- **Record the embedder that actually loaded**, not the configured one. Both
  candidates are 768-dim, so no dimension check would catch a mismatch.
- **Embedder device is explicit, not auto.** The LLM fills the GPU during chat.
- **No `trust_remote_code` for a model that has not been vendored.**
- **Never load models back-to-back to inspect them.** Windows keeps mmap'd pages
  cached after exit; four sequential loads left ~12 GB in the file cache. Read
  GGUF headers instead — milliseconds, and `vocab_only=True` is cheap.
