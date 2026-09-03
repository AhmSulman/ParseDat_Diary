# Plan — settings surface + markdown subsystem (2026-08-24)

Working plan for the current thread of work. Separate from `STATUS.md`/`NEXT.md`
(those are stale — still describe the reverted jina-v5 embedder — and are a
later, deliberately-deferred cleanup; see Phase 3).

## Done

**Markdown ingestion, verified against real content.** `data/input/` now
has 3 real `.md` files (Claude Code's changelog, 5,763 lines, plus two of
MAAN's own planning docs) alongside the 13 PDFs — 16 books, 9,846 chunks,
zero drift. Confirmed: quality gate's `authored=True` bypass passes them,
`Chunker(markdown=True)` attaches the correct nearest-heading `section` to
every chunk (hierarchy respected — `#` then `##`), retrieval returns
markdown hits mixed with PDF hits in one result set, and citations resolve
to `file.md - ## Section Name` instead of a page. No bugs found — this path
existed in code but had never run against real content until now.

## Phase 1 — GUI-configurable settings (done)

Implemented and verified: `config/settings_store.py` (overlay + 22-field
schema + validation), `Config.__init__` layering, `gui/material_app.py`
schema-driven form + save/hot-apply, `gui/material_app.kv` scrollable
"All Settings" card, `tests/test_settings_store.py` (29 tests), plus a live
GUI construction smoke test: built the real form (29 widgets, 22 inputs),
saved a valid change (persisted, confirmed via a fresh `Config()`), saved
an out-of-range value (rejected, prior value intact), reset to default
(overlay key removed, `data/settings.json` back to `{}`). Also fixed
`ServerManager.start()`, which built its launch command without `-b`
despite `LLM_N_BATCH` being in the schema as a server-restart field — the
setting would have saved and silently done nothing. Full suite: 98/98.

**Follow-up fixes, same session:**
- The form was scrollable internally but the *screen* wasn't — three fixed
  cards fighting for space inside a non-scrolling `Screen`, so the page cut
  off. Restructured so the whole Settings screen is one `MDScrollView`;
  removed the redundant nested scrollview (a scrollview inside a
  fixed-height card, nested inside nothing that itself scrolled — the
  actual bug). Verified: content height 3480 vs viewport, confirmed
  scrollable via a live layout pass, not just visual inspection.
- Auto-restart llama-server on save when a `"server"`-tagged field changes
  (GPU layers, context, threads, batch, binary path) — reuses
  `ServerManager`, runs in a background thread, reconnects `rag.llm` after.
  No more taskkill + retype the `CLAUDE.md` launch command by hand.
  Caught and fixed a real staleness bug on the way: `ServerManager` is a
  process-wide singleton (`get_manager()`) that can live the whole GUI
  session, but `start()` was reading `self.cfg` captured at `__init__` —
  a Settings change would save correctly but a restart would silently
  relaunch with the *old* values. `start()` now reads a fresh `Config()`
  for ctx/GPU layers/threads/batch. `"app"`-tagged fields (backend/URL,
  OCR, ingest workers, web server host/port, CPU workers) still need the
  GUI/console process relaunched by hand — a full self-restart is a
  separate, riskier piece of work not done here.

Decisions recap below, for reference.

**Decisions already made:**
- Storage: `data/settings.json`, a standalone overlay file — deleting it
  (or the whole `config/settings_store.py` module) drops every override
  and the app falls straight back to `config.py` defaults. Nothing else
  depends on the file existing.
- `Config.__init__` layers the overlay onto the class defaults as instance
  attributes, only for keys the overlay actually contains — existing
  direct class-attribute mutations (`Config.EMBED_MODEL = x`, used by
  tests and the embedder switcher) keep working untouched.
- Settings screen renders itself from `SETTINGS_SCHEMA` (one list of field
  specs in `settings_store.py`) rather than one hand-built KV widget per
  field — adding a future tunable is one schema entry.
- Surface: every `Config` field *except* `EMBED_TRUST_REMOTE_CODE` and
  `EMBED_FALLBACK` — both are deliberate lockdown decisions recorded in
  `CLAUDE.md` ("No trust_remote_code", "NO FALLBACK, deliberately"), not
  something a settings screen should let slip by accident.
- `LLM_MODEL_PATH` and `EMBED_MODEL`/`EMBED_DIM` keep their existing
  dedicated pickers rather than duplicating a text field — those pickers
  get fixed to persist through the overlay (currently
  `cfg.__class__.LLM_MODEL_PATH = ...` and `Config.EMBED_MODEL = ...` are
  in-memory only and lost on restart).
- New field: `LLAMA_SERVER_BIN` (currently a hardcoded candidate-dir list
  in `brain/server_manager.py`, and not even in `Config`).
- Bundled fix: `SERVER_HOST` default `0.0.0.0` → `127.0.0.1`.
  `NEXT.md` already flagged this as exposing the admin `clean`/`sync`
  routes (file deletion) to the whole network; the Settings screen becomes
  the explicit, visible way to open it up instead of that being a silent
  default.
- Restart semantics are labeled honestly per field, not glossed over:
  `None` (hot-applied immediately on Save — top_k, char budget,
  temperature, max_tokens, neighbor radius), `"server"` (needs
  `ServerManager` to relaunch llama-server — GPU layers, context, threads,
  binary path), `"reindex"` (also needs `main.py reindex` — chunk
  size/overlap), `"app"` (needs the running `gui`/`console`/`server`
  process relaunched — backend/URL, OCR, async workers, web server
  host/port, CPU workers).

**Files:**
- NEW `config/settings_store.py` — overlay load/save (atomic, mtime-cached
  so `Config()` stays cheap), `SETTINGS_SCHEMA`, `validate()`.
- EDIT `config/config.py` — overlay-layering `__init__`, `LLAMA_SERVER_BIN`
  field, `SERVER_HOST` default fix.
- EDIT `gui/material_app.py` — schema-driven settings form builder, Save
  handler (validate → `update_overlay()` → hot-apply the safe subset on
  `app.rag` → toast what still needs restart/reindex), fix the two
  in-memory-only pickers to persist.
- EDIT `gui/material_app.kv` — scrollable container for the generated form.
- NEW `tests/test_settings_store.py` — round-trip, corrupt-file safety,
  atomic write, schema validation bounds.

## Log-file ingestion (done, ad hoc — not originally in this plan)

`.log` added to `core/sources.py`'s `TEXT_EXTS`, alongside `.md`/`.markdown`/
`.txt` — same no-OCR, `authored=True` quality-gate-bypass path, default
(non-markdown) chunker. Verified against the real `logs/app.log` (697 lines,
~60 KB, copied into `data/input/` as `maan-app.log`): 63 chunks, no
crashes, retrieval surfaces genuinely relevant lines, citations degrade to
the bare filename (`locate()` already handled the no-page/no-section case
from the markdown work). 17 books, zero drift.

**Known limitation, not fixed:** a log file changes on every app run; a
"book" in this system is indexed once and the checkpoint marks it done
forever. Re-ingesting an updated `app.log` later will just say
"already indexed" and skip it — nothing here makes it periodically
refresh a growing log. If ongoing log search actually matters, that's a
`forget` + re-`ingest` cycle for now, or a future dedicated feature — not
scoped or built.

**Also fixed this session:** `config/settings_store.py`'s `load_overlay()`
did a per-call `os.path.getmtime()` disk stat to safely detect an
externally-changed file — reasonable in isolation, expensive in practice,
since `Config()` is constructed constantly (once per RAG query in
`brain/llm.py`'s `build_rag_prompt`, once per `Chunker`/`Retriever`/
`Embedder`, every GUI callback). Switched to a plain in-memory cache: load
once per process, `save_overlay()`/`update_overlay()` update the cache
directly since those are the only way *this* process changes the file, and
an explicit `invalidate_cache()` exists for the rare cross-process case.
Measured: ~1 microsecond per `Config()` now, matching the old free
class-attribute-lookup cost. 100/100 tests passing (2 new, replacing the
now-inapplicable mtime-staleness test).

## Answer quality investigation (done — one fix shipped, one deferred by choice)

User-reported: answers "too dumb," not citing, not using book content. Root
caused with real evidence (actual session history + logs), not guessed:

1. **Wrong chat template — fixed.** `build_rag_prompt()` hardcoded a
   Mistral `[INST]...[/INST]` wrapper regardless of the loaded model. The
   configured model (DeepSeek-R1-Distill-Qwen-7B, architecture `qwen2`)
   has its own embedded template (`<|User|>`/`<|Assistant|>`, read directly
   from the GGUF's `tokenizer.chat_template` metadata) with nothing to do
   with `[INST]`. Verified via controlled side-by-side on identical
   content: the `[INST]` version produced a malformed unpaired `</think>`
   and a shallow answer; the model's real template produced clean, paired
   reasoning specifically grounded in the excerpt. Fixed in `brain/llm.py`
   (`build_rag_prompt` → `build_rag_messages`, `LocalLLM.generate` now uses
   `create_chat_completion`) and `brain/llm_server.py` (switched
   `/completion` → `/v1/chat/completions`), plus the one call site in
   `brain/rag.py`. Fixes this for every model this app lists as supported,
   not just DeepSeek — Mistral, Phi-2, Llama-2, and OpenChat each use a
   different native format, and only a genuine Mistral Instruct build was
   ever actually correct before. 100/100 tests still passing.

2. **Retrieval precision — real, pre-existing, NOT fixed.** Re-ran the
   user's actual real question ("maths behind machine learning") through
   the FIXED pipeline end-to-end: still uncited. Raw top-15 hits are all
   from mml-book.pdf pages 1–22 (the preface), never reaching the actual
   math chapters. Found a publisher disclaimer repeated 18× throughout the
   book polluting the index with near-duplicate low-content chunks. Matches
   `OKF/STATUS.md`'s already-documented, already-deferred finding ("a query
   about database replication returned a bibliography entry"). Two options
   presented, user chose to defer the decision:
   - **Boilerplate stripping** (cheap, fast, no new deps): detect text
     repeated near-verbatim across many pages during `core/normalize.py`'s
     pass, drop it before chunking. Needs a reindex. Narrower — won't
     touch legitimately-relevant-but-low-content pages like a genuine TOC
     or a short definitional preface passage.
   - **Reranker** (the "real" fix, bigger lift): cross-encoder
     (`bge-reranker-base` per `OKF/NEXT.md`'s own deferred-ideas list)
     re-scores top ~20 raw hits before they reach the LLM. New model
     download, ~0.3s/query added latency on GPU, more surface area to
     build and test. Handles the general case, not just repeated
     boilerplate.
   - Neither implemented this session — explicitly deferred, tradeoffs
     recorded for whenever it's picked back up.

3. **Also surfaced, not fixed (separate, already-known issue):** GPU
   offload for the in-process `LocalLLM` fallback is silently not working —
   a verification query took 6 minutes on CPU despite `GPU layers: 35`
   being logged. Matches the already-documented
   `llama_supports_gpu_offload() == False` limitation; llama-server (not
   running during this investigation) is the existing documented fix,
   unrelated to tonight's two bugs above.

## Phase 2 — markdown subsystem (scoping now)

Four gaps named, all in scope:

1. **Real structural understanding.** `core/normalize.py:163` is a bare
   heading-only regex. Extend it (still no new dependency — stays
   consistent with the project's offline/vendoring stance) to recognize
   fenced code blocks, tables, and lists as structural units, so the
   chunker treats them as break points the way it already does headings —
   a table or code fence currently has no protection from being split
   mid-row.
2. **Read-tab rendering.** The GUI preview shows raw markdown source
   (`#`, `**`, etc. visible as text) for `.md` library files. Add a
   markdown → Kivy-markup converter (bold/italic/headers/code spans) so
   `.md` files render, while plain `.txt`/PDF text keeps passing through
   unchanged.
3. **Export a single answer as markdown**, not just a whole session —
   factor `storage/history.py`'s markdown formatting into a shared
   function, add an "Export answer" action in the Ask AI tab.
4. **Pattern search across the library** — exact/regex search over
   `data/txt/*.txt`, complementing FAISS semantic search for cases where
   the user wants a literal term, not a meaning-match. New
   `storage/textsearch.py`, exposed as `main.py search <pattern>` and a
   GUI search box.

Not yet broken into a file-level design — doing that after Phase 1 lands.

## Phase 3 — deferred

Storage/retrieval modularization (documenting the storage schema in one
place, resolving the `brain/rag_pipeline.py` wrapper duplication,
refreshing stale `OKF/STATUS.md`/`NEXT.md`). Explicitly deferred per
"first make it work then we'll get on with this" — revisit once Phases 1–2
are working.
