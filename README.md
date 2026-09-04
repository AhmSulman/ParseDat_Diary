```
██████╗ ██████╗
██╔══██╗██╔══██╗
██████╔╝██║  ██║
██╔═══╝ ██║  ██║
██║     ██████╔╝
╚═╝     ╚═════╝
```

# ParseDat_Diary

**Learn from your books — not just search them.**

A local, offline RAG system that answers from your own library with page-level
citations, running entirely on one laptop GPU. No cloud, no API key, no
telemetry. Every answer names the book and page it came from, and every citation
is checked against the passages the model was actually given.

> **Status: strong foundation, unfinished product.** The retrieval-and-answer
> loop runs end to end and is fast. The thing it is *becoming* — a tutor that
> questions you back and remembers what you understand — is designed and not yet
> built. The rating below is honest about that gap.

---

## Quick start

Double-click one of these:

| | |
|---|---|
| **`Launch GUI (GPU).bat`** | KivyMD desktop app — Library, Read, Ask AI, Settings |
| **`Launch Console (GPU).bat`** | Terminal console — ask, tune, manage |

Both start `llama-server` with full GPU offload first. Neither calls
`llama-server.exe` directly — they go through `main.py start`, so the memory
guard runs and a load that would thrash the machine is refused with a reason
instead of freezing the desktop.

```bash
_venv\Scripts\python.exe main.py doctor     # health check - run this first
_venv\Scripts\python.exe main.py memory     # what can load right now
_venv\Scripts\python.exe main.py ingest     # index anything new in data/input
```

---

## Measured, on an RTX 4050 Laptop (6 GB)

Numbers from this machine, not from a spec sheet.

| | CPU build | CUDA build |
|---|---|---|
| Prompt eval (4,057-token retrieval prompt) | 32 tok/s — **127 s** | **1,810 tok/s — 2.2 s** |
| Generation | 7.66 tok/s | **33.5 tok/s** |
| Full cited answer | ~4 minutes | **~41 s** |

Prompt eval is the number that matters: reading the retrieved passages used to be
two and a half minutes of *every single answer*.

| | |
|---|---|
| Library | 4 books · 4,475 chunks · zero drift |
| Embeddings | BAAI/bge-base-en-v1.5, 768-dim, vendored offline |
| Index | FAISS HNSWFlat, neighbour expansion, arithmetic de-overlap |
| Generation | DeepSeek-R1-Distill-Qwen-7B Q4_K_M, fully GPU-resident (~4.3 GB) |
| Tests | 129, stdlib `unittest`, no model loaded, suite runs in 0.3 s |

---

## The architecture, and why it earns its keep

This is the part of the project that deserves credit. Most of these decisions
exist because something failed first, and each one is recorded with the
measurement that justified it rather than as an assertion.

### One source of truth, and six things that answer to it

`data/input/` is the **sole authority** for what is in the library. Six derived
stores hang off it — index, metadata, manifest, checkpoint, extracted text,
categories — and `doctor` reconciles all six on demand, reporting *drift* rather
than a file count. A library where "how many books do you have" has more than one
answer is a library that will lie to you, and this design makes that state
detectable instead of invisible.

### A book is done only when its vectors land

The old pipeline marked a book done as soon as its text hit disk, and saved the
index once at the very end. A crash mid-run threw away every vector from that
session while the checkpoint already said "done" — so the next run skipped those
books **permanently**. That is how the library once reached 18 marked done, 7
actually indexed, and 11 unreachable holes.

Now `mark_done()` runs only after the vectors are in the index, `save()` runs
after every book, and a book that fails to embed is marked **failed** so it is
retried. Ordering as a correctness property, not a detail.

### The merge is arithmetic, not fuzzy

Retrieved chunks are expanded to their neighbours and stitched back into
continuous passages. The overlap is removed by subtraction —
`drop = prev.char_end - cur.char_start` — not by string matching. That is exact
and cheap, and it is correct *only* because chunk offsets are recorded against
the stored text after stripping. Get that wrong and every merged passage is
silently corrupted at the seam with no error raised. The invariant is written
down where it can be found, and tested directly.

### Hallucinations you can catch without reading the book

Every `[N]` the model emits is validated against the passages actually supplied.
A citation outside that range means the model referenced something it was never
given — one of the very few hallucination classes that is **mechanically
detectable**, no human and no second model required. Cheap, and genuinely rare
in RAG systems.

### Offline because of *how* it loads, not because of a flag

The embedder weights are vendored inside the project and loaded **by filesystem
path**. A path is not a repo id, so sentence-transformers treats it as a plain
directory and never consults the hub — there is no lookup to fail and no cache to
lose. This matters because the hub is contacted even for fully cached models: a
HEAD request checks for updates, and with no network it retries and fails, so a
421 MB model sitting on disk still would not load. Verified by pointing
`HF_ENDPOINT` at a dead host.

There is deliberately **no fallback model**: a fallback would swap the embedder
underneath an index built with a different one. Failing loudly with "restore
`data/models/embedder/`" is more useful than degrading quietly.

### Guards that check the thing that actually fails

- **Model match, not dimension.** Both embedder candidates are 768-dim, so a
  dimension check would pass while returning confident nonsense. The retriever
  compares the model *name* recorded in the manifest and refuses to search on a
  mismatch.
- **Commit, not free RAM.** Windows refuses an allocation when the commit charge
  reaches the commit limit; free physical RAM is not that limit. Guarding on RAM
  believed in ~4.8 GB of headroom that did not exist — the size of a 7B model,
  and six logged crashes.
- **Explicit device, not auto-detect.** `ingest` puts the embedder on CUDA,
  `chat` puts it on CPU. Auto-detecting globally puts both the embedder and the
  LLM on a 6 GB card and OOMs mid-answer.
- **Allowlist, not "kill the biggest".** The memory reclaimer stops model hosts
  only. Of six logged exhaustion events the top consumers were VirtualDJ at
  6.02 GB and Edge at 9.75 GB — real work. And `nvidia-smi` cannot support a
  smarter version: on this GPU it reports per-process VRAM as `[N/A]` and lists
  `explorer.exe` among GPU users.

### Tests that run in 0.3 seconds

129 tests, stdlib `unittest`, and **not one of them loads a model**. Where
constructing the real object is expensive, the logic is mirrored standalone and
tested directly — the de-overlap arithmetic, the memory guard against an injected
memory reading, the reasoning suppressor as a plain generator over tokens. The
KivyMD layer, which cannot be constructed headlessly on Windows at all, is tested
by parsing the `.py` with `ast` and the `.kv` with a regex and asserting every
bound name resolves.

### Documentation that records the measurement, not the opinion

`CLAUDE.md` carries a **"never fix these"** list — never insert spaces at
`[a-z][A-Z]` (6,119 corpus hits are identifiers like `MutableMapping`), never
collapse leading whitespace (Python listings are indent-significant). Both look
like defects and are not. `OKF/NEXT.md` records *corrected* decisions explicitly,
so a stale conclusion cannot be quietly restored.

---

## Honest rating

| Area | Score | Why |
|---|---|---|
| **Data integrity** | 9/10 | Sole-authority membership, ordered checkpoints, drift reporting, arithmetic de-overlap. Hard to corrupt, and corruption is visible when it happens. |
| **Offline guarantee** | 9/10 | Vendored-by-path loading verified against a dead host. No `trust_remote_code`, no silent fallback. |
| **Engineering discipline** | 9/10 | 129 fast tests, decisions recorded with their measurements, invariants written where they will be found. |
| **Memory safety** | 8/10 | Prices loads against the limit the OS actually enforces. Six logged crashes traced and prevented. Conservative rather than precise. |
| **Speed** | 8/10 | ~9× faster end to end on GPU; the 7B fits in 6 GB with an 8-bit KV cache. |
| **Citations** | 6/10 | Out-of-range citations are mechanically detectable — genuinely good. But the model frequently ignores the citation instruction, and an uncited answer is only *flagged*, not fixed. |
| **Retrieval precision** | **4/10** | Measured and weak. *"How does database replication handle failures"* now returns the right chapters. *"Maths behind machine learning"* still returns the book's **preface**. Within-book boilerplate pollution is unsolved. |
| **Answer quality** | **3/10** | Barely measured. No golden set, no grader, no score. One verified answer is one data point. |
| **The actual product** | **1/10** | The tutor loop — questions, evidence-grounded grading, a memory of what you understand — is specced and **entirely unbuilt**. |

### **Overall: 6.5/10 as a local RAG tool. 1/10 against what it is meant to become.**

The engineering is better than the product. The foundation is offline, fast,
self-checking and genuinely hard to corrupt — the kind of base most projects
never bother to build. What sits on top of it is still a search box, and the two
things standing between it and being useful are both *measured*: retrieval that
returns prefaces, and a model that will not cite.

Neither is a mystery. Both are next.

---

## What is deliberately not done

- **Retrieval scoping.** Answers draw on the whole library; they should draw on
  1–3 books and a handful of sections. The structural fix for the preface
  problem, and unbuilt.
- **The learning loop.** Question generation, evidence-grounded grading, a
  per-concept model of understanding.
- **Grader calibration.** Nothing may be built on LLM-generated grades until ~50
  hand-graded answers agree with the model ≥80% of the time. If they don't, the
  design changes rather than ships.
- **Token budgeting.** `CONTEXT_CHAR_BUDGET` counts characters. Code-dense text
  runs 1.82 chars/token against 4.32 for prose, so 20,000 characters can be
  ~11,000 tokens against an 8,192 context — and llama.cpp drops the front of the
  prompt silently.
- **Dual index.** bge vs jina-v5, A/B'd on real queries instead of argued about.
- **ONNX GPU OCR.** Advertised in `core/gpu_ocr.py`, never executes. Tesseract
  CPU is the only path that runs. Wire it or delete the branch.
- **Android client.** `android_main.py` exists, untested against the current server.

Full punch list: [`OKF/NEXT.md`](OKF/NEXT.md) · Current state:
[`OKF/STATUS.md`](OKF/STATUS.md)

---

## How it works

```
data/input/*.pdf|md|txt|log
    → core/async_pipeline.py    4 async workers
        → core/gpu_ocr.py       Tesseract (CPU)
        → core/normalize.py     ligatures, de-hyphenation; idempotent
        → core/quality.py       PASS/FAIL → data/quarantine/
        → brain/chunker.py      1200/200 + page and heading locators
        → brain/embedder.py     bge-base, vendored, 768-dim
        → brain/retriever.py    FAISS HNSWFlat
    → data/cache/parsedat.index + parsedat_meta.json + parsedat_manifest.json

question
    → retriever      top-6, expanded ±1, merged into continuous passages
    → llm            numbered excerpts with page spans
    → llama-server   streamed answer on GPU, model's own chat template
    → validator      every [N] checked against what was actually supplied
```

---

## Requirements

- Windows, Python 3.12, venv at `_venv` (**not** `.venv`)
- NVIDIA GPU for the fast path — a **CUDA** llama.cpp build in the project root.
  A CPU build will run, ~9× slower, and `-ngl` will do nothing.
- Tesseract 5.5 for scanned PDFs
- Commit headroom. `main.py memory` will tell you, and `--free` will reclaim some.

---

## Licence

See [LICENSE](LICENSE).
