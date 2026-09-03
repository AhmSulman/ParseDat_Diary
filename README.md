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

> **Status: working foundation, unfinished product.** The retrieval-and-answer
> loop runs end to end and is fast. The thing it is *becoming* — a tutor that
> questions you back and remembers what you understand — is not built yet. The
> rating below is honest about the gap.

---

## Quick start

Double-click one of these:

| | |
|---|---|
| **`Launch GUI (GPU).bat`** | KivyMD desktop app — library, reader, Ask AI, settings |
| **`Launch Console (GPU).bat`** | Terminal console — ask, tune, manage |

Both start `llama-server` with full GPU offload first. They go through
`main.py start` rather than calling `llama-server.exe` directly, so the memory
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

Prompt eval is the number that matters: reading the retrieved passages used to
be two and a half minutes of every single answer.

| | |
|---|---|
| Library | 4 books · 4,475 chunks · zero drift |
| Embeddings | BAAI/bge-base-en-v1.5, 768-dim, vendored offline |
| Index | FAISS HNSWFlat, neighbour expansion, arithmetic de-overlap |
| Generation | DeepSeek-R1-Distill-Qwen-7B Q4_K_M via llama-server, fully GPU-resident |
| Tests | 129, stdlib `unittest`, no model loaded |

---

## Honest rating

| Area | Score | Why |
|---|---|---|
| **Data integrity** | 9/10 | `data/input/` is the sole authority; six derived stores reconcile against it. `doctor` reports drift, `sync` repairs it. A book is marked done only after its vectors land, so a crash cannot create silent holes — a bug that once left 11 books invisible. |
| **Offline guarantee** | 9/10 | Embedder weights vendored inside the project and loaded by path, verified with the HuggingFace endpoint pointed at a dead host. No `trust_remote_code`, no silent fallback. |
| **Memory safety** | 8/10 | Loads are priced against available *commit* — the limit Windows actually enforces — not free RAM. Six logged crashes traced and prevented. Still conservative rather than precise. |
| **Speed** | 8/10 | ~9× faster end to end on GPU. The 7B fits in 6 GB of VRAM with an 8-bit KV cache. |
| **Engineering discipline** | 8/10 | 129 tests, every non-obvious decision documented with the measurement behind it. Tests avoid loading models, so the suite runs in 0.3 s. |
| **Citations** | 6/10 | Every `[N]` is validated against the supplied passages, so an out-of-range citation is a mechanically detectable hallucination. But the model frequently ignores the citation instruction entirely, and an uncited answer is only *flagged*, not fixed. |
| **Retrieval precision** | **4/10** | Measured and weak. *"How does database replication handle failures"* now returns the right chapters. *"Maths behind machine learning"* still returns the book's **preface**, never the maths. Boilerplate pollution inside a single book is unsolved. |
| **Answer quality** | **3/10** | Barely measured. There is no golden set, no grader, no score. One verified answer is one data point. |
| **The actual product** | **1/10** | The tutor loop — questions, evidence-grounded grading, a memory of what you understand — is designed and specced, and **none of it is built**. |

**Overall: 6/10 as a local RAG tool. 1/10 against what it is meant to become.**

The foundation is genuinely solid — offline, fast, self-checking, hard to
corrupt. What sits on top of it is still a search box.

---

## What is deliberately not done

- **Retrieval scoping.** Answers draw on the whole library; they should draw on
  1–3 books and a handful of sections. This is the structural fix for the
  preface problem and it is not built.
- **The learning loop.** Question generation, evidence-grounded grading, a
  per-concept model of what you understand. Specced, unbuilt.
- **Grader calibration.** Nothing may be built on LLM-generated grades until
  ~50 hand-graded answers agree with the model ≥80% of the time.
- **Token budgeting.** `CONTEXT_CHAR_BUDGET` counts characters. Code-dense text
  runs 1.82 chars/token against 4.32 for prose, so 20,000 characters can be
  ~11,000 tokens against an 8,192 context — and llama.cpp drops the front of
  the prompt silently.
- **ONNX GPU OCR.** Advertised in `core/gpu_ocr.py`, never executes. Tesseract
  CPU is the only path that runs. Either wire it or delete the branch.
- **Android client.** `android_main.py` exists and is untested against the
  current server.

Full punch list: [`OKF/NEXT.md`](OKF/NEXT.md). Current state:
[`OKF/STATUS.md`](OKF/STATUS.md).

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
    → llama-server   streamed answer, GPU
    → validator      every [N] checked against what was actually supplied
```

Three rules are load-bearing; breaking any of them loses data silently, and all
three are explained in [`CLAUDE.md`](CLAUDE.md):

1. **Mark a book done only after its vectors land.**
2. **`reindex` enumerates `data/input/`, never `data/txt/`** — walking the text
   directory resurrects deleted books.
3. **Chunk offsets are recorded after stripping** — de-overlap is arithmetic,
   and pre-strip offsets corrupt every merged passage with no error raised.

---

## Requirements

- Windows, Python 3.12, venv at `_venv` (**not** `.venv`)
- NVIDIA GPU for the fast path — a CUDA llama.cpp build in the project root
- Tesseract 5.5 for scanned PDFs
- A commit limit with real headroom. `main.py memory` will tell you.

---

## Licence

See [LICENSE](LICENSE).
