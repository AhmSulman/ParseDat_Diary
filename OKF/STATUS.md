# Status — what's actually built (2026-09-04)

ParseDat_Diary is a local, offline RAG system being rebuilt into a personal
learning system: documents in `data/input/` → FAISS index → a local LLM answers
from them with page-level citations. Windows, Python 3.12.0, venv is `_venv`.

Read `NEXT.md` for the punch list and the decisions not to re-argue.

## Start it

```
_venv\Scripts\python.exe main.py console     # terminal: ask, tune, manage
_venv\Scripts\python.exe main.py gui         # KivyMD desktop app
_venv\Scripts\python.exe main.py doctor      # health check — run this first
```

Answers need a model server running. The binaries now live inside the project
and are found automatically, so this is only needed to run one by hand:

```
llama-b10034-bin-win-cpu-x64\llama-server.exe ^
  -m data\models\DeepSeek-R1-Distill-Qwen-7B-Q4_K_M_2.gguf ^
  -ngl 0 -c 8192 -t 12 -ctk q8_0 -ctv q8_0 --port 8084
```

Killswitch: `taskkill /IM llama-server.exe /F`

## Library

**4 documents, 4,475 chunks, zero drift.** Reset from 17 on 2026-09-04 — the
other 13 are archived at `F:\Desktop\Library_Archive\`, not deleted.

| Book | Chunks | Why it is here |
|---|---|---|
| `mathematics-for-machine-learning.pdf` | 1,026 | #1 in the reading plan; a known retrieval **failure** case |
| `designing-data-intensive-applications.pdf` | 1,795 | #2 in the reading plan; the other known **failure** case |
| `fundamentals-of-data-engineering.pdf` | 1,141 | #3 in the reading plan |
| `deep-learning-with-python.pdf` | 513 | regression baseline |

`data/input/README.md` is the reading plan and is deliberately **not** ingested —
`list_sources` skips READMEs.

## Working now

**Ingestion** — PDFs (PyMuPDF + Tesseract OCR) and `.md`/`.txt`/`.log` (read
directly, no OCR). Text is normalised, quality-gated, chunked at 1200/200 with
exact offsets and page or heading locators.

**Retrieval** — bge-base-en-v1.5, 768-dim, vendored into `data/models/embedder/`.
Top-6 hits expand to their neighbours and merge into continuous passages,
de-overlapped arithmetically from stored offsets. Page furniture is stripped
before embedding and before prompting, never from stored text.

**Answers** — DeepSeek-R1 via `llama-server` over HTTP, rendered with the model's
**own** chat template read from the GGUF. Inline `[N]` citations resolve to book
+ page, and every marker is validated against the sources actually supplied.

**Settings** — every tunable is editable from the GUI, stored as an overlay in
`data/settings.json`. Server-tagged changes relaunch llama-server automatically.

**Memory guard** — loads are priced against available **commit**, not free
physical RAM, and refused rather than allowed to thrash. See `NEXT.md`.

**Tests** — 116, stdlib `unittest`:
`_venv\Scripts\python.exe -m unittest discover -s tests -q`

## Retrieval baseline (2026-09-04, measured)

The first honest measurement this project has had. Re-run these after any
retrieval change.

| Query | Result | Verdict |
|---|---|---|
| *"how does database replication handle failures"* | all 5 hits DDIA pp. 220–312 — the real replication and transaction chapters | **fixed** (was a WSBPEL bibliography entry) |
| *"maths behind machine learning"* | mml-book pp. 7, 17, 8, 7, 1–3 | **still failing** — the preface, never the maths chapters |
| *"what is a tensor in deep learning"* | mml-book p.164–165 first, deep-learning-with-python p.25 second | plausible but not clean |

The corpus reset fixed the **cross-book** failure and did nothing for the
**within-book** one. What remains is preface and boilerplate pollution inside a
single book — the case for boilerplate stripping or a reranker, both in `NEXT.md`.

## Known-not-done

- **The pagefile is the binding constraint.** Only ~6 GB of commit is available
  against a 27.8 GB limit, so the guard correctly refuses both Qwen3-4B and the
  7B. Fixing it is a Windows setting, not code — see `NEXT.md`.
- **Retrieval precision** — measured above, not fixed.
- **`CONTEXT_CHAR_BUDGET` is in characters, not tokens.** 20,000 chars is ~4,600
  tokens of prose but ~11,000 of code-dense text, against `-c 8192`. llama.cpp
  drops the front of the prompt silently when that overflows.
- **GPU generation.** `torch` CUDA works; the llama.cpp build on disk is CPU-only
  and the 7B does not fit in the ~4 GB of free VRAM anyway.
- **ONNX GPU OCR is fiction** — `core/gpu_ocr.py` advertises it, the session load
  is commented out, Tesseract CPU is the only path that runs.
- **Answer quality is barely measured.** The table above is retrieval only;
  nothing has graded an actual answer.
