# MAAN — Chat with Books

```
███╗   ███╗ █████╗  █████╗ ███╗   ██╗
████╗ ████║██╔══██╗██╔══██╗████╗  ██║
██╔████╔██║███████║███████║██╔██╗ ██║
██║╚██╔╝██║██╔══██║██╔══██║██║╚██╗██║
██║ ╚═╝ ██║██║  ██║██║  ██║██║ ╚████║
╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝
```

**Local AI. Your GPU. Your Data. No Cloud. No API Key.**

RTX 4050 · ONNX CUDA · FAISS · Local LLM (llama.cpp) · RAG · Windows Service

---

## 🚀 Quick Start

### Step 1 — Install Python 3.10+
https://www.python.org/downloads/

### Step 2 — Install dependencies
```
pip install -r requirements.txt
```

For GPU LLM support (RTX 4050):
```
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

### Step 3 — Download a local LLM model
Go to: https://huggingface.co/TheBloke

Recommended for RTX 4050 (6GB VRAM):
- `Mistral-7B-Instruct-v0.2.Q4_K_M.gguf` (~4.4GB) ← Best
- `phi-2.Q4_K_M.gguf` (~1.7GB) ← Fastest

Put the `.gguf` file in: `data/models/`

Then set `LLM_MODEL_PATH` in `config/config.py`.

### Step 4 — Drop your PDFs
Put all PDF files into: `data/input/`

### Step 5 — Ingest (extract + index)
```
python main.py ingest
```

### Step 6 — Chat!
```
python main.py chat
```

---

## 📋 All Commands

| Command | What it does |
|---------|-------------|
| `python main.py ingest` | Extract PDFs + build search index |
| `python main.py ingest --reset` | Re-process everything from scratch |
| `python main.py chat` | Chat with your books (terminal) |
| `python main.py chat --model path/to/model.gguf` | Use a specific model |
| `python main.py chat --gpu-layers 35` | Control GPU VRAM usage |
| `python main.py server` | Launch REST API on port 8000 |
| `python main.py search "your query"` | Search without chatting |
| `python main.py service install` | Install as Windows service |
| `python main.py service start/stop/remove` | Manage service |

---

## 🏗️ Architecture

```
 PDFs
  │
  ▼
[INGEST] PyMuPDF reads pages
  │
  ├─ Digital text? → [TEXT EXTRACTOR] → text
  │
  └─ Scanned page? → [GPU OCR] → ONNX CUDA / Tesseract → text
                          ↑
                     RTX 4050 VRAM
  │
  ▼
[CHUNKER] Split text into ~1000 char overlapping chunks
  │
  ▼
[EMBEDDER] sentence-transformers → 384-dim vectors
  │
  ▼
[FAISS INDEX] Vector database saved to data/cache/
  │
  ▼
  ┌─────────── At chat time ───────────┐
  │                                    │
  │  Your Question                     │
  │       ↓                            │
  │  [ENCODE] question → vector        │
  │       ↓                            │
  │  [RETRIEVE] top-5 similar chunks   │
  │       ↓                            │
  │  [PROMPT] question + book excerpts │
  │       ↓                            │
  │  [LOCAL LLM] llama.cpp + RTX 4050  │
  │       ↓                            │
  │  Your Answer (streamed live)       │
  └────────────────────────────────────┘
```

---

## ⚙️ Configuration (`config/config.py`)

| Setting | Default | Description |
|---------|---------|-------------|
| `LLM_MODEL_PATH` | `data/models/model.gguf` | Your downloaded model |
| `LLM_GPU_LAYERS` | `35` | GPU layers (RTX 4050 = 35 for 7B Q4) |
| `LLM_CONTEXT_SIZE` | `4096` | Context window (tokens) |
| `LLM_TEMPERATURE` | `0.7` | Creativity (0=precise, 1=creative) |
| `ASYNC_WORKERS` | `4` | Parallel OCR workers |
| `CHUNK_SIZE` | `1000` | Chars per text chunk |
| `SEARCH_TOP_K` | `5` | Results retrieved per query |
| `SERVER_PORT` | `8000` | Web API port |

---

## 🖥️ Windows Service (Auto-start)

Run as Administrator:
```
service\install_service.bat
```

Or manually:
```
python main.py service install
python main.py service start
```

The service:
- Starts automatically when Windows boots
- Restarts itself after crashes (3 retries: 5s, 10s, 30s)
- Runs the web API on port 8000

---

## 🌐 Web API

Start server:
```
python main.py server
```

Endpoints:
```
GET  http://localhost:8000/         → Health check
GET  http://localhost:8000/status   → Index stats + model info
POST http://localhost:8000/chat     → Ask a question (streaming)
POST http://localhost:8000/search   → Search without answering
```

Chat example (curl):
```bash
curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the main thesis of this book?"}'
```

---

## 🔧 GPU OCR — Plug In Your Model

To use a real ONNX GPU OCR model (instead of Tesseract):

1. Export your model to ONNX format (PaddleOCR, TrOCR, EasyOCR all support this)
2. Put the `.onnx` file in `data/models/`
3. Set `ONNX_MODEL_PATH` in `config/config.py`
4. Uncomment the session loading in `core/gpu_ocr.py → _init_gpu()`
5. Fill in the pre/post-processing in `_infer_onnx()`

---

## 📁 Folder Structure

```
MAAN/
├── main.py              ← Entry point
├── core/
│   ├── async_pipeline.py ← 10x speed async engine
│   ├── gpu_ocr.py        ← RTX 4050 batch OCR
│   ├── extract_text.py   ← Digital text extraction
│   └── ingest.py         ← PDF loader
├── brain/
│   ├── chunker.py        ← Smart text splitter
│   ├── embedder.py       ← Text → vector
│   ├── retriever.py      ← FAISS vector DB
│   ├── llm.py            ← Local LLM (llama.cpp)
│   └── rag.py            ← Full RAG pipeline
├── chat/
│   ├── cli.py            ← Terminal chat interface
│   └── server.py         ← FastAPI web API
├── storage/
│   ├── exporter.py       ← Save .txt + .json
│   ├── checkpoint.py     ← Crash-safe progress
│   └── cache.py          ← Deduplication
├── service/
│   ├── windows_service.py ← Windows service daemon
│   └── install_service.bat ← One-click installer
├── config/config.py      ← All settings
├── data/
│   ├── input/            ← PUT PDFs HERE
│   ├── models/           ← PUT .gguf MODEL HERE
│   ├── txt/              ← Extracted text output
│   ├── json/             ← JSON output
│   └── cache/            ← FAISS index
└── logs/app.log          ← Full activity log
```
