"""
MAAN Web API Server
====================
Exposes MAAN as a local HTTP API using FastAPI.

Endpoints:
  GET  /              → Health check
  POST /chat          → Ask a question (streaming response)
  POST /search        → Semantic search only
  GET  /status        → Index stats

Run: python main.py server --host 0.0.0.0 --port 8000
"""

import json
from brain.rag import RAGPipeline
from logs.logger import log


def run_server(host: str = "0.0.0.0", port: int = 8000, model_path: str = None):
    try:
        from fastapi import FastAPI
        from fastapi.responses import StreamingResponse, JSONResponse
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
        import uvicorn
    except ImportError:
        log.error("❌ FastAPI not installed → pip install fastapi uvicorn")
        return

    app = FastAPI(title="MAAN - Chat with Books", version="3.2")

    # Allow browser access from any origin (for local UI)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    rag = RAGPipeline(model_path=model_path)
    rag.setup()

    class ChatRequest(BaseModel):
        question: str
        top_k: int = 5

    class SearchRequest(BaseModel):
        query: str
        top_k: int = 5

    @app.get("/")
    def root():
        return {"service": "MAAN - Chat with Books", "status": "running"}

    @app.get("/status")
    def status():
        return {
            "indexed_chunks": rag.retriever.doc_count,
            "llm_loaded": rag.llm.is_loaded(),
            "model": rag.llm.model_path,
        }

    @app.post("/chat")
    def chat(req: ChatRequest):
        """Stream LLM answer token-by-token (Server-Sent Events)."""
        def stream():
            for token in rag.answer(req.question, stream=True):
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.post("/search")
    def search(req: SearchRequest):
        """Return top-K relevant chunks without generating an answer."""
        chunks = rag.retriever.search(req.query, k=req.top_k)
        return {"query": req.query, "results": chunks}

    log.info(f"🌐 MAAN server starting at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")
