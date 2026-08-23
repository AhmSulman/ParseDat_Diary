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


def run_server(host: str = "0.0.0.0", port: int = 8000, model_path: str = None,
               allow_admin: bool = False):
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
        # Reports BOTH counts. It previously exposed only doc_count, which is a
        # chunk total despite the name, so clients had no way to show how many
        # books existed.
        return {
            "books": rag.manifest.book_count or rag.retriever.book_count,
            "indexed_chunks": rag.retriever.chunk_count,
            "llm_loaded": rag.llm.is_loaded(),
            "model": rag.llm.model_path,
            "embed_model": rag.manifest.settings.get("embed_model"),
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

    # Library dashboard. Read-only routes are always on; clean/sync exist only
    # with --allow-admin and are refused from anything but loopback, because
    # host defaults to 0.0.0.0 and those routes delete files.
    try:
        from chat.dashboard import register
        register(app, allow_admin=allow_admin)
    except Exception as e:
        log.warning(f"Dashboard not registered: {e}")

    log.info(f"MAAN server at http://{host}:{port}")
    log.info(f"   library dashboard: http://127.0.0.1:{port}/ui")
    if allow_admin:
        log.warning("   admin routes ENABLED (loopback only)")
    uvicorn.run(app, host=host, port=port, log_level="warning")
