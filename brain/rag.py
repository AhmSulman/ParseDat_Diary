"""
RAG Pipeline — Retrieval-Augmented Generation
===============================================
The full AI answering system:

  Your Question
       ↓
  [ENCODER] sentence-transformers → 384-dim vector
       ↓
  [VECTOR DB] FAISS similarity search → top-K chunks
       ↓
  [PROMPT BUILDER] question + book excerpts → full prompt
       ↓
  [DECODER] local LLM (llama.cpp + RTX 4050) → answer
       ↓
  Your Answer (streamed live, token by token)

Think of it as: a researcher who finds the relevant book pages,
reads them, then answers your question in their own words.
"""

from brain.retriever import Retriever
from brain.llm import LocalLLM
from logs.logger import log
from config.config import Config


class RAGPipeline:
    def __init__(self, model_path: str = None, gpu_layers: int = None):
        cfg = Config()
        self.retriever = Retriever()
        self.llm = LocalLLM(model_path=model_path, gpu_layers=gpu_layers)
        self.top_k = cfg.SEARCH_TOP_K

    def setup(self) -> bool:
        """Load the vector index and LLM model."""
        log.info("🔄 Loading MAAN RAG pipeline...")
        self.retriever.load()

        if self.retriever.doc_count == 0:
            log.warning("⚠️  No documents indexed yet. Run: python main.py ingest")

        llm_ok = self.llm.load()
        return llm_ok

    def answer(self, question: str, stream: bool = True):
        """
        Answer a question using RAG.

        1. Retrieves relevant chunks from the vector DB
        2. Builds a prompt with those chunks as context
        3. Streams the LLM's answer

        Args:
            question: User's question string
            stream:   If True, yields tokens as generated

        Yields: str tokens (stream mode) or returns str (non-stream)
        """
        # Step 1: Retrieve relevant chunks
        chunks = self.retriever.search(question, k=self.top_k)

        if not chunks:
            yield "I couldn't find relevant information in your books. Try ingesting more PDFs first."
            return

        sources = list(set(c["source"] for c in chunks))
        log.info(f"📚 Found {len(chunks)} relevant chunks from: {sources}")

        # Step 2: Build RAG prompt
        prompt = self.llm.build_rag_prompt(question, chunks)

        # Step 3: Generate answer (streamed)
        yield from self.llm.generate(prompt, stream=stream)

        # Step 4: Show sources
        yield f"\n\n📚 Sources: {', '.join(sources)}"

    def get_sources(self, question: str) -> list[dict]:
        """Just retrieve relevant chunks without generating an answer."""
        return self.retriever.search(question, k=self.top_k)
