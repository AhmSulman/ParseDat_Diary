"""
RAG Pipeline Wrapper — Ensures compatibility with GUI
This module provides the RAGPipeline class that the GUI expects.
"""

from brain.rag import RAGPipeline as _RAGPipeline
from logs.logger import log

class RAGPipeline(_RAGPipeline):
    """Extended RAGPipeline with GUI-friendly helpers."""

    def query(self, question: str) -> str:
        """Collect streamed tokens into one string."""
        try:
            return "".join(self.query_stream(question))
        except Exception as e:
            log.error(f"Query error: {e}")
            return f"Error: {str(e)[:200]}"

    def query_stream(self, question: str):
        """Yield answer tokens from RAG + local LLM (same as ``answer()``)."""
        yield from self.answer(question, stream=True)


__all__ = ["RAGPipeline"]

