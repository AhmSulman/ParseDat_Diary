"""
Smart Text Chunker
==================
Splits long PDF text into smaller pieces (chunks) for better search.

WHY? Because:
  - Embedding a 200-page book as ONE vector loses detail
  - Embedding sentence by sentence = too granular, loses context
  - Chunking into ~500 word blocks = sweet spot for RAG

Think of it like cutting a book into chapters, then paragraphs.
Each chunk is independently searchable.

Uses sliding window overlap so context isn't lost at chunk boundaries.
"""

from config.config import Config


class Chunker:
    def __init__(self):
        cfg = Config()
        self.chunk_size = cfg.CHUNK_SIZE          # chars per chunk
        self.overlap = cfg.CHUNK_OVERLAP           # overlap between chunks

    def chunk(self, text: str) -> list[str]:
        """
        Split text into overlapping chunks.

        Args:
            text: Full document text

        Returns:
            List of chunk strings
        """
        if not text or not text.strip():
            return []

        text = text.strip()

        # If text is short enough, return as single chunk
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # Try to end at a sentence boundary (. or \n)
            if end < len(text):
                # Search backwards for a good break point
                for sep in ["\n\n", "\n", ". ", "! ", "? "]:
                    pos = text.rfind(sep, start + self.chunk_size // 2, end)
                    if pos != -1:
                        end = pos + len(sep)
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move forward with overlap (so we don't lose context at edges)
            start = end - self.overlap
            if start >= len(text):
                break

        return chunks

    def chunk_with_meta(self, text: str, source: str) -> list[dict]:
        """
        Chunk text and attach metadata to each chunk.

        Returns:
            List of dicts: [{"text": "...", "source": "file.pdf", "chunk_id": 0}, ...]
        """
        chunks = self.chunk(text)
        return [
            {"text": c, "source": source, "chunk_id": i}
            for i, c in enumerate(chunks)
        ]
