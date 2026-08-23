"""
RAG Pipeline — retrieval-augmented generation
==============================================

  question
     -> embedder            query -> 768-dim vector
     -> FAISS               top-k chunks
     -> neighbour expansion chunks +/-1, merged into continuous passages
     -> prompt builder      library header + numbered, page-located excerpts
     -> local LLM           streamed answer with inline [N] citations
     -> validator           every [N] checked against the sources supplied

The last step is the point. A citation outside the supplied range means the
model referenced something it was never given — a hallucination detectable
without a human opening the book.
"""

from brain.llm import LocalLLM
from brain.retriever import Retriever
from config.config import Config
from logs.logger import log
from storage.manifest import Manifest


class RAGPipeline:
    def __init__(self, model_path: str | None = None, gpu_layers: int | None = None):
        cfg = Config()
        self.cfg = cfg
        # CPU for the embedder here: chat loads the 7B model, and there is no
        # VRAM left for a second model. One short query costs ~15 ms on CPU.
        self.retriever = Retriever(device="cpu")
        self.llm = LocalLLM(model_path=model_path, gpu_layers=gpu_layers)
        self.manifest = Manifest()
        self.top_k = cfg.SEARCH_TOP_K

    def setup(self) -> bool:
        log.info("Loading MAAN RAG pipeline...")
        loaded = self.retriever.load()
        self.manifest.load()

        if not loaded or self.retriever.chunk_count == 0:
            log.warning("No documents indexed. Run: python main.py ingest")
        else:
            log.info(
                f"Library: {self.manifest.book_count or self.retriever.book_count} "
                f"books, {self.retriever.chunk_count} chunks"
            )
        return self.llm.load()

    # ── answering ─────────────────────────────────────────────────────────────
    def answer(self, question: str, stream: bool = True):
        """Yield answer tokens, then a reference footer resolving each [N]."""
        passages = self.retriever.search_with_context(question, k=self.top_k)

        if not passages:
            yield ("I could not find anything relevant in your books. "
                   "If you have just added PDFs, run: python main.py ingest")
            return

        titles = self.manifest.titles() or sorted(
            {p["source"] for p in passages if p.get("source")}
        )
        log.info(f"{len(passages)} passages from {len({p['source'] for p in passages})} book(s)")

        prompt = self.llm.build_rag_prompt(question, passages, library_titles=titles)

        collected = []
        for token in self.llm.generate(prompt, stream=stream):
            collected.append(token)
            yield token

        yield from self._citation_footer("".join(collected), passages)

    def _citation_footer(self, answer: str, passages: list[dict]):
        """
        Resolve the [N] markers the model emitted to book and page, and flag any
        that do not correspond to a supplied source.
        """
        check = self.llm.validate_citations(answer, len(passages))

        if check["cited"]:
            yield "\n\nReferences:"
            for n in check["cited"]:
                p = passages[n - 1]
                ps, pe = p.get("page_start"), p.get("page_end")
                if ps and pe and ps != pe:
                    loc = f", p.{ps}-{pe}"
                elif ps:
                    loc = f", p.{ps}"
                else:
                    loc = ""
                yield f"\n  [{n}] {p.get('source', 'unknown')}{loc}"

        if check["invalid"]:
            bad = ", ".join(f"[{n}]" for n in check["invalid"])
            log.warning(
                f"Hallucinated citation(s) {bad} — only {check['n_sources']} sources supplied"
            )
            yield (f"\n\n[!] {bad} do not match any excerpt provided "
                   f"(there were {check['n_sources']}). Treat those claims as unverified.")

        if check["uncited"]:
            log.warning("Answer made substantive claims with no citations")
            yield "\n\n[!] This answer cited no excerpt, so it could not be verified."

    def get_sources(self, question: str) -> list[dict]:
        return self.retriever.search_with_context(question, k=self.top_k)
