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
        self.llm = self._pick_backend(model_path, gpu_layers)
        self.manifest = Manifest()
        self.top_k = cfg.SEARCH_TOP_K
        self.char_budget = cfg.CONTEXT_CHAR_BUDGET

    def _pick_backend(self, model_path, gpu_layers):
        """
        Prefer a running llama-server, fall back to the in-process build.

        Chosen at construction rather than configured once, so the app still
        works when the server is not running instead of failing outright.
        """
        cfg = self.cfg
        if cfg.LLM_BACKEND in ("auto", "server"):
            try:
                from brain.llm_server import LlamaServerLLM
                srv = LlamaServerLLM(base_url=cfg.LLM_SERVER_URL,
                                     model_path=model_path, gpu_layers=gpu_layers)
                if srv.load():
                    return srv
                if cfg.LLM_BACKEND == "server":
                    log.error(f"LLM_BACKEND='server' but {cfg.LLM_SERVER_URL} is unreachable")
                    return srv
                log.info("No llama-server found — using the in-process model")
            except Exception as e:
                log.warning(f"Server backend unavailable: {str(e)[:160]}")
        return LocalLLM(model_path=model_path, gpu_layers=gpu_layers)

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

        ok = self.llm.load()

        # Fit the retrieval budget to the server's REAL context window.
        # llama-server is launched separately, so its -c may be smaller than
        # config assumes (e.g. -c 4096 against a 20,000-char budget). llama.cpp
        # would silently drop the front of the prompt — the passages — and the
        # model would answer confidently from a fragment with nothing logged.
        fit = getattr(self.llm, "usable_context_chars", None)
        if callable(fit):
            usable = fit(reserve_tokens=self.cfg.LLM_ANSWER_RESERVE)
            if usable and usable < self.char_budget:
                shrink = max(1, round(self.top_k * usable / self.char_budget))
                log.warning(
                    f"Server context fits ~{usable:,} chars, not "
                    f"{self.char_budget:,}. Reducing top_k {self.top_k}->{shrink}. "
                    f"Restart llama-server with -c 8192 for full retrieval."
                )
                self.char_budget = usable
                self.top_k = shrink
        return ok

    # ── answering ─────────────────────────────────────────────────────────────
    def answer(self, question: str, stream: bool = True):
        """Yield answer tokens, then a reference footer resolving each [N]."""
        passages = self.retriever.search_with_context(
            question, k=self.top_k, char_budget=self.char_budget)

        if not passages:
            yield ("I could not find anything relevant in your books. "
                   "If you have just added PDFs, run: python main.py ingest")
            return

        titles = self.manifest.titles() or sorted(
            {p["source"] for p in passages if p.get("source")}
        )
        log.info(f"{len(passages)} passages from {len({p['source'] for p in passages})} book(s)")

        prompt = self.llm.build_rag_prompt(question, passages, library_titles=titles)

        # Reasoning models emit <think>...</think> before answering. That is
        # working-out, not the answer: it must not be shown as prose, and it
        # must not be scanned for citations — a [3] mentioned while thinking is
        # not a claim. Suppress it live, keep it out of the validated text.
        collected: list[str] = []
        buf = ""
        thinking = False
        announced = False

        for token in self.llm.generate(prompt, stream=stream):
            collected.append(token)
            buf += token

            while buf:
                if not thinking:
                    i = buf.find("<think>")
                    if i == -1:
                        # Hold back a possible partial "<think>" split across tokens.
                        keep = 7
                        if len(buf) > keep:
                            yield buf[:-keep]
                            buf = buf[-keep:]
                        break
                    if i:
                        yield buf[:i]
                    buf = buf[i + 7:]
                    thinking = True
                    if not announced:
                        announced = True
                        yield "[thinking…]"
                else:
                    j = buf.find("</think>")
                    if j == -1:
                        buf = buf[-8:]          # keep a possible split close tag
                        break
                    buf = buf[j + 8:]
                    thinking = False
        if buf and not thinking:
            yield buf

        from brain.llm_server import strip_thinking
        yield from self._citation_footer(strip_thinking("".join(collected)), passages)

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
