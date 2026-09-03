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


_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"

# Characters to wait before concluding a model is NOT showing its working.
# <think> arrives at the very start when it arrives at all, so this only has to
# outlast the first token or two. It was previously 6,000 — the whole answer had
# to exceed that before anything was released, so a model emitting no tags at
# all had every answer under 6,000 characters buffered forever and then reported
# as "cut off". Measured: a correct 1,940-character answer was discarded.
_DECIDE_AFTER = 200
_HEARTBEAT_EVERY = 350


def suppress_reasoning(tokens):
    """
    Yield display text from a token stream, dropping any reasoning block.

    Reasoning is working-out, not the answer: it must not be shown as prose and
    must not be scanned for citations, since a [3] mentioned while thinking is
    not a claim.

    Three shapes are handled, because which one arrives depends on how the
    prompt was sent:

      <think>...</think>answer   the chat template owns the tags (current path,
                                 /v1/chat/completions)
      ...</think>answer          a bare close, no opener — R1 through the older
                                 /completion path with a raw prompt
      answer                     no tags at all, which is also what R1 does for
                                 many questions. This is the common case and the
                                 one that used to be destroyed.
    """
    buf = ""
    state = "unknown"          # unknown -> thinking | plain
    announced = False
    ticks = 0

    for token in tokens:
        if state == "plain":
            yield token
            continue

        buf += token

        if state == "unknown":
            opened = buf.find(_THINK_OPEN)
            closed = buf.find(_THINK_CLOSE)
            if opened != -1 and (closed == -1 or opened < closed):
                state = "thinking"
                buf = buf[opened + len(_THINK_OPEN):]
            elif closed != -1:
                buf = buf[closed + len(_THINK_CLOSE):]
                state = "plain"
                if buf:
                    yield buf.lstrip()
                    buf = ""
                continue
            elif len(buf) >= _DECIDE_AFTER:
                # Far enough in with no opening tag: this model is answering
                # directly. Release what is held and stream from here, so a
                # plain answer is not withheld until the stream closes.
                state = "plain"
                yield buf
                buf = ""
                continue
            else:
                continue

        if state == "thinking":
            closed = buf.find(_THINK_CLOSE)
            if closed != -1:
                buf = buf[closed + len(_THINK_CLOSE):]
                state = "plain"
                if announced:
                    yield "]" + chr(10) + chr(10)
                if buf:
                    yield buf.lstrip()
                    buf = ""
                continue

            # Heartbeat while reasoning is held back. Without it the UI shows
            # nothing for the hundreds of tokens spent thinking, which reads as
            # a freeze — the GUI was reported as "not responding" while working
            # normally.
            if not announced and len(buf) > 40:
                announced = True
                yield "[reasoning"
            if announced:
                dots = len(buf) // _HEARTBEAT_EVERY
                if dots > ticks:
                    ticks = dots
                    yield "."

    if state == "thinking":
        if announced:
            yield "]"
        yield "\n[answer was cut off during reasoning — raise -c or n_predict]"
    elif buf:
        yield buf


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
        log.info("Loading ParseDat_Diary RAG pipeline...")
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

        messages = self.llm.build_rag_messages(question, passages, library_titles=titles)

        # Reasoning is dropped by suppress_reasoning(); `collected` keeps the
        # raw stream so citations are validated against what the model actually
        # produced, not against the display text.
        collected: list[str] = []

        def _tapped():
            for token in self.llm.generate(messages, stream=stream):
                collected.append(token)
                yield token

        yield from suppress_reasoning(_tapped())

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
                # locate() handles both: a page for PDFs, the nearest heading
                # for markdown, which has no pages to cite.
                from brain.llm import locate
                yield chr(10) + f"  [{n}] {locate(passages[n - 1])}"

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
