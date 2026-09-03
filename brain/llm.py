"""
Local LLM Engine — Your Private AI Brain
==========================================
Runs a local Large Language Model entirely on YOUR machine.
No internet. No API key. No data sent anywhere.

Uses llama-cpp-python which runs .gguf quantized models.
Supports GPU acceleration via CUDA (RTX 4050).

HOW TO GET A MODEL:
  1. Go to: https://huggingface.co/TheBloke
  2. Pick any model (e.g. Mistral-7B-Instruct-v0.2-GGUF)
  3. Download a .gguf file (Q4_K_M = good balance of quality/speed)
  4. Put it in: data/models/
  5. Set MODEL_PATH in config.py  OR  pass --model path/to/model.gguf

RECOMMENDED MODELS for RTX 4050 (6GB VRAM):
  - Mistral-7B-Instruct-v0.2.Q4_K_M.gguf     (~4.4GB) ← Best overall
  - phi-2.Q4_K_M.gguf                          (~1.7GB) ← Fastest
  - llama-2-7b-chat.Q4_K_M.gguf               (~4.1GB) ← Good chat
  - openchat-3.5-0106.Q4_K_M.gguf             (~4.1GB) ← Great reasoning
"""

import os
from logs.logger import log
from config.config import Config


class LocalLLM:
    def __init__(self, model_path: str | None = None, gpu_layers: int | None = None):
        cfg = Config()
        self.model_path = model_path or cfg.LLM_MODEL_PATH
        self.gpu_layers = gpu_layers if gpu_layers is not None else cfg.LLM_GPU_LAYERS
        self.context_size = cfg.LLM_CONTEXT_SIZE
        self.max_tokens = cfg.LLM_MAX_TOKENS   # -1 = unlimited
        self.temperature = cfg.LLM_TEMPERATURE
        self.n_batch = cfg.LLM_N_BATCH
        self.n_threads = cfg.LLM_N_THREADS
        self._llm = None

    def load(self):
        """Load the model into memory (GPU layers go to RTX 4050 VRAM)."""
        if not self.model_path or not os.path.exists(self.model_path):
            log.warning(f"⚠️  Model not found: {self.model_path} — attempting auto-download…")
            try:
                from data.models.auto_download import download_model
                self.model_path = download_model()
            except Exception as dl_err:
                log.error(f"❌ Auto-download failed: {dl_err}")
                log.error("   Manually place a .gguf in data/models/ and set LLM_MODEL_PATH")
                return False

        try:
            from llama_cpp import Llama

            log.info(f"🧠 Loading LLM: {os.path.basename(self.model_path)}")
            log.info(f"   GPU layers : {self.gpu_layers} → RTX 4050 VRAM")
            log.info(f"   Context    : {self.context_size} tokens")
            log.info(f"   Batch size : {self.n_batch} (faster prompt eval)")
            log.info(f"   Threads    : {self.n_threads}")

            self._llm = Llama(
                model_path=self.model_path,
                n_gpu_layers=self.gpu_layers,
                n_ctx=self.context_size,
                n_batch=self.n_batch,      # larger = faster prompt processing
                n_threads=self.n_threads,  # use all CPU cores for non-GPU layers
                use_mmap=True,             # memory-mapped loading (faster cold start)
                use_mlock=False,           # let OS manage physical memory
                verbose=False,
            )

            log.info("✅ LLM loaded and ready!")
            return True

        except ImportError:
            log.error("❌ llama-cpp-python not installed")
            log.error("   GPU: pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121")
            log.error("   CPU: pip install llama-cpp-python")
            return False

        except Exception as e:
            log.error(f"❌ Failed to load LLM: {e}")
            return False

    def is_loaded(self) -> bool:
        return self._llm is not None

    def generate(self, messages: list[dict], stream: bool = True):
        """
        Generate a response to a chat-formatted message list.

        Args:
            messages: [{"role": "system"/"user", "content": ...}, ...] — see
                      build_rag_messages(). Passed to llama-cpp-python's
                      create_chat_completion(), which renders the GGUF's own
                      embedded chat template (tokenizer.chat_template) rather
                      than a hardcoded one.

                      This used to take a single pre-formatted prompt string
                      wrapped in a hardcoded Mistral-style [INST]...[/INST]
                      template, applied regardless of which model was actually
                      loaded. Measured against the configured model
                      (DeepSeek-R1-Distill-Qwen-7B, architecture qwen2, real
                      template uses <|User|>/<|Assistant|>): the [INST] format
                      produced a malformed, unpaired </think> with the real
                      answer discarded as "reasoning" by the stripper, and a
                      shallow response that skipped engaging with the excerpt.
                      The model's own template produced a correctly paired
                      <think>...</think> block that specifically reasoned from
                      the supplied excerpt. Same fix generalizes to every other
                      model this app lists as supported (Mistral, Phi-2,
                      Llama-2, OpenChat) — each has a different native format,
                      and hardcoding one meant every model but a Mistral
                      Instruct build was silently mismatched.

            stream:   If True, yields text tokens as they are generated (live output)

        Yields (stream=True): str tokens
        Returns (stream=False): str full response
        """
        if not self._llm:
            yield "❌ Model not loaded. Run: python main.py chat --model path/to/model.gguf"
            return

        max_tokens = None if self.max_tokens is None or self.max_tokens < 0 else self.max_tokens

        try:
            output = self._llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,        # None = no hard cap
                temperature=self.temperature,
                stream=stream,
            )

            if stream:
                for chunk in output:
                    delta = chunk["choices"][0].get("delta", {})
                    token = delta.get("content") or ""
                    if token:
                        yield token
            else:
                yield output["choices"][0]["message"]["content"]

        except Exception as e:
            yield f"⚠️  Generation error: {e}"

    def build_rag_messages(self, question: str, context_chunks: list[dict],
                           library_titles: list[str] | None = None,
                           char_budget: int | None = None) -> list[dict]:
        """
        Build the RAG chat messages: a system message with the grounded library
        header, numbered sources with page spans, and citation rules, plus the
        user's question as its own message.

        Two changes that matter:

        1. NO PER-CHUNK TRUNCATION. The old builder cut every chunk at 600
           chars. That was a no-op at the old 400-char chunk size and would now
           silently discard half of every 1200-char chunk. Budget is enforced
           across the whole context instead, so nothing is cut mid-passage
           without the caller knowing.

        2. THE LIBRARY COUNT IS STATED, NOT INFERRED. "How many books do you
           have?" was previously unanswerable — nothing ever told the model. It
           guessed. ~200 tokens of header removes an entire class of
           hallucination.

        Returns messages, not a formatted string — see generate()'s docstring
        for why: the model's own chat template renders these, instead of a
        hardcoded format that only matched one specific model family.
        """
        budget = char_budget or Config().CONTEXT_CHAR_BUDGET

        header = ""
        if library_titles:
            listed = "\n".join(f"  - {t}" for t in library_titles)
            header = (
                f"LIBRARY: {len(library_titles)} book(s) are indexed and "
                f"searchable:\n{listed}\n\n"
                "That list is complete. If asked how many books there are, or "
                "which ones, answer from it exactly and do not guess.\n\n"
            )

        parts, used = [], 0
        for i, chunk in enumerate(context_chunks, 1):
            source = chunk.get("source", "unknown")
            text = chunk.get("chunk", "")
            loc = locate(chunk)

            block = f"\n[{i}] {loc}\n{text}\n"
            if used + len(block) > budget and parts:
                break
            parts.append(block)
            used += len(block)

        context_text = "".join(parts)

        system = f"""You are ParseDat_Diary. You answer questions using only the book excerpts provided below.

{header}RULES:
- Use ONLY the numbered excerpts below. Do not add outside knowledge.
- After EVERY claim, cite the excerpt it came from as [1], [2], etc.
- If a claim draws on several excerpts, cite them all, like [2][5].
- Cite only numbers that appear below. Never invent a citation.
- If the excerpts do not answer the question, say so plainly instead of
  guessing. An honest "not in these books" is the correct answer.
- Answer fully. Do not truncate.

EXCERPTS:
{context_text}"""

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ]

    @staticmethod
    def validate_citations(answer: str, n_sources: int) -> dict:
        """
        Check the [N] markers in an answer against the sources actually supplied.

        A citation outside 1..n_sources is a mechanically detectable
        hallucination: the model referenced material it was never given. That is
        one of the few hallucination classes that can be caught without a human
        reading the source, so it is worth catching every time.

        Returns: {cited, invalid, uncited, n_sources}
        """
        import re
        found = [int(x) for x in re.findall(r"\[(\d{1,3})\]", answer or "")]
        cited = sorted(set(n for n in found if 1 <= n <= n_sources))
        invalid = sorted(set(n for n in found if not (1 <= n <= n_sources)))
        # Substantive answer with no citations at all is also suspect.
        uncited = bool(answer and len(answer.strip()) > 200 and not found)
        return {
            "cited": cited,
            "invalid": invalid,
            "uncited": uncited,
            "n_sources": n_sources,
        }

def locate(chunk: dict) -> str:
    """
    Human-readable location of a passage.

    PDFs cite a page. Markdown has no pages, so it cites the nearest heading —
    "notes.md - ## Configuration" is checkable in a way the bare filename is not.
    """
    source = chunk.get("source", "unknown")
    ps, pe = chunk.get("page_start"), chunk.get("page_end")
    if ps and pe and ps != pe:
        return f"{source}, p.{ps}-{pe}"
    if ps:
        return f"{source}, p.{ps}"
    section = chunk.get("section")
    if section:
        return f"{source} - {section}"
    return source
