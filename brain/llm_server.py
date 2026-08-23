"""
llama-server HTTP client — GPU generation without Python bindings
==================================================================
Talks to a running `llama-server` over HTTP instead of loading the model
in-process through llama-cpp-python.

WHY NOT llama-cpp-python
------------------------
It is a dead end on this machine:
  - 0.3.23 (installed) is CPU-only: llama_supports_gpu_offload() == False,
    so LLM_GPU_LAYERS=35 was being silently ignored
  - the newest prebuilt CUDA wheel is 0.3.4, which predates Qwen3 and would
    reject Qwen3-4B outright
  - building 0.3.23 from source needs CUDA 13.2 against build scripts written
    before CUDA 13 existed

llama-server sidesteps all of it: one binary, every architecture, real GPU
offload, and it exposes -ngl / -ctk / -ctv for the memory tuning this box needs.

INTERFACE COMPATIBILITY
-----------------------
This deliberately mirrors brain/llm.py::LocalLLM — `load()`, `is_loaded()`,
`generate(prompt, stream=True)`, plus the inherited `build_rag_prompt()` and
`validate_citations()`. brain/rag.py therefore needs no changes: it already
consumes a streaming generator, and all the citation work carries over.
"""

from __future__ import annotations

import json

from brain.llm import LocalLLM
from config.config import Config
from logs.logger import log


class LlamaServerLLM(LocalLLM):
    """
    Drop-in replacement for LocalLLM backed by an HTTP llama-server.

    Subclasses LocalLLM purely to inherit build_rag_prompt() and
    validate_citations() — the prompt format, the grounded library header and
    the citation validator are backend-independent and must not be duplicated.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8080",
                 model_path: str | None = None, gpu_layers: int | None = None,
                 timeout: float = 300.0):
        super().__init__(model_path=model_path, gpu_layers=gpu_layers)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._ready = False
        self._server_model: str | None = None
        # The server's REAL context window, not what config wishes it were.
        self.server_ctx: int | None = None

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def load(self) -> bool:
        """
        Confirm a server is reachable and note which model it holds.

        Does not start the server — process management is the caller's job, so
        that a crashed model never takes MAAN down with it.
        """
        try:
            import httpx
        except ImportError:
            log.error("httpx not installed — required for the llama-server backend")
            return False

        try:
            with httpx.Client(timeout=10.0) as c:
                r = c.get(f"{self.base_url}/v1/models")
                r.raise_for_status()
                data = r.json().get("data") or []
                if data:
                    self._server_model = data[0].get("id")

                # Ask the server how much context it ACTUALLY has. It is launched
                # independently, so its -c may not match what config assumes —
                # and a mismatch means llama-server silently drops the front of
                # the prompt. That would truncate the retrieved passages without
                # any error, and the model would answer from a fragment while
                # looking perfectly healthy. Detect it instead of assuming.
                try:
                    p = c.get(f"{self.base_url}/props")
                    if p.status_code == 200:
                        j = p.json()
                        ctx = (j.get("default_generation_settings", {}).get("n_ctx")
                               or j.get("n_ctx"))
                        if ctx:
                            self.server_ctx = int(ctx)
                except Exception:
                    pass
        except Exception as e:
            log.error(f"llama-server unreachable at {self.base_url}: {str(e)[:160]}")
            return False

        self._ready = True
        name = (self._server_model or "?").replace("\\", "/").split("/")[-1]
        log.info(f"llama-server ready at {self.base_url} — {name}"
                 + (f", n_ctx={self.server_ctx}" if self.server_ctx else ""))
        return True

    def usable_context_chars(self, reserve_tokens: int = 1200) -> int | None:
        """
        How many characters of retrieved context this server can actually take.

        Returns None when the context size is unknown (leave the caller's budget
        alone). Otherwise: (n_ctx - reserve for the answer and prompt scaffold)
        converted at the usual ~4 chars/token.

        `reserve_tokens` must be generous for reasoning models — DeepSeek-R1
        spends 500-2000 tokens inside <think> before writing anything.
        """
        if not self.server_ctx:
            return None
        usable = max(0, self.server_ctx - reserve_tokens)
        return usable * 4

    def is_loaded(self) -> bool:
        return self._ready

    # ── generation ────────────────────────────────────────────────────────────
    def generate(self, prompt: str, stream: bool = True):
        """
        Yield answer tokens from llama-server.

        Uses the /completion endpoint rather than the chat API: the prompt is
        already fully formatted by build_rag_prompt(), and a chat endpoint would
        wrap it in a second template, corrupting the citation instructions.
        """
        if not self._ready:
            yield "Model server not reachable. Start llama-server, then retry."
            return

        try:
            import httpx
        except ImportError:
            yield "httpx not installed."
            return

        payload = {
            "prompt": prompt,
            "n_predict": self.max_tokens if self.max_tokens and self.max_tokens > 0 else -1,
            "temperature": self.temperature,
            "stop": ["</s>", "[INST]", "User:", "\n\nQuestion:"],
            "stream": bool(stream),
            "cache_prompt": True,
        }

        try:
            with httpx.Client(timeout=self.timeout) as c:
                if not stream:
                    r = c.post(f"{self.base_url}/completion", json=payload)
                    r.raise_for_status()
                    yield r.json().get("content", "")
                    return

                with c.stream("POST", f"{self.base_url}/completion", json=payload) as r:
                    r.raise_for_status()
                    for line in r.iter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        body = line[6:].strip()
                        if body == "[DONE]":
                            break
                        try:
                            chunk = json.loads(body)
                        except json.JSONDecodeError:
                            continue
                        token = chunk.get("content", "")
                        if token:
                            yield token
                        if chunk.get("stop"):
                            break
        except Exception as e:
            log.error(f"llama-server generation failed: {str(e)[:200]}")
            yield f"\n[generation error: {str(e)[:160]}]"


def strip_thinking(text: str) -> str:
    """
    Remove <think>...</think> blocks from a reasoning model's output.

    DeepSeek-R1 emits 500-2000 tokens of chain-of-thought before answering. That
    reasoning is not the answer and must not reach the user, nor be scanned for
    citations — a [3] mentioned while thinking is not a claim being made.

    Handles the unterminated case: if generation was cut off mid-thought, there
    is no closing tag and everything after <think> is discarded.
    """
    import re
    if "<think>" not in text:
        return text
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "<think>" in text:                      # truncated mid-thought
        text = text.split("<think>", 1)[0]
    return text.strip()
