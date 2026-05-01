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
    def __init__(self, model_path: str = None, gpu_layers: int = None):
        cfg = Config()
        self.model_path = model_path or cfg.LLM_MODEL_PATH
        self.gpu_layers = gpu_layers if gpu_layers is not None else cfg.LLM_GPU_LAYERS
        self.context_size = cfg.LLM_CONTEXT_SIZE
        self.max_tokens = cfg.LLM_MAX_TOKENS
        self.temperature = cfg.LLM_TEMPERATURE
        self._llm = None

    def load(self):
        """Load the model into memory (GPU layers go to RTX 4050 VRAM)."""
        if not self.model_path or not os.path.exists(self.model_path):
            log.error(f"❌ Model not found: {self.model_path}")
            log.error("   Download a .gguf model from https://huggingface.co/TheBloke")
            log.error("   Then set LLM_MODEL_PATH in config/config.py")
            return False

        try:
            from llama_cpp import Llama

            log.info(f"🧠 Loading LLM: {os.path.basename(self.model_path)}")
            log.info(f"   GPU layers: {self.gpu_layers} → RTX 4050 VRAM")

            self._llm = Llama(
                model_path=self.model_path,
                n_gpu_layers=self.gpu_layers,   # How many layers run on GPU
                n_ctx=self.context_size,         # Context window size
                n_batch=512,                     # Batch size for prompt processing
                verbose=False,
            )

            log.info("✅ LLM loaded and ready!")
            return True

        except ImportError:
            log.error("❌ llama-cpp-python not installed")
            log.error("   GPU version: pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121")
            log.error("   CPU version: pip install llama-cpp-python")
            return False

        except Exception as e:
            log.error(f"❌ Failed to load LLM: {e}")
            return False

    def is_loaded(self) -> bool:
        return self._llm is not None

    def generate(self, prompt: str, stream: bool = True):
        """
        Generate a response to the prompt.

        Args:
            prompt:  Full prompt string (system + context + question)
            stream:  If True, yields text tokens as they are generated (live output)

        Yields (stream=True): str tokens
        Returns (stream=False): str full response
        """
        if not self._llm:
            yield "❌ Model not loaded. Run: python main.py chat --model path/to/model.gguf"
            return

        try:
            output = self._llm(
                prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stop=["</s>", "[INST]", "User:", "\n\nQuestion:"],
                stream=stream,
                echo=False,
            )

            if stream:
                for chunk in output:
                    token = chunk["choices"][0]["text"]
                    yield token
            else:
                yield output["choices"][0]["text"]

        except Exception as e:
            yield f"⚠️  Generation error: {e}"

    def build_rag_prompt(self, question: str, context_chunks: list[dict]) -> str:
        """
        Build a RAG (Retrieval-Augmented Generation) prompt.

        Combines retrieved document chunks with the user's question
        so the LLM can answer based on the actual book content.
        """
        context_text = ""
        for i, chunk in enumerate(context_chunks, 1):
            source = chunk.get("source", "unknown")
            text = chunk.get("chunk", "")[:800]
            context_text += f"\n[Source {i}: {source}]\n{text}\n"

        prompt = f"""[INST] You are MAAN, an AI assistant that answers questions based on the provided book excerpts.
Answer clearly and helpfully using ONLY the context below.
If the answer is not in the context, say so honestly.

CONTEXT:
{context_text}

QUESTION: {question} [/INST]
ANSWER: """

        return prompt
