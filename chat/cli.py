"""
MAAN Terminal Chat
==================
Talk to your books directly from the terminal.

Usage:
    python main.py chat
    python main.py chat --model data/models/mistral-7b.gguf
    python main.py chat --gpu-layers 35

Commands during chat:
    /quit or /exit    — Exit
    /clear            — Clear conversation history
    /sources <query>  — Show source chunks without answering
    /help             — Show commands
"""

import sys
from brain.rag import RAGPipeline
from logs.logger import log


BANNER = """
╔══════════════════════════════════════════════════════╗
║   M A A N  —  Chat with Books                       ║
║   Type your question. /help for commands.            ║
╚══════════════════════════════════════════════════════╝
"""

HELP_TEXT = """
Commands:
  /quit, /exit      → Exit MAAN
  /clear            → Clear chat history
  /sources <query>  → Show retrieved chunks for a query
  /help             → Show this message
"""


class ChatCLI:
    def __init__(self, model_path: str = None, gpu_layers: int = 35):
        self.rag = RAGPipeline(model_path=model_path, gpu_layers=gpu_layers)

    def run(self):
        print(BANNER)

        if not self.rag.setup():
            print("❌ Could not load LLM. Check config.py for LLM_MODEL_PATH.")
            print("   Download a model: https://huggingface.co/TheBloke")
            sys.exit(1)

        print("✅ Ready! Ask anything about your books.\n")

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\n👋 Goodbye!")
                break

            if not user_input:
                continue

            # ── Commands ──────────────────────────────────────────────────────
            if user_input.lower() in ("/quit", "/exit"):
                print("👋 Goodbye!")
                break

            elif user_input.lower() == "/clear":
                print("\033[2J\033[H")  # Clear terminal
                print(BANNER)
                continue

            elif user_input.lower() == "/help":
                print(HELP_TEXT)
                continue

            elif user_input.lower().startswith("/sources "):
                query = user_input[9:].strip()
                chunks = self.rag.get_sources(query)
                if chunks:
                    print(f"\n📚 Top {len(chunks)} chunks for: '{query}'\n")
                    for i, c in enumerate(chunks, 1):
                        print(f"  [{i}] {c['source']}  score={c['score']:.4f}")
                        print(f"      {c['chunk'][:300].strip()}...\n")
                else:
                    print("No chunks found. Ingest some PDFs first.\n")
                continue

            # ── Normal chat ───────────────────────────────────────────────────
            print("\nMaAN: ", end="", flush=True)
            try:
                for token in self.rag.answer(user_input, stream=True):
                    print(token, end="", flush=True)
                print("\n")
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
