"""
███╗   ███╗ █████╗  █████╗ ███╗   ██╗
████╗ ████║██╔══██╗██╔══██╗████╗  ██║
██╔████╔██║███████║███████║██╔██╗ ██║
██║╚██╔╝██║██╔══██║██╔══██║██║╚██╗██║
██║ ╚═╝ ██║██║  ██║██║  ██║██║ ╚████║
╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝

Chat with Books — Local AI. Your GPU. Your Data.
=================================================
RTX 4050 · ONNX CUDA · FAISS · Local LLM · RAG
"""

import argparse
import asyncio
import sys
import os

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(__file__))

from logs.logger import log


def print_banner():
    print("""
╔══════════════════════════════════════════════════════╗
║   M A A N  —  Chat with Books                       ║
║   Local AI · RTX 4050 · No Cloud · No Censorship    ║
╚══════════════════════════════════════════════════════╝
""")


def main():
    print_banner()

    parser = argparse.ArgumentParser(
        description="MAAN — Chat with Books (Local AI RAG System)"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── ingest ────────────────────────────────────────────────────────────────
    ingest_p = subparsers.add_parser("ingest", help="Process PDFs from data/input/")
    ingest_p.add_argument("--reset", action="store_true", help="Re-process all PDFs")
    ingest_p.add_argument("--workers", type=int, default=4, help="Async worker count")

    # ── chat ──────────────────────────────────────────────────────────────────
    chat_p = subparsers.add_parser("chat", help="Chat with your books (CLI)")
    chat_p.add_argument("--model", type=str, default=None, help="Path to .gguf model")
    chat_p.add_argument("--gpu-layers", type=int, default=35, help="GPU layers for LLM")

    # ── server ────────────────────────────────────────────────────────────────
    server_p = subparsers.add_parser("server", help="Launch web API server")
    server_p.add_argument("--host", default="0.0.0.0")
    server_p.add_argument("--port", type=int, default=8000)
    server_p.add_argument("--model", type=str, default=None)

    # ── search ────────────────────────────────────────────────────────────────
    search_p = subparsers.add_parser("search", help="Semantic search without chat")
    search_p.add_argument("query", type=str, help="Search query")
    search_p.add_argument("--top-k", type=int, default=5)

    # ── service ───────────────────────────────────────────────────────────────
    svc_p = subparsers.add_parser("service", help="Windows service management")
    svc_p.add_argument("action", choices=["install", "remove", "start", "stop", "status"])

    args = parser.parse_args()

    if args.command == "ingest" or args.command is None:
        from core.async_pipeline import AsyncPipeline
        if args.command and args.reset:
            from storage.checkpoint import Checkpoint
            Checkpoint().reset()
        asyncio.run(AsyncPipeline().run())

    elif args.command == "chat":
        from chat.cli import ChatCLI
        ChatCLI(model_path=args.model, gpu_layers=args.gpu_layers).run()

    elif args.command == "server":
        from chat.server import run_server
        run_server(host=args.host, port=args.port, model_path=args.model)

    elif args.command == "search":
        from brain.retriever import Retriever
        r = Retriever()
        r.load()
        results = r.search(args.query, k=args.top_k)
        print(f"\n🔍 Top {args.top_k} results for: '{args.query}'\n")
        for i, res in enumerate(results, 1):
            print(f"  [{i}] {res['source']}  (score: {res['score']:.4f})")
            print(f"      {res['chunk'][:200].strip()}...\n")

    elif args.command == "service":
        from service.windows_service import ServiceManager
        ServiceManager().handle(args.action)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
