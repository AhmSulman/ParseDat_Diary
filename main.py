"""
██████╗ ██████╗
██╔══██╗██╔══██╗
██████╔╝██║  ██║
██╔═══╝ ██║  ██║
██║     ██████╔╝
╚═╝     ╚═════╝

ParseDat_Diary — learn from your books, not just search them.
==============================================================
Local AI · Your GPU · Your data · RTX 4050 · FAISS · Local LLM · RAG
"""

import argparse
import asyncio
import sys
import os

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(__file__))

# Cap BLAS thread pools BEFORE numpy / torch / faiss are imported.
#
# OpenBLAS allocates per-thread scratch buffers from a fixed pool. The GUI runs
# the RAG pipeline on a worker thread while Kivy drives the main loop, and
# numpy, faiss and torch all sit on the same OpenBLAS, so the pool is exhausted
# and every allocation fails:
#     OpenBLAS error: Memory allocation still failed after 10 retries
# It looks like an out-of-memory failure and is not — it happened with 4.5 GB
# free. Setting these after the import has no effect; the pool is sized at load.
# OPENBLAS must be 1: 4 still aborted. The GUI only ever runs BLAS on single
# vectors (one query embedding, one FAISS probe), so 1 thread costs nothing
# measurable, while the abort was fatal and killed the process outright.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

# Windows: force UTF-8 so box-drawing chars in the banner don't crash
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from logs.logger import log


def print_banner():
    print("""
╔══════════════════════════════════════════════════════╗
║   P A R S E D A T _ D I A R Y  —  Chat with Books    ║
║   Local AI · RTX 4050 · No Cloud · No Censorship     ║
╚══════════════════════════════════════════════════════╝
""")


def _print_removed(removed: dict):
    total = sum(len(v) for v in removed.values())
    if not total:
        print("Nothing to remove — already consistent.")
        return
    for kind, items in removed.items():
        if items:
            print(f"  {kind}: {len(items)}")
            for i in items[:10]:
                print(f"      - {i}")
            if len(items) > 10:
                print(f"      ... and {len(items) - 10} more")
    print(f"\nRemoved {total} item(s).")


def _doctor():
    """
    Read-only reconciliation report.

    The counts below are the ones that used to disagree: 18 books marked done,
    7 actually indexed, 13 PDFs on disk. Any nonzero drift means a question
    about the library has more than one answer.
    """
    from storage.library import LibraryService

    rep = LibraryService().report()
    c = rep["counts"]

    print("LIBRARY")
    print(f"  PDFs on disk      : {c['pdfs_on_disk']}")
    print(f"  Books indexed     : {c['books_indexed']}")
    print(f"  Chunks indexed    : {c['chunks_indexed']:,}")
    print(f"  Checkpoint 'done' : {c['checkpoint_done']}")
    print(f"  Extracted .txt    : {c['txt_files']}")
    print(f"  Quarantined       : {c['quarantined']}")

    if rep["settings"]:
        s = rep["settings"]
        print("\nINDEX BUILT WITH")
        print(f"  embedder : {s.get('embed_model')} ({s.get('embed_dim')}-dim)")
        print(f"  chunking : {s.get('chunk_size')} / {s.get('chunk_overlap')} overlap")

    if rep["holes"]:
        print(f"\nHOLES — marked done but never indexed ({len(rep['holes'])}):")
        print("  These are invisible to the AI. Fix: python main.py reindex")
        for h in rep["holes"]:
            print(f"      - {h}")

    orph = rep["orphans"]
    if any(orph.values()):
        print("\nORPHANS — derived data whose PDF is gone:")
        for kind, items in orph.items():
            for i in items:
                print(f"      [{kind}] {i}")
        print("  Fix: python main.py sync")

    if rep["pending"]:
        print(f"\nPENDING — not yet processed ({len(rep['pending'])}):")
        for p in rep["pending"]:
            print(f"      - {p}")
        print("  Fix: python main.py ingest")

    if rep["quarantine"]:
        print("\nQUARANTINED — failed the quality gate, never indexed:")
        for q in rep["quarantine"]:
            print(f"      - {q.get('source')}")
            for r in q.get("reasons", [])[:3]:
                print(f"          {r}")

    print()
    if rep["healthy"]:
        print(f"HEALTHY — every store agrees at {c['pdfs_on_disk']} book(s).")
    else:
        print(f"DRIFT: {rep['drift']} inconsistency(ies). See the fixes above.")


def _clean(args):
    from storage.library import LibraryService

    scopes = dict(
        index=args.index or args.all,
        checkpoint=args.checkpoint or args.all,
        orphans=args.orphans or args.all,
        quarantine=args.quarantine or args.all,
        text=args.text or args.all,
    )
    if not any(scopes.values()):
        print("Nothing selected. Choose --index, --checkpoint, --orphans, "
              "--quarantine, --text, or --all.")
        return

    # --text forces a full re-OCR of every book, which is the expensive,
    # hard-to-undo one. Confirm it explicitly.
    if (args.all or args.text) and not args.yes:
        picked = ", ".join(k for k, v in scopes.items() if v)
        print(f"About to delete: {picked}")
        if scopes["text"]:
            print("  --text removes extracted text: every book must be re-OCR'd.")
        if input("Type 'yes' to continue: ").strip().lower() != "yes":
            print("Aborted.")
            return

    _print_removed(LibraryService().clean(**scopes))
    print("\nRe-run 'python main.py doctor' to confirm.")


def main():
    print_banner()

    parser = argparse.ArgumentParser(
        description="ParseDat_Diary — Chat with Books (Local AI RAG System)"
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
    server_p.add_argument("--allow-admin", action="store_true",
                          help="Enable clean/sync endpoints (localhost only)")

    # ── search ────────────────────────────────────────────────────────────────
    search_p = subparsers.add_parser("search", help="Semantic search without chat")
    search_p.add_argument("query", type=str, help="Search query")
    search_p.add_argument("--top-k", type=int, default=5)

    # ── doctor ────────────────────────────────────────────────────────────────
    subparsers.add_parser(
        "doctor", help="Report library state and any drift (read-only)"
    )

    # ── clean ─────────────────────────────────────────────────────────────────
    clean_p = subparsers.add_parser("clean", help="Remove derived data")
    clean_p.add_argument("--index", action="store_true",
                         help="Drop the vector index, metadata and manifest")
    clean_p.add_argument("--checkpoint", action="store_true",
                         help="Reset the ingest checkpoint")
    clean_p.add_argument("--orphans", action="store_true",
                         help="Remove derived data whose PDF is gone")
    clean_p.add_argument("--quarantine", action="store_true",
                         help="Empty data/quarantine/")
    clean_p.add_argument("--text", action="store_true",
                         help="Delete extracted text (forces re-OCR)")
    clean_p.add_argument("--all", action="store_true",
                         help="Everything derived. Asks for confirmation.")
    clean_p.add_argument("--yes", action="store_true", help="Skip confirmation")

    # ── sync ──────────────────────────────────────────────────────────────────
    subparsers.add_parser(
        "sync", help="Reconcile all derived data to data/input/ (after deleting PDFs)"
    )

    # ── reindex ───────────────────────────────────────────────────────────────
    reindex_p = subparsers.add_parser(
        "reindex", help="Rebuild vectors from extracted text (no OCR)"
    )
    reindex_p.add_argument("--device", default=None,
                           help="cuda or cpu (default: auto)")

    # ── gui ───────────────────────────────────────────────────────────────────
    # ── console ───────────────────────────────────────────────────────────────
    console_p = subparsers.add_parser(
        "console", help="Interactive terminal: ask, tune, ingest, forget books"
    )
    console_p.add_argument("--model", type=str, default=None)
    console_p.add_argument("--gpu-layers", type=int, default=None)

    gui_p = subparsers.add_parser("gui", help="Launch Material Design GUI")
    gui_p.add_argument("--model", type=str, default=None, help="Path to .gguf model")
    gui_p.add_argument("--gpu-layers", type=int, default=None, help="GPU layers for LLM")

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
        run_server(host=args.host, port=args.port, model_path=args.model,
                   allow_admin=args.allow_admin)

    elif args.command == "search":
        from brain.retriever import Retriever
        r = Retriever()
        r.load()
        results = r.search(args.query, k=args.top_k)
        print(f"\n🔍 Top {args.top_k} results for: '{args.query}'\n")
        for i, res in enumerate(results, 1):
            print(f"  [{i}] {res['source']}  (score: {res['score']:.4f})")
            print(f"      {res['chunk'][:200].strip()}...\n")

    elif args.command == "doctor":
        _doctor()

    elif args.command == "clean":
        _clean(args)

    elif args.command == "sync":
        from storage.library import LibraryService
        removed = LibraryService().clean(orphans=True)
        _print_removed(removed)
        print("\nRe-run 'python main.py doctor' to confirm.")

    elif args.command == "reindex":
        from core.reindex import reindex
        st = reindex(device=args.device) if args.device else reindex()
        print(f"\nReindexed {st['indexed']} book(s), {st['chunks']:,} chunks "
              f"in {st['elapsed']}s")
        print(f"  embedder   : {st['embed_model']} on {st['device']}")
        if st["quarantined"]:
            print(f"  quarantined: {st['quarantined']} (see data/quarantine/)")
        if st["needs_ingest"]:
            print(f"  need ingest: {len(st['needs_ingest'])} book(s) have no "
                  f"extracted text yet -> run 'python main.py ingest'")
            for b in st["needs_ingest"]:
                print(f"      - {b}")
        if st["failed"]:
            print(f"  failed     : {st['failed']}")

    elif args.command == "console":
        from chat.console import Console
        Console(model_path=args.model, gpu_layers=args.gpu_layers).run()

    elif args.command == "gui":
        from gui.material_app import run_gui
        run_gui(model_path=args.model, gpu_layers=args.gpu_layers)

    elif args.command == "service":
        from service.windows_service import ServiceManager
        ServiceManager().handle(args.action)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
