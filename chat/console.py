"""
MAAN Console — the terminal interface
======================================
One place to tune retrieval, manage the library, and ask questions, without
editing config files or remembering flag names.

Every knob shows its range and what changing it costs. Knobs that invalidate
the index (chunk size, embedder) are marked, because changing one silently
would leave the stored vectors describing text that no longer exists.

Settings changed here persist to data/console_settings.json and are applied to
the live pipeline immediately where that is possible.
"""

from __future__ import annotations

import json
import os
import time

from config.config import Config
from logs.logger import log

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS_FILE = os.path.join(_ROOT, "data", "console_settings.json")

# name -> (attr, type, min, max, needs_reindex, help)
KNOBS: dict[str, tuple] = {
    "top_k": ("SEARCH_TOP_K", int, 1, 20, False,
              "Passages retrieved per question. More = broader but slower and "
              "can bury the answer."),
    "radius": ("NEIGHBOR_RADIUS", int, 0, 3, False,
               "Neighbouring chunks pulled around each hit. 0 = fragments, "
               "1 = full thought, 2+ = long context."),
    "budget": ("CONTEXT_CHAR_BUDGET", int, 2000, 40000, False,
               "Max characters of book text sent to the model. Must fit the "
               "server's context (~4 chars/token)."),
    "temperature": ("LLM_TEMPERATURE", float, 0.0, 1.5, False,
                    "Randomness. 0 = deterministic, 0.7 = balanced, "
                    ">1.0 = creative and less faithful."),
    "reserve": ("LLM_ANSWER_RESERVE", int, 300, 4000, False,
                "Tokens held back for the answer. Reasoning models need 1500+ "
                "because they think before answering."),
    "chunk_size": ("CHUNK_SIZE", int, 400, 4000, True,
                   "Characters per chunk. ~1200 = one full idea."),
    "chunk_overlap": ("CHUNK_OVERLAP", int, 0, 1000, True,
                      "Characters shared between neighbours. Must stay below "
                      "half of chunk_size."),
}


def _load_saved() -> dict:
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save(d: dict) -> None:
    os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
    tmp = SETTINGS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)
    os.replace(tmp, SETTINGS_FILE)


def apply_saved_settings() -> dict:
    """Push saved knob values onto Config before anything reads them."""
    saved = _load_saved()
    for name, val in saved.items():
        spec = KNOBS.get(name)
        if not spec:
            continue
        attr, typ, lo, hi, _, _ = spec
        try:
            v = typ(val)
        except (TypeError, ValueError):
            continue
        if lo <= v <= hi:
            setattr(Config, attr, v)
    return saved


class Console:
    def __init__(self, model_path=None, gpu_layers=None):
        self.cfg = Config()
        self.saved = apply_saved_settings()
        self.show_reasoning = bool(self.saved.get("show_reasoning", False))
        self.rag = None
        self._model_path = model_path
        self._gpu_layers = gpu_layers

        from storage.history import History
        self.history = History()
        # Resume the latest conversation rather than silently opening a blank
        # one, so closing the app does not lose the thread you were on.
        last = self.history.latest_session_id()
        if last:
            self.history.set_current(last)

    # ── plumbing ──────────────────────────────────────────────────────────────
    def _ensure_rag(self) -> bool:
        if self.rag is not None:
            return True
        print("Loading library and model…")
        from brain.rag import RAGPipeline
        self.rag = RAGPipeline(model_path=self._model_path,
                               gpu_layers=self._gpu_layers)
        ok = self.rag.setup()
        if not ok:
            print("  ! No LLM reachable. Retrieval works; answers will not.")
            print("    Start one:  llama-server -m <model>.gguf -c 8192 --port 8084")
        return True

    # ── commands ──────────────────────────────────────────────────────────────
    def cmd_help(self, _=""):
        print("""
  ask <question>      Ask your books (or just type a question)
  books               List indexed books with chunk counts
  ingest              Process new PDFs from data/input/
  forget <name>       Remove ONE book from the index (PDF stays on disk)
  doctor              Library health and drift
  set                 Show all knobs, ranges and current values
  set <knob> <value>  Change a knob (e.g. set top_k 8)
  reasoning on|off    Show or hide the model's chain-of-thought
  history             Replay the current conversation
  sessions            List every past conversation
  new [title]         Start a fresh conversation (previous one is kept)
  open <n>            Reopen conversation n from 'sessions'
  export [all]        Save conversation(s) as markdown
  reindex             Rebuild all vectors from cached text
  clear               Clear the screen
  exit                Quit
""")

    def cmd_set(self, arg: str = ""):
        parts = arg.split()
        if not parts:
            print("\n  KNOB            CURRENT      RANGE            EFFECT")
            print("  " + "-" * 74)
            for name, (attr, typ, lo, hi, reidx, helptext) in KNOBS.items():
                cur = getattr(self.cfg, attr, "?")
                mark = "  [needs reindex]" if reidx else ""
                print(f"  {name:<15} {str(cur):<12} {f'{lo}-{hi}':<16} {helptext}{mark}")
            print(f"\n  reasoning       {'on' if self.show_reasoning else 'off':<12} "
                  f"{'on/off':<16} Show the model's thinking before its answer.")
            print(f"\n  Saved to {SETTINGS_FILE}")
            return

        if len(parts) < 2:
            print("  Usage: set <knob> <value>")
            return
        name, raw = parts[0], parts[1]
        spec = KNOBS.get(name)
        if not spec:
            print(f"  Unknown knob '{name}'. Type 'set' to list them.")
            return

        attr, typ, lo, hi, reidx, _ = spec
        try:
            val = typ(raw)
        except ValueError:
            print(f"  '{raw}' is not a valid {typ.__name__}")
            return
        if not (lo <= val <= hi):
            print(f"  Out of range: {name} must be {lo}-{hi}")
            return
        # chunk_overlap >= chunk_size//2 can stall the chunker's advance.
        if name == "chunk_overlap" and val >= self.cfg.CHUNK_SIZE // 2:
            print(f"  Must stay below half of chunk_size "
                  f"({self.cfg.CHUNK_SIZE // 2}) or chunking cannot advance.")
            return

        setattr(Config, attr, val)
        self.cfg = Config()
        self.saved[name] = val
        _save(self.saved)

        if self.rag is not None and name in ("top_k", "budget"):
            self.rag.top_k = self.cfg.SEARCH_TOP_K
            self.rag.char_budget = self.cfg.CONTEXT_CHAR_BUDGET
        print(f"  {name} = {val}" + ("   RUN 'reindex' for this to take effect."
                                     if reidx else "   (applied)"))

    def cmd_reasoning(self, arg: str = ""):
        a = arg.strip().lower()
        if a not in ("on", "off"):
            print(f"  reasoning is {'on' if self.show_reasoning else 'off'}. "
                  f"Usage: reasoning on|off")
            return
        self.show_reasoning = (a == "on")
        self.saved["show_reasoning"] = self.show_reasoning
        _save(self.saved)
        print(f"  reasoning {a}" + ("  (you'll see the model think — it is "
                                    "verbose)" if self.show_reasoning else ""))

    def cmd_books(self, _=""):
        from storage.manifest import Manifest
        m = Manifest()
        if not m.books:
            print("  No books indexed. Put PDFs in data/input/ and run 'ingest'.")
            return
        rows = sorted(m.books.values(), key=lambda b: -b.get("n_chunks", 0))
        print(f"\n  {'BOOK':<50} {'CHUNKS':>7} {'PAGES':>6}")
        print("  " + "-" * 66)
        for b in rows:
            print(f"  {b['source'][:50]:<50} {b.get('n_chunks', 0):>7} "
                  f"{b.get('n_pages', 0):>6}")
        print("  " + "-" * 66)
        print(f"  {len(rows)} books, {m.chunk_count:,} chunks\n")

    def cmd_forget(self, arg: str = ""):
        """Remove one book's memory. Matches on any part of the name."""
        q = arg.strip()
        if not q:
            print("  Usage: forget <part of book name>    (see 'books')")
            return
        from storage.library import LibraryService
        from storage.manifest import Manifest

        matches = [s for s in Manifest().sources() if q.lower() in s.lower()]
        if not matches:
            print(f"  No indexed book matches '{q}'")
            return
        if len(matches) > 1:
            print(f"  '{q}' matches {len(matches)} books — be more specific:")
            for s in matches:
                print(f"      {s}")
            return

        src = matches[0]
        if input(f"  Remove '{src[:60]}' from memory? [y/N] ").strip().lower() != "y":
            print("  Cancelled.")
            return

        # Drops the manifest entry, checkpoint entry and cached text. The
        # vectors go on the next reindex, which is also what rebuilds the
        # index without them — HNSW cannot delete rows in place.
        LibraryService().purge_book(src)
        print(f"  Removed {src[:60]}")
        print("  Run 'reindex' to drop its vectors from the search index.")
        self.rag = None


    def cmd_export(self, arg: str = ""):
        """Write conversations to markdown next to their JSON records."""
        if arg.strip().lower() == "all":
            n = self.history.export_all_markdown()
            print(f"  Exported {n} conversation(s) to data/sessions/*.md")
            return
        path = self.history.write_markdown()
        if path:
            print(f"  Wrote {path}")
            print("  Tip: .md files in data/input/ are ingestable — you can add "
                  "a chat back into the library.")
        else:
            print("  Nothing to export in this conversation yet.")

    def cmd_ingest(self, _=""):
        import asyncio
        from core.async_pipeline import AsyncPipeline
        print("  Ingesting new PDFs from data/input/ …")
        asyncio.run(AsyncPipeline().run())
        self.rag = None
        print("  Done. Ask away.")

    def cmd_reindex(self, _=""):
        from core.reindex import reindex
        print("  Rebuilding vectors from cached text (no OCR)…")
        st = reindex()
        print(f"  {st['indexed']} books, {st['chunks']:,} chunks in {st['elapsed']}s "
              f"({st['embed_model']} on {st['device']})")
        if st["needs_ingest"]:
            print(f"  {len(st['needs_ingest'])} book(s) still need 'ingest'")
        self.rag = None

    def cmd_doctor(self, _=""):
        from main import _doctor
        _doctor()

    def cmd_ask(self, question: str):
        q = question.strip()
        if not q:
            return
        self._ensure_rag()
        print()
        t = time.perf_counter()
        parts = []
        try:
            for tok in self.rag.answer(q, stream=True):
                parts.append(tok)
                print(tok, end="", flush=True)
        except KeyboardInterrupt:
            print(chr(10) + "  [interrupted]")
        elapsed = time.perf_counter() - t
        answer = "".join(parts)

        # Persist the turn. Answers used to vanish the moment the next question
        # was asked, with no transcript and no record after closing the app.
        try:
            self.history.add_turn(
                q, answer, elapsed=elapsed,
                citations=self._citations_from(answer),
                model=getattr(self.rag.llm, "_server_model", None),
            )
        except Exception as e:
            log.warning(f"History not saved: {e}")
        print(chr(10)*2 + f"  ({elapsed:.1f}s, {len(answer)} chars)" + chr(10))

    @staticmethod
    def _citations_from(answer: str) -> list[dict]:
        """Recover the reference footer, so a saved turn stays checkable."""
        import re
        out = []
        for m in re.finditer(r"^\s*\[(\d+)\]\s+(.+?)(?:,\s*p\.(\d+))?\s*$",
                             answer, re.MULTILINE):
            out.append({"n": int(m.group(1)),
                        "source": m.group(2).strip(),
                        "page_start": int(m.group(3)) if m.group(3) else None})
        return out

    # ── history ───────────────────────────────────────────────────────────────
    def cmd_new(self, arg: str = ""):
        sid = self.history.new_session(arg.strip() or None)
        print(f"  Started a new conversation ({sid}). Previous one is saved.")

    def cmd_sessions(self, _=""):
        rows = self.history.sessions()
        if not rows:
            print("  No conversations yet.")
            return
        cur = self.history.session_id
        print(chr(10) + f"  {chr(32)*2} {'WHEN':<17} {'TURNS':>5}  TITLE")
        print("  " + "-" * 66)
        for r in rows[:20]:
            mark = "*" if r["id"] == cur else " "
            when = r.get("created_at", "")[:16].replace("T", " ")
            print(f"  {mark:2} {when:<17} {r['n_turns']:>5}  {r['title'][:38]}")
        print("  " + "-" * 66)
        print(f"  {len(rows)} conversation(s). '*' = current. "
              f"Use 'open <n>' for the nth, or 'history' for this one." + chr(10))

    def cmd_open(self, arg: str = ""):
        rows = self.history.sessions()
        if not arg.strip().isdigit():
            print("  Usage: open <number from 'sessions'>")
            return
        n = int(arg.strip())
        if not (1 <= n <= len(rows)):
            print(f"  Pick 1-{len(rows)}")
            return
        sid = rows[n - 1]["id"]
        self.history.set_current(sid)
        print(f"  Now in: {rows[n-1]['title']}")
        print(self.history.transcript(sid, limit=3) or "  (empty)")

    def cmd_history(self, _=""):
        sid = self.history.session_id
        if not sid:
            print("  No conversation open. Ask something, or 'sessions' to list.")
            return
        text = self.history.transcript(sid)
        print()
        print(text if text.strip() else "  (nothing asked yet in this conversation)")


    # ── loop ──────────────────────────────────────────────────────────────────
    def run(self):
        cfg = self.cfg
        print("=" * 66)
        print("  MAAN console — type 'help' for commands, 'exit' to quit")
        print("=" * 66)
        try:
            from storage.manifest import Manifest
            m = Manifest()
            print(f"  Library : {m.book_count} books, {m.chunk_count:,} chunks")
            print(f"  Embedder: {m.settings.get('embed_model', cfg.EMBED_MODEL)}")
        except Exception:
            pass
        print(f"  Model   : {cfg.LLM_SERVER_URL}  (backend: {cfg.LLM_BACKEND})")
        print(f"  Retrieval: top_k={cfg.SEARCH_TOP_K} radius={cfg.NEIGHBOR_RADIUS} "
              f"budget={cfg.CONTEXT_CHAR_BUDGET:,} chars")
        print()

        table = {
            "help": self.cmd_help, "?": self.cmd_help,
            "set": self.cmd_set, "reasoning": self.cmd_reasoning,
            "books": self.cmd_books, "forget": self.cmd_forget,
            "ingest": self.cmd_ingest, "reindex": self.cmd_reindex,
            "doctor": self.cmd_doctor, "ask": self.cmd_ask,
            "new": self.cmd_new, "sessions": self.cmd_sessions,
            "open": self.cmd_open, "history": self.cmd_history,
            "export": self.cmd_export,
            "clear": lambda _="": os.system("cls" if os.name == "nt" else "clear"),
        }

        while True:
            try:
                line = input("maan> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            if line.lower() in ("exit", "quit", ":q"):
                break

            head, _, rest = line.partition(" ")
            fn = table.get(head.lower())
            if fn:
                try:
                    fn(rest)
                except Exception as e:
                    log.error(f"{head} failed: {e}")
                    print(f"  ! {str(e)[:200]}")
            else:
                # Anything unrecognised is treated as a question, so the common
                # case needs no command at all.
                self.cmd_ask(line)

        print("Bye.")
