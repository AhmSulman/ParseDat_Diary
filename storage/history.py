"""
Conversation history — sessions that survive the app closing
=============================================================
Every answer used to overwrite the last one and vanish on exit: no transcript,
no way to start a fresh conversation, no record of anything asked before.

This stores conversations as sessions on disk. Each holds an ordered list of
turns, and each assistant turn keeps the citations it produced, so an answer
stays checkable against the exact pages it came from long after it scrolled off.

One file per session under data/sessions/, plus an index for listing. Per-file
rather than one big blob so a corrupt session loses one conversation instead of
the whole history, and so appending a turn never rewrites unrelated ones.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESSIONS_DIR = os.path.join(_ROOT, "data", "sessions")
INDEX_FILE = os.path.join(SESSIONS_DIR, "_index.json")

_TITLE_MAX = 60


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_write(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _read(path: str) -> dict:
    """Never raises. A corrupt or missing file reads as empty."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}


class History:
    """Sessions of question/answer turns, persisted to disk."""

    def __init__(self, sessions_dir: str | None = None):
        self.dir = sessions_dir or SESSIONS_DIR
        os.makedirs(self.dir, exist_ok=True)
        self.session_id: str | None = None

    # ── paths ─────────────────────────────────────────────────────────────────
    def _path(self, session_id: str) -> str:
        # Session ids are generated here, never user input, but keep the join
        # safe anyway so a crafted id cannot escape the directory.
        safe = "".join(c for c in session_id if c.isalnum() or c in "-_")
        return os.path.join(self.dir, f"{safe}.json")

    def _index_path(self) -> str:
        return os.path.join(self.dir, "_index.json")

    # ── sessions ──────────────────────────────────────────────────────────────
    def new_session(self, title: str | None = None) -> str:
        sid = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
        payload = {
            "id": sid,
            "title": (title or "New conversation")[:_TITLE_MAX],
            "created_at": _now(),
            "turns": [],
        }
        _atomic_write(self._path(sid), payload)
        self.session_id = sid
        self._reindex()
        return sid

    def ensure_session(self) -> str:
        """Current session, creating one on first use."""
        if self.session_id is None:
            self.new_session()
        return self.session_id  # type: ignore[return-value]

    def load_session(self, session_id: str) -> dict:
        return _read(self._path(session_id))

    def set_current(self, session_id: str) -> bool:
        if os.path.exists(self._path(session_id)):
            self.session_id = session_id
            return True
        return False

    def delete_session(self, session_id: str) -> bool:
        p = self._path(session_id)
        if not os.path.exists(p):
            return False
        os.remove(p)
        if self.session_id == session_id:
            self.session_id = None
        self._reindex()
        return True

    # ── turns ─────────────────────────────────────────────────────────────────
    def add_turn(self, question: str, answer: str, *,
                 citations: list[dict] | None = None,
                 elapsed: float | None = None,
                 model: str | None = None) -> None:
        sid = self.ensure_session()
        data = self.load_session(sid)
        if not data:
            return
        data.setdefault("turns", []).append({
            "at": _now(),
            "question": question,
            "answer": answer,
            "citations": citations or [],
            "elapsed_s": round(elapsed, 1) if elapsed else None,
            "model": model,
        })
        # First question names the conversation, so the list is browsable
        # without opening anything.
        if data.get("title") in (None, "", "New conversation"):
            data["title"] = question.strip().replace("\n", " ")[:_TITLE_MAX] or "Untitled"
        _atomic_write(self._path(sid), data)
        self._reindex()

    def turns(self, session_id: str | None = None) -> list[dict]:
        sid = session_id or self.session_id
        if not sid:
            return []
        return self.load_session(sid).get("turns", [])

    def transcript(self, session_id: str | None = None, limit: int | None = None) -> str:
        """The conversation as readable text, for display or export."""
        rows = self.turns(session_id)
        if limit:
            rows = rows[-limit:]
        out = []
        for t in rows:
            out.append(f"You: {t['question']}")
            out.append(t["answer"].strip())
            if t.get("citations"):
                cites = ", ".join(
                    f"[{c.get('n')}] {c.get('source', '?')}"
                    + (f" p.{c['page_start']}" if c.get("page_start") else "")
                    for c in t["citations"]
                )
                out.append(f"    sources: {cites}")
            out.append("")
        return "\n".join(out)

    # ── listing ───────────────────────────────────────────────────────────────
    def _reindex(self) -> None:
        """Rebuild the listing from the session files themselves."""
        rows = []
        for fn in os.listdir(self.dir):
            if not fn.endswith(".json") or fn.startswith("_"):
                continue
            d = _read(os.path.join(self.dir, fn))
            if not d.get("id"):
                continue
            rows.append({
                "id": d["id"],
                "title": d.get("title", "Untitled"),
                "created_at": d.get("created_at", ""),
                "n_turns": len(d.get("turns", [])),
            })
        rows.sort(key=lambda r: r["created_at"], reverse=True)
        _atomic_write(self._index_path(), {"sessions": rows})

    def sessions(self) -> list[dict]:
        idx = _read(self._index_path()).get("sessions")
        if isinstance(idx, list):
            return idx
        self._reindex()
        return _read(self._index_path()).get("sessions", [])

    def latest_session_id(self) -> str | None:
        s = self.sessions()
        return s[0]["id"] if s else None
