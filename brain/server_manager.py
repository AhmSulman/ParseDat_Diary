"""
llama-server process manager — pick a model and actually load it
=================================================================
Generation runs through llama-server, so "the model" is whatever that process
was started with. Setting llm.model_path does nothing: the GUI's model picker
looked like it worked and silently changed nothing.

This starts, stops and swaps that process, so choosing a model in the UI has a
real effect.

MEMORY IS GUARDED, NOT ASSUMED
------------------------------
Loading a 7B here twice caused near-freezes: Windows keeps mmap'd model pages
cached after a process exits, so loads accumulate. Every start() therefore
checks free RAM against the model's size plus KV headroom and REFUSES rather
than letting the machine thrash. Only one server runs at a time.
"""

from __future__ import annotations

import os
import socket
import subprocess
import time

from config.config import Config
from logs.logger import log

# Where a llama.cpp binary release may live. First hit wins.
_CANDIDATE_DIRS = [
    r"E:\llama-b10034-bin-win-cpu-x64",
    r"E:\llama.cpp",
    r"C:\llama.cpp",
]
_EXE = "llama-server.exe" if os.name == "nt" else "llama-server"

# Free RAM that must remain AFTER the model is resident, in MB. Below this
# Windows starts thrashing and the UI stops responding.
_RAM_FLOOR_MB = 2500


def find_server_binary() -> str | None:
    """Locate llama-server, preferring an explicit config path."""
    cfg = Config()
    explicit = getattr(cfg, "LLAMA_SERVER_BIN", None)
    if explicit and os.path.exists(explicit):
        return explicit
    for d in _CANDIDATE_DIRS:
        p = os.path.join(d, _EXE)
        if os.path.exists(p):
            return p
    from shutil import which
    return which("llama-server")


def port_open(port: int, host: str = "127.0.0.1", timeout: float = 1.5) -> bool:
    s = socket.socket()
    s.settimeout(timeout)
    try:
        s.connect((host, port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def free_ram_mb() -> int | None:
    """Free physical memory, or None if it cannot be determined."""
    try:
        import ctypes

        class _MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        st = _MS()
        st.dwLength = ctypes.sizeof(_MS)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return int(st.ullAvailPhys / (1024 * 1024))
    except Exception:
        pass
    return None


class ServerManager:
    """Owns at most one llama-server process."""

    def __init__(self):
        self.cfg = Config()
        self.proc: subprocess.Popen | None = None
        self.model_path: str | None = None
        self.port: int = self._port_from_url(self.cfg.LLM_SERVER_URL)

    @staticmethod
    def _port_from_url(url: str) -> int:
        try:
            return int(url.rsplit(":", 1)[1].split("/")[0])
        except (IndexError, ValueError):
            return 8084

    # ── state ─────────────────────────────────────────────────────────────────
    def is_running(self) -> bool:
        return port_open(self.port)

    def running_model(self) -> str | None:
        """Ask the server which model it holds — it may not be ours."""
        if not self.is_running():
            return None
        try:
            import httpx
            with httpx.Client(timeout=5.0) as c:
                data = c.get(f"http://127.0.0.1:{self.port}/v1/models").json()
            items = data.get("data") or []
            if items:
                return os.path.basename(str(items[0].get("id", "")).replace("\\", "/"))
        except Exception:
            pass
        return None

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def can_load(self, model_path: str) -> tuple[bool, str]:
        """
        Whether this model fits in RAM right now.

        Checked before every start because the failure mode is a frozen desktop,
        not an exception — and a model that "almost" fits still thrashes.
        """
        if not os.path.exists(model_path):
            return False, f"not found: {os.path.basename(model_path)}"
        need_mb = os.path.getsize(model_path) / (1024 * 1024)
        need_mb += 700                       # KV cache + runtime overhead
        free = free_ram_mb()
        if free is None:
            return True, ""                  # cannot tell; do not block

        # SWITCHING frees the current model first. Without this, swapping
        # DeepSeek (4.4 GB) for Qwen3 (2.3 GB) is refused for lack of memory
        # that the swap itself would release — every switch would be blocked
        # once one model was loaded.
        reclaim_mb = 0.0
        if self.model_path and os.path.exists(self.model_path):
            if os.path.abspath(self.model_path) != os.path.abspath(model_path):
                reclaim_mb = os.path.getsize(self.model_path) / (1024 * 1024) + 700

        available = free + reclaim_mb
        if available - need_mb < _RAM_FLOOR_MB:
            extra = (f" (+{reclaim_mb/1024:.1f} GB freed by unloading "
                     f"{os.path.basename(self.model_path)})" if reclaim_mb else "")
            return False, (
                f"needs ~{need_mb/1024:.1f} GB, only {free/1024:.1f} GB free"
                f"{extra}; close something first"
            )
        return True, ""

    def stop(self, timeout: float = 10.0) -> None:
        """Stop the server we started. Frees its RAM."""
        if self.proc and self.proc.poll() is None:
            log.info("Stopping llama-server…")
            self.proc.terminate()
            try:
                self.proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None
        self.model_path = None

    def start(self, model_path: str, *, ctx: int | None = None,
              n_gpu_layers: int | None = None, threads: int | None = None,
              wait: float = 240.0) -> tuple[bool, str]:
        """
        Start llama-server on `model_path`, replacing any server we own.

        Returns (ok, message). Never raises on a bad model — the caller is
        usually a UI callback.

        Reads a FRESH Config() for any of ctx/n_gpu_layers/threads/batch the
        caller leaves unset, rather than self.cfg captured at __init__. This
        manager is a process-wide singleton (get_manager()) that can live for
        the whole GUI session, so self.cfg would otherwise still hold whatever
        was configured when the app started — a Settings-screen change would
        save correctly but a "restart server" would silently relaunch with the
        stale values.
        """
        exe = find_server_binary()
        if not exe:
            return False, ("llama-server not found. Set LLAMA_SERVER_BIN in "
                           "the Settings screen (or config/config.py).")

        ok, why = self.can_load(model_path)
        if not ok:
            return False, why

        self.stop()
        if port_open(self.port):
            return False, (f"port {self.port} is already in use by another "
                           f"llama-server. Stop it first.")

        cfg = self.cfg = Config()      # refresh: overlay may have changed since __init__
        ctx = ctx or cfg.LLM_CONTEXT_SIZE
        ngl = cfg.LLM_GPU_LAYERS if n_gpu_layers is None else n_gpu_layers
        threads = threads if threads is not None else cfg.LLM_N_THREADS
        batch = cfg.LLM_N_BATCH

        cmd = [exe, "-m", model_path, "-c", str(ctx), "-ngl", str(ngl),
               "-t", str(threads), "-tb", str(threads), "-b", str(batch),
               "--port", str(self.port), "--host", "127.0.0.1"]

        log_path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "logs", "llama-server.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        log.info(f"Starting llama-server: {os.path.basename(model_path)} "
                 f"(ctx={ctx}, ngl={ngl})")
        try:
            fh = open(log_path, "w", encoding="utf-8", errors="replace")
            self.proc = subprocess.Popen(
                cmd, stdout=fh, stderr=subprocess.STDOUT,
                cwd=os.path.dirname(exe),      # the exe needs its sibling DLLs
            )
        except Exception as e:
            return False, f"could not launch: {str(e)[:160]}"

        deadline = time.time() + wait
        while time.time() < deadline:
            if self.proc.poll() is not None:
                return False, f"server exited during load — see {log_path}"
            if port_open(self.port):
                self.model_path = model_path
                return True, f"{os.path.basename(model_path)} loaded on port {self.port}"
            time.sleep(1.0)

        self.stop()
        return False, f"timed out after {wait:.0f}s loading {os.path.basename(model_path)}"


_manager: ServerManager | None = None


def get_manager() -> ServerManager:
    """Process-wide manager, so two UI screens cannot start two servers."""
    global _manager
    if _manager is None:
        _manager = ServerManager()
    return _manager
