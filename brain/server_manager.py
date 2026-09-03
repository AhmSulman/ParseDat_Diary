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
prices the load and REFUSES rather than letting the machine thrash. Only one
server runs at a time.

The price is charged against COMMIT, not free physical RAM — see
memory_status(). Free RAM is not the limit Windows enforces, and guarding on it
believed in ~4.8 GB of headroom that did not exist.
"""

from __future__ import annotations

import glob
import os
import socket
import subprocess
import time

from config.config import Config
from logs.logger import log

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Absolute fallbacks, checked only after the project itself. These are tied to a
# drive letter and so do not survive the project moving; the project-local
# lookup below does.
_CANDIDATE_DIRS = [
    r"E:\llama-b10034-bin-win-cpu-x64",
    r"E:\llama.cpp",
    r"C:\llama.cpp",
]
_EXE = "llama-server.exe" if os.name == "nt" else "llama-server"


def _project_candidates() -> list[str]:
    """
    llama.cpp builds unzipped inside the project, newest-looking first.

    Globbed rather than named so dropping a newer build in needs no code change,
    and reverse-sorted so `llama-b10034-...` wins over `llama-b9000-...`. Binaries
    kept here move with the project and survive a change of drive letter, which
    is why they are preferred over the absolute paths above.
    """
    out = []
    for d in sorted(glob.glob(os.path.join(_ROOT, "llama*")), reverse=True):
        if os.path.isdir(d):
            out.append(d)
    return out


# Commit that must remain available AFTER the model is resident, in MB.
#
# COMMIT, NOT FREE PHYSICAL RAM. Windows refuses an allocation when the commit
# charge reaches the commit limit (RAM + pagefile), and that is what actually
# fires — measured on this machine: 10,853 MB physically free while only
# 6,039 MB of commit remained available, with six Resource-Exhaustion (2004)
# events already logged. Guarding on free RAM believed in 4.8 GB of headroom
# that did not exist, almost exactly the size of a 7B model.
_COMMIT_FLOOR_MB = 2500

# Runtime cost beyond the weights: compute buffers, the graph, the process.
_RUNTIME_OVERHEAD_MB = 600

# KV cache allowance per 8k of context, in MB. Deliberately the WORST case
# measured across the models in use rather than an average: Qwen3-4B carries
# 36 layers x 8 KV heads = 1,152 MB at 8k f16, MORE than the 7B's 448 MB
# despite being half its size. A flat allowance under-counted it by ~800 MB.
# Being conservative means refusing a marginal load, which is the correct
# failure for a machine whose alternative is a frozen desktop.
_KV_WORST_MB_PER_8K = 1200


def find_server_binary() -> str | None:
    """Locate llama-server, preferring an explicit config path."""
    cfg = Config()
    explicit = getattr(cfg, "LLAMA_SERVER_BIN", None)
    if explicit and os.path.exists(explicit):
        return explicit
    for d in _project_candidates() + _CANDIDATE_DIRS:
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


def memory_status() -> dict | None:
    """
    Physical AND commit availability in MB, or None if it cannot be read.

    ullAvailPageFile is Windows' "available commit" despite the name — it is
    commit limit minus commit charge, not free space in pagefile.sys. It is the
    number that decides whether an allocation succeeds, so it is the number the
    guard compares against. The old code read this struct and then returned
    only ullAvailPhys, throwing the useful field away.
    """
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
            mb = 1024 * 1024
            return {
                "phys_free_mb": int(st.ullAvailPhys / mb),
                "commit_free_mb": int(st.ullAvailPageFile / mb),
                "commit_limit_mb": int(st.ullTotalPageFile / mb),
            }
    except Exception:
        pass
    return None


def free_ram_mb() -> int | None:
    """Free physical memory, or None. Kept for callers that only want RAM."""
    st = memory_status()
    return st["phys_free_mb"] if st else None


def free_vram_mb() -> int | None:
    """Free VRAM via nvidia-smi, or None when there is no NVIDIA GPU."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return int(out.stdout.strip().splitlines()[0])
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
    # Seams so the guard can be tested without a real file or a real machine.
    def _exists(self, path: str) -> bool:
        return os.path.exists(path)

    def _size_mb(self, path: str) -> float:
        return os.path.getsize(path) / (1024 * 1024)

    def _cost_mb(self, path: str, ctx: int) -> float:
        """Commit this model will charge: weights + KV for `ctx` + runtime."""
        return (self._size_mb(path)
                + _KV_WORST_MB_PER_8K * (ctx / 8192)
                + _RUNTIME_OVERHEAD_MB)

    def can_load(self, model_path: str, *, ctx: int | None = None,
                 status: dict | None = ...) -> tuple[bool, str]:
        """
        Whether this model fits right now, judged on COMMIT rather than RAM.

        Checked before every start because the failure mode is a frozen desktop,
        not an exception — and a model that "almost" fits still thrashes.

        `status` is injectable for tests; the default reads the live machine.
        A reading that cannot be taken does not block: an unknown is not a no.
        """
        if not self._exists(model_path):
            return False, f"not found: {os.path.basename(model_path)}"

        if ctx is None:
            ctx = Config().LLM_CONTEXT_SIZE
        if status is ...:
            status = memory_status()
        if not status:
            return True, ""                  # cannot tell; do not block

        need_mb = self._cost_mb(model_path, ctx)

        # SWITCHING frees the current model first. Without this, swapping
        # DeepSeek (4.4 GB) for Qwen3 (2.3 GB) is refused for lack of memory
        # that the swap itself would release — every switch would be blocked
        # once one model was loaded.
        reclaim_mb = 0.0
        if self.model_path and self._exists(self.model_path):
            if os.path.abspath(self.model_path) != os.path.abspath(model_path):
                reclaim_mb = self._cost_mb(self.model_path, ctx)

        commit_free = status["commit_free_mb"]
        available = commit_free + reclaim_mb
        if available - need_mb < _COMMIT_FLOOR_MB:
            extra = (f", +{reclaim_mb/1024:.1f} GB freed by unloading "
                     f"{os.path.basename(self.model_path)}" if reclaim_mb else "")
            return False, (
                f"needs ~{need_mb/1024:.1f} GB of commit at ctx={ctx:,}, but only "
                f"{commit_free/1024:.1f} GB is available{extra}. "
                f"Close something, lower the context, or enlarge the pagefile "
                f"(commit limit is {status['commit_limit_mb']/1024:.1f} GB)."
            )
        return True, ""

    def vram_warning(self, model_path: str, n_gpu_layers: int) -> str | None:
        """
        Advisory only — VRAM is not a blocking check.

        We cannot know how much of a model `-ngl N` will actually place on the
        GPU without reading its layer count, so this reports a likely shortfall
        rather than refusing. Free VRAM also moves during a session as the
        desktop grows, so a hard gate here would refuse loads that would work.
        """
        if n_gpu_layers <= 0:
            return None
        free = free_vram_mb()
        if free is None or not self._exists(model_path):
            return None
        size = self._size_mb(model_path)
        if size > free:
            return (f"{os.path.basename(model_path)} is {size/1024:.1f} GB but only "
                    f"{free/1024:.1f} GB of VRAM is free — llama.cpp will keep the "
                    f"remaining layers on the CPU.")
        return None

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

        # Settings are resolved BEFORE the guard runs: the KV cache scales with
        # the context size, so a guard that does not know ctx cannot price the
        # load it is being asked to approve.
        cfg = self.cfg = Config()      # refresh: overlay may have changed since __init__
        ctx = ctx or cfg.LLM_CONTEXT_SIZE
        ngl = cfg.LLM_GPU_LAYERS if n_gpu_layers is None else n_gpu_layers
        threads = threads if threads is not None else cfg.LLM_N_THREADS
        batch = cfg.LLM_N_BATCH

        ok, why = self.can_load(model_path, ctx=ctx)
        if not ok:
            return False, why

        warn = self.vram_warning(model_path, ngl)
        if warn:
            log.warning(warn)

        self.stop()
        if port_open(self.port):
            return False, (f"port {self.port} is already in use by another "
                           f"llama-server. Stop it first.")

        # -ctk/-ctv q8_0 halve the KV cache. At 8k that is 1,152 MB -> 576 MB
        # for Qwen3-4B, whose 36 layers x 8 KV heads make its cache larger than
        # the 7B's despite the model being half the size. The quality cost of an
        # 8-bit KV cache is negligible; the memory is not.
        cmd = [exe, "-m", model_path, "-c", str(ctx), "-ngl", str(ngl),
               "-t", str(threads), "-tb", str(threads), "-b", str(batch),
               "-ctk", "q8_0", "-ctv", "q8_0",
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
