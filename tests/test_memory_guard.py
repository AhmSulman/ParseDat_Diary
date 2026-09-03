"""
Tests for the llama-server memory guard.

    _venv\\Scripts\\python.exe -m unittest discover -s tests -v

The guard exists because the failure mode is a frozen desktop, not an
exception. It previously compared against FREE PHYSICAL RAM, which is the
wrong number: Windows refuses an allocation when COMMIT is exhausted, and
commit is RAM plus pagefile. On the machine this was found on, physical free
read 10,853 MB while available commit was 6,039 MB — the guard believed in
4.8 GB of headroom that did not exist, almost exactly the size of a 7B model.
Six Resource-Exhaustion (2004) events had been logged.

No model is loaded and no process is started here: `can_load` takes an
injectable memory reading.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import brain.server_manager as sm  # noqa: E402


def status(phys, commit, limit=27816):
    return {"phys_free_mb": phys, "commit_free_mb": commit, "commit_limit_mb": limit}


class _Fake(sm.ServerManager):
    """ServerManager that never touches the filesystem or a real process."""

    def __init__(self, sizes: dict, loaded: str | None = None):
        self._sizes = sizes
        self.proc = None
        self.model_path = loaded
        self.port = 8084

    def _size_mb(self, path):
        return self._sizes[path]

    def _exists(self, path):
        return path in self._sizes


class TestCommitIsTheBindingConstraint(unittest.TestCase):
    def test_refuses_when_commit_is_short_despite_ample_physical_ram(self):
        """The exact bug: plenty of free RAM, not enough commit."""
        m = _Fake({"7b.gguf": 4466})
        ok, why = m.can_load("7b.gguf", ctx=8192,
                             status=status(phys=10853, commit=6039))
        self.assertFalse(ok)
        self.assertIn("commit", why.lower())

    def test_allows_when_commit_is_ample(self):
        m = _Fake({"7b.gguf": 4466})
        ok, why = m.can_load("7b.gguf", ctx=8192,
                             status=status(phys=10853, commit=20000))
        self.assertTrue(ok, why)

    def test_physical_ram_alone_is_not_enough_to_pass(self):
        """A reading that would have passed the old guard must now fail."""
        m = _Fake({"7b.gguf": 4466})
        # Old guard: 4466 + 700 = 5166 needed, 10853 free -> passed.
        ok, _ = m.can_load("7b.gguf", ctx=8192,
                           status=status(phys=10853, commit=5200))
        self.assertFalse(ok)


class TestKVAllowance(unittest.TestCase):
    def test_allowance_scales_with_context(self):
        """Qwen3-4B needs 1,152 MB of KV at 8k f16 - the flat 700 MB was short."""
        m = _Fake({"4b.gguf": 2382})
        tight = status(phys=20000, commit=6000)
        ok_small, _ = m.can_load("4b.gguf", ctx=2048, status=tight)
        ok_large, _ = m.can_load("4b.gguf", ctx=32768, status=tight)
        self.assertTrue(ok_small)
        self.assertFalse(ok_large, "a 32k context must cost more than a 2k one")

    def test_reports_the_number_it_used(self):
        m = _Fake({"4b.gguf": 2382})
        _, why = m.can_load("4b.gguf", ctx=8192, status=status(20000, 1000))
        self.assertIn("GB", why)


class TestSwitching(unittest.TestCase):
    def test_switching_counts_the_memory_the_swap_will_release(self):
        """Without this, every switch is refused once one model is loaded."""
        m = _Fake({"7b.gguf": 4466, "4b.gguf": 2382}, loaded="7b.gguf")
        ok, why = m.can_load("4b.gguf", ctx=8192,
                             status=status(phys=8000, commit=4000))
        self.assertTrue(ok, why)

    def test_reloading_the_same_model_reclaims_nothing_twice(self):
        m = _Fake({"7b.gguf": 4466}, loaded="7b.gguf")
        ok, _ = m.can_load("7b.gguf", ctx=8192,
                           status=status(phys=8000, commit=3000))
        self.assertFalse(ok)


class TestDegradesSafely(unittest.TestCase):
    def test_unknown_memory_does_not_block(self):
        m = _Fake({"7b.gguf": 4466})
        ok, _ = m.can_load("7b.gguf", ctx=8192, status=None)
        self.assertTrue(ok)

    def test_missing_file_is_refused(self):
        m = _Fake({})
        ok, why = m.can_load("nope.gguf", ctx=8192, status=status(20000, 20000))
        self.assertFalse(ok)
        self.assertIn("not found", why)


class TestRealReadingIsWellFormed(unittest.TestCase):
    def test_memory_status_returns_commit_fields(self):
        st = sm.memory_status()
        if st is None:
            self.skipTest("memory status unavailable on this platform")
        for key in ("phys_free_mb", "commit_free_mb", "commit_limit_mb"):
            self.assertIn(key, st)
            self.assertGreater(st[key], 0)
        # Commit available can never exceed the commit limit.
        self.assertLessEqual(st["commit_free_mb"], st["commit_limit_mb"])

class TestRelativeModelPaths(unittest.TestCase):
    """
    start() launches the subprocess with cwd set to the llama.cpp folder,
    because the exe needs its sibling DLLs. A relative model path therefore
    resolves against THAT directory, not the project — the server exits with
    "failed to open GGUF file ... No such file or directory" for a file that
    plainly exists, while can_load() (which checks from this process's cwd)
    passes. Observed on a real load, 2026-09-04.
    """

    def test_start_makes_the_model_path_absolute(self):
        captured = {}

        class _M(sm.ServerManager):
            def __init__(self):
                self.proc = None
                self.model_path = None
                self.port = 8084

            def can_load(self, model_path, *, ctx=None, status=...):
                captured["seen_by_guard"] = model_path
                return False, "stop here"       # halt before launching anything

        m = _M()
        real = sm.find_server_binary()
        if not real:
            self.skipTest("no llama-server binary on this machine")
        m.start("data/models/some-model.gguf", ctx=8192)
        self.assertTrue(os.path.isabs(captured["seen_by_guard"]),
                        f"path reached the guard still relative: {captured['seen_by_guard']}")

if __name__ == "__main__":
    unittest.main()
