"""
Tests for LibraryService reconciliation.

    _venv\\Scripts\\python.exe -m unittest discover -s tests -v

The classification under test is what made the original failure invisible: a
book "marked done but never indexed" (HOLE) looked identical to one simply not
processed yet (PENDING). Eleven holes went unnoticed for exactly that reason.

Everything runs against a temp tree; the real data/ is never touched.
"""

import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import storage.library as L        # noqa: E402
import storage.manifest as M       # noqa: E402
import storage.checkpoint as C     # noqa: E402
from config.config import Config   # noqa: E402


class LibraryFixture(unittest.TestCase):
    """Redirects every store into a temp tree."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="parsedat_lib_")
        self.input_dir = os.path.join(self.tmp, "input")
        self.txt_dir = os.path.join(self.tmp, "txt")
        self.cache = os.path.join(self.tmp, "cache")
        for d in (self.input_dir, self.txt_dir, self.cache):
            os.makedirs(d, exist_ok=True)

        self._saved = {
            "input": Config.INPUT_DIR,
            "txt": L._TXT, "quar": L._QUAR, "cats": L._CATEGORIES,
            "idx": L.INDEX_FILE, "meta": L.META_FILE, "man": L.MANIFEST_FILE,
            "m_file": M.MANIFEST_FILE, "c_state": C.STATE_FILE, "c_dir": C._CKPT_DIR,
        }
        Config.INPUT_DIR = self.input_dir
        L._TXT = self.txt_dir
        L._QUAR = os.path.join(self.tmp, "quarantine")
        L._CATEGORIES = os.path.join(self.tmp, "categories.json")
        L.INDEX_FILE = os.path.join(self.cache, "parsedat.index")
        L.META_FILE = os.path.join(self.cache, "parsedat_meta.json")
        L.MANIFEST_FILE = M.MANIFEST_FILE = os.path.join(self.cache, "manifest.json")
        C._CKPT_DIR = os.path.join(self.tmp, "checkpoints")
        C.STATE_FILE = os.path.join(C._CKPT_DIR, "state.json")

    def tearDown(self):
        Config.INPUT_DIR = self._saved["input"]
        L._TXT, L._QUAR, L._CATEGORIES = (
            self._saved["txt"], self._saved["quar"], self._saved["cats"])
        L.INDEX_FILE, L.META_FILE, L.MANIFEST_FILE = (
            self._saved["idx"], self._saved["meta"], self._saved["man"])
        M.MANIFEST_FILE = self._saved["m_file"]
        C.STATE_FILE, C._CKPT_DIR = self._saved["c_state"], self._saved["c_dir"]
        shutil.rmtree(self.tmp, ignore_errors=True)

    # helpers
    def add_pdf(self, name):
        with open(os.path.join(self.input_dir, name), "wb") as f:
            f.write(b"%PDF-1.4 fake")

    def add_txt(self, name):
        stem = os.path.splitext(name)[0]
        with open(os.path.join(self.txt_dir, f"{stem}.txt"), "w", encoding="utf-8") as f:
            f.write("text")

    def index_book(self, svc, name, chunks=10):
        svc.manifest.add_book(name, book_id=0, n_chunks=chunks)
        svc.manifest.save()

    def mark_done(self, svc, name):
        svc.checkpoint.mark_done(name)


class TestStatusClassification(LibraryFixture):

    def test_indexed_book_is_healthy(self):
        self.add_pdf("a.pdf")
        svc = L.LibraryService()
        self.index_book(svc, "a.pdf")
        self.mark_done(svc, "a.pdf")
        rep = L.LibraryService().report()
        self.assertEqual(rep["books"][0]["status"], L.INDEXED)
        self.assertTrue(rep["healthy"])
        self.assertEqual(rep["drift"], 0)

    def test_marked_done_but_not_indexed_is_a_HOLE(self):
        """The original silent failure: 11 books in this state, invisible."""
        self.add_pdf("a.pdf")
        svc = L.LibraryService()
        self.mark_done(svc, "a.pdf")
        rep = L.LibraryService().report()
        self.assertEqual(rep["books"][0]["status"], L.HOLE)
        self.assertIn("a.pdf", rep["holes"])
        self.assertFalse(rep["healthy"])

    def test_unprocessed_book_is_PENDING_not_a_hole(self):
        """PENDING and HOLE must not be conflated — different fixes."""
        self.add_pdf("a.pdf")
        rep = L.LibraryService().report()
        self.assertEqual(rep["books"][0]["status"], L.PENDING)
        self.assertEqual(rep["holes"], [])

    def test_indexed_book_with_deleted_pdf_is_an_ORPHAN(self):
        self.add_pdf("a.pdf")
        svc = L.LibraryService()
        self.index_book(svc, "a.pdf")
        os.remove(os.path.join(self.input_dir, "a.pdf"))
        rep = L.LibraryService().report()
        self.assertEqual([b["status"] for b in rep["books"]], [L.ORPHAN])
        self.assertIn("a.pdf", rep["orphans"]["index"])
        self.assertGreater(rep["drift"], 0)

    def test_counts_reflect_each_store_independently(self):
        self.add_pdf("a.pdf")
        self.add_pdf("b.pdf")
        self.add_txt("a.pdf")
        svc = L.LibraryService()
        self.index_book(svc, "a.pdf", chunks=7)
        self.mark_done(svc, "a.pdf")
        c = L.LibraryService().report()["counts"]
        self.assertEqual(c["pdfs_on_disk"], 2)
        self.assertEqual(c["books_indexed"], 1)
        self.assertEqual(c["chunks_indexed"], 7)
        self.assertEqual(c["checkpoint_done"], 1)
        self.assertEqual(c["txt_files"], 1)


class TestOrphanPruning(LibraryFixture):

    def test_sync_removes_every_trace_of_a_deleted_book(self):
        self.add_pdf("gone.pdf")
        self.add_txt("gone.pdf")
        svc = L.LibraryService()
        self.index_book(svc, "gone.pdf")
        self.mark_done(svc, "gone.pdf")

        os.remove(os.path.join(self.input_dir, "gone.pdf"))
        L.LibraryService().clean(orphans=True)

        rep = L.LibraryService().report()
        self.assertEqual(rep["counts"]["books_indexed"], 0)
        self.assertEqual(rep["counts"]["checkpoint_done"], 0)
        self.assertEqual(rep["counts"]["txt_files"], 0, "orphan .txt must go")
        self.assertEqual(rep["drift"], 0)

    def test_orphan_txt_is_removed_so_reindex_cannot_resurrect(self):
        """
        The resurrection trap. reindex reads extracted text; leaving an orphan
        .txt behind would re-index a book that was just purged.
        """
        self.add_pdf("keep.pdf")
        self.add_txt("keep.pdf")
        self.add_txt("deleted.pdf")          # no matching PDF
        L.LibraryService().clean(orphans=True)
        remaining = os.listdir(self.txt_dir)
        self.assertIn("keep.txt", remaining)
        self.assertNotIn("deleted.txt", remaining)

    def test_sync_never_touches_the_pdfs_themselves(self):
        self.add_pdf("a.pdf")
        self.add_pdf("b.pdf")
        L.LibraryService().clean(orphans=True)
        self.assertEqual(sorted(os.listdir(self.input_dir)), ["a.pdf", "b.pdf"])

    def test_sync_is_idempotent(self):
        self.add_pdf("a.pdf")
        self.add_txt("a.pdf")
        L.LibraryService().clean(orphans=True)
        first = L.LibraryService().report()["counts"]
        L.LibraryService().clean(orphans=True)
        self.assertEqual(L.LibraryService().report()["counts"], first)

    def test_purge_book_clears_manifest_checkpoint_and_text(self):
        self.add_pdf("a.pdf")
        self.add_txt("a.pdf")
        svc = L.LibraryService()
        self.index_book(svc, "a.pdf")
        self.mark_done(svc, "a.pdf")

        self.assertTrue(L.LibraryService().purge_book("a.pdf"))
        rep = L.LibraryService().report()
        self.assertEqual(rep["counts"]["books_indexed"], 0)
        self.assertEqual(rep["counts"]["checkpoint_done"], 0)
        self.assertFalse(os.path.exists(os.path.join(self.txt_dir, "a.txt")))
        # the PDF itself survives — purge is about derived state
        self.assertTrue(os.path.exists(os.path.join(self.input_dir, "a.pdf")))


class TestCleanScopes(LibraryFixture):

    def test_clean_index_drops_manifest_but_keeps_text(self):
        self.add_pdf("a.pdf")
        self.add_txt("a.pdf")
        svc = L.LibraryService()
        self.index_book(svc, "a.pdf")

        L.LibraryService().clean(index=True)
        rep = L.LibraryService().report()
        self.assertEqual(rep["counts"]["books_indexed"], 0)
        self.assertEqual(rep["counts"]["txt_files"], 1, "text is the reindex cache")

    def test_clean_checkpoint_only_resets_checkpoint(self):
        self.add_pdf("a.pdf")
        svc = L.LibraryService()
        self.index_book(svc, "a.pdf")
        self.mark_done(svc, "a.pdf")

        L.LibraryService().clean(checkpoint=True)
        rep = L.LibraryService().report()
        self.assertEqual(rep["counts"]["checkpoint_done"], 0)
        self.assertEqual(rep["counts"]["books_indexed"], 1)

    def test_clean_with_no_scope_changes_nothing(self):
        self.add_pdf("a.pdf")
        self.add_txt("a.pdf")
        before = L.LibraryService().report()["counts"]
        L.LibraryService().clean()
        self.assertEqual(L.LibraryService().report()["counts"], before)


class TestManifestRobustness(LibraryFixture):

    def test_corrupt_manifest_reads_as_empty_not_raises(self):
        self.add_pdf("a.pdf")
        os.makedirs(os.path.dirname(M.MANIFEST_FILE), exist_ok=True)
        with open(M.MANIFEST_FILE, "w", encoding="utf-8") as f:
            f.write("{ this is not json")
        rep = L.LibraryService().report()          # must not raise
        self.assertEqual(rep["counts"]["books_indexed"], 0)

    def test_manifest_survives_a_non_dict_payload(self):
        os.makedirs(os.path.dirname(M.MANIFEST_FILE), exist_ok=True)
        with open(M.MANIFEST_FILE, "w", encoding="utf-8") as f:
            f.write("[1, 2, 3]")
        self.assertEqual(M.Manifest().book_count, 0)

    def test_dimension_mismatch_is_reported_not_crashed(self):
        m = M.Manifest()
        m.set_settings("model-a", 768, 1200, 200)
        ok, why = m.is_compatible("model-a", 384)
        self.assertFalse(ok)
        self.assertIn("reindex", why)

    def test_model_mismatch_is_reported(self):
        m = M.Manifest()
        m.set_settings("model-a", 768, 1200, 200)
        ok, why = m.is_compatible("model-b", 768)
        self.assertFalse(ok)

    def test_empty_manifest_is_compatible_with_anything(self):
        ok, _ = M.Manifest().is_compatible("anything", 999)
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main(verbosity=2)
