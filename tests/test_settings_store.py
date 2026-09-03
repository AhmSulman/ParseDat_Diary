"""
Tests for the settings overlay (config/settings_store.py).

    _venv\\Scripts\\python.exe -m unittest discover -s tests -v

Everything runs against a temp file; the real data/settings.json is never
touched.
"""

import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config.settings_store as S  # noqa: E402


class SettingsFixture(unittest.TestCase):
    """Redirects the overlay file into a temp tree and resets the mtime cache."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="parsedat_settings_")
        self._saved_file = S.SETTINGS_FILE
        S.SETTINGS_FILE = os.path.join(self.tmp, "settings.json")
        S._cache = None

    def tearDown(self):
        S.SETTINGS_FILE = self._saved_file
        S._cache = None
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestOverlayRoundTrip(SettingsFixture):

    def test_missing_file_reads_as_empty(self):
        self.assertEqual(S.load_overlay(), {})

    def test_save_then_load_round_trips(self):
        S.save_overlay({"LLM_GPU_LAYERS": 20, "SEARCH_TOP_K": 8})
        self.assertEqual(S.load_overlay(), {"LLM_GPU_LAYERS": 20, "SEARCH_TOP_K": 8})

    def test_update_overlay_merges_rather_than_replaces(self):
        S.save_overlay({"LLM_GPU_LAYERS": 20})
        S.update_overlay({"SEARCH_TOP_K": 8})
        self.assertEqual(S.load_overlay(),
                          {"LLM_GPU_LAYERS": 20, "SEARCH_TOP_K": 8})

    def test_update_overlay_overwrites_an_existing_key(self):
        S.save_overlay({"LLM_GPU_LAYERS": 20})
        S.update_overlay({"LLM_GPU_LAYERS": 0})
        self.assertEqual(S.load_overlay(), {"LLM_GPU_LAYERS": 0})

    def test_clear_overlay_drops_everything(self):
        S.save_overlay({"LLM_GPU_LAYERS": 20})
        S.clear_overlay()
        self.assertEqual(S.load_overlay(), {})

    def test_corrupt_file_reads_as_empty_not_raises(self):
        with open(S.SETTINGS_FILE, "w", encoding="utf-8") as f:
            f.write("{not json")
        self.assertEqual(S.load_overlay(), {})

    def test_non_dict_json_reads_as_empty(self):
        with open(S.SETTINGS_FILE, "w", encoding="utf-8") as f:
            f.write("[1, 2, 3]")
        self.assertEqual(S.load_overlay(), {})

    def test_write_is_atomic_no_tmp_file_left_behind(self):
        S.save_overlay({"LLM_GPU_LAYERS": 20})
        self.assertFalse(os.path.exists(S.SETTINGS_FILE + ".tmp"))
        self.assertTrue(os.path.exists(S.SETTINGS_FILE))

    def test_a_save_from_this_process_is_visible_immediately(self):
        S.save_overlay({"LLM_GPU_LAYERS": 20})
        self.assertEqual(S.load_overlay()["LLM_GPU_LAYERS"], 20)
        S.save_overlay({"LLM_GPU_LAYERS": 40})
        self.assertEqual(S.load_overlay()["LLM_GPU_LAYERS"], 40)

    def test_load_overlay_reads_disk_at_most_once_per_process(self):
        # The whole point of the cache: Config() is built constantly (once
        # per RAG query, per Chunker/Retriever/Embedder), so repeated calls
        # here must not touch disk again.
        S.save_overlay({"LLM_GPU_LAYERS": 20})
        S._cache = None  # force one real read, like a fresh process would
        calls = []
        real_read = S._read_raw
        S._read_raw = lambda: (calls.append(1) or real_read())
        try:
            for _ in range(50):
                S.load_overlay()
        finally:
            S._read_raw = real_read
        self.assertEqual(len(calls), 1)

    def test_invalidate_cache_forces_a_fresh_read(self):
        S.save_overlay({"LLM_GPU_LAYERS": 20})
        S.load_overlay()
        # Simulate another process editing the file directly on disk.
        with open(S.SETTINGS_FILE, "w", encoding="utf-8") as f:
            import json
            json.dump({"LLM_GPU_LAYERS": 99}, f)
        self.assertEqual(S.load_overlay()["LLM_GPU_LAYERS"], 20)  # still cached
        S.invalidate_cache()
        self.assertEqual(S.load_overlay()["LLM_GPU_LAYERS"], 99)  # re-read


class TestFieldValidation(unittest.TestCase):

    def test_int_in_range_is_accepted(self):
        ok, err, value = S.validate("LLM_GPU_LAYERS", "20")
        self.assertTrue(ok)
        self.assertEqual(value, 20)

    def test_int_above_max_is_rejected(self):
        ok, err, value = S.validate("LLM_GPU_LAYERS", "500")
        self.assertFalse(ok)
        self.assertIsNone(value)

    def test_int_below_min_is_rejected(self):
        ok, err, value = S.validate("LLM_GPU_LAYERS", "-1")
        self.assertFalse(ok)

    def test_non_numeric_is_rejected(self):
        ok, err, value = S.validate("LLM_GPU_LAYERS", "lots")
        self.assertFalse(ok)

    def test_float_field_coerces(self):
        ok, err, value = S.validate("LLM_TEMPERATURE", "0.9")
        self.assertTrue(ok)
        self.assertEqual(value, 0.9)

    def test_choice_field_accepts_listed_value(self):
        ok, err, value = S.validate("LLM_BACKEND", "server")
        self.assertTrue(ok)
        self.assertEqual(value, "server")

    def test_choice_field_rejects_unlisted_value(self):
        ok, err, value = S.validate("LLM_BACKEND", "cloud")
        self.assertFalse(ok)

    def test_negative_one_is_a_valid_max_tokens_sentinel(self):
        ok, err, value = S.validate("LLM_MAX_TOKENS", "-1")
        self.assertTrue(ok)

    def test_unknown_key_is_rejected(self):
        ok, err, value = S.validate("NOT_A_REAL_SETTING", "1")
        self.assertFalse(ok)

    def test_string_field_is_trimmed(self):
        ok, err, value = S.validate("LLAMA_SERVER_BIN", "  C:\\bin\\llama-server.exe  ")
        self.assertTrue(ok)
        self.assertEqual(value, "C:\\bin\\llama-server.exe")


class TestCrossFieldValidation(unittest.TestCase):

    def test_overlap_over_half_of_size_is_rejected(self):
        errors = S.validate_patch(
            {"CHUNK_OVERLAP": 700}, {"CHUNK_SIZE": 1200, "CHUNK_OVERLAP": 200})
        self.assertTrue(errors)

    def test_overlap_under_half_of_size_is_accepted(self):
        errors = S.validate_patch(
            {"CHUNK_OVERLAP": 200}, {"CHUNK_SIZE": 1200, "CHUNK_OVERLAP": 200})
        self.assertFalse(errors)

    def test_checks_against_current_when_only_one_side_changes(self):
        # Only CHUNK_SIZE is in the patch; CHUNK_OVERLAP must come from `current`.
        errors = S.validate_patch(
            {"CHUNK_SIZE": 300}, {"CHUNK_SIZE": 1200, "CHUNK_OVERLAP": 200})
        self.assertTrue(errors)  # 200 >= 300 // 2


class TestRestartLabels(unittest.TestCase):

    def test_groups_changed_keys_by_restart_requirement(self):
        grouped = S.restart_labels(["LLM_GPU_LAYERS", "SEARCH_TOP_K", "CHUNK_SIZE"])
        self.assertEqual(set(grouped["server"]), {"LLM_GPU_LAYERS"})
        self.assertEqual(set(grouped["none"]), {"SEARCH_TOP_K"})
        self.assertEqual(set(grouped["reindex"]), {"CHUNK_SIZE"})

    def test_unknown_key_defaults_to_app_restart(self):
        grouped = S.restart_labels(["NOT_A_REAL_SETTING"])
        self.assertEqual(grouped["app"], ["NOT_A_REAL_SETTING"])


class TestConfigOverlayLayering(SettingsFixture):
    """Config() itself, layered with an overlay pointed at the temp file."""

    def setUp(self):
        super().setUp()
        import config.config as C
        self.C = C

    def test_no_overlay_uses_class_default(self):
        cfg = self.C.Config()
        self.assertEqual(cfg.LLM_GPU_LAYERS, self.C.Config.LLM_GPU_LAYERS)

    def test_overlay_value_overrides_default(self):
        S.save_overlay({"LLM_GPU_LAYERS": 7})
        cfg = self.C.Config()
        self.assertEqual(cfg.LLM_GPU_LAYERS, 7)

    def test_class_default_is_untouched_by_an_instance_override(self):
        S.save_overlay({"LLM_GPU_LAYERS": 7})
        self.C.Config()
        self.assertEqual(self.C.Config.LLM_GPU_LAYERS, 35)

    def test_unknown_overlay_key_is_ignored_not_raised(self):
        S.save_overlay({"NOT_A_REAL_FIELD": 1})
        cfg = self.C.Config()  # must not raise
        self.assertFalse(hasattr(cfg, "NOT_A_REAL_FIELD"))

    def test_direct_class_mutation_still_works_for_keys_outside_the_overlay(self):
        # tests/test_library_service.py relies on exactly this pattern for
        # Config.INPUT_DIR -- must keep working once Config gained __init__.
        saved = self.C.Config.INPUT_DIR
        try:
            self.C.Config.INPUT_DIR = "/tmp/somewhere"
            cfg = self.C.Config()
            self.assertEqual(cfg.INPUT_DIR, "/tmp/somewhere")
        finally:
            self.C.Config.INPUT_DIR = saved


if __name__ == "__main__":
    unittest.main()
