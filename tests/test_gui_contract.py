"""
Contract between material_app.kv and material_app.py.

    _venv\\Scripts\\python.exe -m unittest discover -s tests -v

KV binds callbacks and text by name: `on_release: root.on_reindex()` and
`text: root.library_info`. Nothing checks those names until the widget is built
or the button is pressed, so a rename or a typo is invisible until it crashes
in front of the user — and a KivyMD window cannot be constructed headlessly on
Windows, so no test can catch it by building the app.

This reads both files as text instead: the Python with `ast`, the KV with a
regex. No Kivy import, no window, no model.
"""

import ast
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PY = os.path.join(_ROOT, "gui", "material_app.py")
_KV = os.path.join(_ROOT, "gui", "material_app.kv")
_CLASS = "ParseDatMaterialRoot"


def _root_class():
    with open(_PY, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == _CLASS:
            return node
    raise AssertionError(f"{_CLASS} not found in {_PY}")


def _methods():
    return {n.name for n in _root_class().body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _properties():
    """Class-level assignments, which is how Kivy properties are declared."""
    out = set()
    for node in _root_class().body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    out.add(t.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out.add(node.target.id)
    return out


class TestKvBindings(unittest.TestCase):
    def setUp(self):
        with open(_KV, encoding="utf-8") as fh:
            self.kv = fh.read()

    def _refs(self):
        """
        (called, read) attribute names in the KV.

        The identifier is captured whole and the following character inspected
        separately. A negative lookahead here would let `\\w*` backtrack to a
        shorter match just to satisfy it, so `root.on_ask()` would be read as
        the attribute `on_as`.
        """
        called, read = set(), set()
        for m in re.finditer(r"root\.([A-Za-z_]\w*)(\s*\()?", self.kv):
            (called if m.group(2) else read).add(m.group(1))
        return called, read

    def test_every_called_handler_exists(self):
        called, _ = self._refs()
        missing = sorted(called - _methods())
        self.assertEqual(missing, [], f"KV calls methods that do not exist: {missing}")

    def test_every_referenced_property_exists(self):
        _, read = self._refs()
        missing = sorted(read - (_methods() | _properties()))
        self.assertEqual(missing, [],
                         f"KV reads attributes that do not exist: {missing}")

    def test_the_kv_rule_matches_the_class_name(self):
        self.assertIn(f"<{_CLASS}>", self.kv,
                      "the KV rule name must match the Python class, or Kivy "
                      "silently applies no layout at all")

    def test_library_actions_are_reachable(self):
        """The management actions must exist and be wired into the menu."""
        methods = _methods()
        for name in ("on_reindex", "on_sync_library", "on_clear_index",
                     "remove_book", "show_remove_menu", "refresh_library_info"):
            self.assertIn(name, methods)
        with open(_PY, encoding="utf-8") as fh:
            py = fh.read()
        for label in ("Reindex vectors", "Sync to disk", "Remove a book",
                      "Clear all chunks", "Library statistics"):
            self.assertIn(label, py, f"{label!r} is not offered in any menu")

    def test_destructive_actions_ask_first(self):
        """Nothing that deletes may run straight off a tap."""
        with open(_PY, encoding="utf-8") as fh:
            py = fh.read()
        for name in ("on_clear_index", "remove_book", "on_sync_library", "on_reindex"):
            body = py.split(f"def {name}(", 1)[1].split("\n    def ", 1)[0]
            self.assertIn("_confirm(", body,
                          f"{name} does not confirm before acting")


if __name__ == "__main__":
    unittest.main()
