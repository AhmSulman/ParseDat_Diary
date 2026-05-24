"""
PDF Category Manager
====================
Stores user-defined categories as  data/categories.json.

Format:
{
  "Finance": ["report.pdf", "budget.pdf"],
  "Science": ["paper.pdf"]
}
"""

import json
import os

_ROOT           = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CATEGORIES_FILE = os.path.join(_ROOT, "data", "categories.json")
UNCATEGORIZED   = "Uncategorized"


class CategoryManager:
    def __init__(self):
        os.makedirs(os.path.join(_ROOT, "data"), exist_ok=True)
        self._data: dict[str, list[str]] = {}
        self.load()

    def load(self):
        if os.path.exists(CATEGORIES_FILE):
            try:
                with open(CATEGORIES_FILE, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except Exception:
                self._data = {}

    def save(self):
        with open(CATEGORIES_FILE, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    # ── Category operations ────────────────────────────────────────────────────
    def list_categories(self) -> list[str]:
        return sorted(self._data.keys())

    def create_category(self, name: str):
        if name and name not in self._data:
            self._data[name] = []
            self.save()

    def delete_category(self, name: str):
        self._data.pop(name, None)
        self.save()

    def rename_category(self, old: str, new: str):
        if old in self._data and new and new != old:
            self._data[new] = self._data.pop(old)
            self.save()

    # ── PDF operations ─────────────────────────────────────────────────────────
    def add_pdf(self, category: str, pdf_name: str):
        if category not in self._data:
            self._data[category] = []
        if pdf_name not in self._data[category]:
            self._data[category].append(pdf_name)
            self.save()

    def remove_pdf_from_category(self, category: str, pdf_name: str):
        if category in self._data:
            self._data[category] = [p for p in self._data[category] if p != pdf_name]
            self.save()

    def remove_pdf_everywhere(self, pdf_name: str):
        """Remove a PDF from every category."""
        for cat in self._data:
            self._data[cat] = [p for p in self._data[cat] if p != pdf_name]
        self.save()

    def get_category_of(self, pdf_name: str) -> str:
        """Return the first category that contains this PDF, or Uncategorized."""
        for cat, pdfs in self._data.items():
            if pdf_name in pdfs:
                return cat
        return UNCATEGORIZED

    def pdfs_in_category(self, category: str) -> list[str]:
        if category == UNCATEGORIZED:
            # PDFs in input_dir but not in any category
            from config.config import Config
            cfg = Config()
            all_pdfs  = set(f for f in os.listdir(cfg.INPUT_DIR) if f.lower().endswith(".pdf"))
            all_cat   = {p for pdfs in self._data.values() for p in pdfs}
            return sorted(all_pdfs - all_cat)
        return sorted(self._data.get(category, []))

    def all_categories_with_counts(self) -> dict[str, int]:
        """Return {category: pdf_count} including Uncategorized."""
        result = {k: len(v) for k, v in self._data.items()}
        from config.config import Config
        cfg = Config()
        try:
            all_pdfs = set(f for f in os.listdir(cfg.INPUT_DIR) if f.lower().endswith(".pdf"))
        except Exception:
            all_pdfs = set()
        all_cat   = {p for pdfs in self._data.values() for p in pdfs}
        result[UNCATEGORIZED] = len(all_pdfs - all_cat)
        return result
