"""
ParseDat_Diary — Material Design desktop / Android shell (KivyMD 1.x)
============================================================
PDF library, ingest, preview, and local RAG Q&A.

Run: python main.py gui
"""

from __future__ import annotations

import asyncio
import os
import platform
import shutil
import sys
import threading
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from kivy.clock import Clock
from kivy.lang.builder import Builder
from kivy.metrics import dp
from kivy.properties import ListProperty, NumericProperty, StringProperty
from kivy.core.window import Window

from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.label import MDLabel
from kivymd.uix.menu import MDDropdownMenu
from kivymd.uix.snackbar import Snackbar
from kivymd.uix.textfield import MDTextField

from config.config import Config
from logs.logger import log

# Only set a fixed window size on desktop — Android uses full screen naturally
_IS_ANDROID = platform.system() == "Linux" and "ANDROID_ARGUMENT" in os.environ
if not _IS_ANDROID:
    Window.size = (1280, 860)


# (model, dim, label). The DIMENSION matters: the FAISS index stores vectors of
# one fixed width, so switching to a model of a different dim makes every stored
# vector unusable until a full reindex. The old list offered two 384-dim models
# against a 768-dim index with no warning.
EMBEDDER_OPTIONS = [
    ("BAAI/bge-base-en-v1.5", 768, "BGE Base  ·  420 MB  ·  current, offline"),
    ("sentence-transformers/all-mpnet-base-v2", 768, "MPNet Base  ·  420 MB  ·  needs download"),
    ("sentence-transformers/all-MiniLM-L6-v2", 384, "MiniLM L6  ·  22 MB  ·  fast, FULL REINDEX"),
]

PRESET_MODELS = [
    ("mistral-7b-instruct-v0.2.Q4_K_M.gguf",
     "Mistral 7B Q4  ·  4.4 GB  ·  Best overall",
     "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"),
    ("phi-2.Q4_K_M.gguf",
     "Phi-2 Q4  ·  1.7 GB  ·  Fastest",
     "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf"),
    ("openchat-3.5-0106.Q4_K_M.gguf",
     "OpenChat 3.5 Q4  ·  4.1 GB  ·  Great reasoning",
     "https://huggingface.co/TheBloke/openchat-3.5-0106-GGUF/resolve/main/openchat-3.5-0106.Q4_K_M.gguf"),
]


class ParseDatMaterialRoot(MDBoxLayout):
    """Root widget; loaded from ``material_app.kv``."""

    status_library        = StringProperty("Drop PDFs into data/input or use Add PDFs.")
    library_info          = StringProperty("—")
    ingest_status         = StringProperty("")
    pdf_list_compact      = StringProperty("—")
    pdf_chip_hint         = StringProperty("Choose PDF…")
    status_read_tab       = StringProperty("Open the menu and pick a document.")
    preview_heading       = StringProperty("")
    preview_display       = StringProperty("")
    rag_status_hint       = StringProperty("Loading model and search index…")
    answer_display        = StringProperty("")
    ingest_progress_value = NumericProperty(0)

    # Settings screen
    active_model_label    = StringProperty("—")
    active_embedder_label = StringProperty("—")
    model_list_text       = StringProperty("")
    dl_status             = StringProperty("")
    settings_save_status  = StringProperty("")

    # Category management
    active_category    = StringProperty("All PDFs")
    category_bar_text  = StringProperty("")

    def __init__(self, app_ref: "ParseDatMaterialApp", **kwargs):
        super().__init__(**kwargs)
        self._app = app_ref
        self._selected_pdf: str | None = None
        self._pdf_menu: MDDropdownMenu | None = None
        self._ingest_thread: threading.Thread | None = None
        self._preview_ticket = 0
        self._ask_busy = False
        self._file_chooser_popup = None
        self._settings_inputs: dict = {}

        from storage.categories import CategoryManager
        self._cats = CategoryManager()

        from storage.history import History
        self._history = History()
        self._turn_start = 0
        last = self._history.latest_session_id()
        if last:
            self._history.set_current(last)

        Clock.schedule_once(lambda *_: self.refresh_pdf_list(), 0)
        Clock.schedule_once(lambda *_: self.refresh_library_info(), 0)

    def on_kv_post(self, base_widget):
        super().on_kv_post(base_widget)

        def go_library(_dt=None):
            sm = self.ids.get("sm")
            if sm:
                try:
                    sm.current = "library"
                except Exception:
                    pass

        Clock.schedule_once(go_library, 0)

    def on_pick_upload(self):
        """Open a file chooser — native Kivy popup (works on Android too)."""
        from kivy.uix.popup import Popup
        from kivy.uix.filechooser import FileChooserListView
        from kivy.uix.boxlayout import BoxLayout
        from kivy.uix.button import Button

        content = BoxLayout(orientation="vertical", spacing=dp(8), padding=dp(8))
        fc = FileChooserListView(
            filters=["*.pdf", "*.PDF"],
            multiselect=True,
            path=str(Path.home()),
        )
        content.add_widget(fc)

        btn_row = BoxLayout(size_hint_y=None, height=dp(48), spacing=dp(8))
        btn_cancel = Button(text="Cancel", size_hint_x=0.4)
        btn_ok = Button(text="Add Selected", size_hint_x=0.6)
        btn_row.add_widget(btn_cancel)
        btn_row.add_widget(btn_ok)
        content.add_widget(btn_row)

        popup = Popup(
            title="Select PDF files",
            content=content,
            size_hint=(0.92, 0.88),
        )
        btn_cancel.bind(on_release=popup.dismiss)
        btn_ok.bind(on_release=lambda *_: self._finish_upload(fc.selection, popup))
        popup.open()
        self._file_chooser_popup = popup

    def _finish_upload(self, paths: list[str], popup=None):
        if popup:
            popup.dismiss()
        cfg = Config()
        os.makedirs(cfg.INPUT_DIR, exist_ok=True)
        n = 0
        for p in paths:
            try:
                dest = os.path.join(cfg.INPUT_DIR, os.path.basename(p))
                shutil.copy2(p, dest)
                n += 1
            except Exception as ex:
                log.error(f"Copy failed: {ex}")
        self._toast(f"Added {n} PDF(s).")
        self.refresh_pdf_list()

    # ── Ingest ────────────────────────────────────────────────────────────────
    def on_run_ingest(self):
        if self._ingest_thread and self._ingest_thread.is_alive():
            self._toast("Ingest already running.")
            return
        self.ingest_status = "Extracting → chunking → embedding → indexing…"
        self.ingest_progress_value = 0.08
        Clock.schedule_interval(self._ingest_pulse, 0.45)

        def job():
            try:
                from core.async_pipeline import AsyncPipeline
                asyncio.run(AsyncPipeline().run())
            except Exception as ex:
                log.exception(ex)
                Clock.schedule_once(lambda *_: self._ingest_failed(str(ex)), 0)
                return
            Clock.schedule_once(lambda *_: self._ingest_ok(), 0)

        self._ingest_thread = threading.Thread(target=job, daemon=True)
        self._ingest_thread.start()

    def _ingest_pulse(self, _dt):
        if self._ingest_thread and self._ingest_thread.is_alive():
            self.ingest_progress_value = min(0.92, float(self.ingest_progress_value) + 0.04)
            return
        Clock.unschedule(self._ingest_pulse)

    def _ingest_ok(self):
        Clock.unschedule(self._ingest_pulse)
        self.ingest_progress_value = 1.0
        self.ingest_status = "Done. Index updated in data/cache."
        self.refresh_library_info()
        self._toast("Ingest complete!")
        self.refresh_pdf_list()
        rag = getattr(self._app, "rag", None)
        if rag is not None:
            try:
                rag.retriever.load()
            except Exception as ex:
                log.warning(f"Index reload: {ex}")
            # Report books AND chunks. doc_count is a chunk total despite the
            # name, so the old message never told the user how many books the
            # AI could actually see.
            chunks = rag.retriever.chunk_count
            books = rag.retriever.book_count
            if getattr(self._app, "rag_ok", False):
                self.rag_status_hint = (
                    f"Ready — {books} book(s), {chunks:,} chunk(s). Ask away."
                )

        Clock.schedule_once(lambda *_: setattr(self, "ingest_progress_value", 0), 2.0)
        Clock.schedule_once(lambda *_: setattr(self, "ingest_status", ""), 2.8)

    def _ingest_failed(self, msg: str):
        Clock.unschedule(self._ingest_pulse)
        self.ingest_progress_value = 0
        self.ingest_status = f"Failed: {msg[:280]}"
        self._toast("Ingest failed — see logs.")

    # ── PDF dropdown (Read tab) ────────────────────────────────────────────────
    def on_open_pdf_menu(self, btn):
        cfg = Config()
        pdfs = sorted(f for f in os.listdir(cfg.INPUT_DIR) if f.lower().endswith(".pdf"))
        if not pdfs:
            self._toast("No PDFs yet — use Library → Add PDFs.")
            return
        if self._pdf_menu:
            self._pdf_menu.dismiss()

        def make_pick(name: str):
            def pick(*_a):
                if self._pdf_menu:
                    self._pdf_menu.dismiss()
                self._select_pdf(name)
            return pick

        menu_items = [{"text": p, "on_release": make_pick(p)} for p in pdfs]
        self._pdf_menu = MDDropdownMenu(
            caller=btn,
            items=menu_items,
            width_mult=6,
            max_height=dp(400),
        )
        self._pdf_menu.open()

    def show_more_menu(self, btn):
        if self._pdf_menu:
            self._pdf_menu.dismiss()

        def stats(*_):
            # Same LibraryService the CLI and dashboard use, so this can never
            # report different numbers than `main.py doctor`. It also surfaces
            # drift -- holes and orphans -- which a plain file count cannot see.
            try:
                from storage.library import LibraryService
                rep = LibraryService().report()
                c = rep["counts"]
                if rep["healthy"]:
                    msg = (f"{c['books_indexed']} books · "
                           f"{c['chunks_indexed']:,} chunks · healthy")
                else:
                    bits = []
                    if rep["holes"]:
                        bits.append(f"{len(rep['holes'])} not indexed")
                    if any(rep["orphans"].values()):
                        bits.append("orphans")
                    if rep["pending"]:
                        bits.append(f"{len(rep['pending'])} pending")
                    msg = (f"{c['books_indexed']}/{c['pdfs_on_disk']} indexed · "
                           + ", ".join(bits))
            except Exception as ex:
                log.warning(f"Library stats failed: {ex}")
                msg = "Could not read library state"
            self._toast(msg)
            if self._pdf_menu:
                self._pdf_menu.dismiss()

        def reload_list(*_):
            self.refresh_pdf_list()
            if self._pdf_menu:
                self._pdf_menu.dismiss()

        menu_items = [
            {"text": "Library statistics", "on_release": stats},
            {"text": "Reindex vectors", "on_release": lambda *_: self._menu_then(self.on_reindex)},
            {"text": "Sync to disk", "on_release": lambda *_: self._menu_then(self.on_sync_library)},
            {"text": "Remove a book…", "on_release": lambda *_: self._menu_then(lambda: self.show_remove_menu(btn))},
            {"text": "Clear all chunks", "on_release": lambda *_: self._menu_then(self.on_clear_index)},
            {"text": "Reload PDF list",    "on_release": reload_list},
            {"text": "Config: edit config/config.py", "on_release": lambda *_: self._toast("Edit config/config.py")},
        ]
        self._pdf_menu = MDDropdownMenu(caller=btn, items=menu_items, width_mult=5)
        self._pdf_menu.open()

    # ── Read / preview ─────────────────────────────────────────────────────────
    def _select_pdf(self, pdf_name: str):
        self._selected_pdf = pdf_name
        self.pdf_chip_hint = pdf_name
        self.status_read_tab = pdf_name

        self._preview_ticket += 1
        ticket = self._preview_ticket
        self.preview_display = "Loading…"
        self.preview_heading = ""

        threading.Thread(target=self._preview_worker, args=(pdf_name, ticket), daemon=True).start()

    def _preview_worker(self, pdf_name: str, ticket: int):
        cfg = Config()
        base_stem = os.path.splitext(pdf_name)[0]
        txt_path = os.path.join(cfg.OUTPUT_TXT, base_stem + ".txt")
        pdf_path = os.path.join(cfg.INPUT_DIR, pdf_name)
        note = ""
        text = ""

        try:
            if not os.path.isfile(pdf_path):
                text = f"Missing file:\n{pdf_path}"
                note = "error"
            elif os.path.isfile(txt_path):
                with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                note = f"Extracted text · {os.path.basename(txt_path)}"
                text = text[:120_000]
            else:
                import fitz
                doc = fitz.open(pdf_path)
                parts: list[str] = []
                max_pages = min(len(doc), 120)
                for i in range(max_pages):
                    parts.append(doc.load_page(i).get_text())
                doc.close()
                text = "\n".join(parts)
                note = f"Live extract — pages 1–{max_pages}. Run Ingest for full OCR + search."
                text = text[:120_000]
        except Exception as ex:
            text = str(ex)
            note = "Could not open PDF."

        trimmed = text[:260_000]

        def apply(*_):
            if ticket != self._preview_ticket:
                return
            self.preview_heading = note
            self.preview_display = trimmed or "(no text — try Ingest with OCR)"
            Clock.schedule_once(self._reflow_preview, 0)

        Clock.schedule_once(apply, 0)

    def _reflow_preview(self, *_):
        sc = self.ids.get("preview_scroll")
        lb = self.ids.get("pdf_preview")
        if sc and lb:
            w = max(sc.width - dp(24), dp(220))
            lb.text_size = (w, None)
            lb.texture_update()
            lb.height = max(lb.texture_size[1], dp(80))

    # ── Ask AI — live streaming ────────────────────────────────────────────────
    def on_ask(self):
        fld = self.ids.get("question_field")
        if not fld:
            return
        q = fld.text.strip()
        if not q:
            self._toast("Enter a question first.")
            return
        if self._ask_busy:
            self._toast("Still generating — please wait.")
            return
        if not getattr(self._app, "rag_ok", False) or self._app.rag is None:
            self._toast("Model not ready. Check config and model path.")
            return

        self._ask_busy = True
        fld.text = ""
        # Append to the transcript rather than clearing it. Every answer used to
        # overwrite the previous one, so the conversation vanished a question at
        # a time and nothing survived closing the app.
        if self.answer_display.strip():
            self.answer_display += chr(10)*2 + ("-" * 40) + chr(10)*2
        self.answer_display += "You: " + q + chr(10)*2
        self._turn_start = len(self.answer_display)

        threading.Thread(target=self._ask_worker, args=(q,), daemon=True).start()

    def _ask_worker(self, question: str):
        """Stream tokens directly into answer_display for live output."""
        try:
            rag = self._app.rag
            assert rag is not None
            for token in rag.query_stream(question):
                # Capture token in closure to avoid late-binding
                def push(dt, t=token):
                    self.answer_display += t
                Clock.schedule_once(push, 0)
        except Exception as ex:
            err = str(ex)
            log.exception(ex)
            Clock.schedule_once(lambda *_: setattr(self, "answer_display",
                                                    f"Error: {err[:600]}"), 0)
        finally:
            def done(*_):
                self._ask_busy = False
                # Persist the turn so it survives closing the app.
                try:
                    answer = self.answer_display[getattr(self, "_turn_start", 0):]
                    self._history.add_turn(question, answer)
                except Exception as ex:
                    log.warning(f"History not saved: {ex}")
                Clock.schedule_once(self._reflow_answer, 0)
            Clock.schedule_once(done, 0)

    def _reflow_answer(self, *_):
        sc = self.ids.get("ans_scroll")
        lb = self.ids.get("answer_area")
        if sc and lb:
            w = max(sc.width - dp(24), dp(220))
            lb.text_size = (w, None)
            lb.texture_update()
            lb.height = max(lb.texture_size[1], dp(160))

    # ── Category & PDF management ──────────────────────────────────────────────
    def refresh_pdf_list(self, category: str | None = None):
        cfg = Config()
        os.makedirs(cfg.INPUT_DIR, exist_ok=True)
        if category is None:
            category = self.active_category

        if category == "All PDFs":
            pdfs = sorted(f for f in os.listdir(cfg.INPUT_DIR) if f.lower().endswith(".pdf"))
        else:
            pdfs = [p for p in self._cats.pdfs_in_category(category)
                    if os.path.exists(os.path.join(cfg.INPUT_DIR, p))]

        cat_counts = self._cats.all_categories_with_counts()
        self.category_bar_text = "  |  ".join(
            f"{c} ({n})" for c, n in [("All PDFs", len(
                [f for f in os.listdir(cfg.INPUT_DIR) if f.lower().endswith(".pdf")]
            ))] + list(cat_counts.items())
        )

        if not pdfs:
            self.status_library = f"[{category}]  No PDFs here yet."
            self.pdf_list_compact = "(empty)"
            return
        self.status_library = f"[{category}]  {len(pdfs)} PDF(s)"
        lines = "\n".join(f"  •  {p}" for p in pdfs[:30])
        if len(pdfs) > 30:
            lines += f"\n  …  +{len(pdfs) - 30} more"
        self.pdf_list_compact = lines

    def show_category_menu(self, btn):
        """Dropdown to switch active category filter."""
        cfg = Config()
        all_count = len([f for f in os.listdir(cfg.INPUT_DIR) if f.lower().endswith(".pdf")])
        counts = self._cats.all_categories_with_counts()
        items = [{"text": f"All PDFs  ({all_count})",
                  "on_release": lambda *_: self._set_category("All PDFs")}]
        for cat, n in counts.items():
            items.append({"text": f"{cat}  ({n})",
                          "on_release": lambda *_, c=cat: self._set_category(c)})
        if self._pdf_menu:
            self._pdf_menu.dismiss()
        self._pdf_menu = MDDropdownMenu(caller=btn, items=items, width_mult=5, max_height=dp(360))
        self._pdf_menu.open()

    def _set_category(self, name: str):
        if self._pdf_menu:
            self._pdf_menu.dismiss()
        self.active_category = name
        self.refresh_pdf_list(name)

    def show_pdf_actions_menu(self, btn):
        """Long-press / action button on a PDF — remove or assign to category."""
        cfg = Config()
        pdfs = sorted(f for f in os.listdir(cfg.INPUT_DIR) if f.lower().endswith(".pdf"))
        if not pdfs:
            self._toast("No PDFs in library.")
            return

        def make_remove(pdf: str):
            def do(*_):
                if self._pdf_menu:
                    self._pdf_menu.dismiss()
                self._confirm_remove_pdf(pdf)
            return do

        def make_assign(pdf: str):
            def do(*_):
                if self._pdf_menu:
                    self._pdf_menu.dismiss()
                self._show_assign_category(pdf)
            return do

        items = []
        for p in pdfs[:20]:
            items.append({"text": f"Delete  {p}",    "on_release": make_remove(p)})
            items.append({"text": f"Categorise  {p}", "on_release": make_assign(p)})

        if self._pdf_menu:
            self._pdf_menu.dismiss()
        self._pdf_menu = MDDropdownMenu(caller=btn, items=items, width_mult=8, max_height=dp(420))
        self._pdf_menu.open()

    def _confirm_remove_pdf(self, pdf_name: str):
        """Popup confirmation before deleting a PDF file."""
        from kivy.uix.popup import Popup
        from kivy.uix.boxlayout import BoxLayout
        from kivy.uix.button import Button
        from kivymd.uix.label import MDLabel

        content = BoxLayout(orientation="vertical", spacing=dp(12), padding=dp(16))
        content.add_widget(MDLabel(
            text=f"Delete  '{pdf_name}'  from disk?\nThis cannot be undone.",
            halign="center", adaptive_height=True,
        ))
        btn_row = BoxLayout(size_hint_y=None, height=dp(48), spacing=dp(8))
        btn_no  = Button(text="Cancel",     size_hint_x=0.5)
        btn_yes = Button(text="Delete",     size_hint_x=0.5, background_color=(0.8, 0.1, 0.1, 1))
        btn_row.add_widget(btn_no)
        btn_row.add_widget(btn_yes)
        content.add_widget(btn_row)

        popup = Popup(title="Confirm Delete", content=content, size_hint=(0.8, 0.4))
        btn_no.bind(on_release=popup.dismiss)
        btn_yes.bind(on_release=lambda *_: self._do_remove_pdf(pdf_name, popup))
        popup.open()

    def _do_remove_pdf(self, pdf_name: str, popup=None):
        if popup:
            popup.dismiss()
        cfg = Config()
        path = os.path.join(cfg.INPUT_DIR, pdf_name)
        try:
            os.remove(path)
            self._cats.remove_pdf_everywhere(pdf_name)
            self._toast(f"Deleted {pdf_name}.")
        except Exception as e:
            self._toast(f"Delete failed: {e}")
        self.refresh_pdf_list()

    def _show_assign_category(self, pdf_name: str):
        """Popup to pick or create a category for a PDF."""
        from kivy.uix.popup import Popup
        from kivy.uix.boxlayout import BoxLayout
        from kivy.uix.button import Button
        from kivy.uix.textinput import TextInput
        from kivymd.uix.label import MDLabel

        content = BoxLayout(orientation="vertical", spacing=dp(8), padding=dp(12))
        content.add_widget(MDLabel(text=f"Assign  '{pdf_name}'  to category:", adaptive_height=True))

        existing_cats = self._cats.list_categories()
        popup = Popup(title="Assign Category", content=content, size_hint=(0.85, 0.7))

        for cat in existing_cats:
            b = Button(text=cat, size_hint_y=None, height=dp(44))
            b.bind(on_release=lambda *_, c=cat: (
                self._cats.add_pdf(c, pdf_name),
                self._toast(f"'{pdf_name}' -> {c}"),
                popup.dismiss(),
                self.refresh_pdf_list(),
            ))
            content.add_widget(b)

        new_row = BoxLayout(size_hint_y=None, height=dp(48), spacing=dp(8))
        ti = TextInput(hint_text="New category name…", multiline=False, size_hint_x=0.7)
        btn_create = Button(text="Create", size_hint_x=0.3)
        new_row.add_widget(ti)
        new_row.add_widget(btn_create)
        content.add_widget(new_row)

        def create_and_assign(*_):
            name = ti.text.strip()
            if name:
                self._cats.create_category(name)
                self._cats.add_pdf(name, pdf_name)
                self._toast(f"Created '{name}' and added PDF.")
                popup.dismiss()
                self.refresh_pdf_list()

        btn_create.bind(on_release=create_and_assign)
        popup.open()

    def on_new_category(self):
        """Popup to create a new empty category."""
        from kivy.uix.popup import Popup
        from kivy.uix.boxlayout import BoxLayout
        from kivy.uix.button import Button
        from kivy.uix.textinput import TextInput

        content = BoxLayout(orientation="vertical", spacing=dp(10), padding=dp(12))
        ti = TextInput(hint_text="Category name…", multiline=False, size_hint_y=None, height=dp(48))
        content.add_widget(ti)
        btn_row = BoxLayout(size_hint_y=None, height=dp(48), spacing=dp(8))
        btn_cancel = Button(text="Cancel")
        btn_ok     = Button(text="Create")
        btn_row.add_widget(btn_cancel)
        btn_row.add_widget(btn_ok)
        content.add_widget(btn_row)

        popup = Popup(title="New Category", content=content, size_hint=(0.75, 0.35))
        btn_cancel.bind(on_release=popup.dismiss)

        def do_create(*_):
            name = ti.text.strip()
            if name:
                self._cats.create_category(name)
                self._toast(f"Category '{name}' created.")
                self.refresh_pdf_list()
            popup.dismiss()

        btn_ok.bind(on_release=do_create)
        popup.open()

    # ── Settings: Model & Embedder ─────────────────────────────────────────────
    def refresh_settings(self):
        cfg = Config()
        cur = os.path.basename(cfg.LLM_MODEL_PATH)
        self.active_model_label = cur if os.path.exists(cfg.LLM_MODEL_PATH) else f"{cur}  (not downloaded)"

        # The embedder that ACTUALLY loaded, falling back to the configured name
        # when nothing has loaded yet. brain.embedder no longer exports a
        # module-level MODEL_NAME: the configured model can fall back at load
        # time, so a constant would have reported the wrong one.
        em = cfg.EMBED_MODEL
        app = getattr(self, "_app", None)
        rag = getattr(app, "rag", None) if app else None
        if rag is not None:
            em = getattr(rag.retriever.embedder, "model_name", None) or em
        label = next((l for n, _d, l in EMBEDDER_OPTIONS if n == em), em)
        built = ""
        try:
            from storage.manifest import Manifest
            built_with = Manifest().settings.get("embed_model")
            # Flag a mismatch here: both models are 768-dim, so nothing else
            # would catch an index built with a different embedder.
            if built_with and built_with != em:
                built = f"  (index built with {built_with} — reindex)"
        except Exception:
            pass
        self.active_embedder_label = label + built

        # List local .gguf files
        models_dir = cfg.MODELS_DIR
        os.makedirs(models_dir, exist_ok=True)
        files = [f for f in os.listdir(models_dir) if f.endswith(".gguf")]
        if files:
            self.model_list_text = "\n".join(f"  •  {f}" for f in files)
        else:
            self.model_list_text = "  (none yet — use Download buttons)"

        self.build_settings_form()

    # ── Settings: full parameter form ───────────────────────────────────────────
    def build_settings_form(self):
        """
        Render the settings form from SETTINGS_SCHEMA — one row per Config
        field, grouped, prefilled with the currently effective value.

        Built in Python rather than laid out in the .kv: the schema is the
        single source of truth for what's tunable, so a field added there
        shows up here with no KV edit.
        """
        box = self.ids.get("settings_form_box")
        if box is None:
            return
        box.clear_widgets()
        self._settings_inputs = {}

        from config.settings_store import SETTINGS_SCHEMA

        cfg = Config()
        restart_tag = {
            "server": "  ·  restart server",
            "reindex": "  ·  needs reindex",
            "app": "  ·  restart ParseDat_Diary",
            None: "",
        }

        seen_groups: list[str] = []
        for f in SETTINGS_SCHEMA:
            if f["group"] not in seen_groups:
                seen_groups.append(f["group"])
                box.add_widget(MDLabel(
                    text=f["group"],
                    font_style="Overline", bold=True,
                    theme_text_color="Custom",
                    text_color=(0.76, 0.52, 1, 1),
                    adaptive_height=True,
                    size_hint_y=None, height=dp(28),
                ))

            row = MDBoxLayout(
                orientation="horizontal", spacing=dp(8),
                size_hint_y=None, height=dp(44),
            )
            row.add_widget(MDLabel(
                text=f["label"] + restart_tag.get(f["restart"], ""),
                theme_text_color="Secondary", font_style="Caption",
                size_hint_x=0.55, adaptive_height=True,
            ))
            value = getattr(cfg, f["key"], f.get("default", ""))
            ti = MDTextField(
                text=str(value), size_hint_x=0.45, height=dp(40),
                mode="rectangle",
            )
            self._settings_inputs[f["key"]] = ti
            row.add_widget(ti)
            box.add_widget(row)

    def on_save_all_settings(self):
        """
        Validate every field, persist only what differs from the hardcoded
        default (data/settings.json stays a diff, not a full snapshot), and
        hot-apply the subset that's safe to change on a live pipeline.
        """
        from config.settings_store import (
            SETTINGS_SCHEMA, validate, validate_patch, restart_labels,
            load_overlay, save_overlay,
        )

        cfg = Config()
        current_effective = {f["key"]: getattr(cfg, f["key"]) for f in SETTINGS_SCHEMA}
        class_defaults = {f["key"]: getattr(Config, f["key"]) for f in SETTINGS_SCHEMA}

        candidate = dict(current_effective)
        errors = []
        for f in SETTINGS_SCHEMA:
            key = f["key"]
            widget = self._settings_inputs.get(key)
            if widget is None:
                continue
            ok, err, value = validate(key, widget.text)
            if not ok:
                errors.append(f"{f['label']}: {err}")
                continue
            candidate[key] = value

        errors.extend(validate_patch(candidate, current_effective))

        if errors:
            more = f" (+{len(errors) - 2} more)" if len(errors) > 2 else ""
            self._toast("Fix: " + "; ".join(errors[:2]) + more)
            self.settings_save_status = "Not saved — " + "; ".join(errors[:3]) + more
            return

        overlay = load_overlay()
        changed_keys = []
        for f in SETTINGS_SCHEMA:
            key = f["key"]
            new_val = candidate[key]
            if new_val == class_defaults[key]:
                if key in overlay:
                    del overlay[key]
                    changed_keys.append(key)
            elif overlay.get(key) != new_val:
                overlay[key] = new_val
                changed_keys.append(key)
        save_overlay(overlay)

        if not changed_keys:
            self.settings_save_status = "No changes."
            return

        self._hot_apply_settings(candidate, changed_keys)

        grouped = restart_labels(changed_keys)
        parts = []
        if grouped.get("reindex"):
            parts.append(f"run reindex ({', '.join(grouped['reindex'])})")
        if grouped.get("app"):
            parts.append(f"restart ParseDat_Diary ({', '.join(grouped['app'])})")

        msg = f"Saved {len(changed_keys)} setting(s)."
        if parts:
            msg += "  Still needed: " + "; ".join(parts)
        self.settings_save_status = msg
        self._toast(msg[:220])

        # llama-server is the one thing worth relaunching automatically: it's
        # a background process this app already owns (ServerManager), so
        # there's no reason to make the user taskkill + retype the launch
        # command from CLAUDE.md by hand every time GPU layers/context/
        # threads/batch/binary path change.
        if grouped.get("server"):
            self._auto_restart_llama_server(grouped["server"])

    def _auto_restart_llama_server(self, changed_keys: list[str]):
        """Relaunch llama-server on the currently active model, with the values
        just saved, in the background."""
        from brain.server_manager import get_manager

        app = self._app
        rag = getattr(app, "rag", None) if app else None
        model_path = (getattr(getattr(rag, "llm", None), "model_path", None)
                      or Config().LLM_MODEL_PATH)
        if not model_path or not os.path.exists(model_path):
            self.settings_save_status += "  (no model loaded yet — server " \
                                          "settings will apply on next start)"
            return

        self.settings_save_status = (
            f"Restarting llama-server ({', '.join(changed_keys)})…")

        def job():
            mgr = get_manager()
            ok, msg = mgr.start(model_path)
            Clock.schedule_once(lambda *_: self._server_restarted(ok, msg), 0)

        threading.Thread(target=job, daemon=True).start()

    def _server_restarted(self, ok: bool, msg: str):
        if ok:
            self.settings_save_status = f"llama-server restarted: {msg}"
            self._toast("llama-server restarted with the new settings.")
            app = self._app
            if app and getattr(app, "rag", None) is not None:
                try:
                    app.rag.llm.load()
                except Exception as ex:
                    log.warning(f"Backend reconnect after restart: {ex}")
        else:
            self.settings_save_status = f"llama-server restart failed: {msg}"
            self._toast(f"Restart failed: {msg[:150]}")

    def _hot_apply_settings(self, candidate: dict, changed_keys: list[str]):
        """
        Patch the live RAGPipeline for the handful of fields it's safe to
        change without restarting anything — everything else is read once at
        construction time and genuinely needs the restart the UI names.
        """
        app = self._app
        rag = getattr(app, "rag", None) if app else None
        if rag is None:
            return

        hot = {
            "SEARCH_TOP_K": lambda v: setattr(rag, "top_k", v),
            "CONTEXT_CHAR_BUDGET": lambda v: setattr(rag, "char_budget", v),
            "NEIGHBOR_RADIUS": lambda v: setattr(rag.retriever.cfg, "NEIGHBOR_RADIUS", v),
            "LLM_TEMPERATURE": lambda v: setattr(rag.llm, "temperature", v),
            "LLM_MAX_TOKENS": lambda v: setattr(rag.llm, "max_tokens", v),
        }
        for key in changed_keys:
            fn = hot.get(key)
            if fn is None:
                continue
            try:
                fn(candidate[key])
            except Exception as e:
                log.warning(f"Hot-apply {key} failed: {e}")

    def on_select_model(self, filename: str):
        """
        Actually load a different .gguf, by restarting llama-server on it.

        The old version set llm.model_path and told the user to relaunch. That
        did nothing: generation goes through llama-server, so the model is
        whatever that process was started with. The picker appeared to work and
        silently changed nothing.
        """
        cfg = Config()
        full_path = os.path.join(cfg.MODELS_DIR, filename)
        if not os.path.exists(full_path):
            self._toast(f"Not found: {filename}")
            return

        from brain.server_manager import get_manager
        mgr = get_manager()

        ok, why = mgr.can_load(full_path)
        if not ok:
            # Refuse rather than thrash — an over-committed load freezes the
            # desktop instead of raising.
            self._toast(f"Cannot load: {why}")
            return

        from config.settings_store import update_overlay
        update_overlay({"LLM_MODEL_PATH": full_path})
        self.active_model_label = f"{filename}  (loading…)"
        self._toast(f"Loading {filename}… this takes a minute.")

        def job():
            started, msg = mgr.start(full_path)
            Clock.schedule_once(lambda *_: self._model_loaded(filename, started, msg), 0)

        threading.Thread(target=job, daemon=True).start()

    def _model_loaded(self, filename: str, ok: bool, msg: str):
        """Back on the UI thread once llama-server has finished loading."""
        if not ok:
            self.active_model_label = f"{filename}  (failed)"
            self._toast(f"Load failed: {msg[:120]}")
            return
        self.active_model_label = filename
        # Re-point the pipeline at the now-running server.
        app = self._app
        if app and getattr(app, "rag", None) is not None:
            try:
                app.rag.llm.load()
            except Exception as ex:
                log.warning(f"Backend reconnect: {ex}")
        self._toast(f"{filename} ready.")

    def on_download_model(self, name: str, url: str):
        """Download a preset model in a background thread."""
        cfg = Config()
        dest = os.path.join(cfg.MODELS_DIR, name)
        if os.path.exists(dest):
            self._toast(f"{name} already downloaded.")
            self.on_select_model(name)
            return
        self.dl_status = f"Downloading {name} …"
        self._toast(f"Downloading {name} (~this may take a while)…")

        def worker():
            import urllib.request

            def progress(count, block, total):
                pct = min(100, count * block * 100 // max(total, 1))
                mb  = count * block / 1024 / 1024
                Clock.schedule_once(
                    lambda *_: setattr(self, "dl_status",
                                       f"Downloading {name}: {pct}%  ({mb:.0f} MB)"), 0
                )

            try:
                urllib.request.urlretrieve(url, dest, reporthook=progress)
                Clock.schedule_once(lambda *_: self._dl_done(name), 0)
            except Exception as e:
                if os.path.exists(dest):
                    os.remove(dest)
                Clock.schedule_once(
                    lambda *_: setattr(self, "dl_status", f"Failed: {e}"), 0
                )

        threading.Thread(target=worker, daemon=True).start()

    def _dl_done(self, name: str):
        self.dl_status = f"{name} downloaded."
        self._toast(f"{name} ready!")
        self.refresh_settings()
        self.on_select_model(name)

    def show_model_menu(self, btn):
        """Dropdown: local .gguf files + preset downloads."""
        cfg = Config()
        files = [f for f in os.listdir(cfg.MODELS_DIR) if f.endswith(".gguf")]
        items = []
        for f in files:
            items.append({"text": f"✓ {f}", "on_release": lambda *_, n=f: (
                self._pdf_menu.dismiss() if self._pdf_menu else None,
                self.on_select_model(n),
            )})
        for name, label, url in PRESET_MODELS:
            tag = "⬇" if not os.path.exists(os.path.join(cfg.MODELS_DIR, name)) else "✓"
            items.append({
                "text": f"{tag} {label}",
                "on_release": lambda *_, n=name, u=url: (
                    self._pdf_menu.dismiss() if self._pdf_menu else None,
                    self.on_download_model(n, u),
                ),
            })
        if self._pdf_menu:
            self._pdf_menu.dismiss()
        from kivymd.uix.menu import MDDropdownMenu
        self._pdf_menu = MDDropdownMenu(caller=btn, items=items, width_mult=7, max_height=dp(380))
        self._pdf_menu.open()

    def show_embedder_menu(self, btn):
        """Dropdown: fast local embedder options (no API required)."""
        cur = Config().EMBED_MODEL

        items = []
        for name, dim, label in EMBEDDER_OPTIONS:
            tag = "●" if name == cur else "○"
            warn = "" if dim == Config().EMBED_DIM else f"  ⚠ {dim}-dim"
            items.append({
                "text": f"{tag} {label}{warn}",
                "on_release": lambda *_, n=name, d=dim, l=label: (
                    self._pdf_menu.dismiss() if self._pdf_menu else None,
                    self._switch_embedder(n, d, l),
                ),
            })
        if self._pdf_menu:
            self._pdf_menu.dismiss()
        from kivymd.uix.menu import MDDropdownMenu
        self._pdf_menu = MDDropdownMenu(caller=btn, items=items, width_mult=8, max_height=dp(280))
        self._pdf_menu.open()

    def _switch_embedder(self, name: str, dim: int, label: str):
        """
        Switch the embedder. This INVALIDATES the index.

        Vectors from two different models are not comparable, so the stored
        index describes nothing the new model would produce. The retriever
        refuses to search on a model mismatch rather than returning confident
        nonsense, so the library stays unusable until a reindex. Say so plainly
        instead of the old "re-ingest for it to take effect".
        """
        import brain.embedder as emb_mod
        from config.settings_store import update_overlay

        update_overlay({"EMBED_MODEL": name, "EMBED_DIM": dim})
        emb_mod._model = None           # force a lazy reload on next use
        emb_mod._model_name = None
        self.active_embedder_label = label

        if dim != 768:
            self._toast(f"{name}: {dim}-dim. Index is 768-dim — "
                        f"search is DISABLED until you reindex.")
        else:
            self._toast(f"Embedder → {name}. Run reindex; search is disabled "
                        f"until you do.")

    def on_new_chat(self):
        """Start a fresh conversation. The previous one stays saved."""
        sid = self._history.new_session()
        self.answer_display = ""
        self._turn_start = 0
        self._toast(f"New conversation started. Previous one is saved.")
        log.info(f"New chat session {sid}")

    def on_show_history(self):
        """Reload the current conversation into the answer pane."""
        text = self._history.transcript()
        if not text.strip():
            self._toast("No history in this conversation yet.")
            return
        self.answer_display = text
        self._turn_start = len(text)
        Clock.schedule_once(self._reflow_answer, 0)

    def _menu_then(self, action):
        """Dismiss the dropdown, then run the action on the next frame.

        Opening a dialog while the menu is still closing leaves the dialog
        behind the overlay and it cannot be clicked.
        """
        if self._pdf_menu:
            self._pdf_menu.dismiss()
        Clock.schedule_once(lambda *_: action(), 0.05)

    def _toast(self, text: str):
        try:
            Snackbar(text=text, duration=2.4).open()
        except Exception:
            log.info(text)

    # ── Library management ───────────────────────────────────────────────────
    #
    # Every destructive action goes through _confirm() and every long one
    # through _in_background(), so the UI cannot be left looking frozen and a
    # mis-tap cannot delete an index. All of them call LibraryService, the same
    # object `main.py doctor` uses, so the GUI can never report or do something
    # the CLI would disagree with.
    def _confirm(self, title: str, text: str, on_yes):
        from kivymd.uix.button import MDFlatButton
        from kivymd.uix.dialog import MDDialog

        dialog = MDDialog(title=title, text=text)

        def close(*_):
            dialog.dismiss()

        def go(*_):
            dialog.dismiss()
            on_yes()

        dialog.buttons = [
            MDFlatButton(text="CANCEL", on_release=close),
            MDFlatButton(text="CONTINUE", on_release=go),
        ]
        dialog.open()

    def _in_background(self, label: str, work, done=None):
        """Run `work()` off the UI thread; `done(result)` is called on it."""
        if self._ingest_thread and self._ingest_thread.is_alive():
            self._toast("Another library job is still running.")
            return
        self.ingest_status = label
        self.ingest_progress_value = 0.08
        Clock.schedule_interval(self._ingest_pulse, 0.45)

        def job():
            try:
                result = work()
            except Exception as ex:                     # noqa: BLE001
                log.exception(ex)
                Clock.schedule_once(lambda *_: self._ingest_failed(str(ex)), 0)
                return

            def finish(*_):
                self.ingest_progress_value = 1.0
                self.ingest_status = ""
                self.refresh_pdf_list()
                self.refresh_library_info()
                if done:
                    done(result)

            Clock.schedule_once(finish, 0)

        self._ingest_thread = threading.Thread(target=job, daemon=True)
        self._ingest_thread.start()

    def refresh_library_info(self):
        """One line of truth about the library, from LibraryService."""
        try:
            from storage.library import LibraryService
            rep = LibraryService().report()
            c = rep["counts"]
            s = rep.get("settings") or {}
            bits = [f"{c['books_indexed']} books",
                    f"{c['chunks_indexed']:,} chunks"]
            if s.get("embed_model"):
                bits.append(str(s["embed_model"]).split("/")[-1])
            if rep["healthy"]:
                bits.append("healthy")
            else:
                if rep["holes"]:
                    bits.append(f"{len(rep['holes'])} not indexed")
                if any(rep["orphans"].values()):
                    bits.append("orphans — run Sync")
                if rep["pending"]:
                    bits.append(f"{len(rep['pending'])} pending — run Ingest")
            self.library_info = "  ·  ".join(bits)
        except Exception as ex:                          # noqa: BLE001
            log.warning(f"Library info failed: {ex}")
            self.library_info = "Could not read library state"

    def on_reindex(self):
        """Rebuild vectors from cached text. No OCR, no PDF parsing."""
        def work():
            from core.reindex import reindex
            return reindex(device="cuda")

        self._confirm(
            "Reindex vectors?",
            "Rebuilds every vector from the extracted text already on disk. "
            "No OCR and no re-reading of PDFs. Takes a few minutes.",
            lambda: self._in_background(
                "Re-embedding every chunk…", work,
                lambda r: self._toast(
                    f"Reindexed {r.get('books', '?')} book(s), "
                    f"{r.get('chunks', 0):,} chunks")))

    def on_sync_library(self):
        """Reconcile every derived store against data/input/."""
        def work():
            from storage.library import LibraryService
            return LibraryService().clean(orphans=True)

        self._confirm(
            "Sync to disk?",
            "Removes index entries, text and manifest records belonging to "
            "documents that are no longer in data/input. Nothing in "
            "data/input is touched.",
            lambda: self._in_background(
                "Reconciling derived stores…", work,
                lambda r: self._toast(
                    f"Removed {sum(len(v) for v in r.values())} orphaned item(s)")))

    def on_clear_index(self):
        """Drop vectors and checkpoint, keeping PDFs and extracted text."""
        def work():
            from storage.library import LibraryService
            return LibraryService().clean(index=True, checkpoint=True)

        self._confirm(
            "Clear all chunks?",
            "Deletes the vector index, its metadata and the manifest, and "
            "resets the checkpoint. Your PDFs and extracted text are kept, so "
            "Reindex can rebuild everything without re-reading any PDF.",
            lambda: self._in_background(
                "Clearing index…", work,
                lambda r: self._toast(
                    f"Cleared {sum(len(v) for v in r.values())} file(s). "
                    "Run Reindex to rebuild.")))

    def remove_book(self, source: str):
        """Delete one document and every derived record of it."""
        def work():
            from storage.library import LibraryService
            svc = LibraryService()
            ok = svc.purge_book(source)
            path = os.path.join(Config().INPUT_DIR, source)
            if os.path.exists(path):
                os.remove(path)
            return ok

        self._confirm(
            "Remove this book?",
            f"Deletes {source} from data/input and removes its chunks, text "
            f"and manifest entry. This cannot be undone from inside the app.",
            lambda: self._in_background(
                f"Removing {source}…", work,
                lambda ok: self._toast(
                    f"Removed {source}" if ok else f"Could not remove {source}")))

    def show_remove_menu(self, btn):
        """Pick a book to remove."""
        from storage.library import LibraryService
        books = sorted(LibraryService().pdfs_on_disk())
        if not books:
            self._toast("No documents in data/input.")
            return
        if self._pdf_menu:
            self._pdf_menu.dismiss()

        def make(src):
            def go(*_):
                if self._pdf_menu:
                    self._pdf_menu.dismiss()
                self.remove_book(src)
            return go

        self._pdf_menu = MDDropdownMenu(
            caller=btn,
            items=[{"text": b, "on_release": make(b)} for b in books],
            width_mult=6, max_height=dp(400),
        )
        self._pdf_menu.open()


class ParseDatMaterialApp(MDApp):
    def __init__(
        self,
        model_path: str | None = None,
        gpu_layers: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_path = model_path
        self.gpu_layers = gpu_layers
        self.rag = None
        self.rag_ok = False
        self._root: ParseDatMaterialRoot | None = None

    def build(self):
        self.title = "ParseDat_Diary — Chat with Books"

        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "DeepPurple"
        self.theme_cls.accent_palette  = "Amber"

        kv_path = Path(__file__).with_name("material_app.kv")
        if not kv_path.is_file():
            log.error(f"Missing KV: {kv_path}")
            return MDBoxLayout()
        Builder.load_file(str(kv_path))

        root = ParseDatMaterialRoot(app_ref=self)
        self._root = root

        threading.Thread(target=self._load_rag_background, daemon=True).start()
        return root

    def _load_rag_background(self):
        try:
            from brain.rag_pipeline import RAGPipeline
            rag = RAGPipeline(model_path=self.model_path, gpu_layers=self.gpu_layers)
            ok = bool(rag.setup())
        except Exception as ex:
            log.exception(ex)
            Clock.schedule_once(lambda *_: self._rag_ui_fail(str(ex)), 0)
            return

        def apply(*_):
            self.rag = rag
            self.rag_ok = ok
            if self._root is None:
                return
            if ok:
                chunks = rag.retriever.chunk_count
                books = rag.retriever.book_count
                self._root.rag_status_hint = (
                    f"Ready — {books} book(s), {chunks:,} chunk(s). Ask away."
                )
                self._root._toast("AI ready.")
            else:
                self._root.rag_status_hint = (
                    "LLM not loaded. Set LLM_MODEL_PATH or pass --model. "
                    "Ingest and Read still work."
                )

        Clock.schedule_once(apply, 0)

    def _rag_ui_fail(self, msg: str):
        self.rag_ok = False
        if self._root:
            self._root.rag_status_hint = f"Init error: {msg[:220]}"


def run_gui(model_path: str | None = None, gpu_layers: int | None = None):
    ParseDatMaterialApp(model_path=model_path, gpu_layers=gpu_layers).run()


if __name__ == "__main__":
    mp     = os.environ.get("PARSEDAT_MODEL_PATH") or None
    layers = os.environ.get("PARSEDAT_GPU_LAYERS")
    gl     = int(layers) if layers and str(layers).isdigit() else None
    run_gui(model_path=mp, gpu_layers=gl)
