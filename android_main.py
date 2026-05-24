"""
MAAN Android Client
====================
Connects to a MAAN desktop server (FastAPI at port 8000).
The heavy AI runs on your PC/GPU; this app is the mobile front-end.

Run on Android via buildozer.
To start the server on your desktop: python main.py server
"""

from __future__ import annotations

import json
import os
import threading

from kivy.clock import Clock
from kivy.lang import Builder
from kivy.metrics import dp, sp
from kivy.uix.popup import Popup
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.properties import StringProperty, NumericProperty

from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.snackbar import Snackbar

KV = """
#:import dp kivy.metrics.dp
#:import sp kivy.metrics.sp

<AndroidRoot>:
    orientation: 'vertical'
    md_bg_color: [0.05, 0.04, 0.09, 1]

    MDTopAppBar:
        title: 'MAAN'
        subtitle: 'Chat with Books'
        elevation: 6
        md_bg_color: app.theme_cls.primary_color
        specific_text_color: 1, 1, 1, 1

    ScreenManager:
        id: sm
        size_hint_y: 1

        # ── Chat screen ────────────────────────────────────────
        Screen:
            name: 'chat'
            MDBoxLayout:
                orientation: 'vertical'
                spacing: dp(10)
                padding: dp(12)

                # Server status
                MDCard:
                    size_hint_y: None
                    height: dp(40)
                    padding: dp(10), dp(6)
                    radius: [dp(12)]
                    elevation: 2
                    md_bg_color: [0.14, 0.10, 0.24, 1]
                    MDLabel:
                        text: root.server_status
                        font_style: 'Caption'
                        theme_text_color: 'Secondary'
                        adaptive_height: True

                # Answer display
                MDCard:
                    elevation: 4
                    padding: dp(12)
                    radius: [dp(16)]
                    size_hint_y: 1
                    md_bg_color: [0.10, 0.08, 0.17, 1]
                    ScrollView:
                        id: ans_scroll
                        bar_width: dp(4)
                        bar_color: app.theme_cls.accent_color
                        MDLabel:
                            id: answer_lbl
                            text: root.answer_text
                            theme_text_color: 'Primary'
                            adaptive_height: True
                            size_hint_y: None
                            height: self.texture_size[1]
                            line_height: 1.6
                            padding: dp(6), dp(6)
                            font_size: sp(14)

                # Input row
                MDCard:
                    padding: dp(6)
                    radius: [dp(14)]
                    elevation: 3
                    size_hint_y: None
                    height: dp(110)
                    md_bg_color: [0.14, 0.10, 0.24, 1]
                    MDTextField:
                        id: q_field
                        hint_text: 'Ask a question about your PDFs…'
                        multiline: True
                        mode: 'fill'
                        fill_color_normal: 0, 0, 0, 0
                        fill_color_focus: 0, 0, 0, 0
                        size_hint_y: None
                        height: dp(100)

                MDFillRoundFlatIconButton:
                    icon: 'send'
                    text: 'Ask'
                    size_hint_x: 1
                    height: dp(52)
                    md_bg_color: app.theme_cls.accent_color
                    theme_text_color: 'Custom'
                    text_color: 0, 0, 0, 1
                    on_release: root.on_ask()

        # ── Settings screen ────────────────────────────────────
        Screen:
            name: 'settings'
            MDBoxLayout:
                orientation: 'vertical'
                spacing: dp(14)
                padding: dp(16)

                MDLabel:
                    text: 'Settings'
                    font_style: 'H5'
                    bold: True
                    theme_text_color: 'Primary'
                    adaptive_height: True

                MDCard:
                    padding: dp(12)
                    radius: [dp(14)]
                    elevation: 3
                    size_hint_y: None
                    height: dp(90)
                    md_bg_color: [0.14, 0.10, 0.24, 1]
                    MDTextField:
                        id: server_field
                        hint_text: 'MAAN server URL'
                        text: root.server_url
                        mode: 'fill'
                        fill_color_normal: 0, 0, 0, 0
                        fill_color_focus: 0, 0, 0, 0
                        size_hint_y: None
                        height: dp(78)

                MDFillRoundFlatIconButton:
                    icon: 'connection'
                    text: 'Connect'
                    size_hint_x: 1
                    height: dp(48)
                    md_bg_color: app.theme_cls.primary_color
                    on_release: root.on_connect()

                Widget:
                    size_hint_y: 1

    # ── Bottom nav ─────────────────────────────────────────────
    MDBottomNavigation:
        size_hint_y: None
        height: dp(56)
        selected_color_background: app.theme_cls.accent_color

        MDBottomNavigationItem:
            name: 'tab_chat'
            text: 'Ask AI'
            icon: 'robot'
            on_tab_press: sm.current = 'chat'

        MDBottomNavigationItem:
            name: 'tab_settings'
            text: 'Server'
            icon: 'server'
            on_tab_press: sm.current = 'settings'
"""


class AndroidRoot(MDBoxLayout):
    server_status = StringProperty("Tap Server tab → enter desktop IP → Connect")
    answer_text   = StringProperty("Your answers will appear here.")
    server_url    = StringProperty("http://192.168.1.100:8000")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._busy = False

    def on_connect(self):
        fld = self.ids.get("server_field")
        if fld and fld.text.strip():
            self.server_url = fld.text.strip()
        self._check_server()

    def _check_server(self):
        self.server_status = f"Connecting to {self.server_url} …"

        def worker():
            try:
                import requests
                r = requests.get(f"{self.server_url}/status", timeout=5)
                data = r.json()
                chunks = data.get("indexed_chunks", 0)
                llm    = data.get("llm_loaded", False)
                msg = f"Connected ✓  |  {chunks:,} chunks  |  LLM: {'✓' if llm else '✗'}"
            except Exception as e:
                msg = f"Cannot reach server: {e}"
            Clock.schedule_once(lambda *_: setattr(self, "server_status", msg), 0)

        threading.Thread(target=worker, daemon=True).start()

    def on_ask(self):
        fld = self.ids.get("q_field")
        if not fld:
            return
        q = fld.text.strip()
        if not q:
            self._toast("Type a question first.")
            return
        if self._busy:
            self._toast("Waiting for previous answer…")
            return
        self._busy = True
        fld.text = ""
        self.answer_text = ""

        threading.Thread(target=self._ask_worker, args=(q,), daemon=True).start()

    def _ask_worker(self, question: str):
        try:
            import requests
            payload = {"question": question, "top_k": 8}
            with requests.post(
                f"{self.server_url}/chat",
                json=payload,
                stream=True,
                timeout=120,
            ) as resp:
                for line in resp.iter_lines():
                    if line:
                        text = line.decode("utf-8", errors="replace")
                        if text.startswith("data:"):
                            text = text[5:].strip()
                        tok = text
                        Clock.schedule_once(
                            lambda dt, t=tok: setattr(self, "answer_text", self.answer_text + t),
                            0,
                        )
        except Exception as e:
            Clock.schedule_once(
                lambda *_: setattr(self, "answer_text", f"Error: {e}"), 0
            )
        finally:
            def done(*_):
                self._busy = False
            Clock.schedule_once(done, 0)

    def _toast(self, text: str):
        try:
            Snackbar(text=text, duration=2.2).open()
        except Exception:
            pass


class MaanAndroidApp(MDApp):
    def build(self):
        self.title = "MAAN"
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "DeepPurple"
        self.theme_cls.accent_palette  = "Amber"
        Builder.load_string(KV)
        return AndroidRoot()


if __name__ == "__main__":
    MaanAndroidApp().run()
