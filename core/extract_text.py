class TextExtractor:
    def run(self, page) -> str:
        try:
            text = page.get_text("text").strip()
            return text if len(text) > 10 else ""
        except Exception:
            return ""
