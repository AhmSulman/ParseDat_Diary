import os, json
from datetime import datetime
from logs.logger import log

_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TXT_DIR = os.path.join(_ROOT, "data", "txt")
_JSON_DIR = os.path.join(_ROOT, "data", "json")

class Exporter:
    def __init__(self):
        os.makedirs(_TXT_DIR, exist_ok=True)
        os.makedirs(_JSON_DIR, exist_ok=True)

    def save(self, pdf_name: str, text: str):
        base = os.path.splitext(pdf_name)[0]
        txt_path  = os.path.join(_TXT_DIR, f"{base}.txt")
        json_path = os.path.join(_JSON_DIR, f"{base}.json")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        payload = {"source": pdf_name, "extracted_at": datetime.utcnow().isoformat()+"Z",
                   "char_count": len(text), "text": text}
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        log.info(f"Saved: {txt_path}")
