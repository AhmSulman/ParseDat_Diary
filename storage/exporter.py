import os, json
from datetime import datetime
from logs.logger import log

class Exporter:
    def __init__(self):
        os.makedirs("data/txt", exist_ok=True)
        os.makedirs("data/json", exist_ok=True)

    def save(self, pdf_name: str, text: str):
        base = os.path.splitext(pdf_name)[0]
        with open(f"data/txt/{base}.txt", "w", encoding="utf-8") as f:
            f.write(text)
        payload = {"source": pdf_name, "extracted_at": datetime.utcnow().isoformat()+"Z",
                   "char_count": len(text), "text": text}
        with open(f"data/json/{base}.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        log.info(f"   💾 Saved: data/txt/{base}.txt + .json")
