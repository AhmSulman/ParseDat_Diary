import json
import os
from datetime import datetime, timezone

from core.normalize import normalize
from logs.logger import log

_ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TXT_DIR  = os.path.join(_ROOT, "data", "txt")
_QUAR_DIR = os.path.join(_ROOT, "data", "quarantine")


class Exporter:
    """
    Writes extracted book text to data/txt/.

    Text is NORMALISED before it hits disk, so every later stage — chunking,
    embedding, reindex — reads repaired text. Normalisation is idempotent, so
    re-running over an already-clean file is a no-op.

    data/json/ is no longer written. storage/exporter.py was the only thing that
    ever produced it and nothing anywhere read it back: 7.6 MB duplicating
    data/txt/ with a metadata wrapper. Book-level metadata now lives in
    storage/manifest.py, which is actually consumed.
    """

    def __init__(self):
        os.makedirs(_TXT_DIR, exist_ok=True)

    def save(self, pdf_name: str, text: str) -> str:
        """Normalise and write the book text. Returns the path written."""
        base = os.path.splitext(pdf_name)[0]
        txt_path = os.path.join(_TXT_DIR, f"{base}.txt")

        cleaned = normalize(text)

        tmp = txt_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(cleaned)
        os.replace(tmp, txt_path)

        log.info(f"Saved: {os.path.basename(txt_path)} ({len(cleaned):,} chars)")
        return txt_path

    def quarantine(self, pdf_name: str, text: str, report) -> str:
        """
        Park a book that failed the quality gate, with the reasons.

        Quarantined books are never chunked and never indexed. They are kept
        rather than deleted so the decision stays inspectable — a gate that
        silently drops books is indistinguishable from a bug.
        """
        os.makedirs(_QUAR_DIR, exist_ok=True)
        base = os.path.splitext(pdf_name)[0]

        with open(os.path.join(_QUAR_DIR, f"{base}.txt"), "w", encoding="utf-8") as f:
            f.write(text)

        report_path = os.path.join(_QUAR_DIR, f"{base}.report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump({
                "source": pdf_name,
                "quarantined_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "reasons": list(getattr(report, "reasons", [])),
                "metrics": dict(getattr(report, "metrics", {})),
            }, f, ensure_ascii=False, indent=2)

        log.warning(f"QUARANTINED {pdf_name}: {'; '.join(report.reasons)}")
        return report_path
