"""
Model Auto-Downloader
=====================
Run this once (or it runs automatically on first launch) to
pull the default LLM model from HuggingFace.

Usage:
    python data/models/auto_download.py
"""

import os
import sys
import urllib.request

# ── Edit these to swap models ──────────────────────────────────
MODEL_FILENAME = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
DOWNLOAD_URL   = (
    "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
    "/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
)
# ──────────────────────────────────────────────────────────────

DEST = os.path.join(os.path.dirname(__file__), MODEL_FILENAME)


def _progress(count, block, total):
    done = count * block
    pct  = min(100, done * 100 // max(total, 1))
    bar  = "█" * (pct // 2) + "░" * (50 - pct // 2)
    mb   = done / 1024 / 1024
    sys.stdout.write(f"\r  [{bar}] {pct:3d}%  {mb:6.1f} MB")
    sys.stdout.flush()


def download_model():
    if os.path.exists(DEST):
        size_gb = os.path.getsize(DEST) / 1024 / 1024 / 1024
        print(f"✅  Model already present: {MODEL_FILENAME}  ({size_gb:.2f} GB)")
        return DEST

    print(f"⬇️   Downloading {MODEL_FILENAME} (~4.4 GB) …")
    print(f"    Source: {DOWNLOAD_URL}")
    print()
    try:
        urllib.request.urlretrieve(DOWNLOAD_URL, DEST, reporthook=_progress)
    except Exception as e:
        if os.path.exists(DEST):
            os.remove(DEST)
        print(f"\n❌  Download failed: {e}")
        sys.exit(1)

    print(f"\n✅  Saved to {DEST}")
    return DEST


if __name__ == "__main__":
    download_model()
