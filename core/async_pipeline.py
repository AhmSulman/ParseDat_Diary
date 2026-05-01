"""
Async Pipeline — 10x Speed Engine
===================================
Instead of processing PDFs one-by-one (slow), this does everything
in parallel using Python's asyncio + queues.

Think of it like a restaurant:
  Old way: 1 chef cooks 1 meal, serves it, then cooks the next.
  New way: Kitchen reads orders, cooks batch, serves batch — all at once.

FLOW:
  [PDF files] → read_queue → [GPU OCR workers] → write_queue → [Disk writers]
                                     ↓
                              [Embedder queue] → [FAISS indexer]
"""

import asyncio
import os
import time
from pathlib import Path

import fitz  # PyMuPDF

from core.gpu_ocr import GPUOCRBatch
from core.extract_text import TextExtractor
from storage.exporter import Exporter
from storage.checkpoint import Checkpoint
from brain.chunker import Chunker
from brain.retriever import Retriever
from config.config import Config
from logs.logger import log


class AsyncPipeline:
    def __init__(self):
        self.cfg = Config()
        self.text_extractor = TextExtractor()
        self.gpu_ocr = GPUOCRBatch()
        self.exporter = Exporter()
        self.checkpoint = Checkpoint()
        self.chunker = Chunker()
        self.retriever = Retriever()

        # Async queues — the "conveyor belts" between workers
        self.read_queue: asyncio.Queue = asyncio.Queue(maxsize=32)
        self.write_queue: asyncio.Queue = asyncio.Queue(maxsize=32)
        self.embed_queue: asyncio.Queue = asyncio.Queue(maxsize=64)

        self._stats = {"done": 0, "skipped": 0, "failed": 0, "pages": 0}

    # ── Entry point ───────────────────────────────────────────────────────────
    async def run(self):
        pdfs = self._list_pdfs()

        if not pdfs:
            log.warning(f"⚠️  No PDFs in '{self.cfg.INPUT_DIR}'. Drop some and retry.")
            return

        log.info(f"🚀 MAAN Pipeline starting — {len(pdfs)} PDFs, {self.cfg.ASYNC_WORKERS} workers")
        self.retriever.load()

        t_start = time.perf_counter()

        # Run all workers concurrently
        await asyncio.gather(
            self._reader_worker(pdfs),                             # 1 reader
            *[self._ocr_worker(i) for i in range(self.cfg.ASYNC_WORKERS)],  # N OCR workers
            self._writer_worker(),                                 # 1 writer
            self._embed_worker(),                                  # 1 embedder
        )

        self.retriever.save()
        elapsed = time.perf_counter() - t_start

        log.info("=" * 55)
        log.info(f"✅ Done in {elapsed:.1f}s")
        log.info(f"   Processed : {self._stats['done']}")
        log.info(f"   Skipped   : {self._stats['skipped']}")
        log.info(f"   Failed    : {self._stats['failed']}")
        log.info(f"   Pages     : {self._stats['pages']}")
        log.info(f"   Speed     : {self._stats['pages'] / max(elapsed, 1):.1f} pages/sec")
        log.info("=" * 55)

    # ── Stage 1: Reader ───────────────────────────────────────────────────────
    async def _reader_worker(self, pdfs: list):
        """Reads PDFs from disk and pushes pages into the read_queue."""
        for pdf_name in pdfs:
            if self.checkpoint.is_done(pdf_name):
                self._stats["skipped"] += 1
                log.info(f"⏭️  Skip: {pdf_name}")
                continue

            path = os.path.join(self.cfg.INPUT_DIR, pdf_name)
            try:
                # fitz.open is CPU-bound — run in thread so we don't block the event loop
                doc = await asyncio.to_thread(fitz.open, path)
                pages = list(doc)
                log.info(f"📄 Queued: {pdf_name} ({len(pages)} pages)")

                # Push the whole document as one job
                await self.read_queue.put({"name": pdf_name, "pages": pages})

            except Exception as e:
                log.error(f"❌ Read failed: {pdf_name} — {e}")
                self.checkpoint.mark_failed(pdf_name)
                self._stats["failed"] += 1

        # Signal workers that reading is done (one sentinel per worker)
        for _ in range(self.cfg.ASYNC_WORKERS):
            await self.read_queue.put(None)

    # ── Stage 2: OCR Workers (N parallel) ────────────────────────────────────
    async def _ocr_worker(self, worker_id: int):
        """
        Pulls a PDF from read_queue, extracts text (GPU OCR if needed),
        then pushes result to write_queue.
        """
        while True:
            job = await self.read_queue.get()

            if job is None:  # Sentinel — no more work
                self.read_queue.task_done()
                break

            pdf_name = job["name"]
            pages = job["pages"]
            extracted_pages = []

            try:
                # Collect pages that need OCR (no digital text)
                text_pages, ocr_needed = [], []

                for pg_num, page in enumerate(pages, 1):
                    text = await asyncio.to_thread(self.text_extractor.run, page)
                    if text:
                        extracted_pages.append(f"--- Page {pg_num} ---\n{text}")
                        text_pages.append(pg_num)
                    else:
                        ocr_needed.append((pg_num, page))

                # GPU batch OCR for scanned pages
                if ocr_needed:
                    log.info(f"   🖼️  [{worker_id}] {pdf_name}: {len(ocr_needed)} pages → GPU OCR")
                    pg_nums = [p[0] for p in ocr_needed]
                    pg_objs = [p[1] for p in ocr_needed]

                    ocr_texts = await asyncio.to_thread(
                        self.gpu_ocr.infer_batch, pg_objs
                    )

                    for pg_num, ocr_text in zip(pg_nums, ocr_texts):
                        if ocr_text:
                            extracted_pages.append(f"--- Page {pg_num} (OCR) ---\n{ocr_text}")

                self._stats["pages"] += len(pages)

                # Sort pages back into order
                full_text = "\n\n".join(
                    sorted(extracted_pages, key=lambda x: int(
                        x.split("---")[1].strip().split()[1]
                    ) if "---" in x else 0)
                )

                await self.write_queue.put({
                    "name": pdf_name,
                    "text": full_text or "[NO TEXT EXTRACTED]",
                    "pages": len(pages),
                })

            except Exception as e:
                log.error(f"❌ OCR failed: {pdf_name} — {e}")
                self.checkpoint.mark_failed(pdf_name)
                self._stats["failed"] += 1

            self.read_queue.task_done()

        # Signal writer when last OCR worker is done
        await self.write_queue.put(None)

    # ── Stage 3: Writer ───────────────────────────────────────────────────────
    async def _writer_worker(self):
        """Saves extracted text to disk and forwards to embedder."""
        none_count = 0

        while True:
            job = await self.write_queue.get()

            if job is None:
                none_count += 1
                if none_count >= self.cfg.ASYNC_WORKERS:
                    await self.embed_queue.put(None)
                    break
                continue

            try:
                await asyncio.to_thread(
                    self.exporter.save, job["name"], job["text"]
                )
                self.checkpoint.mark_done(job["name"])
                self._stats["done"] += 1

                # Forward to embedder
                await self.embed_queue.put(job)
                log.info(f"   💾 Saved: {job['name']}")

            except Exception as e:
                log.error(f"❌ Write failed: {job['name']} — {e}")

    # ── Stage 4: Embedder (background) ───────────────────────────────────────
    async def _embed_worker(self):
        """Chunks text and adds to FAISS vector index."""
        while True:
            job = await self.embed_queue.get()

            if job is None:
                break

            try:
                chunks = self.chunker.chunk(job["text"])
                for chunk in chunks:
                    await asyncio.to_thread(
                        self.retriever.add, chunk, {"source": job["name"]}
                    )
            except Exception as e:
                log.warning(f"⚠️  Embed failed: {job['name']} — {e}")

    def _list_pdfs(self) -> list:
        os.makedirs(self.cfg.INPUT_DIR, exist_ok=True)
        return sorted(
            f for f in os.listdir(self.cfg.INPUT_DIR)
            if f.lower().endswith(".pdf")
        )
