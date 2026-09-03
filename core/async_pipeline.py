"""
Async Pipeline — PDF -> text -> quality gate -> chunks -> vectors
==================================================================
Stages run concurrently over asyncio queues:

  [PDFs] -> read_queue -> [OCR workers] -> write_queue -> [writer]
                                                            |
                                                    embed_queue -> [embedder]

CHECKPOINT ORDERING IS LOAD-BEARING
-----------------------------------
The old pipeline marked a book "done" in the WRITER stage, as soon as its text
hit disk — before it was ever embedded. `retriever.save()` then ran once, at the
very end of run(). So any crash, or the user closing the window mid-run, threw
away every vector from that session while the checkpoint had already recorded
those books as complete.

The next `ingest` skipped them. Permanently. That is how the library reached
18 books marked done, 7 actually indexed, and 11 holes that no re-run could
heal — the "it gets dumber and dumber" behaviour.

Two rules now hold, and both matter:

  1. `mark_done()` is called ONLY after the book's vectors are in the index —
     in the embed stage, never the writer.
  2. `retriever.save()` runs after EACH book, not once at the end, so a crash
     costs at most the book in flight.

A book that fails to embed is marked failed, not done, so the next run retries.
"""

import asyncio
import os
import time

import fitz  # PyMuPDF

from brain.chunker import Chunker
from brain.retriever import Retriever
from config.config import Config
from core.extract_text import TextExtractor
from core.gpu_ocr import GPUOCRBatch
from core.normalize import normalize
from core.quality import score
from core.sources import is_markdown, is_pdf, list_sources, read_text_source
from logs.logger import log
from storage.checkpoint import Checkpoint
from storage.exporter import Exporter
from storage.manifest import Manifest, file_sha256


class AsyncPipeline:
    def __init__(self):
        self.cfg = Config()
        self.text_extractor = TextExtractor()
        self.gpu_ocr = GPUOCRBatch()
        self.exporter = Exporter()
        self.checkpoint = Checkpoint()
        self.chunker = Chunker()
        self.retriever = Retriever()
        self.manifest = Manifest()

        self.read_queue: asyncio.Queue = asyncio.Queue(maxsize=32)
        self.write_queue: asyncio.Queue = asyncio.Queue(maxsize=32)
        self.embed_queue: asyncio.Queue = asyncio.Queue(maxsize=64)

        self._stats = {
            "done": 0, "skipped": 0, "failed": 0,
            "pages": 0, "chunks": 0, "quarantined": 0,
        }

    # ── Entry point ───────────────────────────────────────────────────────────
    async def run(self):
        pdfs = self._list_pdfs()
        if not pdfs:
            log.warning(f"No PDFs in '{self.cfg.INPUT_DIR}'. Drop some in and retry.")
            return

        log.info(f"ParseDat_Diary pipeline: {len(pdfs)} PDFs, {self.cfg.ASYNC_WORKERS} workers")
        self.retriever.load()

        t0 = time.perf_counter()
        await asyncio.gather(
            self._reader_worker(pdfs),
            *[self._ocr_worker(i) for i in range(self.cfg.ASYNC_WORKERS)],
            self._writer_worker(),
            self._embed_worker(),
        )
        # The model that ACTUALLY loaded, not the configured one: the configured
        # embedder can fall back, and the fallback is also 768-dim, so recording
        # the wrong name would let this index later be queried with a different
        # model at matching dimensions — silent garbage no dim check catches.
        resolved = self.retriever.embedder.model_name or self.cfg.EMBED_MODEL
        self.manifest.set_settings(
            embed_model=resolved,
            embed_dim=self.cfg.EMBED_DIM,
            chunk_size=self.cfg.CHUNK_SIZE,
            chunk_overlap=self.cfg.CHUNK_OVERLAP,
        )
        if resolved != self.cfg.EMBED_MODEL:
            log.warning(
                f"Configured embedder '{self.cfg.EMBED_MODEL}' did not load; "
                f"index built with fallback '{resolved}'."
            )

        self.retriever.save()
        self.manifest.save()

        elapsed = time.perf_counter() - t0
        s = self._stats
        log.info("=" * 55)
        log.info(f"Done in {elapsed:.1f}s")
        log.info(f"   Indexed     : {s['done']}")
        log.info(f"   Skipped     : {s['skipped']}")
        log.info(f"   Quarantined : {s['quarantined']}")
        log.info(f"   Failed      : {s['failed']}")
        log.info(f"   Pages       : {s['pages']}")
        log.info(f"   Chunks      : {s['chunks']}")
        log.info(f"   Library     : {self.manifest.book_count} books, "
                 f"{self.manifest.chunk_count} chunks")
        log.info("=" * 55)

    # ── Stage 1: Reader ───────────────────────────────────────────────────────
    async def _reader_worker(self, pdfs: list):
        seen_hashes = self.manifest.sha_index()

        for pdf_name in pdfs:
            if self.checkpoint.is_done(pdf_name) and self.manifest.has(pdf_name):
                # Only skip when BOTH agree. The checkpoint alone used to be
                # enough, which is exactly how books became permanently skipped
                # without ever reaching the index.
                self._stats["skipped"] += 1
                log.info(f"Skip (already indexed): {pdf_name}")
                continue

            path = os.path.join(self.cfg.INPUT_DIR, pdf_name)

            # Content-hash dedup: the same book re-added under a different
            # filename must not be indexed twice.
            try:
                sha = await asyncio.to_thread(file_sha256, path)
                if sha in seen_hashes and seen_hashes[sha] != pdf_name:
                    log.info(f"Skip (same content as '{seen_hashes[sha]}'): {pdf_name}")
                    self._stats["skipped"] += 1
                    continue
            except OSError:
                sha = ""

            try:
                if not is_pdf(pdf_name):
                    # Already text: reading it through PyMuPDF and Tesseract
                    # would be pointless and lossy. Hand it straight to the
                    # writer with no pages and no OCR.
                    body = await asyncio.to_thread(read_text_source, path)
                    log.info(f"Queued: {pdf_name} (text, {len(body):,} chars)")
                    await self.write_queue.put({
                        "name": pdf_name, "text": body,
                        "pages": 0, "sha256": sha,
                    })
                    continue

                doc = await asyncio.to_thread(fitz.open, path)
                pages = list(doc)
                log.info(f"Queued: {pdf_name} ({len(pages)} pages)")
                await self.read_queue.put(
                    {"name": pdf_name, "pages": pages, "sha256": sha}
                )
            except Exception as e:
                log.error(f"Read failed: {pdf_name} - {e}")
                self.checkpoint.mark_failed(pdf_name)
                self._stats["failed"] += 1

        for _ in range(self.cfg.ASYNC_WORKERS):
            await self.read_queue.put(None)

    # ── Stage 2: OCR workers ─────────────────────────────────────────────────
    async def _ocr_worker(self, worker_id: int):
        while True:
            job = await self.read_queue.get()
            if job is None:
                self.read_queue.task_done()
                break

            pdf_name, pages = job["name"], job["pages"]
            try:
                extracted: list[tuple[int, str]] = []
                ocr_needed = []

                for pg_num, page in enumerate(pages, 1):
                    text = await asyncio.to_thread(self.text_extractor.run, page)
                    if text:
                        extracted.append((pg_num, f"--- Page {pg_num} ---\n{text}"))
                    else:
                        ocr_needed.append((pg_num, page))

                if ocr_needed:
                    log.info(f"   [{worker_id}] {pdf_name}: {len(ocr_needed)} pages -> OCR")
                    texts = await asyncio.to_thread(
                        self.gpu_ocr.infer_batch, [p for _, p in ocr_needed]
                    )
                    for (pg_num, _), ocr_text in zip(ocr_needed, texts):
                        if ocr_text:
                            extracted.append(
                                (pg_num, f"--- Page {pg_num} (OCR) ---\n{ocr_text}")
                            )

                self._stats["pages"] += len(pages)

                # Sort by the page number carried alongside, rather than by
                # re-parsing it out of the header string.
                extracted.sort(key=lambda t: t[0])
                full_text = "\n\n".join(t for _, t in extracted)

                await self.write_queue.put({
                    "name": pdf_name,
                    "text": full_text,
                    "pages": len(pages),
                    "sha256": job.get("sha256", ""),
                })
            except Exception as e:
                log.error(f"OCR failed: {pdf_name} - {e}")
                self.checkpoint.mark_failed(pdf_name)
                self._stats["failed"] += 1

            self.read_queue.task_done()

        await self.write_queue.put(None)

    # ── Stage 3: Writer — normalise, quality-gate, persist text ──────────────
    async def _writer_worker(self):
        none_count = 0
        while True:
            job = await self.write_queue.get()
            if job is None:
                none_count += 1
                if none_count >= self.cfg.ASYNC_WORKERS:
                    await self.embed_queue.put(None)
                    break
                continue

            name = job["name"]
            try:
                text = normalize(job["text"])

                report = await asyncio.to_thread(score, text, not is_pdf(name))
                if not report.passed:
                    # Discarded, not repaired: quality over coverage.
                    await asyncio.to_thread(
                        self.exporter.quarantine, name, text, report
                    )
                    self.checkpoint.mark_failed(name)
                    self._stats["quarantined"] += 1
                    continue

                await asyncio.to_thread(self.exporter.save, name, text)
                job["text"] = text
                await self.embed_queue.put(job)
            except Exception as e:
                log.error(f"Write failed: {name} - {e}")
                self.checkpoint.mark_failed(name)
                self._stats["failed"] += 1

    # ── Stage 4: Embedder — the ONLY place a book is marked done ─────────────
    async def _embed_worker(self):
        while True:
            job = await self.embed_queue.get()
            if job is None:
                break

            name = job["name"]
            try:
                # Markdown gets heading-aware chunking and section locators;
                # a .md file has no pages to cite.
                chunker = (Chunker(markdown=True) if is_markdown(name)
                           else self.chunker)
                rows = chunker.chunk_with_meta(
                    job["text"], name, book_id=self.manifest.book_count
                )
                if not rows:
                    log.warning(f"No chunks produced: {name}")
                    self.checkpoint.mark_failed(name)
                    self._stats["failed"] += 1
                    continue

                texts = [r.pop("text") for r in rows]
                await asyncio.to_thread(self.retriever.add_batch, texts, rows)

                self.manifest.add_book(
                    name,
                    book_id=rows[0]["book_id"],
                    n_chunks=len(rows),
                    n_pages=job.get("pages", 0),
                    char_count=len(job["text"]),
                    sha256=job.get("sha256", ""),
                )

                # Persist BEFORE marking done: if this crashes, the book is
                # retried rather than skipped forever.
                await asyncio.to_thread(self.retriever.save)
                self.manifest.save()

                self.checkpoint.mark_done(name)
                self._stats["done"] += 1
                self._stats["chunks"] += len(rows)
                log.info(f"   Indexed {len(rows)} chunks from {name}")
            except Exception as e:
                log.warning(f"Embed failed: {name} - {e}")
                self.checkpoint.mark_failed(name)
                self._stats["failed"] += 1

    def _list_pdfs(self) -> list:
        return list_sources(self.cfg.INPUT_DIR)
