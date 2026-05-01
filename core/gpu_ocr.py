"""
GPU OCR Batch Engine — RTX 4050 Accelerated
=============================================
Runs OCR on MULTIPLE pages at once using your GPU.

Instead of: page1 → OCR → result, page2 → OCR → result  (SLOW)
We do:      [page1, page2, page3, ...] → GPU → [results]  (FAST)

Two modes:
  1. ONNX Runtime + CUDA EP  (if onnxruntime-gpu installed)  ← RTX 4050 mode
  2. Tesseract CPU            (fallback — always works)

For real ONNX OCR models, use:
  - PaddleOCR exported to ONNX
  - TrOCR (Microsoft) exported to ONNX
  - EasyOCR exported to ONNX
"""

import io
import numpy as np
from logs.logger import log


class GPUOCRBatch:
    def __init__(self):
        self.mode = "cpu"        # Start assuming CPU
        self.session = None      # ONNX session (if GPU available)
        self.batch_size = 8      # Pages per GPU batch

        self._init_gpu()

    # ── Initialization ────────────────────────────────────────────────────────
    def _init_gpu(self):
        """Try to set up ONNX Runtime with CUDA. Fall back to CPU gracefully."""
        try:
            import onnxruntime as ort

            available = ort.get_available_providers()
            log.info(f"🔧 ONNX Providers available: {available}")

            if "CUDAExecutionProvider" in available:
                # RTX 4050 detected!
                self.mode = "onnx_cuda"
                log.info("🚀 GPU OCR Engine: ONNX Runtime + CUDA (RTX mode)")
                log.info("   To load a model: set ONNX_MODEL_PATH in config.py")
                # Example:
                # self.session = ort.InferenceSession(
                #     "data/models/ocr_model.onnx",
                #     providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
                # )
            else:
                log.info("ℹ️  CUDA not available via ONNX. Using CPU OCR.")

        except ImportError:
            log.info("ℹ️  onnxruntime-gpu not installed. Using Tesseract CPU OCR.")

        # Check Tesseract availability as fallback
        self._tesseract_ok = self._check_tesseract()

    def _check_tesseract(self) -> bool:
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            log.info("✅ Tesseract available as OCR backend")
            return True
        except Exception:
            log.warning("⚠️  Tesseract not found — scanned pages will be skipped")
            log.warning("    Windows install: https://github.com/UB-Mannheim/tesseract/wiki")
            return False

    # ── Public API ────────────────────────────────────────────────────────────
    def infer_batch(self, pages: list) -> list[str]:
        """
        Run OCR on a list of PDF page objects.
        Returns a list of text strings (one per page).

        Args:
            pages: List of fitz.Page objects

        Returns:
            List of str — extracted text per page
        """
        if not pages:
            return []

        results = []

        # Process in sub-batches (GPU has limited VRAM)
        for i in range(0, len(pages), self.batch_size):
            batch = pages[i: i + self.batch_size]
            images = [self._page_to_image_bytes(p) for p in batch]

            if self.mode == "onnx_cuda" and self.session is not None:
                batch_results = self._infer_onnx(images)
            elif self._tesseract_ok:
                batch_results = self._infer_tesseract(images)
            else:
                batch_results = ["[OCR UNAVAILABLE]"] * len(images)

            results.extend(batch_results)

        return results

    # ── Page → Image ──────────────────────────────────────────────────────────
    def _page_to_image_bytes(self, page, dpi: int = 200) -> bytes:
        """Render a PDF page to PNG bytes."""
        try:
            pix = page.get_pixmap(dpi=dpi)
            return pix.tobytes("png")
        except Exception as e:
            log.error(f"Page render failed: {e}")
            return b""

    # ── ONNX GPU Inference ────────────────────────────────────────────────────
    def _infer_onnx(self, image_bytes_list: list) -> list[str]:
        """
        Batch inference via ONNX Runtime + CUDA.

        ── HOW TO PLUG IN YOUR MODEL ─────────────────────────────────────
        1. Export your OCR model to ONNX format
        2. Set ONNX_MODEL_PATH in config.py
        3. Uncomment the session loading in _init_gpu()
        4. Replace the preprocessing/postprocessing below with your model's
           expected input/output format.
        ──────────────────────────────────────────────────────────────────
        """
        results = []
        for img_bytes in image_bytes_list:
            if not img_bytes:
                results.append("")
                continue
            try:
                # ── Preprocess image for your model ──
                from PIL import Image
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                arr = np.array(img, dtype=np.float32) / 255.0
                arr = arr.transpose(2, 0, 1)[np.newaxis]  # NCHW

                # ── Run ONNX inference ────────────────
                # outputs = self.session.run(None, {"input": arr})
                # text = decode_output(outputs[0])

                # Placeholder until model is loaded:
                results.append("[ONNX_GPU_SLOT: load model in config]")

            except Exception as e:
                log.error(f"ONNX infer error: {e}")
                results.append("")
        return results

    # ── Tesseract CPU Fallback ────────────────────────────────────────────────
    def _infer_tesseract(self, image_bytes_list: list) -> list[str]:
        """Run Tesseract OCR on a batch of images (CPU, reliable)."""
        import pytesseract
        from PIL import Image

        results = []
        for img_bytes in image_bytes_list:
            if not img_bytes:
                results.append("")
                continue
            try:
                img = Image.open(io.BytesIO(img_bytes))
                text = pytesseract.image_to_string(img, lang="eng")
                results.append(text.strip())
            except Exception as e:
                log.error(f"Tesseract error: {e}")
                results.append("")
        return results
