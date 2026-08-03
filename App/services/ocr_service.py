"""OCR / text-extraction service for AktaSense.

Extracts raw text from an uploaded document and produces the metadata
needed by the OCR Preview component (Blueprint Section 7, ocr_service.py).

Supported inputs:
- PDF       — native text layer via Pipeline, PaddleOCR fallback per page.
- PNG/JPG   — OCR directly via PaddleOCR.
- Camera    — treated as an image upload.

This module never modifies the Pipeline; it only calls it.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Optional, Tuple

from streamlit.runtime.uploaded_file_manager import UploadedFile

from Pipeline import predict as pipeline_predict

LOGGER = logging.getLogger(__name__)

# ── Quality thresholds (Blueprint Section 7) ───────────────────────────────

_QUALITY_GOOD_MIN = 500
_QUALITY_FAIR_MIN = 100


def _classify_quality(char_count: int) -> str:
    """Map a character count to a quality tier: good | fair | poor."""
    if char_count > _QUALITY_GOOD_MIN:
        return "good"
    if char_count > _QUALITY_FAIR_MIN:
        return "fair"
    return "poor"


def _suffix_of(filename: str) -> str:
    if not filename or "." not in filename:
        return ""
    return filename.rsplit(".", 1)[-1].lower()


def _save_to_temp(uploaded_file: UploadedFile) -> Path:
    """Persist uploaded bytes to a temp file and return its path."""
    suffix = _suffix_of(getattr(uploaded_file, "name", "") or "upload")
    with tempfile.NamedTemporaryFile(
        suffix=f".{suffix}" if suffix else "", delete=False
    ) as handle:
        uploaded_file.seek(0)
        handle.write(uploaded_file.read())
        return Path(handle.name)


def _ocr_image(image_path: Path) -> str:
    """OCR a single image via PaddleOCR and return joined text lines."""
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise ValueError(
            "PaddleOCR tidak tersedia. Install `paddleocr` untuk "
            "memproses file gambar."
        ) from exc

    ocr = PaddleOCR(use_angle_cls=True, lang="id", show_log=False)
    result = ocr.ocr(str(image_path), cls=True)
    lines: list[str] = []
    for block in result or []:
        for line in block or []:
            lines.append(str(line[1][0]))
    return "\n".join(lines)


def _count_pdf_pages(pdf_path: Path) -> int:
    """Return the number of pages in a PDF (0 if unreadable)."""
    try:
        import fitz
        with fitz.open(str(pdf_path)) as doc:
            return int(doc.page_count)
    except Exception:
        return 0


# ── Public API ─────────────────────────────────────────────────────────────

def extract_and_preview(
    uploaded_file: UploadedFile,
) -> Tuple[str, int, int, str]:
    """Extract text and derive preview metadata from an uploaded file.

    Args:
        uploaded_file: Streamlit UploadedFile (PDF or image).

    Returns:
        ``(raw_text, page_count, char_count, quality)`` where ``quality``
        is ``"good"`` / ``"fair"`` / ``"poor"``.

    Raises:
        ValueError: If no text could be extracted, the format is
            unsupported, or a required library is missing.
    """
    filename = getattr(uploaded_file, "name", "") or "upload"
    ext = _suffix_of(filename)

    tmp_path = _save_to_temp(uploaded_file)
    try:
        if ext == "pdf":
            raw_text = pipeline_predict.extract_text(tmp_path)
            page_count = _count_pdf_pages(tmp_path)
        elif ext in ("png", "jpg", "jpeg"):
            raw_text = _ocr_image(tmp_path)
            page_count = 1
        else:
            raise ValueError(
                "Format file tidak didukung untuk ekstraksi teks. "
                "Gunakan PDF, PNG, JPG, atau JPEG."
            )
    finally:
        tmp_path.unlink(missing_ok=True)

    if not raw_text or not raw_text.strip():
        raise ValueError("Tidak ada teks yang berhasil diekstraksi dari dokumen ini.")

    char_count = len(raw_text.strip())
    quality = _classify_quality(char_count)
    LOGGER.info("Extracted %d chars (%d pages), quality=%s", char_count, page_count, quality)
    return raw_text, page_count, char_count, quality
