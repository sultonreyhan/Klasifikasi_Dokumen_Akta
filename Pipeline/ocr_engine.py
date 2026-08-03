"""Shared, Streamlit-safe PaddleOCR resource for document extraction.

PaddleOCR 3.x initializes PaddleX (PDX) process-wide.  Creating more than
one ``PaddleOCR`` instance in a Streamlit process therefore raises
``PDX has already been initialized``.  Every OCR caller must obtain the
engine through :func:`get_ocr_engine` rather than construct it directly.
"""

from __future__ import annotations

import logging
from typing import Any

import streamlit as st

LOGGER = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def get_ocr_engine() -> Any:
    """Create the application-wide PaddleOCR engine exactly once.

    ``st.cache_resource`` persists this object across Streamlit reruns and
    pages, so image uploads, single predictions, and batch predictions all
    reuse the same PaddleX/PDX initialization.
    """
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise RuntimeError(
            "PaddleOCR tidak tersedia. Install `paddleocr` untuk memproses OCR."
        ) from exc

    LOGGER.info("Initializing shared PaddleOCR/PaddleX resource.")
    return PaddleOCR(use_angle_cls=True, lang="id", show_log=False)
