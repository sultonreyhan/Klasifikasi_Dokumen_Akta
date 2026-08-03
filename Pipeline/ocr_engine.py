"""Shared, Streamlit-safe PaddleOCR resource for document extraction.

PaddleOCR 3.x initializes PaddleX (PDX) process-wide.  Creating more than
one ``PaddleOCR`` instance in a Streamlit process therefore raises
``PDX has already been initialized``.  Every OCR caller must obtain the
engine through :func:`get_ocr_engine` rather than construct it directly.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import streamlit as st

LOGGER = logging.getLogger(__name__)

# PaddleX 3.x eagerly initializes a process-global repository manager while
# being imported. Streamlit may re-execute application modules without
# restarting the Python process, which attempts that initialization a second
# time and raises ``PDX has already been initialized``. PaddleOCR initializes
# its inference pipeline lazily, so deferring this import-time setup is safe.
# Disable PaddleX eager initialization, oneDNN crash on CPU, and connectivity check.
os.environ["PADDLE_PDX_EAGER_INIT"] = "False"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "False"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"


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
            "Runtime OCR tidak tersedia. Pastikan `paddleocr` dan "
            "`paddlepaddle` terpasang."
        ) from exc

    LOGGER.info("Initializing shared PaddleOCR/PaddleX resource.")
    return PaddleOCR(
        lang="id",
        use_textline_orientation=True,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
    )
