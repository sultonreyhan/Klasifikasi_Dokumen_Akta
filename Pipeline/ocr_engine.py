"""Shared, Streamlit-safe OCR resource for document extraction.

AktaSense uses **RapidOCR** (PP-OCR models executed through ONNX Runtime) as its
OCR engine.  It was chosen over PaddleOCR/PaddlePaddle because:

- it is a pure-Python wheel with a single ``onnxruntime`` dependency, so it
  installs cleanly on Windows, macOS and Linux (including Streamlit Cloud);
- it needs far less memory than the PaddlePaddle stack (~150 MB vs ~500 MB),
  which matters on the 1 GB Community Cloud limit;
- it avoids the process-wide PaddleX initialisation that historically caused
  ``PDX has already been initialized`` and ``OCR runtime unavailable`` errors
  in Streamlit reruns.

Every OCR caller must obtain the engine through :func:`get_ocr_engine` (which
is cached process-wide by Streamlit) and must pass images through
:func:`recognize_text`.  Callers must **not** parse RapidOCR's output format
themselves — this module is the single place that knows it.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Union

import numpy as np
import streamlit as st

LOGGER = logging.getLogger(__name__)

# Keep ONNX Runtime from spawning more worker threads than needed on small
# instances (also reduces peak memory on Streamlit Community Cloud).
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("ORT_MAX_THREADS", "2")

# A RapidOCR engine instance accepts these image inputs.
ImageInput = Union[str, Path, np.ndarray, Any]


@st.cache_resource(show_spinner=False)
def get_ocr_engine() -> Any:
    """Create the application-wide RapidOCR engine exactly once.

    ``st.cache_resource`` persists this object across Streamlit reruns and
    pages, so image uploads, single predictions, batch predictions and the
    camera all reuse the same ONNX Runtime session.

    Returns:
        A configured :class:`rapidocr.RapidOCR` instance.

    Raises:
        RuntimeError: If RapidOCR (or its ONNX Runtime dependency) is missing.
    """
    try:
        from rapidocr import RapidOCR
    except ImportError as exc:
        raise RuntimeError(
            "Runtime OCR tidak tersedia. Pastikan `rapidocr` dan "
            "`onnxruntime` terpasang."
        ) from exc

    LOGGER.info("Initializing shared RapidOCR/ONNX Runtime resource.")
    return RapidOCR()


def recognize_text(engine: Any, image: ImageInput) -> str:
    """Run OCR on one image and return the recognized lines as text.

    This is the single place that understands RapidOCR's output contract, so
    callers stay decoupled from the engine implementation.

    Args:
        engine: A RapidOCR engine from :func:`get_ocr_engine`.
        image: Image path, :class:`numpy.ndarray` (H, W, 3) or a PIL image.

    Returns:
        The recognized lines joined by newlines, or an empty string when no
        text is found.

    Raises:
        RuntimeError: If the engine fails to process the image.
    """
    if image is None:
        return ""

    try:
        output = engine(image)
    except Exception as exc:  # pragma: no cover - defensive guard
        LOGGER.warning("OCR inference failed: %s", exc)
        raise RuntimeError(f"OCR gagal diproses: {exc}") from exc

    lines = _extract_lines(output)
    LOGGER.info("OCR produced %d line(s).", len(lines))
    return "\n".join(lines)


def _extract_lines(output: Any) -> list[str]:
    """Normalize a RapidOCR result object into a list of recognized lines.

    RapidOCR >= 3.x returns a ``RapidOCROutput`` with a ``txts`` tuple of
    recognized strings.  Some versions / engines return a list of
    ``[box, text, score]`` items instead.  Both are handled here.
    """
    if output is None:
        return []

    # Modern RapidOCR (>= 3.x): object with .txts
    txts = getattr(output, "txts", None)
    if txts is not None:
        lines = [str(t) for t in txts if str(t).strip()]
        return lines

    # Legacy RapidOCR (< 3.x): iterable of [box, text, score] entries.
    if isinstance(output, (list, tuple)):
        lines = []
        for item in output:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                text = item[1]
                if str(text).strip():
                    lines.append(str(text))
        return lines

    return []
