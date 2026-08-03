"""Error display component for AktaSense.

Renders validation, OCR, and prediction errors with an actionable
suggestion (Blueprint Section 16, Layer 1–4). V1.1 design pass: Lucide
icons instead of emoji.
"""

from __future__ import annotations

import streamlit as st

from App.utils.icons import icon_markdown

# ── Default messages (Indonesian, per Blueprint Section 16) ────────────────

DEFAULT_VALIDATION_TITLE = "Format file tidak didukung."
DEFAULT_VALIDATION_SUGGESTION = "Gunakan file PDF, PNG, JPG, atau JPEG."

DEFAULT_OCR_TITLE = "Teks tidak berhasil diekstraksi dari dokumen ini."
DEFAULT_OCR_SUGGESTION = (
    "Kemungkinan penyebab: dokumen terenkripsi, resolusi terlalu rendah, "
    "atau format tidak didukung. Pastikan dokumen memiliki resolusi "
    "minimal 150 DPI."
)

DEFAULT_PREDICTION_TITLE = "Klasifikasi tidak berhasil."
DEFAULT_PREDICTION_SUGGESTION = (
    "Silakan coba lagi. Jika masalah berlanjut, hubungi administrator."
)


# ── Public API ─────────────────────────────────────────────────────────────

def render_validation_error(message: str | None = None,
                            suggestion: str | None = None) -> None:
    """Render a red error banner for validation failures."""
    title = message or DEFAULT_VALIDATION_TITLE
    hint = suggestion or DEFAULT_VALIDATION_SUGGESTION
    st.error(f"**{icon_markdown('x-circle', 14, 'error')} {title}**\n\n{hint}")


def render_ocr_error(message: str | None = None,
                     suggestion: str | None = None) -> None:
    """Render a warning banner for OCR / extraction failures."""
    title = message or DEFAULT_OCR_TITLE
    hint = suggestion or DEFAULT_OCR_SUGGESTION
    st.warning(
        f"**{icon_markdown('alert-triangle', 14, 'peringatan')} {title}**\n\n{hint}"
    )


def render_prediction_error(message: str | None = None,
                            suggestion: str | None = None) -> None:
    """Render a red error banner for prediction failures."""
    title = message or DEFAULT_PREDICTION_TITLE
    hint = suggestion or DEFAULT_PREDICTION_SUGGESTION
    st.error(f"**{icon_markdown('x-circle', 14, 'error')} {title}**\n\n{hint}")


def render_model_missing_error() -> None:
    """Render the permanent banner when ML artifacts are unavailable."""
    st.warning(
        f"**{icon_markdown('alert-triangle', 14, 'peringatan')} "
        "Model belum tersedia.**\n\n"
        "Silakan jalankan proses training terlebih dahulu.\n\n"
        "`python Pipeline/train.py`"
    )
