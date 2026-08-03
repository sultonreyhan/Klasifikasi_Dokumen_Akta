"""Loading indicator component for AktaSense.

Renders a spinner with a contextual message (Blueprint Section 6,
Component: loading_indicator.py).
"""

from __future__ import annotations

import streamlit as st

from App.utils.icons import icon

# ── Message helpers ────────────────────────────────────────────────────────

EXTRACT_PDF_NATIVE_MSG = "Mengekstraksi teks dari dokumen PDF..."
EXTRACT_PDF_SCAN_MSG = (
    "Membaca teks dari gambar dokumen... (proses lebih lama)"
)
EXTRACT_IMAGE_MSG = "Mengekstraksi teks dari gambar..."
PREDICT_MSG = "Mengklasifikasi dokumen..."
PREPARING_MSG = "Menyiapkan hasil..."

_BATCH_WAIT_MSG = (
    "Mohon tunggu, jangan tutup atau refresh halaman ini."
)


# ── Public API ─────────────────────────────────────────────────────────────

def render_loading(stage_label: str) -> None:
    """Render a spinner with the given contextual label.

    Args:
        stage_label: Message shown inside the spinner.
    """
    st.spinner(stage_label)


def batch_wait_message() -> None:
    """Render the persistent wait note for batch processing."""
    st.caption(
        f"{icon('info', 13)} {_BATCH_WAIT_MSG}", unsafe_allow_html=True
    )
