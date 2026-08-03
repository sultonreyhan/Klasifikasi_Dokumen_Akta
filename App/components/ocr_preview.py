"""OCR preview component for AktaSense.

Renders extraction statistics, a quality badge, and the read-only
extracted text (Blueprint Section 5.2, Stage 2; Blueprint Section 6).
V1.1 design pass: updated quality palette + Lucide icon title.
"""

from __future__ import annotations

import streamlit as st

from App.utils.icons import icon

# ── Quality badge rendering ────────────────────────────────────────────────

_QUALITY_BADGES = {
    "good": ("Baik", "#16A34A"),
    "fair": ("Cukup", "#F59E0B"),
    "poor": ("Rendah", "#DC2626"),
}


def _quality_html(quality: str) -> str:
    label, color = _QUALITY_BADGES.get(quality, ("Tidak Diketahui", "#94A3B8"))
    return (
        f"<span style='color:{color};font-weight:700;'>"
        f"{icon('shield-check', 13, class_='')} {label}</span>"
    )


def render_ocr_stats(page_count: int, char_count: int, quality: str | None) -> None:
    """Render halaman | karakter | kualitas in a three-column strip."""
    col1, col2, col3 = st.columns(3)
    col1.metric("Halaman", int(page_count))
    col2.metric("Karakter", f"{int(char_count):,}")
    if quality:
        col3.markdown(
            "**Kualitas OCR**\n\n" + _quality_html(quality),
            unsafe_allow_html=True,
        )


def render_ocr_quality_badge(quality: str | None) -> None:
    """Render just the quality badge (used in summary contexts)."""
    st.markdown(_quality_html(quality or "unknown"), unsafe_allow_html=True)


def render_ocr_text_display(ocr_text: str) -> None:
    """Render the extracted text in a read-only monospace text area."""
    st.text_area(
        label="Teks yang berhasil diekstraksi",
        value=ocr_text,
        height=300,
        disabled=True,
    )


def render_ocr_preview(
    ocr_text: str,
    page_count: int,
    char_count: int,
    quality: str | None,
) -> None:
    """Render the full OCR preview section.

    Args:
        ocr_text: Extracted text to display.
        page_count: Number of pages extracted.
        char_count: Number of characters in ``ocr_text``.
        quality: ``"good"`` / ``"fair"`` / ``"poor"`` / None.
    """
    st.markdown(
        f'<div class="akta-page-title" style="font-size:1.4rem;">'
        f'{icon("scan-text", 18)}<span>Tinjauan Hasil Ekstraksi</span></div>',
        unsafe_allow_html=True,
    )
    render_ocr_stats(page_count, char_count, quality)

    if quality == "poor":
        st.warning(
            "Kualitas teks yang berhasil diekstraksi rendah. "
            "Hasil klasifikasi mungkin kurang akurat. Pastikan dokumen "
            "memiliki resolusi yang cukup."
        )

    render_ocr_text_display(ocr_text)

    col_ok, col_cancel = st.columns([1, 1])
    continue_pressed = col_ok.button(
        f"{icon('arrow-right', 15)} Lanjut ke Klasifikasi",
        type="primary",
        width="stretch",
        key="ocr_continue",
    )
    cancel_pressed = col_cancel.button(
        f"{icon('rotate-ccw', 15)} Ganti Dokumen",
        width="stretch",
        key="ocr_cancel",
    )
    return continue_pressed, cancel_pressed
