"""Prediction result component for AktaSense.

Renders the predicted class badge and its taxonomy tag
(Blueprint Section 5.2, Stage 4; Blueprint Section 6).
"""

from __future__ import annotations

import streamlit as st

_BADGE_CSS = """
<style>
.akta-class-badge {
    display: inline-block;
    background: #EEF2FD;
    border: 1px solid #1B4FD8;
    color: #1B4FD8;
    border-radius: 6px;
    padding: 6px 14px;
    font-size: 18px;
    font-weight: 700;
}
.akta-taxonomy-tag {
    display: inline-block;
    background: #FDF8EC;
    border: 1px solid #D4A017;
    color: #92620A;
    border-radius: 6px;
    padding: 4px 12px;
    font-size: 13px;
    font-weight: 600;
}
</style>
"""


def render_class_badge(display_name: str) -> None:
    """Render the primary predicted-class badge."""
    st.markdown(
        f'<span class="akta-class-badge">{display_name}</span>',
        unsafe_allow_html=True,
    )


def render_taxonomy_tag(taxonomy: str) -> None:
    """Render the taxonomy group tag."""
    st.markdown(
        f'<span class="akta-taxonomy-tag">{taxonomy}</span>',
        unsafe_allow_html=True,
    )


def render_prediction_result(
    predicted_class: str,
    display_name: str,
    taxonomy: str,
) -> None:
    """Render the full Prediction Result card.

    Args:
        predicted_class: Raw model label (e.g. ``"ajb"``).
        display_name: Human-readable name (e.g. ``"Akta Jual Beli"``).
        taxonomy: Taxonomy group string.
    """
    st.markdown(_BADGE_CSS, unsafe_allow_html=True)

    st.markdown("### 🎯 Jenis Akta")
    col_badge, col_tag = st.columns([1.2, 1])
    with col_badge:
        render_class_badge(display_name)
    with col_tag:
        render_taxonomy_tag(taxonomy)

    st.caption(
        "Hasil AI bersifat indikatif. Verifikasi tetap diperlukan."
    )
