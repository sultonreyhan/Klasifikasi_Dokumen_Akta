"""Prediction result component for AktaSense.

Renders the predicted class as a focal result card with its taxonomy tag
(Blueprint Section 5.2, Stage 4; Blueprint Section 6).
"""

from __future__ import annotations

import streamlit as st

from App.utils.icons import icon


def render_prediction_result(
    predicted_class: str,
    display_name: str,
    taxonomy: str,
) -> None:
    """Render the full Prediction Result focal card.

    Args:
        predicted_class: Raw model label (e.g. ``"ajb"``). Unused in the
            visual but kept in the signature for API stability.
        display_name: Human-readable name (e.g. ``"Akta Jual Beli"``).
        taxonomy: Taxonomy group string.
    """
    st.markdown(
        f'<div class="akta-result-hero">'
        f'<div class="akta-result-label">{icon("target", 15)} '
        f'Jenis Akta Terdeteksi</div>'
        f'<h2 class="akta-result-name">{display_name}</h2>'
        f'<span class="akta-result-tag">{taxonomy}</span>'
        f'<p class="akta-result-note">Hasil AI bersifat indikatif. '
        f'Verifikasi tetap diperlukan.</p>'
        f'</div>',
        unsafe_allow_html=True,
    )
