"""Prediction insight component for AktaSense.

Renders the V1 SHAP feature-importance explanation for a single
prediction (Blueprint Section 21). V1.1 design pass: Altair chart colours
supporting features (positive) primary blue and reducing features
(negative) slate, matching the explanatory note.
"""

from __future__ import annotations

from typing import Any, Dict, List

import altair as alt
import pandas as pd
import streamlit as st

from App.utils.icons import icon

_PRIMARY = "#2563EB"
_MUTED = "#CBD5E1"

_INSIGHT_NOTE = (
    "AktaSense menganalisis pola semantik dalam teks dokumen Anda "
    "menggunakan model AI. Grafik di atas menunjukkan aspek-aspek "
    "semantik yang paling mempengaruhi hasil klasifikasi dokumen ini.\n\n"
    "Semakin panjang batang, semakin besar pengaruh aspek tersebut "
    "terhadap prediksi. Batang berwarna biru menunjukkan aspek yang "
    "mendukung prediksi ini, sedangkan batang abu-abu menunjukkan "
    "aspek yang mengurangi keyakinan model."
)


def _extract_top_contributors(
    shap_explanation: Dict[str, Any],
    predicted_class: str,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """Pull the top contributors for the predicted class from SHAP data."""
    classes = shap_explanation.get("classes", {}) if shap_explanation else {}
    class_data = classes.get(predicted_class, {}) if classes else {}
    contributors = class_data.get("top_contributors", [])
    return list(contributors)[:top_k]


def render_shap_top_features(top_contributors: List[Dict[str, Any]]) -> None:
    """Render the SHAP top-feature horizontal bar chart."""
    if not top_contributors:
        st.caption("Tidak ada data fitur yang dapat ditampilkan.")
        return

    # Positive (supporting) → Primary Blue, negative → Slate.
    rows = [
        {
            "Fitur": c["feature"],
            "Kontribusi": float(c["value"]),
            "Arah": "Mendukung" if float(c["value"]) >= 0 else "Mengurangi",
        }
        for c in top_contributors
    ]
    # Order: most influential first (largest |value|), matching Pipeline sort.
    rows.sort(key=lambda r: abs(r["Kontribusi"]), reverse=True)

    source = pd.DataFrame(rows)

    chart = (
        alt.Chart(source)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X(
                "Kontribusi:Q",
                axis=alt.Axis(grid=False, title=None),
            ),
            y=alt.Y(
                "Fitur:N",
                sort=alt.SortField("Kontribusi", order="descending"),
                axis=alt.Axis(title=None),
            ),
            color=alt.condition(
                alt.datum.Arah == "Mendukung",
                alt.value(_PRIMARY),
                alt.value(_MUTED),
            ),
            tooltip=["Fitur", alt.Tooltip("Kontribusi:Q", format=".3f")],
        )
        .properties(height=320)
    )

    st.altair_chart(chart, width="stretch", theme=None)


def render_insight_note() -> None:
    """Render the contextual explanation note (Blueprint Section 21.4)."""
    st.info(_INSIGHT_NOTE)


def render_prediction_insight(
    shap_explanation: Dict[str, Any],
    predicted_class: str,
) -> None:
    """Render the full Prediction Insight panel (inside an expander).

    Args:
        shap_explanation: The ``shap_explanation`` dict from a
            :class:`Pipeline.predict.PredictionResult`.
        predicted_class: Raw label of the predicted class.
    """
    st.markdown(
        f'<div class="akta-page-title" style="font-size:1.2rem;">'
        f'{icon("lightbulb", 18)}<span>Analisis Dokumen</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown("**Fitur semantik paling berpengaruh**")

    contributors = _extract_top_contributors(shap_explanation, predicted_class)
    render_shap_top_features(contributors)
    render_insight_note()
