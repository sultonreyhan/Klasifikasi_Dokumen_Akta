"""Prediction insight component for AktaSense.

Renders the V1 SHAP feature-importance explanation for a single
prediction (Blueprint Section 21).

V1 scope (confirmed C2): SHAP feature importance only — token-level
highlighting and document-location mapping are postponed.
"""

from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

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

    # Positive (supporting) → Royal Blue, negative → Slate.
    rows = [
        {
            "Fitur": c["feature"],
            "Kontribusi": float(c["value"]),
        }
        for c in top_contributors
    ]
    # Order: most influential first (largest |value|), matching Pipeline sort.
    rows.sort(key=lambda r: abs(r["Kontribusi"]), reverse=True)

    st.bar_chart(
        rows,
        x="Fitur",
        y="Kontribusi",
        height=320,
        width="stretch",
    )


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
    st.markdown("### 🔍 Analisis Dokumen")
    st.markdown("**Fitur semantik paling berpengaruh**")

    contributors = _extract_top_contributors(shap_explanation, predicted_class)
    render_shap_top_features(contributors)
    render_insight_note()
