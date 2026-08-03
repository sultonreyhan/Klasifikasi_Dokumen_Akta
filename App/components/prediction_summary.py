"""Prediction summary component for AktaSense.

Renders the confidence meter and the per-class probability chart
(Blueprint Section 5.2, Stage 4; Blueprint Section 6).
"""

from __future__ import annotations

from typing import Dict

import streamlit as st

from App.utils.confidence_classifier import CONFIDENCE_COLORS
from App.utils.label_mapper import get_display_name


def render_confidence_meter(
    confidence_score: float,
    confidence_label: str,
) -> None:
    """Render the confidence score as label + percentage + progress bar."""
    color = CONFIDENCE_COLORS.get(confidence_label, "#94A3B8")
    pct = round(confidence_score * 100, 1)

    col1, col2 = st.columns([1.2, 1])
    col1.markdown(
        f"**Tingkat Keyakinan:** "
        f"<span style='color:{color}; font-weight:700;'>{confidence_label}</span>",
        unsafe_allow_html=True,
    )
    col2.markdown(f"**{pct:.1f}%**")
    st.progress(min(max(confidence_score, 0.0), 1.0))


def render_probability_chart(
    probability_distribution: Dict[str, float],
    highlight_class: str | None = None,
) -> None:
    """Render a horizontal bar chart of all class probabilities.

    Args:
        probability_distribution: ``{label: probability}`` from the model.
        highlight_class: Optional raw label to highlight (the predicted class).
    """
    # Convert labels to display names, keep highlight on the predicted one.
    labels: list[str] = []
    values: list[float] = []
    colors: list[str] = []

    for label, prob in probability_distribution.items():
        display = get_display_name(label)
        labels.append(display)
        values.append(float(prob))
        colors.append("#1B4FD8" if label == highlight_class else "#94A3B8")

    # st.bar_chart is horizontal for long category lists and keeps it simple.
    chart_data = {
        "Jenis Akta": labels,
        "Probabilitas": values,
    }
    st.bar_chart(
        chart_data,
        x="Jenis Akta",
        y="Probabilitas",
        height=320,
        width="stretch",
    )


def render_prediction_summary(
    confidence_score: float,
    confidence_label: str,
    probability_distribution: Dict[str, float],
    highlight_class: str | None = None,
) -> None:
    """Render the full Prediction Summary card (inside an expander)."""
    st.markdown("### 📊 Ringkasan Prediksi")
    render_confidence_meter(confidence_score, confidence_label)
    st.divider()
    st.markdown("**Distribusi Probabilitas**")
    render_probability_chart(probability_distribution, highlight_class)
