"""Prediction summary component for AktaSense.

Renders the confidence meter and the per-class probability chart
(Blueprint Section 5.2, Stage 4; Blueprint Section 6). V1.1 design pass:
Altair chart highlights the predicted class in primary blue.
"""

from __future__ import annotations

from typing import Dict

import altair as alt
import pandas as pd
import streamlit as st

from App.utils.confidence_classifier import CONFIDENCE_COLORS
from App.utils.icons import icon
from App.utils.label_mapper import get_display_name

_PRIMARY = "#2563EB"
_MUTED = "#CBD5E1"


def render_confidence_meter(
    confidence_score: float,
    confidence_label: str,
) -> None:
    """Render the confidence score as label + percentage + progress bar."""
    color = CONFIDENCE_COLORS.get(confidence_label, "#94A3B8")
    pct = round(confidence_score * 100, 1)

    st.markdown(
        f'<div style="display:flex;align-items:center;justify-content:space-between;'
        f'flex-wrap:wrap;gap:8px;margin-bottom:10px;">'
        f'<span class="akta-confidence">'
        f'{icon("gauge", 15)} Tingkat Keyakinan: '
        f'<span style="color:{color};">{confidence_label}</span></span>'
        f'<span style="font-size:1.3rem;font-weight:800;color:var(--akta-text);">'
        f'{pct:.1f}%</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.progress(min(max(confidence_score, 0.0), 1.0))


def render_probability_chart(
    probability_distribution: Dict[str, float],
    highlight_class: str | None = None,
) -> None:
    """Render a horizontal bar chart of all class probabilities.

    The predicted class is drawn in primary blue; the rest in slate.

    Args:
        probability_distribution: ``{label: probability}`` from the model.
        highlight_class: Optional raw label to highlight (the predicted class).
    """
    labels: list[str] = []
    values: list[float] = []
    highlights: list[str] = []

    for label, prob in probability_distribution.items():
        labels.append(get_display_name(label))
        values.append(float(prob))
        highlights.append("Prediksi" if label == highlight_class else "Lainnya")

    source = pd.DataFrame(
        {"Jenis Akta": labels, "Probabilitas": values, "Sorotan": highlights}
    )

    chart = (
        alt.Chart(source)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X(
                "Probabilitas:Q",
                axis=alt.Axis(format=".0%", grid=False, title=None),
                scale=alt.Scale(domain=[0, 1]),
            ),
            y=alt.Y(
                "Jenis Akta:N",
                sort=alt.SortField("Probabilitas", order="descending"),
                axis=alt.Axis(title=None),
            ),
            color=alt.condition(
                alt.datum.Sorotan == "Prediksi",
                alt.value(_PRIMARY),
                alt.value(_MUTED),
            ),
            tooltip=[
                "Jenis Akta",
                alt.Tooltip("Probabilitas:Q", format=".1%"),
            ],
        )
        .properties(height=320)
    )

    st.altair_chart(chart, width="stretch", theme=None)


def render_prediction_summary(
    confidence_score: float,
    confidence_label: str,
    probability_distribution: Dict[str, float],
    highlight_class: str | None = None,
) -> None:
    """Render the full Prediction Summary card (inside an expander)."""
    st.markdown(
        f'<div class="akta-page-title" style="font-size:1.2rem;">'
        f'{icon("bar-chart-3", 18)}<span>Ringkasan Prediksi</span></div>',
        unsafe_allow_html=True,
    )
    render_confidence_meter(confidence_score, confidence_label)
    st.divider()
    st.markdown(
        f'<div class="akta-note-strip">{icon("check-circle", 14)} '
        f'<b>Biru</b> = kelas terprediksi, <b>abu-abu</b> = kelas lainnya.</div>',
        unsafe_allow_html=True,
    )
    render_probability_chart(probability_distribution, highlight_class)
