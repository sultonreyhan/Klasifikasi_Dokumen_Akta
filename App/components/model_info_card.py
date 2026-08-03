"""Model info card component for AktaSense.

Renders read-only model metadata on the Landing page
(Blueprint Section 5.1; data always read dynamically from
``Models/training_metadata.json`` — confirmed decision C1).
V1.1 design pass: Lucide icon + pill chips from the design system.
"""

from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

from App.utils.icons import icon
from App.utils.label_mapper import get_all_classes_with_display


def _class_chips(label_list: List[str]) -> str:
    """Build inline HTML pill chips for every class display name."""
    chips = []
    for item in get_all_classes_with_display(label_list):
        chips.append(
            f'<span class="akta-chip">{item["display_name"]}</span>'
        )
    return "".join(chips)


def render_model_info(metadata: Dict[str, Any]) -> None:
    """Render the Model Info card.

    Args:
        metadata: The dict returned by
            :func:`App.services.metadata_service.load_model_metadata`.
            If empty, a "metadata unavailable" note is shown.
    """
    st.markdown(
        f'<div class="akta-page-title" style="font-size:1.4rem;">'
        f'{icon("database", 18)}<span>Model AI</span></div>',
        unsafe_allow_html=True,
    )

    if not metadata:
        st.caption("Metadata model tidak tersedia.")
        return

    label_list = metadata.get("label_list", [])
    num_classes = metadata.get("num_classes", len(label_list))
    oob_score = metadata.get("oob_score")
    embedding_model = metadata.get("embedding_model", "-")
    timestamp = metadata.get("timestamp_utc")

    col1, col2, col3 = st.columns(3)
    col1.metric("Jenis Model", "Random Forest")
    col2.metric("Jumlah Kelas", num_classes)
    col3.metric("OOB Score", f"{oob_score:.2f}" if oob_score is not None else "-")

    st.markdown("**Kelas yang didukung:**")
    if label_list:
        st.markdown(_class_chips(label_list), unsafe_allow_html=True)
    else:
        st.caption("Tidak ada kelas yang tersedia.")

    st.caption(f"Embedding: {embedding_model}")
    if timestamp:
        st.caption(f"Dilatih: {timestamp}")
