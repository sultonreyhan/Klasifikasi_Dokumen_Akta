"""Batch result table component for AktaSense.

Renders summary statistics (modern stat cards) and the results dataframe
for a batch prediction run (Blueprint Section 5.3, Stage 3; Blueprint
Section 6). V1.1 design pass: Lucide icons + stat cards.
"""

from __future__ import annotations

from typing import List

import pandas as pd
import streamlit as st

from App.services.prediction_service import BatchPredictionRecord
from App.utils.icons import icon

_RESULT_COLUMNS = {
    "Nama File":          "filename",
    "Kelas Prediksi":     "display_name",
    "Keyakinan":          "confidence_pct",
    "Waktu Proses (s)":   "processing_time",
    "Status":             "status",
}

_STAT_CONFIG = {
    "success": ("check-circle", "Berhasil", "#16A34A", "#F0FDF4"),
    "failed": ("x-circle", "Gagal", "#DC2626", "#FEF2F2"),
    "total": ("file-text", "Total Dokumen", "#2563EB", "#EFF4FF"),
}


def _to_dataframe(results: List[BatchPredictionRecord]) -> pd.DataFrame:
    rows = []
    for record in results:
        rows.append(
            {
                "filename": record.filename,
                "display_name": record.display_name or "-",
                "confidence_pct": f"{record.confidence * 100:.1f}%",
                "processing_time": f"{record.processing_time:.2f}",
                "status": record.status,
            }
        )
    df = pd.DataFrame(rows, columns=list(_RESULT_COLUMNS.values()))
    return df.rename(columns={v: k for k, v in _RESULT_COLUMNS.items()})


def _stat_card(key: str, value: int) -> str:
    icon_name, label, fg, bg = _STAT_CONFIG[key]
    return (
        f'<div class="akta-stat-card">'
        f'<span class="akta-stat-icon" style="background:{bg};color:{fg};">'
        f'{icon(icon_name, 22)}</span>'
        f'<div><div class="akta-stat-value">{value}</div>'
        f'<div class="akta-stat-label">{label}</div></div>'
        f'</div>'
    )


def render_summary_stats(results: List[BatchPredictionRecord]) -> None:
    """Render the berhasil | gagal | total summary strip."""
    total = len(results)
    succeeded = sum(1 for r in results if r.status == "Berhasil")
    failed = total - succeeded

    col1, col2, col3 = st.columns(3)
    col1.markdown(_stat_card("success", succeeded), unsafe_allow_html=True)
    col2.markdown(_stat_card("failed", failed), unsafe_allow_html=True)
    col3.markdown(_stat_card("total", total), unsafe_allow_html=True)


def render_result_dataframe(results: List[BatchPredictionRecord]) -> None:
    """Render the sortable results table via st.dataframe."""
    if not results:
        st.caption("Belum ada hasil untuk ditampilkan.")
        return
    st.dataframe(_to_dataframe(results), width="stretch")


def render_batch_result_table(
    results: List[BatchPredictionRecord],
) -> None:
    """Render the full batch result section (stats + table)."""
    st.markdown(
        f'<div class="akta-page-title" style="font-size:1.4rem;">'
        f'{icon("list", 18)}<span>Hasil Prediksi Batch</span></div>',
        unsafe_allow_html=True,
    )
    render_summary_stats(results)
    st.divider()
    render_result_dataframe(results)

    if all(r.status == "Gagal" for r in results):
        st.warning(
            "Tidak ada hasil yang dapat diekspor. Semua file gagal diproses."
        )
