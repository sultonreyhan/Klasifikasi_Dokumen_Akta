"""Batch result table component for AktaSense.

Renders summary statistics and the results dataframe for a batch
prediction run (Blueprint Section 5.3, Stage 3; Blueprint Section 6).
"""

from __future__ import annotations

from typing import List

import pandas as pd
import streamlit as st

from App.services.prediction_service import BatchPredictionRecord

_RESULT_COLUMNS = {
    "Nama File":          "filename",
    "Kelas Prediksi":     "display_name",
    "Keyakinan":          "confidence_pct",
    "Waktu Proses (s)":   "processing_time",
    "Status":             "status",
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


def render_summary_stats(results: List[BatchPredictionRecord]) -> None:
    """Render the berhasil | gagal | total summary strip."""
    total = len(results)
    succeeded = sum(1 for r in results if r.status == "Berhasil")
    failed = total - succeeded

    col1, col2, col3 = st.columns(3)
    col1.metric("✅ Berhasil", succeeded)
    col2.metric("❌ Gagal", failed)
    col3.metric("📄 Total", total)


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
    st.subheader("📋 Hasil Prediksi Batch")
    render_summary_stats(results)
    st.divider()
    render_result_dataframe(results)

    if all(r.status == "Gagal" for r in results):
        st.warning(
            "Tidak ada hasil yang dapat diekspor. Semua file gagal diproses."
        )
