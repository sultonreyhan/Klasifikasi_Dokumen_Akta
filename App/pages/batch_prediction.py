"""Batch Prediction page for AktaSense.

State machine: idle → validating → processing → result.
(Blueprint Section 5.3, 12.4, 13.2, 20, 22.)
"""

import os
# Configure PaddleX environment variables process-wide before any imports
os.environ["PADDLE_PDX_EAGER_INIT"] = "False"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "False"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure the project root is on sys.path when this page is imported directly.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from App.components.batch_result_table import render_batch_result_table
from App.components.error_display import (
    render_model_missing_error,
    render_validation_error,
)
from App.components.loading_indicator import batch_wait_message
from App.services.export_service import (
    EXCEL_MIME,
    build_download_name,
    export_batch_to_excel,
)
from App.services.prediction_service import run_batch_prediction
from App.utils.file_validator import (
    BATCH_ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE_MB,
    validate_batch,
)
from App.utils.icons import icon_heading, icon_markdown
from App.utils.session_helpers import init_session_state, reset_batch_state

init_session_state()

st.markdown(
    icon_heading("folder", "Prediksi Batch"), unsafe_allow_html=True
)

# ── Model availability gate ────────────────────────────────────────────────
if not st.session_state.get("pipeline_ready"):
    render_model_missing_error()
    st.stop()

stage = st.session_state["batch_stage"]


# ── Stage: idle ────────────────────────────────────────────────────────────

if stage == "idle":
    st.markdown(
        "Unggah banyak dokumen PDF untuk diklasifikasikan sekaligus. "
        "Hasil dapat diekspor ke Excel."
    )

    uploaded_files = st.file_uploader(
        label="Upload Dokumen Akta (PDF)",
        type=sorted(BATCH_ALLOWED_EXTENSIONS),
        accept_multiple_files=True,
        key="batch_uploader",
        help=f"Format: PDF. Ukuran maksimum {MAX_FILE_SIZE_MB:.0f} MB per file.",
    )

    st.info(
        "Setiap PDF diperlakukan sebagai satu dokumen. "
        "Prediksi Batch hanya menerima file PDF."
    )

    if uploaded_files:
        st.session_state["batch_files"] = list(uploaded_files)
        st.session_state["batch_validation"] = validate_batch(list(uploaded_files))
        st.session_state["batch_error"] = None
        st.session_state["batch_stage"] = "validating"
        st.rerun()


# ── Stage: validating ──────────────────────────────────────────────────────

elif stage == "validating":
    files = st.session_state.get("batch_files", [])
    validation = st.session_state.get("batch_validation", [])

    if not files or not validation:
        reset_batch_state()
        st.rerun()

    if st.session_state.get("batch_error"):
        render_validation_error(message=st.session_state["batch_error"])
        st.session_state["batch_error"] = None

    st.subheader(
        f"{icon_markdown('file-check', 15, 'validasi')} Validasi File"
    )

    table_rows = []
    for idx, item in enumerate(validation, start=1):
        table_rows.append(
            {
                "No": idx,
                "Nama File": item["filename"],
                "Ukuran (MB)": item.get("size_mb"),
                "Status": "Valid" if item["valid"] else "Tidak Valid",
                "Keterangan": item.get("reason") or "-",
            }
        )
    st.dataframe(pd.DataFrame(table_rows), width="stretch")

    valid_files = [f for f, v in zip(files, validation) if v["valid"]]
    valid_count = len(valid_files)

    st.markdown(
        f"**Valid:** {valid_count} file &nbsp;|&nbsp; "
        f"**Tidak valid:** {len(validation) - valid_count} file"
    )

    col_process, col_reset = st.columns([1, 1])
    if col_process.button(
        f"{icon_markdown('play', 15, 'play')} Proses {valid_count} File Valid",
        type="primary",
        disabled=valid_count == 0,
        width="stretch",
        key="batch_proceed",
    ):
        st.session_state["batch_total"] = valid_count
        st.session_state["batch_progress"] = 0
        st.session_state["batch_stage"] = "processing"
        st.rerun()

    if col_reset.button(
        f"{icon_markdown('rotate-ccw', 15, 'rotate')} Batal / Pilih Ulang",
        width="stretch",
        key="batch_cancel",
    ):
        reset_batch_state()
        st.rerun()


# ── Stage: processing ──────────────────────────────────────────────────────

elif stage == "processing":
    files = st.session_state.get("batch_files", [])
    validation = st.session_state.get("batch_validation", [])
    valid_files = [f for f, v in zip(files, validation) if v["valid"]]

    if not valid_files:
        st.session_state["batch_error"] = "Tidak ada file valid untuk diproses."
        st.session_state["batch_stage"] = "validating"
        st.rerun()

    st.subheader(f"{icon_markdown('refresh-cw', 15, 'refresh')} Memproses Dokumen")

    progress_bar = st.progress(0.0)
    status_text = st.empty()
    batch_wait_message()

    total = len(valid_files)

    def on_progress(done: int, _total: int) -> None:
        progress_bar.progress(done / _total)
        current = valid_files[done - 1] if done - 1 < len(valid_files) else None
        current_name = getattr(current, "name", "") if current else ""
        status_text.text(
            f"Memproses: {current_name} ({done} dari {_total} file selesai)"
        )

    results = run_batch_prediction(valid_files, progress_callback=on_progress)

    progress_bar.progress(1.0)
    status_text.text("Semua file selesai diproses.")

    st.session_state["batch_results"] = results
    st.session_state["batch_stage"] = "result"
    st.rerun()


# ── Stage: result ──────────────────────────────────────────────────────────

elif stage == "result":
    results = st.session_state.get("batch_results", [])

    if not results:
        reset_batch_state()
        st.rerun()

    render_batch_result_table(results)

    if any(r.status == "Berhasil" for r in results):
        metadata = st.session_state.get("model_metadata", {}) or {}
        model_version = metadata.get("timestamp_utc", None)

        try:
            excel_bytes = export_batch_to_excel(
                results,
                model_version=model_version,
                include_optional=False,
            )
            st.download_button(
                label=f"{icon_markdown('download', 15, 'download')} Download Hasil Excel",
                data=excel_bytes,
                file_name=build_download_name(),
                mime=EXCEL_MIME,
                type="primary",
                key="batch_export",
            )
        except Exception as exc:
            st.error(f"Gagal membuat file Excel: {exc}")

    if st.button(
        f"{icon_markdown('rotate-ccw', 15, 'rotate')} Prediksi Ulang",
        type="secondary",
        width="stretch",
        key="batch_reset",
    ):
        reset_batch_state()
        st.rerun()
