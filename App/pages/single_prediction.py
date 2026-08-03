"""Single Prediction page for AktaSense.

State machine: idle → validating → extracting → predicting → result.
(Blueprint Section 5.2, 12, 13, 14, 15, 16.)
"""

import streamlit as st

from App.components.document_builder import render_document_builder
from App.components.error_display import render_model_missing_error, render_validation_error
from App.components.loading_indicator import (
    EXTRACT_IMAGE_MSG,
    EXTRACT_PDF_NATIVE_MSG,
    PREDICT_MSG,
)
from App.components.ocr_preview import render_ocr_preview
from App.components.prediction_insight import render_prediction_insight
from App.components.prediction_result import render_prediction_result
from App.components.prediction_summary import render_prediction_summary
from App.services.ocr_service import extract_and_preview
from App.services.prediction_service import run_single_prediction
from App.utils.confidence_classifier import classify as classify_confidence
from App.utils.file_validator import validate_single
from App.utils.icons import icon, icon_heading, inline_icon
from App.utils.label_mapper import get_display_name, get_taxonomy
from App.utils.session_helpers import init_session_state, reset_single_state

init_session_state()

st.markdown(
    icon_heading("file-text", "Prediksi Dokumen"), unsafe_allow_html=True
)

# ── Model availability gate ────────────────────────────────────────────────
if not st.session_state.get("pipeline_ready"):
    render_model_missing_error()
    st.stop()

stage = st.session_state["single_stage"]


# ── Helpers ────────────────────────────────────────────────────────────────

def _extraction_message(source_name: str) -> str:
    ext = (source_name or "").rsplit(".", 1)[-1].lower() if "." in (source_name or "") else ""
    if ext == "pdf":
        return EXTRACT_PDF_NATIVE_MSG
    return EXTRACT_IMAGE_MSG


# ── Stage: idle ────────────────────────────────────────────────────────────

if stage == "idle":
    if st.session_state.get("single_error"):
        render_validation_error(message=st.session_state["single_error"])
        st.session_state["single_error"] = None

    st.markdown(
        "Unggah atau ambil foto dokumen akta untuk mengetahui jenisnya."
    )

    uploaded_file, source = render_document_builder()

    if uploaded_file is not None:
        st.session_state["single_uploaded_file"] = uploaded_file
        st.session_state["single_file_source"] = source
        if st.button(
            f"{icon('play', 15)} Proses Dokumen",
            type="primary",
            width="stretch",
            key="single_proceed",
        ):
            st.session_state["single_stage"] = "validating"
            st.rerun()


# ── Stage: validating → extracting (OCR) ───────────────────────────────────

elif stage == "validating":
    uploaded_file = st.session_state.get("single_uploaded_file")
    source_name = getattr(uploaded_file, "name", "") or ""

    ok, error = validate_single(uploaded_file)
    if not ok:
        st.session_state["single_error"] = error or "Dokumen tidak valid."
        st.session_state["single_stage"] = "idle"
        st.rerun()

    with st.spinner(_extraction_message(source_name)):
        try:
            raw_text, page_count, char_count, quality = extract_and_preview(
                uploaded_file
            )
        except Exception as exc:
            st.session_state["single_error"] = str(exc)
            st.session_state["single_stage"] = "idle"
            st.rerun()

    st.session_state["single_raw_text"] = raw_text
    st.session_state["single_page_count"] = page_count
    st.session_state["single_char_count"] = char_count
    st.session_state["single_ocr_quality"] = quality
    st.session_state["single_stage"] = "extracting"
    st.rerun()


# ── Stage: extracting (OCR preview) ────────────────────────────────────────

elif stage == "extracting":
    continue_pressed, cancel_pressed = render_ocr_preview(
        ocr_text=st.session_state["single_raw_text"],
        page_count=st.session_state["single_page_count"],
        char_count=st.session_state["single_char_count"],
        quality=st.session_state["single_ocr_quality"],
    )

    if continue_pressed:
        st.session_state["single_stage"] = "predicting"
        st.rerun()
    if cancel_pressed:
        reset_single_state()
        st.rerun()


# ── Stage: predicting ──────────────────────────────────────────────────────

elif stage == "predicting":
    raw_text = st.session_state.get("single_raw_text", "")
    try:
        with st.spinner(PREDICT_MSG):
            result, processing_time = run_single_prediction(
                text=raw_text,
                with_shap=True,
            )
    except Exception as exc:
        st.session_state["single_error"] = str(exc)
        st.session_state["single_stage"] = "idle"
        st.rerun()

    st.session_state["single_result"] = result
    st.session_state["single_processing_time"] = processing_time
    st.session_state["single_stage"] = "result"
    st.rerun()


# ── Stage: result ──────────────────────────────────────────────────────────

elif stage == "result":
    result = st.session_state.get("single_result")
    if result is None:
        st.session_state["single_stage"] = "idle"
        st.rerun()

    st.success(f"{inline_icon('check-circle', 15)} Klasifikasi Selesai")

    display_name = get_display_name(result.predicted_class)
    taxonomy = get_taxonomy(result.predicted_class)
    confidence_label = classify_confidence(result.confidence_score)

    render_prediction_result(
        predicted_class=result.predicted_class,
        display_name=display_name,
        taxonomy=taxonomy,
    )

    with st.expander("Ringkasan Prediksi", expanded=True):
        render_prediction_summary(
            confidence_score=result.confidence_score,
            confidence_label=confidence_label,
            probability_distribution=result.probability_distribution,
            highlight_class=result.predicted_class,
        )

    with st.expander("Analisis Dokumen", expanded=False):
        render_prediction_insight(
            shap_explanation=result.shap_explanation,
            predicted_class=result.predicted_class,
        )

    st.caption(
        f"Waktu pemrosesan: {st.session_state.get('single_processing_time', 0.0):.2f} detik."
    )

    if st.button(
        f"{icon('rotate-ccw', 15)} Prediksi Dokumen Lain",
        type="secondary",
        width="stretch",
        key="single_reset",
    ):
        reset_single_state()
        st.rerun()
