"""Prediction service for AktaSense — the single gateway to the ML Pipeline.

This module is the ONLY layer allowed to import ``Pipeline`` modules.
Pages and components must call these functions instead of calling
the Pipeline directly (Blueprint Section 7).

Sprint 1 scope:
- :func:`load_pipeline_resources` — fully implemented, cached.
- :func:`run_single_prediction` — implemented in Sprint 3.
- :func:`run_batch_prediction`  — implemented in Sprint 4.
"""

from __future__ import annotations

import logging
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from Pipeline import config as pipeline_config
from Pipeline import embedding as pipeline_embedding
from Pipeline import predict as pipeline_predict
from Pipeline import train as pipeline_train

from App.utils.confidence_classifier import classify as classify_confidence
from App.utils.label_mapper import get_display_name, get_taxonomy

LOGGER = logging.getLogger(__name__)


class ModelArtifactsMissingError(FileNotFoundError):
    """Raised only when a required local model artifact is unavailable."""


def _validate_model_artifacts() -> None:
    """Log model artifact availability and fail before attempting deserialization."""
    model_dir = pipeline_config.MODEL_DIRECTORY
    model_path = pipeline_config.MODEL_ARTIFACT_PATH
    encoder_path = pipeline_config.LABEL_ENCODER_PATH

    if model_dir.is_dir():
        LOGGER.info("✓ Folder Models ditemukan: %s", model_dir)
    else:
        LOGGER.error("✗ Folder Models tidak ditemukan: %s", model_dir)

    missing_paths: list[Path] = []
    for artifact_path in (model_path, encoder_path):
        if artifact_path.is_file():
            LOGGER.info("✓ %s ditemukan", artifact_path.name)
        else:
            LOGGER.error("✗ %s tidak ditemukan: %s", artifact_path.name, artifact_path)
            missing_paths.append(artifact_path)

    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise ModelArtifactsMissingError(f"Model artifact tidak ditemukan: {missing}")


# ── Result record (Blueprint Section 20.1) ─────────────────────────────────

@dataclass
class BatchPredictionRecord:
    """One row of a batch prediction result (no SHAP per Blueprint C5)."""

    filename: str
    predicted_class: str
    display_name: str
    taxonomy: str
    confidence: float
    confidence_label: str
    processing_time: float
    status: str                       # "Berhasil" | "Gagal"
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict representation."""
        return {
            "filename": self.filename,
            "predicted_class": self.predicted_class,
            "display_name": self.display_name,
            "taxonomy": self.taxonomy,
            "confidence": self.confidence,
            "confidence_label": self.confidence_label,
            "processing_time": self.processing_time,
            "status": self.status,
            "error_message": self.error_message,
        }


# ── Pipeline resource loading (cached) ─────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_pipeline_resources() -> Dict[str, Any]:
    """Load all ML artifacts exactly once per session.

    Loads:
    - Random Forest classifier   (Models/random_forest.pkl)
    - Label encoder             (Models/label_encoder.pkl)
    - IndoBERT model + device   (HuggingFace)
    - IndoBERT tokenizer        (HuggingFace)

    Returns:
        Dict with keys ``clf``, ``encoder``, ``embedding_model``,
        ``tokenizer``, ``device``, and ``class_names``.

    Raises:
        FileNotFoundError / ValueError: Propagated from Pipeline if the
            artifacts are missing or the pipeline fails to load.
    """
    LOGGER.info("Loading pipeline resources (first time).")

    _validate_model_artifacts()

    clf = pipeline_train.load_model(pipeline_config.MODEL_ARTIFACT_PATH)
    encoder = pipeline_train.load_label_encoder(pipeline_config.LABEL_ENCODER_PATH)
    embedding_model, device = pipeline_embedding.load_model(device=pipeline_config.DEVICE)
    tokenizer = pipeline_embedding.load_tokenizer()

    class_names = [str(c) for c in encoder.classes_]

    resources: Dict[str, Any] = {
        "clf": clf,
        "encoder": encoder,
        "embedding_model": embedding_model,
        "tokenizer": tokenizer,
        "device": device,
        "class_names": class_names,
    }

    LOGGER.info(
        "Pipeline resources loaded: %d classes, device=%s",
        len(class_names),
        device,
    )
    return resources


# ── Prediction functions ───────────────────────────────────────────────────

def run_single_prediction(
    text: str,
    with_shap: bool = True,
) -> Tuple[Any, float]:
    """Predict the class of a single document from its extracted text.

    Reuses the locked Pipeline (:func:`Pipeline.predict.predict_document`)
    with the session-cached ML resources. No inference logic is duplicated
    in the App layer.

    Args:
        text: The cleaned/raw extracted document text (already produced
            by :func:`App.services.ocr_service.extract_and_preview`).
        with_shap: Whether to compute the SHAP explanation
            (Blueprint Section 21, V1 scope).

    Returns:
        ``(PredictionResult, processing_time)``.

    Raises:
        ValueError: If prediction, embedding, or SHAP fails
            (message propagated from the Pipeline).
    """
    resources = load_pipeline_resources()

    started = time.perf_counter()
    result = pipeline_predict.predict_document(
        text=text,
        clf=resources["clf"],
        encoder=resources["encoder"],
        embedding_model=resources["embedding_model"],
        tokenizer=resources["tokenizer"],
        device=resources["device"],
        with_shap=with_shap,
    )
    processing_time = time.perf_counter() - started

    LOGGER.info(
        "Single prediction -> %s (conf=%.4f, %.2fs)",
        result.predicted_class,
        result.confidence_score,
        processing_time,
    )
    return result, processing_time


def run_batch_prediction(
    file_list: List[Any],
    progress_callback: Optional[Any] = None,
) -> List[BatchPredictionRecord]:
    """Predict classes for a list of PDF documents (no SHAP per C5).

    Each file is processed independently. A failure in one file does not
    stop the rest of the batch — the failed file is recorded with a
    ``"Gagal"`` status and processing continues.

    Args:
        file_list: List of validated Streamlit UploadedFile (PDF only).
        progress_callback: Optional callable invoked after every file with
            ``(files_done, total)``.

    Returns:
        A list of :class:`BatchPredictionRecord`, one per input file.
    """
    resources = load_pipeline_resources()
    total = len(file_list)
    records: List[BatchPredictionRecord] = []

    for index, uploaded_file in enumerate(file_list, start=1):
        filename = getattr(uploaded_file, "name", "") or f"file_{index}"
        started = time.perf_counter()
        tmp_path: Optional[Path] = None

        try:
            tmp_path = _save_to_temp(uploaded_file)
            raw_text = pipeline_predict.extract_text(tmp_path)
            result = pipeline_predict.predict_document(
                text=raw_text,
                clf=resources["clf"],
                encoder=resources["encoder"],
                embedding_model=resources["embedding_model"],
                tokenizer=resources["tokenizer"],
                device=resources["device"],
                with_shap=False,
            )
            elapsed = time.perf_counter() - started
            records.append(
                BatchPredictionRecord(
                    filename=filename,
                    predicted_class=result.predicted_class,
                    display_name=get_display_name(result.predicted_class),
                    taxonomy=get_taxonomy(result.predicted_class),
                    confidence=float(result.confidence_score),
                    confidence_label=classify_confidence(result.confidence_score),
                    processing_time=elapsed,
                    status="Berhasil",
                    error_message=None,
                )
            )
        except Exception as exc:
            elapsed = time.perf_counter() - started
            LOGGER.warning("Batch file %s failed: %s", filename, exc)
            records.append(
                BatchPredictionRecord(
                    filename=filename,
                    predicted_class="",
                    display_name="",
                    taxonomy="",
                    confidence=0.0,
                    confidence_label="RENDAH",
                    processing_time=elapsed,
                    status="Gagal",
                    error_message=str(exc),
                )
            )
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

        if progress_callback is not None:
            progress_callback(index, total)

    LOGGER.info(
        "Batch completed: %d/%d succeeded.",
        sum(1 for r in records if r.status == "Berhasil"),
        total,
    )
    return records


def _save_to_temp(uploaded_file: Any) -> Path:
    """Persist an uploaded file's bytes to a temp file and return its path."""
    filename = getattr(uploaded_file, "name", "") or "upload"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    with tempfile.NamedTemporaryFile(
        suffix=f".{ext}" if ext else "", delete=False
    ) as handle:
        uploaded_file.seek(0)
        handle.write(uploaded_file.read())
        return Path(handle.name)


# ── Self-check ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    res = load_pipeline_resources()
    print("Loaded classes:", res["class_names"])
    print("Device:", res["device"])
