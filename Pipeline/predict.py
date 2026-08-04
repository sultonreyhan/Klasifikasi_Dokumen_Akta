"""Document prediction pipeline for AktaSense.

End-to-end inference for a single document: text extraction (native PDF text
layer with OCR fallback), cleaning, IndoBERT embedding, Random Forest
prediction, probability calculation, label decoding and SHAP explanation.

This module contains **all** prediction business logic. The Streamlit
application must only call :func:`predict_document` and render the returned
:class:`PredictionResult`; it must not reimplement any inference logic.

Public API
----------
- :func:`load_model`
- :func:`load_label_encoder`
- :func:`extract_text`
- :func:`clean_text`
- :func:`generate_embedding`
- :func:`predict`
- :func:`predict_probability`
- :func:`decode_prediction`
- :func:`generate_shap_explanation`
- :func:`predict_document`
- :class:`PredictionResult`
"""

import logging
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from Pipeline import config
from Pipeline import embedding
from Pipeline import train as train_pipeline
from Pipeline.ocr_engine import get_ocr_engine, recognize_text

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Result object
# --------------------------------------------------------------------------- #

@dataclass
class PredictionResult:
    """Structured result of a single document prediction."""

    predicted_class: str
    confidence_score: float
    probability_distribution: Dict[str, float]
    shap_explanation: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dictionary of the result."""
        return asdict(self)


# --------------------------------------------------------------------------- #
# Model & encoder loading
# --------------------------------------------------------------------------- #

def load_model(path: Optional[Path] = None):
    """Load the trained Random Forest artifact.

    Args:
        path: Path to ``random_forest.pkl``; defaults to
            ``config.MODEL_ARTIFACT_PATH``.

    Returns:
        The fitted Random Forest classifier.

    Raises:
        FileNotFoundError: If the artifact does not exist.
    """
    return train_pipeline.load_model(path)


def load_label_encoder(path: Optional[Path] = None):
    """Load the persisted :class:`LabelEncoder`.

    Args:
        path: Path to ``label_encoder.pkl``; defaults to
            ``config.LABEL_ENCODER_PATH``.

    Returns:
        The fitted label encoder.

    Raises:
        FileNotFoundError: If the artifact does not exist.
    """
    return train_pipeline.load_label_encoder(path)


# --------------------------------------------------------------------------- #
# Text extraction
# --------------------------------------------------------------------------- #

def extract_text(source: Path) -> str:
    """Extract raw text from a document file.

    Supported formats:

    - ``.pdf``    native text layer; OCR (RapidOCR) fallback when a page has
                  no embedded text (graceful skip if the OCR engine is absent)
    - ``.docx``   via ``python-docx``

    Args:
        source: Path to the document.

    Returns:
        The extracted raw text.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file type is unsupported or the text is unreadable.
    """
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"Document not found: {source}")

    suffix = source.suffix.lower()
    LOGGER.info("Extracting text from %s (%s)", source.name, suffix)

    if suffix == ".pdf":
        return _extract_pdf(source)
    if suffix == ".docx":
        return _extract_docx(source)
    raise ValueError(f"Unsupported file type: {suffix}. Use .pdf or .docx.")


def _extract_pdf(source: Path) -> str:
    """Extract text from a PDF using the native text layer, with OCR fallback."""
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise ValueError("PyMuPDF is required to read PDF documents.") from exc

    parts: List[str] = []
    try:
        with fitz.open(str(source)) as document:
            for page in document:
                page_text = page.get_text().strip()
                if page_text:
                    parts.append(page_text)
                    continue
                # No embedded text -> try OCR fallback.
                ocr = _ocr_page(page)
                if ocr:
                    parts.append(ocr)
                else:
                    LOGGER.warning(
                        "Page %d has no text layer and OCR unavailable; skipped.",
                        page.number + 1,
                    )
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Failed to read PDF {source.name}: {exc}") from exc

    text = "\n".join(parts)
    if not text.strip():
        raise ValueError(f"No text could be extracted from {source.name}.")
    return text


def _ocr_page(page) -> Optional[str]:
    """Best-effort OCR of a page using the shared RapidOCR engine."""
    try:
        ocr = get_ocr_engine()
        pixmap = page.get_pixmap(dpi=200)
        image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height, pixmap.width, pixmap.n
        )
        if image.shape[2] == 4:
            image = image[:, :, :3]
        elif image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        text = recognize_text(ocr, image)
        return text or None
    except Exception as exc:  # pragma: no cover - OCR is best-effort
        LOGGER.warning("OCR failed for a page: %s", exc)
        return None


def _extract_docx(source: Path) -> str:
    """Extract text from a ``.docx`` document."""
    try:
        import docx
    except ImportError as exc:
        raise ValueError("python-docx is required to read .docx documents.") from exc

    try:
        document = docx.Document(str(source))
        paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
    except Exception as exc:
        raise ValueError(f"Failed to read DOCX {source.name}: {exc}") from exc

    text = "\n".join(paragraphs)
    if not text.strip():
        raise ValueError(f"No text could be extracted from {source.name}.")
    return text


# --------------------------------------------------------------------------- #
# Text cleaning
# --------------------------------------------------------------------------- #

def clean_text(text: str) -> str:
    """Normalize raw document text consistently with the training data.

    Minimal, semantics-preserving cleaning only (mirrors the dataset phase):
    Unicode NFC normalization, line-ending normalization, whitespace
    collapsing and blank-line collapsing. No stopwords, stemming, lowercasing
    or punctuation removal.

    Args:
        text: Raw text to clean.

    Returns:
        The cleaned text.

    Raises:
        ValueError: If the input is empty or only whitespace.
    """
    if text is None or not str(text).strip():
        raise ValueError("Cannot clean an empty document.")

    cleaned = str(text)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")  # line endings
    cleaned = unicodedata_normalize(cleaned)                    # NFC
    cleaned = re.sub(r"[ \t]+", " ", cleaned)                   # inline whitespace
    cleaned = re.sub(r" +\n", "\n", cleaned)                    # trailing inline ws
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)                # blank-line collapse
    cleaned = cleaned.strip()

    if not cleaned:
        raise ValueError("Document became empty after cleaning.")
    return cleaned


def unicodedata_normalize(text: str) -> str:
    """Apply Unicode NFC normalization to a string."""
    import unicodedata

    return unicodedata.normalize("NFC", text)


# --------------------------------------------------------------------------- #
# Feature generation
# --------------------------------------------------------------------------- #

def generate_embedding(
    text: str,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    device: str = config.DEVICE,
) -> np.ndarray:
    """Compute the 768-dimensional IndoBERT embedding for cleaned text.

    Args:
        text: Cleaned document text.
        model: Optional already-loaded IndoBERT model; otherwise loaded once.
        tokenizer: Optional already-loaded tokenizer; otherwise loaded once.
        device: Device hosting the model.

    Returns:
        The ``768``-dimensional feature vector.

    Raises:
        ValueError: If the text cannot be embedded.
    """
    own_model = model is None or tokenizer is None
    if own_model:
        model, device = embedding.load_model(device=device)
        tokenizer = embedding.load_tokenizer()

    try:
        vector = embedding.generate_embedding(
            model, tokenizer, text, device=device, max_length=config.MAX_LENGTH
        )
    except Exception as exc:
        raise ValueError(f"Embedding generation failed: {exc}") from exc

    LOGGER.info("Generated embedding shape=%s dtype=%s", vector.shape, vector.dtype)
    return vector


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #

def predict(
    clf,
    vector: np.ndarray,
) -> np.ndarray:
    """Return the raw class-index prediction for an embedding vector.

    Args:
        clf: The fitted Random Forest classifier.
        vector: A ``(768,)`` embedding vector.

    Returns:
        A length-1 integer array with the predicted class index.

    Raises:
        ValueError: If prediction fails.
    """
    matrix = np.asarray(vector, dtype=np.float32).reshape(1, -1)
    try:
        pred = clf.predict(matrix)
    except Exception as exc:
        raise ValueError(f"Random Forest prediction failed: {exc}") from exc
    return pred


def predict_probability(
    clf,
    vector: np.ndarray,
) -> np.ndarray:
    """Return the class-probability vector for an embedding vector.

    Args:
        clf: The fitted Random Forest classifier.
        vector: A ``(768,)`` embedding vector.

    Returns:
        A ``(n_classes,)`` probability array (rows sum to 1).

    Raises:
        ValueError: If probability computation fails.
    """
    matrix = np.asarray(vector, dtype=np.float32).reshape(1, -1)
    try:
        proba = clf.predict_proba(matrix)[0]
    except Exception as exc:
        raise ValueError(f"Probability computation failed: {exc}") from exc
    return proba


def decode_prediction(
    encoder,
    proba: np.ndarray,
) -> Tuple[str, float]:
    """Decode probabilities into the predicted class and confidence score.

    Args:
        encoder: The fitted :class:`LabelEncoder`.
        proba: Probability vector from :func:`predict_probability`.

    Returns:
        ``(predicted_class, confidence_score)``.

    Raises:
        ValueError: If the probability vector is empty.
    """
    if proba is None or len(proba) == 0:
        raise ValueError("Empty probability vector cannot be decoded.")

    best_index = int(np.argmax(proba))
    predicted_class = str(encoder.inverse_transform([best_index])[0])
    confidence = float(proba[best_index])
    return predicted_class, confidence


# --------------------------------------------------------------------------- #
# SHAP explanation
# --------------------------------------------------------------------------- #

def generate_shap_explanation(
    clf,
    vector: np.ndarray,
    class_names: List[str],
    top_k: int = 10,
) -> Dict[str, Any]:
    """Produce a JSON-serializable local SHAP explanation for one document.

    Uses :class:`shap.TreeExplainer`, the exact explainer for tree models, on a
    single embedding vector.

    Args:
        clf: The fitted Random Forest classifier.
        vector: The ``(768,)`` embedding vector.
        class_names: Ordered class label strings.
        top_k: Number of top contributing features to report.

    Returns:
        A dictionary with the base value, per-class top contributors, and
        global feature importances.

    Raises:
        ValueError: If SHAP fails for the given sample.
    """
    import shap

    matrix = np.asarray(vector, dtype=np.float32).reshape(1, -1)
    try:
        explainer = shap.TreeExplainer(clf)
        raw = explainer.shap_values(matrix, check_additivity=False)
    except Exception as exc:
        raise ValueError(f"SHAP explanation failed: {exc}") from exc

    base = np.asarray(explainer.expected_value)
    feature_count = matrix.shape[1]
    names = [f"dim_{i}" for i in range(feature_count)]

    def _top_contrib(contrib: np.ndarray) -> List[Dict[str, Any]]:
        order = np.argsort(np.abs(contrib))[::-1][:top_k]
        return [
            {
                "feature": names[int(i)],
                "index": int(i),
                "value": float(contrib[int(i)]),
            }
            for i in order
        ]

    def _base_value(class_index: int) -> float:
        if base.ndim == 0:
            return float(base)
        if base.ndim == 1 and len(base) == 1:
            return float(base[0])
        return float(base[class_index])

    per_class: Dict[str, Any] = {}
    if isinstance(raw, list):
        # per-class list of (1, f) arrays
        for c, class_name in enumerate(class_names):
            contrib = np.asarray(raw[c])[0]
            per_class[class_name] = {
                "base_value": _base_value(c),
                "top_contributors": _top_contrib(contrib),
            }
    else:
        values = np.asarray(raw)
        if values.ndim == 3:
            # (1, f, c) -> classes on the last axis
            for c, class_name in enumerate(class_names):
                contrib = values[0, :, c]
                per_class[class_name] = {
                    "base_value": _base_value(c),
                    "top_contributors": _top_contrib(contrib),
                }
        else:
            # (1, f) single output
            per_class[class_names[0]] = {
                "base_value": _base_value(0),
                "top_contributors": _top_contrib(values[0]),
            }

    return {
        "explainer": "shap.TreeExplainer",
        "classes": per_class,
        "feature_count": feature_count,
    }


# --------------------------------------------------------------------------- #
# End-to-end prediction
# --------------------------------------------------------------------------- #

def predict_document(
    source: Optional[Path] = None,
    text: Optional[str] = None,
    clf: Optional[Any] = None,
    encoder: Optional[Any] = None,
    embedding_model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    device: str = config.DEVICE,
    with_shap: bool = True,
) -> PredictionResult:
    """Predict the class of a single document end-to-end.

    Exactly one of ``source`` (a file path) or ``text`` (raw document text)
    must be provided.

    Args:
        source: Path to a ``.pdf``/``.docx`` document.
        text: Raw document text (bypasses file extraction).
        clf: Optional pre-loaded classifier; otherwise loaded once.
        encoder: Optional pre-loaded encoder; otherwise loaded once.
        embedding_model: Optional pre-loaded IndoBERT model.
        tokenizer: Optional pre-loaded tokenizer.
        device: Device hosting the embedding model.
        with_shap: Whether to compute the SHAP explanation.

    Returns:
        A :class:`PredictionResult` with the predicted class, confidence,
        probability distribution, SHAP explanation, metadata and timestamp.

    Raises:
        ValueError: If both/neither of ``source`` and ``text`` are provided,
            or if any inference step fails.
    """
    if (source is None) == (text is None):
        raise ValueError("Provide exactly one of `source` or `text`.")

    # 1. Load artifacts once, lazily.
    if clf is None:
        clf = load_model()
    if encoder is None:
        encoder = load_label_encoder()
    class_names = [str(c) for c in encoder.classes_]

    # 2. Text extraction + cleaning.
    if source is not None:
        raw_text = extract_text(source)
    else:
        raw_text = str(text)
    cleaned = clean_text(raw_text)

    # 3. Embedding.
    vector = generate_embedding(cleaned, embedding_model, tokenizer, device=device)

    # 4. Prediction.
    proba = predict_probability(clf, vector)
    predicted_class, confidence = decode_prediction(encoder, proba)

    # 5. SHAP explanation.
    if with_shap:
        shap_explanation = generate_shap_explanation(clf, vector, class_names)
    else:
        shap_explanation = {"explainer": None, "classes": {}, "feature_count": int(vector.shape[0])}

    # 6. Assemble result.
    probability_distribution = {
        name: float(prob)
        for name, prob in zip(class_names, proba)
    }
    metadata = {
        "source": str(source) if source else None,
        "text_length": int(len(cleaned)),
        "embedding_dimension": int(vector.shape[0]),
        "embedding_model": config.MODEL_NAME,
        "model": type(clf).__name__,
    }

    result = PredictionResult(
        predicted_class=predicted_class,
        confidence_score=confidence,
        probability_distribution=probability_distribution,
        shap_explanation=shap_explanation,
        metadata=metadata,
    )
    LOGGER.info(
        "Prediction complete -> %s (confidence=%.4f)",
        predicted_class,
        confidence,
    )
    return result


# --------------------------------------------------------------------------- #
# Optional entrypoint (smoke test only, writes nothing)
# --------------------------------------------------------------------------- #

def main() -> None:
    """Smoke-test :func:`predict_document` on a text sample."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sample = (
        "AKTA PENDIRIAN PERSEROAN TERBATAS. Pada hari ini telah hadir para "
        "pemegang saham dan menyetujui anggaran dasar perusahaan. Notaris "
        "membacakan akta kepada para pihak."
    )
    result = predict_document(text=sample)
    print(result.to_dict())


if __name__ == "__main__":
    main()
