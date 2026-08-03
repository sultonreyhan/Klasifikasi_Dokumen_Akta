"""IndoBERT feature-engineering pipeline for AktaSense.

This module turns the cleaned dataset (``Training_Dataset.csv``) into dense
semantic feature vectors using Indonesian-IndoBERT with mean pooling. Every
public function is reusable by later sprints (random forest training,
prediction, evaluation); **no** artifact is generated or saved here.

Public API
----------
- :func:`load_dataset`
- :func:`validate_dataset`
- :func:`load_tokenizer`
- :func:`load_model`
- :func:`tokenize`
- :func:`mean_pooling`
- :func:`generate_embedding`
- :func:`build_feature_matrix`
- :func:`export_feature_matrix`
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm

from Pipeline import config

LOGGER = logging.getLogger(__name__)

# Pooling strategy applied to produce one fixed-size vector per document.
POOLING_STRATEGY: str = "mean_pooling"


# --------------------------------------------------------------------------- #
# Dataset loading & validation
# --------------------------------------------------------------------------- #

def load_dataset(
    dataset_path: Optional[Path] = None,
    text_column: str = config.COLUMN_TEXT,
    label_column: str = config.COLUMN_LABEL,
    taxonomy_column: str = config.COLUMN_TAXONOMY,
    filename_column: str = config.COLUMN_FILENAME,
) -> pd.DataFrame:
    """Load the training dataset from a CSV file.

    Args:
        dataset_path: Path to the CSV. Defaults to ``config.TRAINING_DATASET_PATH``.
        text_column: Name of the column holding the cleaned document text.
        label_column: Name of the column holding the string class label.
        taxonomy_column: Name of the column holding the taxonomy group.
        filename_column: Name of the column holding the source file name.

    Returns:
        A DataFrame containing the loaded dataset.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If a required column is missing from the CSV.
    """
    path = dataset_path or config.TRAINING_DATASET_PATH
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    LOGGER.info("Loading dataset from %s", path)
    frame = pd.read_csv(path)
    frame.columns = [str(col) for col in frame.columns]

    required = [text_column, label_column, taxonomy_column, filename_column]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing required column(s) in dataset: {missing}")

    LOGGER.info("Loaded %d rows with columns %s", len(frame), list(frame.columns))
    return frame


def validate_dataset(
    frame: pd.DataFrame,
    text_column: str = config.COLUMN_TEXT,
    label_column: str = config.COLUMN_LABEL,
) -> Dict[str, Any]:
    """Validate a dataset for empty text, missing labels and duplicates.

    Args:
        frame: The DataFrame produced by :func:`load_dataset`.
        text_column: Column containing the document text.
        label_column: Column containing the class label.

    Returns:
        A dictionary reporting the number of empty-text rows, missing-label
        rows, duplicate rows, total row count and a ``valid`` flag.

    Raises:
        ValueError: If the requested columns are absent from the frame.
    """
    if text_column not in frame.columns or label_column not in frame.columns:
        raise ValueError("validate_dataset requires both the text and label columns.")

    empty_text = int(
        frame[text_column].isna().sum()
        + (frame[text_column].astype(str).str.strip() == "").sum()
    )
    missing_labels = int(frame[label_column].isna().sum())
    duplicates = int(frame.duplicated().sum())

    report = {
        "total_rows": int(len(frame)),
        "empty_text": empty_text,
        "missing_labels": missing_labels,
        "duplicate_rows": duplicates,
        "valid": empty_text == 0 and missing_labels == 0 and duplicates == 0,
    }

    LOGGER.info(
        "Validation: total=%d empty_text=%d missing_labels=%d duplicates=%d valid=%s",
        report["total_rows"],
        report["empty_text"],
        report["missing_labels"],
        report["duplicate_rows"],
        report["valid"],
    )
    return report


# --------------------------------------------------------------------------- #
# Model & tokenizer loading
# --------------------------------------------------------------------------- #

def load_tokenizer(model_name: Optional[str] = None):
    """Load the official IndoBERT tokenizer.

    Args:
        model_name: Hugging Face identifier of the tokenizer. Defaults to
            ``config.MODEL_NAME``.

    Returns:
        The Hugging Face tokenizer instance.
    """
    name = model_name or config.MODEL_NAME
    LOGGER.info("Loading tokenizer %s", model_name or config.MODEL_NAME)
    return AutoTokenizer.from_pretrained(name)


def load_model(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
) -> Tuple[torch.nn.Module, str]:
    """Load the IndoBERT model and move it to a target device.

    Args:
        model_name: Hugging Face identifier of the model. Defaults to
            ``config.MODEL_NAME``.
        device: Target device string; defaults to ``config.DEVICE``.

    Returns:
        A ``(model, device)`` tuple. The model is switched to evaluation mode.

    Raises:
        RuntimeError: If the device string is not a supported device.
    """
    name = model_name or config.MODEL_NAME
    target = device or config.DEVICE

    LOGGER.info("Loading model %s on device=%s", name, target)
    model = AutoModel.from_pretrained(name)

    if target in ("cuda", "cpu"):
        model.to(target)
    else:
        raise RuntimeError(f"Unsupported device: {target}")

    model.eval()
    return model, target


# --------------------------------------------------------------------------- #
# Tokenization
# --------------------------------------------------------------------------- #

def tokenize(
    tokenizer: Any,
    documents: Sequence[str],
    max_length: int = config.MAX_LENGTH,
    device: str = config.DEVICE,
):
    """Tokenize a list of documents into model-ready tensors.

    Documents are padded/truncated so each one maps to a single
    ``(max_length,)`` sequence. For very long documents prefer the window-aware
    path in :func:`generate_embedding` to avoid information loss.

    Args:
        tokenizer: The IndoBERT tokenizer from :func:`load_tokenizer`.
        documents: An ordered list of raw (cleaned) document strings.
        max_length: Maximum number of tokens per sequence.
        device: Device that will hold the generated tensors.

    Returns:
        ``(input_ids, attention_mask)`` both shaped ``(batch, max_length)`` and
        moved to ``device``. Returns ``(None, None)`` for an empty input.
    """
    if not documents:
        return None, None

    encoded = tokenizer(
        list(documents),
        add_special_tokens=True,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    LOGGER.debug(
        "Tokenized %d documents -> %d sequences (max_length=%d)",
        len(documents),
        input_ids.shape[0],
        max_length,
    )
    return input_ids, attention_mask


# --------------------------------------------------------------------------- #
# Mean pooling & embedding
# --------------------------------------------------------------------------- #

def mean_pooling(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Aggregate token embeddings into a fixed-size vector using masked mean.

    Args:
        last_hidden_state: Tensor of shape ``(batch, seq_len, hidden)``.
        attention_mask: Binary tensor of shape ``(batch, seq_len)`` that marks
            real tokens (1) and padding (0).

    Returns:
        Tensor of shape ``(batch, hidden)``; padding positions are excluded.
    """
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


def generate_embedding(
    model: torch.nn.Module,
    tokenizer: Any,
    text: str,
    device: str = config.DEVICE,
    max_length: int = config.MAX_LENGTH,
) -> np.ndarray:
    """Compute the 768-dimensional embedding for a single document.

    The document is tokenized and forwarded through the model. Long documents
    are split into sliding windows of ``max_length`` tokens and the per-window
    mean-pooled vectors are then averaged, so the entire text is represented.

    Args:
        model: The loaded IndoBERT model.
        tokenizer: The loaded IndoBERT tokenizer.
        text: A single document's cleaned text.
        device: Device hosting the model.
        max_length: Maximum number of tokens per window.

    Returns:
        A ``768``-dimensional :class:`numpy.ndarray` for the document.

    Raises:
        ValueError: If the document tokenizes to zero tokens.
    """
    tokens = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
        padding=False,
        return_tensors="pt",
    )
    ids = tokens["input_ids"][0]
    if ids.numel() == 0:
        raise ValueError("Cannot embed an empty document.")

    window_vectors = []
    start = 0
    total = ids.shape[0]
    while start < total:
        chunk = ids[start : start + max_length].unsqueeze(0).to(device)
        attn = torch.ones_like(chunk)
        with torch.no_grad():
            output = model(input_ids=chunk, attention_mask=attn)
        window_vectors.append(mean_pooling(output.last_hidden_state, attn).squeeze(0))
        start += max_length

    doc_vector = torch.stack(window_vectors).mean(dim=0)
    return doc_vector.cpu().numpy().astype(np.float32)


# --------------------------------------------------------------------------- #
# Feature matrix construction & export
# --------------------------------------------------------------------------- #

def build_feature_matrix(
    model: torch.nn.Module,
    tokenizer: Any,
    documents: Sequence[str],
    labels: Optional[Sequence[str]] = None,
    device: str = config.DEVICE,
    max_length: int = config.MAX_LENGTH,
    batch_size: int = config.BATCH_SIZE,
    show_progress: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Build the feature matrix ``X`` and the encoded label vector ``y``.

    Documents are embedded via :func:`generate_embedding` and stacked into a
    single ``(n_samples, hidden_size)`` float matrix.

    Args:
        model: The loaded IndoBERT model.
        tokenizer: The loaded IndoBERT tokenizer.
        documents: Ordered list of document texts.
        labels: Optional aligned string labels per document. When provided,
            unique labels are sorted and encoded to integer ``y``.
        device: Device hosting the model.
        max_length: Maximum token window per document.
        batch_size: Reserved for batched inference; retained for API stability.
        show_progress: Whether to show a tqdm progress bar.

    Returns:
        A ``(X, y)`` tuple; ``y`` is ``None`` when ``labels`` is ``None``.

    Raises:
        ValueError: If the number of labels does not match the documents.
    """
    if labels is not None and len(labels) != len(documents):
        raise ValueError("Number of labels must match number of documents.")

    hidden_size = model.config.hidden_size
    vectors = np.empty((len(documents), hidden_size), dtype=np.float32)
    y: Optional[np.ndarray] = None

    if labels is not None:
        unique = sorted(set(str(lab) for lab in labels))
        mapping = {lab: i for i, lab in enumerate(unique)}
        y = np.array([mapping[str(lb)] for lb in labels], dtype=np.int64)
        LOGGER.info("Encoded %d classes: %s", len(unique), unique)

    iterator = tqdm(documents, desc="Embedding documents", disable=not show_progress)
    for index, text in enumerate(iterator):
        vectors[index] = generate_embedding(
            model, tokenizer, text, device=device, max_length=max_length
        )

    LOGGER.info("Built feature matrix shape=%s dtype=%s", vectors.shape, vectors.dtype)
    return vectors, y


def export_feature_matrix(
    feature_matrix: np.ndarray,
    labels: Optional[np.ndarray],
    output_path: Path,
    filenames: Optional[List[str]] = None,
) -> Path:
    """Export an embedding feature matrix to a compressed ``.npz`` artifact.

    Args:
        feature_matrix: The ``X`` matrix from :func:`build_feature_matrix`.
        labels: The integer ``y`` vector; may be ``None``.
        output_path: Destination file; its parent directory is created if needed.
        filenames: Optional aligned list of source file names.

    Returns:
        The path to the written artifact.

    Raises:
        ValueError: If the label length does not match the feature matrix rows.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if labels is not None and labels.shape[0] != feature_matrix.shape[0]:
        raise ValueError("Labels and feature_matrix row counts must match.")

    payload: Dict[str, Any] = {"X": feature_matrix}
    if labels is not None:
        payload["y"] = labels
    if filenames is not None:
        payload["filenames"] = np.asarray(filenames, dtype=object)

    np.savez_compressed(out, **payload)
    LOGGER.info("Exported feature matrix to %s (shape=%s)", out, feature_matrix.shape)
    return out


# --------------------------------------------------------------------------- #
# Optional entrypoint (opt-in smoke test only, writes nothing)
# --------------------------------------------------------------------------- #

def main() -> None:
    """Optional end-to-end run invoked only when executed directly.

    This never writes an artifact; it smoke-tests each reusable function so
    the pipeline can be validated during development.
    """
    logging.basicConfig(level=logging.INFO)
    frame = load_dataset()

    print(f"Validation: {validate_dataset(frame)}")

    tokenizer = load_tokenizer()
    model, device = load_model()

    X, _ = build_feature_matrix(
        model,
        tokenizer,
        frame[config.COLUMN_TEXT].tolist()[:4],
        device=device,
        show_progress=True,
    )
    LOGGER.info("Smoke test produced X shape=%s", X.shape)
    LOGGER.info("Pipeline ready for the next sprint.")


if __name__ == "__main__":
    main()