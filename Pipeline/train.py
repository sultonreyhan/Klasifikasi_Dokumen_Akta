"""Random Forest training pipeline for AktaSense.

This module trains the document classifier described in the locked Model
Architecture. Features are produced by the existing :mod:`Pipeline.embedding`
module (IndoBERT + mean pooling); the classifier is :class:`~sklearn.ensemble.RandomForestClassifier`.

This module is modular and reusable. It only handles training
(and loading for downstream use): it does **not** evaluate,
**not** compute SHAP and **not** run inference.

Public API
----------
- :func:`load_dataset`
- :func:`validate_dataset`
- :func:`prepare_features`
- :func:`prepare_labels`
- :func:`split_dataset`
- :func:`build_random_forest`
- :func:`train_model`
- :func:`save_model`
- :func:`save_label_encoder`
- :func:`save_training_metadata`
- :func:`load_model`
- :func:`load_label_encoder`
- :func:`load_training_metadata`
- :func:`main`
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from Pipeline import config
from Pipeline import embedding

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Dataset loading & validation
# --------------------------------------------------------------------------- #

def load_dataset(dataset_path: Optional[Path] = None) -> pd.DataFrame:
    """Load the training dataset through the shared embedding module.

    Args:
        dataset_path: CSV path; defaults to ``config.TRAINING_DATASET_PATH``.

    Returns:
        The loaded :class:`pandas.DataFrame`.

    Raises:
        FileNotFoundError: If the dataset does not exist.
        ValueError: If a required column is missing.
    """
    LOGGER.info("Loading training dataset")
    return embedding.load_dataset(dataset_path)


def validate_dataset(frame: pd.DataFrame) -> Dict[str, Any]:
    """Validate the dataset and abort on a fatal error.

    Args:
        frame: DataFrame to validate.

    Returns:
        The validation report dictionary.

    Raises:
        ValueError: If the dataset is empty, missing text/labels or duplicated.
    """
    report = embedding.validate_dataset(frame)
    LOGGER.info("Validation report: %s", report)
    if not report["valid"]:
        raise ValueError(
            "Dataset is not ready for training. Review the validation report."
        )
    return report


# --------------------------------------------------------------------------- #
# Feature & label preparation
# --------------------------------------------------------------------------- #

def prepare_features(
    model,
    tokenizer,
    frame: pd.DataFrame,
    device: str = config.DEVICE,
    show_progress: bool = True,
) -> np.ndarray:
    """Generate the embedding feature matrix from document texts.

    Reuses the embedding pipeline and never duplicates its logic.

    Args:
        model: The loaded IndoBERT model (from ``embedding.load_model``).
        tokenizer: The loaded IndoBERT tokenizer (from ``embedding.load_tokenizer``).
        frame: The loaded dataset.
        device: Device that hosts the model.
        show_progress: Whether to show the tqdm progress bar.

    Returns:
        ``X`` matrix of shape ``(n_samples, 768)``.

    Raises:
        ValueError: If the feature generation fails for any document.
    """
    LOGGER.info("Preparing features for %d documents", len(frame))
    documents = frame[config.COLUMN_TEXT].astype(str).tolist()
    if not documents:
        raise ValueError("No documents available for feature generation.")

    matrix, _ = embedding.build_feature_matrix(
        model,
        tokenizer,
        documents,
        device=device,
        max_length=config.MAX_LENGTH,
        show_progress=show_progress,
    )
    LOGGER.info("Feature matrix ready: shape=%s dtype=%s", matrix.shape, matrix.dtype)
    return matrix


def prepare_labels(
    frame: pd.DataFrame, encoder: Optional[LabelEncoder] = None
) -> Tuple[np.ndarray, LabelEncoder]:
    """Encode string labels to integers with a :class:`LabelEncoder`.

    Args:
        frame: The loaded dataset.
        encoder: Optional pre-existing encoder; otherwise a new one is fitted.

    Returns:
        A tuple of the integer labels vector and the fitted encoder.
    """
    raw = frame[config.COLUMN_LABEL].astype(str).tolist()
    if encoder is None:
        encoder = LabelEncoder()
        labels = encoder.fit_transform(raw)
    else:
        labels = encoder.transform(raw)

    LOGGER.info("Encoded %d samples into %d classes", len(labels), len(encoder.classes_))
    return labels.astype(np.int64), encoder


# --------------------------------------------------------------------------- #
# Train/test split
# --------------------------------------------------------------------------- #

def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = config.TEST_SIZE,
    random_state: Optional[int] = config.RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Partition the feature matrix into stratified train/test splits.

    Args:
        X: Feature matrix.
        y: Integer labels.
        test_size: Fraction of samples held out for testing.
        random_state: Random seed for reproducibility.

    Returns:
        A ``(X_train, X_test, y_train, y_test)`` tuple.

    Raises:
        ValueError: If a class has too few samples to stratify.
    """
    LOGGER.info("Splitting dataset (test_size=%.2f, random_state=%s)", test_size, random_state)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    LOGGER.info(
        "Split sizes -> train=%d test=%d",
        len(X_train),
        len(X_test),
    )
    return X_train, X_test, y_train, y_test


# --------------------------------------------------------------------------- #
# Model construction & training
# --------------------------------------------------------------------------- #

def build_random_forest() -> RandomForestClassifier:
    """Instantiate the Random Forest classifier from configuration.

    Returns:
        An unconfigured ``RandomForestClassifier`` instance.

    Raises:
        ValueError: If any configured hyperparameter is invalid.
    """
    params: Dict[str, Any] = {
        "n_estimators": config.N_ESTIMATORS,
        "max_depth": config.MAX_DEPTH,
        "min_samples_split": config.MIN_SAMPLES_SPLIT,
        "min_samples_leaf": config.MIN_SAMPLES_LEAF,
        "class_weight": config.CLASS_WEIGHT,
        "n_jobs": config.N_JOBS,
        "random_state": config.RANDOM_SEED,
        "oob_score": True,  # out-of-bag estimate for the tiny dataset
    }

    clf = RandomForestClassifier(**params)
    LOGGER.info("Built RandomForestClassifier with params=%s", params)
    return clf


def train_model(
    clf: RandomForestClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> RandomForestClassifier:
    """Fit the random forest on the training split.

    Args:
        clf: An unfitted classifier from :func:`build_random_forest`.
        X_train: Training feature matrix.
        y_train: Training integer labels.

    Returns:
        The fitted classifier.

    Raises:
        RuntimeError: If the model fails during fitting.
    """
    if not len(X_train):
        raise ValueError("Cannot train on an empty training set.")

    LOGGER.info("Training random forest on %d samples", len(X_train))
    try:
        clf.fit(X_train, y_train)
    except Exception as exc:  # pragma: no cover - defensive guard
        raise RuntimeError(f"Random forest fitting failed: {exc}") from exc

    LOGGER.info("Training complete (train accuracy=%.4f)", clf.score(X_train, y_train))
    return clf


# --------------------------------------------------------------------------- #
# Serialization
# --------------------------------------------------------------------------- #

def save_model(clf: RandomForestClassifier, path: Optional[Path] = None) -> Path:
    """Serialize the fitted Random Forest via ``joblib``.

    Args:
        clf: The fitted classifier.
        path: Destination path; defaults to ``config.MODEL_ARTIFACT_PATH``.

    Returns:
        The path where the model was written.

    Raises:
        OSError: If the file cannot be written.
    """
    out = Path(path) if path else config.MODEL_ARTIFACT_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        joblib.dump(clf, out)
    except Exception as exc:  # pragma: no cover - defensive guard
        raise OSError(f"Failed to write model to {out}: {exc}") from exc
    LOGGER.info("Saved model to %s", out)
    return out


def save_label_encoder(encoder: LabelEncoder, output_path: Optional[Path] = None) -> Path:
    """Serialize the fitted :class:`LabelEncoder` via ``joblib``.

    Args:
        encoder: The fitted label encoder.
        output_path: Destination path; defaults to ``config.LABEL_ENCODER_PATH``.

    Returns:
        The path where the encoder was written.

    Raises:
        OSError: If the file cannot be written.
    """
    out = Path(output_path) if output_path else config.LABEL_ENCODER_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        joblib.dump(encoder, out)
    except Exception as exc:  # pragma: no cover - defensive guard
        raise OSError(f"Failed to write label encoder to {out}: {exc}") from exc
    LOGGER.info("Saved label encoder to %s", out)
    return out


def save_training_metadata(
    metadata: Dict[str, Any],
    output_path: Optional[Path] = None,
) -> Path:
    """Persist training metadata as formatted JSON.

    Args:
        metadata: Arbitrary JSON-serializable metadata to store.
        output_path: Destination path; defaults to ``config.TRAINING_METADATA_PATH``.

    Returns:
        The path where the metadata was written.

    Raises:
        OSError: If the file cannot be written.
    """
    out = Path(output_path) if output_path else config.TRAINING_METADATA_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)
    except (OSError, TypeError) as exc:
        raise OSError(f"Failed to write training metadata to {out}: {exc}") from exc
    LOGGER.info("Saved training metadata to %s", out)
    return out


def load_model(path: Optional[Path] = None) -> RandomForestClassifier:
    """Deserialize a previously trained Random Forest via ``joblib``.

    Intended for downstream prediction / the application.

    Args:
        path: Source path; defaults to ``config.MODEL_ARTIFACT_PATH``.

    Returns:
        The deserialized fitted classifier.

    Raises:
        FileNotFoundError: If the artifact does not exist.
        OSError: If the artifact cannot be read.
    """
    source = Path(path) if path else config.MODEL_ARTIFACT_PATH
    if not source.exists():
        raise FileNotFoundError(f"Model artifact not found: {source}")
    try:
        clf = joblib.load(source)
    except Exception as exc:  # pragma: no cover - defensive guard
        raise OSError(f"Failed to load model from {source}: {exc}") from exc
    LOGGER.info("Loaded model from %s", source)
    return clf


def load_label_encoder(path: Optional[Path] = None) -> LabelEncoder:
    """Deserialize the fitted :class:`LabelEncoder` via ``joblib``.

    Args:
        path: Source path; defaults to ``config.LABEL_ENCODER_PATH``.

    Returns:
        The deserialized fitted encoder.

    Raises:
        FileNotFoundError: If the artifact does not exist.
        OSError: If the artifact cannot be read.
    """
    source = Path(path) if path else config.LABEL_ENCODER_PATH
    if not source.exists():
        raise FileNotFoundError(f"Label encoder artifact not found: {source}")
    try:
        encoder = joblib.load(source)
    except Exception as exc:  # pragma: no cover - defensive guard
        raise OSError(f"Failed to load label encoder from {source}: {exc}") from exc
    LOGGER.info("Loaded label encoder from %s", source)
    return encoder


def load_training_metadata(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load the persisted training metadata JSON.

    Args:
        path: Source path; defaults to ``config.TRAINING_METADATA_PATH``.

    Returns:
        The metadata dictionary.

    Raises:
        FileNotFoundError: If the artifact does not exist.
        ValueError: If the JSON cannot be parsed.
    """
    source = Path(path) if path else config.TRAINING_METADATA_PATH
    if not source.exists():
        raise FileNotFoundError(f"Training metadata not found: {source}")
    try:
        with open(source, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Failed to parse training metadata from {source}: {exc}") from exc


# --------------------------------------------------------------------------- #
# Pipeline orchestration
# --------------------------------------------------------------------------- #

def _build_metadata(
    clf: RandomForestClassifier,
    X_train: np.ndarray,
    X_test: np.ndarray,
    encoder: LabelEncoder,
    validation: Dict[str, Any],
    elapsed_seconds: float,
) -> Dict[str, Any]:
    """Assemble a JSON-serializable training metadata dictionary."""
    return {
        "model": "RandomForestClassifier",
        "embedding_model": config.MODEL_NAME,
        "embedding_pooling": embedding.POOLING_STRATEGY,
        "embedding_dimension": int(X_train.shape[1]),
        "num_samples": int(validation["total_rows"]),
        "num_classes": int(len(encoder.classes_)),
        "label_list": [str(c) for c in encoder.classes_],
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "test_fraction": config.TEST_SIZE,
        "oob_score": float(clf.oob_score_) if getattr(clf, "oob_score_", None) else None,
        "hyperparameters": clf.get_params(deep=False),
        "validation": validation,
        "training_time_seconds": round(elapsed_seconds, 3),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    """End-to-end training entry point.

    Loads the dataset, generates embeddings via the embedding module, splits,
    trains a Random Forest, and persists the model, label encoder and metadata.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    started = datetime.now(timezone.utc)

    try:
        frame = load_dataset()
        validation = validate_dataset(frame)
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.error("Dataset stage failed: %s", exc)
        raise

    try:
        tokenizer = embedding.load_tokenizer()
        model, device = embedding.load_model()
    except Exception as exc:
        LOGGER.error("Model loading failed: %s", exc)
        raise

    try:
        X = prepare_features(model, tokenizer, frame, device=device)
    except Exception as exc:
        LOGGER.error("Feature generation failed: %s", exc)
        raise

    y, encoder = prepare_labels(frame)

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    clf = build_random_forest()
    clf = train_model(clf, X_train, y_train)

    save_model(clf)
    save_label_encoder(encoder)

    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    metadata = _build_metadata(clf, X_train, X_test, encoder, validation, elapsed)
    save_training_metadata(metadata)

    LOGGER.info(
        "Training pipeline completed in %.2fs. Artifacts written to %s",
        elapsed,
        config.OUTPUT_DIRECTORY,
    )


if __name__ == "__main__":
    main()