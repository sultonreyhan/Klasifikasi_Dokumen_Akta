"""Model evaluation pipeline for AktaSense.

Evaluates the trained :class:`~sklearn.ensemble.RandomForestClassifier` on the
held-out test split of the locked dataset. Features are produced with the same
IndoBERT + mean-pooling pipeline used during training (reused from
:mod:`Pipeline.embedding` and :mod:`Pipeline.train`).

Artifacts (written under ``config.EVALUATION_DIRECTORY``):

- ``metrics.json``              - scalar + per-class performance metrics
- ``classification_report.json``- structured classification report
- ``evaluation_summary.json``   - combined run summary
- ``confusion_matrix.png``      - confusion matrix heat map

Public API
----------
- :func:`load_model`
- :func:`load_label_encoder`
- :func:`prepare_evaluation_data`
- :func:`evaluate_model`
- :func:`save_metrics`
- :func:`save_classification_report`
- :func:`save_evaluation_summary`
- :func:`save_confusion_matrix`
- :func:`main`
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless rendering for the confusion matrix PNG
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

from Pipeline import config
from Pipeline import embedding
from Pipeline import train as train_pipeline

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Artifact loading
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
# Evaluation dataset preparation
# --------------------------------------------------------------------------- #

def prepare_evaluation_data(
    model,
    tokenizer,
    frame: pd.DataFrame,
    device: str = config.DEVICE,
    show_progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the held-out feature matrix, true labels and original strings.

    The dataset is embedded with the shared pipeline and split with the exact
    parameters used at training time (same seed, same stratification), so the
    returned test split matches what the model never saw during fitting.

    Args:
        model: The IndoBERT model (from ``embedding.load_model``).
        tokenizer: The IndoBERT tokenizer.
        frame: Loaded evaluation dataset.
        device: Device hosting the IndoBERT model.
        show_progress: Whether to show the embedding progress bar.

    Returns:
        ``(X_test, y_test, y_test_str)`` where ``y_test`` are integer labels
        and ``y_test_str`` the original string labels for display.
    """
    X = train_pipeline.prepare_features(model, tokenizer, frame, device=device,
                                        show_progress=show_progress)
    y, encoder = train_pipeline.prepare_labels(frame)

    _, X_test, _, y_test = train_pipeline.split_dataset(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
    )

    y_test_str = np.asarray(
        [encoder.inverse_transform([int(v)])[0] for v in y_test], dtype=object
    )
    LOGGER.info("Evaluation data ready: X_test=%s", X_test.shape)
    return X_test, y_test, y_test_str


# --------------------------------------------------------------------------- #
# Metric computation
# --------------------------------------------------------------------------- #

def evaluate_model(
    clf,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
) -> Tuple[Dict[str, Any], Dict[str, Any], np.ndarray]:
    """Compute classification metrics on the held-out test split.

    Args:
        clf: The fitted Random Forest classifier.
        X_test: Test feature matrix.
        y_test: Test integer labels.
        class_names: Ordered class label strings (from the label encoder).

    Returns:
        A ``(metrics, report, cm)`` tuple: a scalar/per-class metrics dict, a
        structured classification report dict and the raw confusion matrix.

    Raises:
        ValueError: If the prediction output is malformed.
    """
    y_pred = clf.predict(X_test)

    n_classes = len(class_names)
    precision = precision_score(y_test, y_pred, average=None, labels=range(n_classes),
                                zero_division=0)
    recall = recall_score(y_test, y_pred, average=None, labels=range(n_classes),
                          zero_division=0)
    f1 = f1_score(y_test, y_pred, average=None, labels=range(n_classes),
                  zero_division=0)
    support = confusion_matrix(y_test, y_pred, labels=range(n_classes)).sum(axis=1)

    accuracy = float(accuracy_score(y_test, y_pred))
    macro_precision = float(np.mean(precision))
    macro_recall = float(np.mean(recall))
    macro_f1 = float(np.mean(f1))

    metrics: Dict[str, Any] = {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1_score": macro_f1,
        "num_test_samples": int(len(y_test)),
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1_score": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(class_names)
        },
    }

    report: Dict[str, Any] = {
        "classes": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1_score": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(class_names)
        },
        "macro_avg": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1_score": macro_f1,
            "support": int(support.sum()),
        },
        "accuracy": accuracy,
        "num_classes": n_classes,
    }

    cm = confusion_matrix(y_test, y_pred, labels=range(n_classes))
    LOGGER.info(
        "Evaluation -> accuracy=%.4f macro_f1=%.4f (%d test samples)",
        accuracy,
        macro_f1,
        len(y_test),
    )
    return metrics, report, cm


# --------------------------------------------------------------------------- #
# Serialization helpers
# --------------------------------------------------------------------------- #

def _write_json(payload: Dict[str, Any], output_path: Path) -> Path:
    """Write a JSON payload to disk, creating the parent directory."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
    except (OSError, TypeError) as exc:
        raise OSError(f"Failed to write JSON to {out}: {exc}") from exc
    LOGGER.info("Wrote %s", out)
    return out


def save_metrics(metrics: Dict[str, Any], output_path: Optional[Path] = None) -> Path:
    """Persist scalar and per-class metrics as JSON.

    Args:
        metrics: The metrics dict from :func:`evaluate_model`.
        output_path: Destination; defaults to ``config.METRICS_PATH``.

    Returns:
        The path to the written artifact.
    """
    return _write_json(metrics, output_path or config.METRICS_PATH)


def save_classification_report(
    report: Dict[str, Any], output_path: Optional[Path] = None
) -> Path:
    """Persist the structured classification report as JSON.

    Args:
        report: The report dict from :func:`evaluate_model`.
        output_path: Destination; defaults to ``config.CLASSIFICATION_REPORT_PATH``.

    Returns:
        The path to the written artifact.
    """
    return _write_json(report, output_path or config.CLASSIFICATION_REPORT_PATH)


def save_evaluation_summary(
    metrics: Dict[str, Any],
    class_names: List[str],
    extra: Optional[Dict[str, Any]] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """Persist a combined run summary (metrics + model/dataset context).

    Args:
        metrics: The metrics dict from :func:`evaluate_model`.
        class_names: Ordered class label strings.
        extra: Optional additional context (e.g. dataset size).
        output_path: Destination; defaults to ``config.EVALUATION_SUMMARY_PATH``.

    Returns:
        The path to the written artifact.
    """
    summary: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": "RandomForestClassifier",
        "embedding_model": config.MODEL_NAME,
        "embedding_pooling": embedding.POOLING_STRATEGY,
        "test_fraction": config.TEST_SIZE,
        "metrics": metrics,
        "classes": class_names,
    }
    if extra:
        summary["context"] = extra
    return _write_json(summary, output_path or config.EVALUATION_SUMMARY_PATH)


def save_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Optional[Path] = None,
) -> Path:
    """Render and persist the confusion matrix as a PNG heat map.

    Args:
        cm: Raw confusion matrix from :func:`evaluate_model`.
        class_names: Ordered class label strings.
        output_path: Destination; defaults to ``config.CONFUSION_MATRIX_PATH``.

    Returns:
        The path to the written artifact.
    """
    out = Path(output_path) if output_path else config.CONFUSION_MATRIX_PATH
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 1.1), max(5, len(class_names))))
    image = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(image, ax=ax)

    tick_marks = np.arange(len(class_names))
    ax.set(
        xticks=tick_marks,
        yticks=tick_marks,
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix - Random Forest",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                int(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )
    fig.tight_layout()
    try:
        fig.savefig(out, dpi=150, bbox_inches="tight")
    except OSError as exc:
        raise OSError(f"Failed to write confusion matrix to {out}: {exc}") from exc
    finally:
        plt.close(fig)

    LOGGER.info("Wrote confusion matrix PNG to %s", out)
    return out


# --------------------------------------------------------------------------- #
# Pipeline orchestration
# --------------------------------------------------------------------------- #

def main() -> None:
    """End-to-end evaluation entry point."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    started = datetime.now(timezone.utc)

    clf = load_model()
    encoder = load_label_encoder()
    class_names = [str(c) for c in encoder.classes_]
    LOGGER.info("Loaded model with %d classes", len(class_names))

    frame = train_pipeline.load_dataset()
    train_pipeline.validate_dataset(frame)

    try:
        indobert = embedding.load_model()
    except Exception as exc:
        LOGGER.error("Failed to load IndoBERT for evaluation: %s", exc)
        raise

    tokenizer = embedding.load_tokenizer()
    X_test, y_test, y_test_str = prepare_evaluation_data(
        indobert[0], tokenizer, frame, device=indobert[1]
    )

    metrics, report, cm = evaluate_model(clf, X_test, y_test, class_names)

    save_metrics(metrics)
    save_classification_report(report)
    save_evaluation_summary(
        metrics,
        class_names,
        extra={"num_samples": int(len(frame)), "classes": class_names},
    )
    save_confusion_matrix(cm, class_names)

    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    LOGGER.info(
        "Evaluation completed in %.2fs. Artifacts written to %s",
        elapsed,
        config.EVALUATION_DIRECTORY,
    )


if __name__ == "__main__":
    main()