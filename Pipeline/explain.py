"""SHAP explainability pipeline for AktaSense.

Generates model-agnostic feature-importance explanations for the trained
:class:`~sklearn.ensemble.RandomForestClassifier`. The tree-based :class:`shap.TreeExplainer`
is used because it is exact and fast for Random Forests.

Features are produced with the same IndoBERT + mean-pooling pipeline reused
from :mod:`Pipeline.embedding` and :mod:`Pipeline.train`; no preprocessing
logic is duplicated here.

Artifacts (written under ``config.EXPORTS_DIRECTORY``):

- ``shap_summary.png``      - SHAP beeswarm summary plot
- ``shap_bar.png``          - mean(|SHAP|) bar plot
- ``force_plot.html``       - interactive force plot (plotly)
- ``waterfall_example.png`` - waterfall explanation of the first sample
- ``shap_summary.json``     - machine-readable importance summary

Public API
----------
- :func:`load_model`
- :func:`load_label_encoder`
- :func:`prepare_explanation_data`
- :func:`build_explainer`
- :func:`compute_shap_values`
- :func:`save_summary_plot`
- :func:`save_bar_plot`
- :func:`save_force_plot`
- :func:`save_waterfall_example`
- :func:`build_json_summary`
- :func:`save_shap_summary`
- :func:`main`
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import numpy as np
import shap
import pandas as pd

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
# Data preparation
# --------------------------------------------------------------------------- #

def prepare_explanation_data(
    model,
    tokenizer,
    frame: pd.DataFrame,
    device: str = config.DEVICE,
    show_progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Build the feature matrix and true labels for SHAP computation.

    Reuses the shared embedding pipeline (:func:`train_pipeline.prepare_features`)
    on the full evaluation dataset.

    Args:
        model: The IndoBERT model (from ``embedding.load_model``).
        tokenizer: The IndoBERT tokenizer.
        frame: Loaded evaluation dataset.
        device: Device hosting the IndoBERT model.
        show_progress: Whether to show the embedding progress bar.

    Returns:
        ``(X, y, class_names)`` where ``X`` is the ``(n, 768)`` feature matrix,
        ``y`` the integer labels and ``class_names`` the ordered labels.
    """
    X = train_pipeline.prepare_features(model, tokenizer, frame, device=device,
                                        show_progress=show_progress)
    y, encoder = train_pipeline.prepare_labels(frame)
    class_names = [str(c) for c in encoder.classes_]
    LOGGER.info("Explanation data ready: X=%s", X.shape)
    return X, y, class_names


# --------------------------------------------------------------------------- #
# SHAP computation
# --------------------------------------------------------------------------- #

def build_explainer(clf):
    """Initialise the SHAP explainer best suited to the Random Forest.

    Args:
        clf: The fitted Random Forest classifier.

    Returns:
        A :class:`shap.TreeExplainer` bound to the classifier.

    Raises:
        ValueError: If the classifier type is not tree-based / unsupported.
    """
    try:
        explainer = shap.TreeExplainer(clf)
    except Exception as exc:
        raise ValueError(f"TreeExplainer could not be built for {type(clf).__name__}: {exc}") from exc
    LOGGER.info("Built %s", type(explainer).__name__)
    return explainer


def compute_shap_values(
    explainer,
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute SHAP values for the evaluation dataset.

    For a multi-output Random Forest the SHAP values are returned per class as
    a ``(n_samples, n_features, n_classes)`` array. A ``(n_samples, n_features)``
    view collapsed over classes (mean absolute) is also returned for display.

    Args:
        explainer: The fitted SHAP explainer.
        X: Feature matrix.
        feature_names: Optional column/feature names (kept for API symmetry).

    Returns:
        ``(shap_values, collapsed)``. ``shap_values`` has the native
        per-class shape while ``collapsed`` is the ``(n, features)`` mean
        absolute contribution over classes.

    Raises:
        ValueError: If SHAP computation fails.
    """
    LOGGER.info("Computing SHAP values on %d samples", X.shape[0])
    try:
        raw = explainer.shap_values(X, check_additivity=False)
    except Exception as exc:
        raise ValueError(f"SHAP value computation failed: {exc}") from exc

    if isinstance(raw, list):
        # per-class list of (n, f) arrays -> (n, f, c)
        values = np.asarray(raw).transpose(1, 2, 0)
    else:
        values = np.asarray(raw)

    if values.ndim == 2:
        # single-output model -> promote to (n, f, 1)
        values = values[..., np.newaxis]

    collapsed = np.abs(values).mean(axis=2)
    LOGGER.info(
        "SHAP values shape=%s collapsed=%s", values.shape, collapsed.shape
    )
    return values, collapsed


# --------------------------------------------------------------------------- #
# Visualizations
# --------------------------------------------------------------------------- #

def _make_feature_names(X: np.ndarray) -> List[str]:
    """Return ``dim_0 ... dim_{f-1}`` names for a bare numpy matrix."""
    return [f"dim_{i}" for i in range(X.shape[1])]


def save_summary_plot(
    shap_values: np.ndarray,
    X: np.ndarray,
    output_path: Optional[Path] = None,
    max_display: int = config.SHAP_TOP_FEATURES,
) -> Path:
    """Render the SHAP beeswarm summary plot.

    Args:
        shap_values: Aggregated SHAP values.
        X: Feature matrix used for feature-value coloring.
        output_path: Destination; defaults to ``config.SHAP_SUMMARY_PATH``.
        max_display: Maximum number of features displayed.

    Returns:
        The path to the written artifact.
    """
    out = Path(output_path) if output_path else config.SHAP_SUMMARY_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        shap.summary_plot(
            shap_values,
            X,
            feature_names=_make_feature_names(X),
            max_display=max_display,
            show=False,
        )
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        plt.close("all")
        raise OSError(f"Failed to render SHAP summary plot: {exc}") from exc
    LOGGER.info("Wrote %s", out)
    return out


def save_bar_plot(
    shap_values: np.ndarray,
    X: np.ndarray,
    output_path: Optional[Path] = None,
    max_display: int = config.SHAP_TOP_FEATURES,
) -> Path:
    """Render the mean-|SHAP| feature-importance bar plot.

    Args:
        shap_values: Aggregated SHAP values.
        X: Feature matrix.
        output_path: Destination; defaults to ``config.SHAP_BAR_PATH``.
        max_display: Maximum number of features displayed.

    Returns:
        The path to the written artifact.
    """
    out = Path(output_path) if output_path else config.SHAP_BAR_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        shap.summary_plot(
            shap_values,
            X,
            feature_names=_make_feature_names(X),
            max_display=max_display,
            plot_type="bar",
            show=False,
        )
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        plt.close("all")
        raise OSError(f"Failed to render SHAP bar plot: {exc}") from exc
    LOGGER.info("Wrote %s", out)
    return out


def save_force_plot(
    explainer,
    shap_values: np.ndarray,
    X: np.ndarray,
    output_path: Optional[Path] = None,
) -> Path:
    """Export an interactive force plot as self-contained HTML.

    The plotly force plot is saved without CDN dependencies so it works offline.
    For multiclass forests the first class slice is displayed.

    Args:
        explainer: The fitted SHAP explainer.
        shap_values: SHAP values (native per-class shape supported).
        X: Feature matrix.
        output_path: Destination; defaults to ``config.FORCE_PLOT_PATH``.

    Returns:
        The path to the written artifact.
    """
    out = Path(output_path) if output_path else config.FORCE_PLOT_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        values = np.asarray(shap_values)
        if values.ndim == 3:
            base_value = np.asarray(explainer.expected_value)[0]
            values_2d = values[..., 0]
        else:
            base_value = explainer.expected_value
            values_2d = values
        html = shap.force_plot(
            base_value,
            values_2d,
            X,
            feature_names=_make_feature_names(X),
            matplotlib=False,
            show=False,
        )
        with open(out, "w", encoding="utf-8") as handle:
            shap.save_html(handle, html)
    except Exception as exc:
        raise OSError(f"Failed to render SHAP force plot: {exc}") from exc
    LOGGER.info("Wrote %s", out)
    return out


def save_waterfall_example(
    explainer,
    shap_values: np.ndarray,
    X: np.ndarray,
    sample_index: int = 0,
    output_path: Optional[Path] = None,
) -> Path:
    """Render the waterfall explanation for a single evaluation sample.

    Args:
        explainer: The fitted SHAP explainer.
        shap_values: SHAP values (native per-class shape supported).
        X: Feature matrix.
        sample_index: Row index of the document to explain.
        output_path: Destination; defaults to ``config.WATERFALL_EXAMPLE_PATH``.

    Returns:
        The path to the written artifact.

    Raises:
        IndexError: If ``sample_index`` is out of bounds.
    """
    if sample_index >= X.shape[0]:
        raise IndexError(f"sample_index {sample_index} out of range for {X.shape[0]} rows")

    out = Path(output_path) if output_path else config.WATERFALL_EXAMPLE_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        values = np.asarray(shap_values)
        if values.ndim == 3:
            base_value = np.asarray(explainer.expected_value)[0]
            sample_values = values[sample_index, :, 0]
        else:
            base_value = explainer.expected_value
            sample_values = values[sample_index]
        explanation = shap.Explanation(
            values=sample_values,
            base_values=base_value,
            data=X[sample_index],
            feature_names=_make_feature_names(X),
        )
        shap.plots.waterfall(explanation, max_display=config.SHAP_TOP_FEATURES, show=False)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        plt.close("all")
        raise OSError(f"Failed to render SHAP waterfall plot: {exc}") from exc
    LOGGER.info("Wrote %s", out)
    return out


# --------------------------------------------------------------------------- #
# JSON summary
# --------------------------------------------------------------------------- #

def build_json_summary(
    shap_values: np.ndarray,
    X: np.ndarray,
    model,
    class_names: List[str],
) -> Dict[str, Any]:
    """Build a machine-readable summary of SHAP importance.

    Args:
        shap_values: Aggregated SHAP values.
        X: Feature matrix.
        model: The fitted Random Forest classifier.
        class_names: Ordered class label strings.

    Returns:
        A JSON-serializable dictionary with the top features and the most
        positive / negative per-sample contributors.
    """
    mean_abs = np.abs(shap_values).mean(axis=0) if shap_values.ndim == 2 else np.abs(shap_values)
    order = np.argsort(mean_abs)[::-1]

    top_features = [
        {
            "rank": int(rank + 1),
            "feature": f"dim_{int(i)}",
            "index": int(i),
            "mean_abs_shap": float(mean_abs[i]),
        }
        for rank, i in enumerate(order[: config.SHAP_TOP_FEATURES])
    ]

    positive_contrib = [
        {
            "sample": int(idx),
            "feature": f"dim_{int(i)}",
            "index": int(i),
            "shap_value": float(shap_values[idx, i]),
        }
        for idx, i in zip(*np.unravel_index(np.argsort(shap_values.ravel())[-5:], shap_values.shape))
    ][::-1]

    negative_contrib = [
        {
            "sample": int(idx),
            "feature": f"dim_{int(i)}",
            "index": int(i),
            "shap_value": float(shap_values[idx, i]),
        }
        for idx, i in zip(*np.unravel_index(np.argsort(shap_values.ravel())[:5], shap_values.shape))
    ]

    summary: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_type": type(model).__name__,
        "explainer": "shap.TreeExplainer",
        "embedding_model": config.MODEL_NAME,
        "embedding_pooling": embedding.POOLING_STRATEGY,
        "embedding_dimension": int(X.shape[1]),
        "num_evaluated_samples": int(X.shape[0]),
        "num_classes": len(class_names),
        "class_labels": class_names,
        "top_features": top_features,
        "top_positive_contributors": positive_contrib,
        "top_negative_contributors": negative_contrib,
    }
    LOGGER.info("Built SHAP JSON summary (top features=%d)", len(top_features))
    return summary


def save_shap_summary(
    summary: Dict[str, Any],
    output_path: Optional[Path] = None,
) -> Path:
    """Persist the SHAP summary JSON.

    Args:
        summary: The summary dict from :func:`build_json_summary`.
        output_path: Destination; defaults to ``config.SHAP_JSON_PATH``.

    Returns:
        The path to the written artifact.
    """
    out = Path(output_path) if output_path else config.SHAP_JSON_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
    except (OSError, TypeError) as exc:
        raise OSError(f"Failed to write SHAP summary to {out}: {exc}") from exc
    LOGGER.info("Wrote %s", out)
    return out


# --------------------------------------------------------------------------- #
# Pipeline orchestration
# --------------------------------------------------------------------------- #

def main() -> None:
    """End-to-end SHAP explainability entry point."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    started = datetime.now(timezone.utc)

    clf = load_model()
    encoder = load_label_encoder()
    class_names = [str(c) for c in encoder.classes_]
    LOGGER.info("Loaded Random Forest with %d classes", len(class_names))

    frame = train_pipeline.load_dataset()
    train_pipeline.validate_dataset(frame)

    indobert = embedding.load_model()
    tokenizer = embedding.load_tokenizer()
    X, y, class_names = prepare_explanation_data(
        indobert[0], tokenizer, frame, device=indobert[1]
    )

    explainer = build_explainer(clf)
    shap_values, collapsed = compute_shap_values(explainer, X)

    save_summary_plot(collapsed, X)
    save_bar_plot(collapsed, X)
    save_force_plot(explainer, shap_values, X)
    save_waterfall_example(explainer, shap_values, X, sample_index=0)

    summary = build_json_summary(collapsed, X, clf, class_names)
    save_shap_summary(summary)

    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    LOGGER.info(
        "SHAP explainability completed in %.2fs. Artifacts written to %s",
        elapsed,
        config.EXPORTS_DIRECTORY,
    )


if __name__ == "__main__":
    main()