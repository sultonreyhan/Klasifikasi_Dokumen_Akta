"""Central configuration for the AktaSense ML pipeline.

All tunable / environment-specific values live here so that
``embedding.py`` (and future modules) stay free of hard-coded
configuration.
"""

from pathlib import Path
from typing import Optional


def _resolve_device() -> str:
    """Return the best available compute device as a string."""
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

# Root of the project (parent of this Pipeline package).
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Absolute path to the training dataset. Treated as READ-ONLY input.
DATASET_DIRECTORY: Path = PROJECT_ROOT / "Dataset"
TRAINING_DATASET_PATH: Path = DATASET_DIRECTORY / "Training_Dataset.csv"

# Directory where generated artifacts (feature matrices, model files) are
# written. Created on-demand by the training stage.
OUTPUT_DIRECTORY: Path = PROJECT_ROOT / "Models"

# Alias for the model-artifact directory (used by downstream prediction/app).
MODEL_DIRECTORY: Path = OUTPUT_DIRECTORY

# Individual artifact paths within the output directory.
MODEL_ARTIFACT_PATH: Path = OUTPUT_DIRECTORY / "random_forest.pkl"
LABEL_ENCODER_PATH: Path = OUTPUT_DIRECTORY / "label_encoder.pkl"
TRAINING_METADATA_PATH: Path = OUTPUT_DIRECTORY / "training_metadata.json"

# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

# Directory where evaluation artifacts are written.
EVALUATION_DIRECTORY: Path = PROJECT_ROOT / "Evaluation"

# Individual evaluation artifact paths.
METRICS_PATH: Path = EVALUATION_DIRECTORY / "metrics.json"
CLASSIFICATION_REPORT_PATH: Path = EVALUATION_DIRECTORY / "classification_report.json"
EVALUATION_SUMMARY_PATH: Path = EVALUATION_DIRECTORY / "evaluation_summary.json"
CONFUSION_MATRIX_PATH: Path = EVALUATION_DIRECTORY / "confusion_matrix.png"

# --------------------------------------------------------------------------- #
# Exports / explainability
# --------------------------------------------------------------------------- #

# Directory where visualization / explainability artifacts are written.
EXPORTS_DIRECTORY: Path = PROJECT_ROOT / "Exports"

# Individual explainability artifact paths.
SHAP_SUMMARY_PATH: Path = EXPORTS_DIRECTORY / "shap_summary.png"
SHAP_BAR_PATH: Path = EXPORTS_DIRECTORY / "shap_bar.png"
FORCE_PLOT_PATH: Path = EXPORTS_DIRECTORY / "force_plot.html"
WATERFALL_EXAMPLE_PATH: Path = EXPORTS_DIRECTORY / "waterfall_example.png"
SHAP_JSON_PATH: Path = EXPORTS_DIRECTORY / "shap_summary.json"

# Feature count used for SHAP displays (top-k aggregated features).
SHAP_TOP_FEATURES: int = 15

# --------------------------------------------------------------------------- #
# Model / embedding
# --------------------------------------------------------------------------- #

# Official Indonesian BERT model identifier (Hugging Face hub).
# ``base-p1`` is the canonical IndoBERT checkpoint supported by the hub.
MODEL_NAME: str = "indobenchmark/indobert-base-p1"

# Maximum sequence length used during tokenization (in tokens).
# IndoBERT supports up to 512 tokens per forward pass.
MAX_LENGTH: int = 512

# Hardware device: auto-detected at runtime (see ``_resolve_device``).
DEVICE: str = _resolve_device()

# Number of documents processed per forward pass.
BATCH_SIZE: int = 8

# Random seed for reproducible tokenization / sampling.
RANDOM_SEED: int = 42

# --------------------------------------------------------------------------- #
# Random Forest
# --------------------------------------------------------------------------- #

# Validation hold-out fraction used by the train/test split.
TEST_SIZE: float = 0.2

# Number of trees in the forest.
N_ESTIMATORS: int = 200

# Maximum depth of the trees (None = unlimited).
MAX_DEPTH: Optional[int] = None

# Minimum number of samples required to split an internal node.
MIN_SAMPLES_SPLIT: int = 2

# Minimum number of samples required at each leaf node.
MIN_SAMPLES_LEAF: int = 1

# Class-weight strategy used to counter class imbalance ('balanced_subsample').
CLASS_WEIGHT: str = "balanced_subsample"

# Number of parallel jobs used during fitting (-1 = use all cores).
N_JOBS: int = -1

# --------------------------------------------------------------------------- #
# Dataset schema
# --------------------------------------------------------------------------- #

# Expected CSV columns produced by the Dataset pipeline.
COLUMN_TEXT: str = "cleaned_text"
COLUMN_LABEL: str = "label"
COLUMN_TAXONOMY: str = "taxonomy"
COLUMN_FILENAME: str = "filename"