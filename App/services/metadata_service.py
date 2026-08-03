"""Model metadata service for AktaSense.

Reads ``Models/training_metadata.json`` (the locked training artifact)
and exposes the data as a plain dict for UI rendering.

The label list is always read from the JSON file — it is NEVER hardcoded
in the App layer (confirmed decision C1).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st

from Pipeline import config as pipeline_config

LOGGER = logging.getLogger(__name__)

# ── Path resolution ────────────────────────────────────────────────────────

# Default artifact path comes from the locked Pipeline config.
DEFAULT_METADATA_PATH: Path = pipeline_config.TRAINING_METADATA_PATH


# ── Cached loader ──────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _read_metadata_json(path_str: str) -> Optional[Dict[str, Any]]:
    """Read and parse the training metadata JSON (cached per session).

    Args:
        path_str: String path to ``training_metadata.json``.

    Returns:
        The parsed dict, or ``None`` if the file is missing or malformed.
    """
    path = Path(path_str)
    if not path.exists():
        LOGGER.warning("Training metadata not found at %s", path)
        return None

    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.error("Failed to parse training metadata from %s: %s", path, exc)
        return None


# ── Public API ─────────────────────────────────────────────────────────────

def load_model_metadata(
    path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Load the training metadata as a JSON-serializable dict.

    Args:
        path: Optional override path to the metadata JSON file.
              Defaults to ``Pipeline.config.TRAINING_METADATA_PATH``.

    Returns:
        The metadata dict (e.g. ``label_list``, ``num_classes``,
        ``oob_score``, ``embedding_model``, ``timestamp_utc``).
        Returns an empty dict if the file is unavailable.
    """
    target = Path(path) if path else DEFAULT_METADATA_PATH
    metadata = _read_metadata_json(str(target))

    if metadata is None:
        LOGGER.warning("Returning empty metadata dict.")
        return {}

    # Normalise keys that UI depends on, guarding against older files.
    metadata.setdefault("label_list", [])
    metadata.setdefault("num_classes", len(metadata.get("label_list", [])))
    metadata.setdefault("embedding_model", "indobenchmark/indobert-base-p1")
    metadata.setdefault("oob_score", None)
    metadata.setdefault("timestamp_utc", None)

    return metadata


# ── Self-check ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    data = load_model_metadata()
    print("num_classes:", data.get("num_classes"))
    print("label_list:", data.get("label_list"))
    print("oob_score:", data.get("oob_score"))
