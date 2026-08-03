"""Label mapping utilities for AktaSense.

Converts raw model label strings (e.g. "ajb") to human-readable
display names and taxonomy group strings as defined in the
Application Blueprint (Section 11).

The mapping table is the source of truth for all UI label rendering.
If a label is not present in the map, a graceful fallback is applied
instead of raising an exception.
"""

from __future__ import annotations

import logging
from typing import List

LOGGER = logging.getLogger(__name__)

# ── Mapping table (Blueprint Section 11.1) ─────────────────────────────────
# Format: label → (display_name, taxonomy_group)

_MAP: dict[str, tuple[str, str]] = {
    "ajb":            ("Akta Jual Beli",                    "Peralihan Hak"),
    "akta_pendirian": ("Akta Pendirian",                    "Pendirian & Perubahan Badan Hukum"),
    "akta_perubahan": ("Akta Perubahan",                    "Pendirian & Perubahan Badan Hukum"),
    "hibah":          ("Akta Hibah",                        "Peralihan Hak"),
    "pernyataan":     ("Akta Pernyataan / Kuasa",           "Perjanjian & Pernyataan"),
    "pkr":            ("Pernyataan Keputusan Rapat",        "Organisasi & Keputusan"),
    "ppjb":           ("Perjanjian Pengikatan Jual Beli",   "Peralihan Hak"),
    "waris":          ("Akta Keterangan Waris",             "Waris & Keluarga"),
}


# ── Internal fallback ──────────────────────────────────────────────────────

def _fallback_display(label: str) -> str:
    """Convert an unknown label to a readable title-case string."""
    return label.replace("_", " ").title()


# ── Public API ─────────────────────────────────────────────────────────────

def get_display_name(label: str) -> str:
    """Return the human-readable display name for a model label.

    Args:
        label: Raw label string from the model (e.g. ``"ajb"``).

    Returns:
        Display name (e.g. ``"Akta Jual Beli"``).
        Falls back to a title-cased version of the label if not found.
    """
    if label in _MAP:
        return _MAP[label][0]
    LOGGER.warning("Label '%s' not in LABEL_DISPLAY_MAP; using fallback.", label)
    return _fallback_display(label)


def get_taxonomy(label: str) -> str:
    """Return the taxonomy group for a model label.

    Args:
        label: Raw label string from the model.

    Returns:
        Taxonomy group string (e.g. ``"Peralihan Hak"``).
        Falls back to ``"Tidak Diketahui"`` if the label is not found.
    """
    if label in _MAP:
        return _MAP[label][1]
    LOGGER.warning("Label '%s' not in LABEL_DISPLAY_MAP; taxonomy unknown.", label)
    return "Tidak Diketahui"


def get_all_classes_with_display(label_list: List[str]) -> List[dict]:
    """Return a list of dicts for every label in label_list.

    Each dict contains ``label``, ``display_name``, and ``taxonomy``.
    Used by model_info_card and probability distribution charts.

    Args:
        label_list: Ordered list of raw label strings from the model.

    Returns:
        List of ``{label, display_name, taxonomy}`` dicts in the same order.
    """
    return [
        {
            "label":        lbl,
            "display_name": get_display_name(lbl),
            "taxonomy":     get_taxonomy(lbl),
        }
        for lbl in label_list
    ]


# ── Self-check ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_labels = ["ajb", "akta_pendirian", "waris", "unknown_label"]
    for lbl in test_labels:
        print(
            "{!r:20} -> {!r:40} | {!r}".format(
                lbl, get_display_name(lbl), get_taxonomy(lbl)
            )
        )
