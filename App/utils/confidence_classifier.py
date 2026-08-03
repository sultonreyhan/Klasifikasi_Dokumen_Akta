"""Confidence level classifier for AktaSense.

Maps a raw probability confidence score (0.0–1.0) to a three-tier
human-readable label as defined in the Application Blueprint
(Section 10.2, Confidence Level Color Mapping).

Thresholds:
    TINGGI  — score ≥ 0.75
    SEDANG  — 0.50 ≤ score < 0.75
    RENDAH  — score < 0.50
"""

from __future__ import annotations

# ── Threshold constants ────────────────────────────────────────────────────

_HIGH_THRESHOLD: float = 0.75
_MID_THRESHOLD: float = 0.50

# ── Color mapping (Blueprint Section 10.2) ─────────────────────────────────
# Maps label → hex color string for UI rendering.

CONFIDENCE_COLORS: dict[str, str] = {
    "TINGGI": "#16A34A",   # Success Green
    "SEDANG": "#F59E0B",   # Warning Amber
    "RENDAH": "#DC2626",   # Error Red
}


# ── Public API ─────────────────────────────────────────────────────────────

def classify(score: float) -> str:
    """Convert a confidence score to a human-readable level label.

    Args:
        score: Probability value in the range [0.0, 1.0].

    Returns:
        ``"TINGGI"``, ``"SEDANG"``, or ``"RENDAH"``.
    """
    if score >= _HIGH_THRESHOLD:
        return "TINGGI"
    if score >= _MID_THRESHOLD:
        return "SEDANG"
    return "RENDAH"


def get_color(score: float) -> str:
    """Return the hex color associated with a confidence score.

    Args:
        score: Probability value in the range [0.0, 1.0].

    Returns:
        Hex color string (e.g. ``"#16A34A"``).
    """
    return CONFIDENCE_COLORS[classify(score)]


# ── Self-check ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for s in [0.90, 0.75, 0.74, 0.50, 0.49, 0.10]:
        print(f"{s:.2f} → {classify(s):6}  {get_color(s)}")
