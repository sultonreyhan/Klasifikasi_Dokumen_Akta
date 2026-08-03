"""Session state helpers for AktaSense.

Responsibilities:
- Initialise all session_state keys (idempotent — only set if absent).
- Provide reset helpers for each page's state.
- Inject the Google Fonts stylesheet (Plus Jakarta Sans + JetBrains Mono).
"""

import streamlit as st

from pathlib import Path

# ── Font injection ─────────────────────────────────────────────────────────

_FONT_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&family=JetBrains+Mono:wght@400&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* OCR text area uses monospace */
textarea[aria-label="Teks yang berhasil diekstraksi"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
}
</style>
"""

# Assets directory lives at the project root (parent of App/).
_ASSETS_STYLE_PATH = (
    Path(__file__).resolve().parent.parent.parent / "Assets" / "style.css"
)


def _assets_css() -> str:
    """Read Assets/style.css if present; return empty string otherwise."""
    try:
        if _ASSETS_STYLE_PATH.exists():
            return f"<style>\n{_ASSETS_STYLE_PATH.read_text(encoding='utf-8')}\n</style>"
    except OSError:
        pass
    return ""


def inject_custom_font() -> None:
    """Inject Plus Jakarta Sans, JetBrains Mono, and the app stylesheet."""
    st.markdown(_FONT_CSS, unsafe_allow_html=True)
    st.markdown(_assets_css(), unsafe_allow_html=True)


# ── Default values ─────────────────────────────────────────────────────────

_APP_DEFAULTS: dict = {
    "model_metadata": None,
    "pipeline_ready": False,
}

_SINGLE_DEFAULTS: dict = {
    "single_stage": "idle",          # idle|validating|extracting|predicting|result
    "single_uploaded_file": None,
    "single_file_source": None,      # "upload" | "camera" | None
    "single_raw_text": "",
    "single_page_count": 0,
    "single_char_count": 0,
    "single_ocr_quality": None,      # "good" | "fair" | "poor" | None
    "single_result": None,
    "single_processing_time": 0.0,
    "single_error": None,
}

_BATCH_DEFAULTS: dict = {
    "batch_stage": "idle",           # idle|validating|processing|result
    "batch_files": [],
    "batch_validation": [],          # List[dict]: {filename, valid, reason, size_mb}
    "batch_results": [],             # List[BatchPredictionRecord]
    "batch_progress": 0,
    "batch_total": 0,
    "batch_current_file": "",
    "batch_error": None,
}


# ── Public API ─────────────────────────────────────────────────────────────

def init_session_state() -> None:
    """Initialise all session_state keys with defaults (idempotent).

    Safe to call on every page load — only sets keys that do not exist yet.
    """
    for key, default in {**_APP_DEFAULTS, **_SINGLE_DEFAULTS, **_BATCH_DEFAULTS}.items():
        if key not in st.session_state:
            st.session_state[key] = default


def reset_single_state() -> None:
    """Reset all single_* keys to their initial values.

    Call when the user clicks 'Prediksi Dokumen Lain'.
    """
    for key, default in _SINGLE_DEFAULTS.items():
        st.session_state[key] = default


def reset_batch_state() -> None:
    """Reset all batch_* keys to their initial values.

    Call when the user clicks 'Prediksi Ulang'.
    """
    for key, default in _BATCH_DEFAULTS.items():
        st.session_state[key] = default
