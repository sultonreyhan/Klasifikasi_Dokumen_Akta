"""AktaSense — Application Entry Point.

Defines the three-page navigation shell and initialises session state
and pipeline resources on startup.

Run with:
    streamlit run App/app.py
(from the project root so that Pipeline/ is on the Python path)
"""

import sys
from pathlib import Path

# Ensure the project root (parent of App/) is on sys.path so that
# `import Pipeline` works from any page or service module.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from App.utils.session_helpers import init_session_state, inject_custom_font
from App.services.metadata_service import load_model_metadata
from App.services.prediction_service import load_pipeline_resources

# ── Page configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="AktaSense",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Font & custom CSS ──────────────────────────────────────────────────────
inject_custom_font()

# ── Session state ──────────────────────────────────────────────────────────
init_session_state()

# ── Load model metadata once (cached) ─────────────────────────────────────
if st.session_state["model_metadata"] is None:
    try:
        st.session_state["model_metadata"] = load_model_metadata()
    except Exception as exc:
        st.session_state["model_metadata"] = None
        # Non-fatal: pages will show an appropriate warning if None.

# ── Sidebar: branding + model status ──────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 AktaSense")
    st.caption("Klasifikasi Dokumen Akta Notaris & PPAT")
    st.divider()

    meta = st.session_state.get("model_metadata")
    if meta:
        n = meta.get("num_classes", "?")
        st.markdown(f"**Model:** v1.0 &nbsp;|&nbsp; **{n} kelas aktif**")
    else:
        st.warning("Metadata model tidak tersedia.")

    st.divider()

    # ── Attempt to pre-load pipeline resources ─────────────────────────
    if not st.session_state["pipeline_ready"]:
        with st.spinner("Memuat model AI..."):
            try:
                load_pipeline_resources()
                st.session_state["pipeline_ready"] = True
            except Exception as exc:
                st.session_state["pipeline_ready"] = False
                st.error(
                    "Model AI tidak ditemukan. "
                    "Pastikan `Models/random_forest.pkl` dan "
                    "`Models/label_encoder.pkl` tersedia.\n\n"
                    "Jalankan `Pipeline/train.py` untuk melatih model terlebih dahulu."
                )

    if st.session_state["pipeline_ready"]:
        st.success("Model dimuat ✅")

# ── Navigation ─────────────────────────────────────────────────────────────
# Page paths are resolved from __file__ (absolute) so the app works
# regardless of the current working directory.
_PAGES_DIR = Path(__file__).resolve().parent / "pages"

landing_page = st.Page(
    _PAGES_DIR / "landing.py",
    title="Beranda",
    icon="🏠",
    default=True,
)
single_page = st.Page(
    _PAGES_DIR / "single_prediction.py",
    title="Prediksi Dokumen",
    icon="📄",
)
batch_page = st.Page(
    _PAGES_DIR / "batch_prediction.py",
    title="Prediksi Batch",
    icon="📂",
)

pg = st.navigation([landing_page, single_page, batch_page])
pg.run()
