"""Landing page for AktaSense.

Hero, mode selection, model info card, and how-it-works
(Blueprint Section 5.1).
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so App/Pipeline imports work
# even when this file is executed as a standalone page script.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from App.components.model_info_card import render_model_info
from App.utils.session_helpers import init_session_state

init_session_state()

# ── Hero ───────────────────────────────────────────────────────────────────

_ASSETS_LOGO = _PROJECT_ROOT / "Assets" / "logo.png"
if _ASSETS_LOGO.exists():
    st.image(str(_ASSETS_LOGO), width=96)

st.markdown("# 📄 **AktaSense**")
st.markdown(
    "**Klasifikasi dokumen akta Notaris dan PPAT menggunakan kecerdasan "
    "buatan.**\n\n"
    "Unggah dokumen akta, dan sistem akan mengidentifikasi jenis akta "
    "secara otomatis beserta penjelasannya."
)

st.divider()

# ── Mode selection ─────────────────────────────────────────────────────────

st.markdown("## Pilih Mode Prediksi")

col_single, col_batch = st.columns(2)

_PAGES_DIR = Path(__file__).resolve().parent

with col_single:
    st.markdown(
        "### 📄 Prediksi Dokumen\n\n"
        "Klasifikasikan **satu dokumen** akta (PDF atau gambar) dan lihat "
        "ringkasan serta analisis hasilnya."
    )
    if st.button("Mulai →", type="primary", width="stretch",
                 key="nav_single"):
        st.switch_page(_PAGES_DIR / "single_prediction.py")

with col_batch:
    st.markdown(
        "### 📂 Prediksi Batch\n\n"
        "Klasifikasikan **banyak dokumen PDF** sekaligus dan ekspor "
        "hasilnya ke file Excel."
    )
    if st.button("Mulai →", type="primary", width="stretch",
                 key="nav_batch"):
        st.switch_page(_PAGES_DIR / "batch_prediction.py")

st.divider()

# ── Model info (dynamic) ───────────────────────────────────────────────────

render_model_info(st.session_state.get("model_metadata", {}) or {})

st.divider()

# ── How it works ───────────────────────────────────────────────────────────

st.markdown("## Cara Kerja")

step1, step2, step3 = st.columns(3)
step1.markdown("### 1️⃣ Upload Dokumen\n\nPilih file PDF atau gambar akta.")
step2.markdown("### 2️⃣ AI Memproses\n\nTeks diekstraksi dan diklasifikasi "
               "oleh model.")
step3.markdown("### 3️⃣ Lihat Hasil\n\nDapatkan jenis akta, keyakinan, dan "
               "analisis.")

st.caption(
    "AktaSense adalah alat bantu indikatif. Verifikasi dokumen secara "
    "manual tetap diperlukan."
)
