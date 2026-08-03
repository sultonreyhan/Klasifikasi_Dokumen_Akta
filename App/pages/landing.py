"""Landing page for AktaSense.

Hero, mode selection, model info card, and how-it-works
(Blueprint Section 5.1). V1.1 design pass: Lucide icons, card layout.
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
from App.utils.icons import icon, icon_markdown
from App.utils.session_helpers import init_session_state

init_session_state()

# ── Hero ───────────────────────────────────────────────────────────────────

_ASSETS_LOGO = _PROJECT_ROOT / "Assets" / "logo.png"
if _ASSETS_LOGO.exists():
    st.image(str(_ASSETS_LOGO), width=72)

st.markdown(
    f'<div class="akta-hero">'
    f'<span class="akta-hero-eyebrow">{icon("sparkles", 13)} '
    f'KLASIFIKASI DOKUMEN DENGAN AI</span>'
    f'<h1>Klasifikasi Dokumen Akta<br/>'
    f'<span class="akta-hero-accent">Notaris &amp; PPAT</span> otomatis</h1>'
    f'<p class="akta-hero-sub">Unggah dokumen akta, dan sistem akan '
    f'mengidentifikasi jenis akta secara otomatis beserta keyakinan dan '
    f'penjelasannya.</p>'
    f'</div>',
    unsafe_allow_html=True,
)

st.divider()

# ── Mode selection ─────────────────────────────────────────────────────────

st.markdown("## Pilih Mode Prediksi")

_PAGES_DIR = Path(__file__).resolve().parent


def _mode_card(icon_name: str, title: str, description: str) -> str:
    return (
        f'<div class="akta-mode-card">'
        f'<span class="akta-mode-icon">{icon(icon_name, 22)}</span>'
        f'<p class="akta-mode-title">{title}</p>'
        f'<p class="akta-mode-desc">{description}</p>'
        f'</div>'
    )


col_single, col_batch = st.columns(2)

with col_single:
    st.markdown(
        _mode_card(
            "file-text",
            "Prediksi Dokumen",
            "Klasifikasikan <b>satu dokumen</b> akta (PDF atau gambar) dan "
            "lihat ringkasan serta analisis hasilnya.",
        ),
        unsafe_allow_html=True,
    )
    if st.button(
        f"{icon_markdown('arrow-right', 15, 'arrow')} Mulai",
        type="primary",
        width="stretch",
        key="nav_single",
    ):
        st.switch_page(_PAGES_DIR / "single_prediction.py")

with col_batch:
    st.markdown(
        _mode_card(
            "folder",
            "Prediksi Batch",
            "Klasifikasikan <b>banyak dokumen PDF</b> sekaligus dan ekspor "
            "hasilnya ke file Excel.",
        ),
        unsafe_allow_html=True,
    )
    if st.button(
        f"{icon_markdown('arrow-right', 15, 'arrow')} Mulai",
        type="primary",
        width="stretch",
        key="nav_batch",
    ):
        st.switch_page(_PAGES_DIR / "batch_prediction.py")

st.divider()

# ── Model info (dynamic) ───────────────────────────────────────────────────

render_model_info(st.session_state.get("model_metadata", {}) or {})

st.divider()

# ── How it works ───────────────────────────────────────────────────────────

st.markdown("## Cara Kerja")

step1, step2, step3 = st.columns(3)

steps = [
    ("1", "Upload Dokumen", "Pilih file PDF atau gambar akta."),
    ("2", "AI Memproses", "Teks diekstraksi dan diklasifikasi oleh model."),
    ("3", "Lihat Hasil", "Dapatkan jenis akta, keyakinan, dan analisis."),
]

for col, (num, title, desc) in zip(
    (step1, step2, step3), steps, strict=True
):
    col.markdown(
        f'<div class="akta-step-card">'
        f'<span class="akta-step-num">{num}</span>'
        f'<p class="akta-step-title">{title}</p>'
        f'<p class="akta-step-desc">{desc}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.caption(
    "AktaSense adalah alat bantu indikatif. Verifikasi dokumen secara "
    "manual tetap diperlukan."
)
