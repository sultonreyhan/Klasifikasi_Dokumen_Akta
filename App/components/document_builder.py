"""Document builder component for AktaSense.

Renders the file uploader and camera input tabs for single prediction
(Blueprint Section 5.2, Stage 0; Blueprint Section 12.1).
"""

from __future__ import annotations

from typing import Optional

import streamlit as st

from App.utils.file_validator import SINGLE_ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB
from App.utils.icons import icon

_UPLOAD_HELP = (
    "Format: PDF, PNG, JPG, JPEG. "
    f"Ukuran maksimum {MAX_FILE_SIZE_MB:.0f} MB."
)

_CAMERA_UNAVAILABLE_MSG = (
    "Kamera tidak tersedia di browser ini. Gunakan tab Upload File."
)


def _render_file_tab() -> Optional[st.runtime.uploaded_file_manager.UploadedFile]:
    """Render the upload-file tab and return the selected file (or None)."""
    uploaded = st.file_uploader(
        label="Upload Dokumen Akta",
        type=sorted(SINGLE_ALLOWED_EXTENSIONS),
        accept_multiple_files=False,
        help=_UPLOAD_HELP,
        key="single_uploader",
    )
    return uploaded


def _render_camera_tab() -> Optional[st.runtime.uploaded_file_manager.UploadedFile]:
    """Render the camera tab with graceful fallback.

    Uses ``st.camera_input`` (Blueprint decision C3). If the widget is
    unsupported by the browser it returns None with no picture; in that
    case an info message directs the user to the upload tab.
    """
    picture = st.camera_input(
        label="Ambil Foto Dokumen",
        key="single_camera",
    )
    if picture is None:
        st.info(_CAMERA_UNAVAILABLE_MSG)
    return picture


def render_document_builder(on_upload_callback=None, on_camera_callback=None):
    """Render the Document Builder (upload + camera tabs).

    Args:
        on_upload_callback: Called with the selected file when the user
            clicks the primary action for an uploaded file.
        on_camera_callback: Called with the captured picture when the user
            clicks the primary action for a camera photo.

    Returns:
        ``(file, source)`` where ``file`` is the selected UploadedFile
        (or None) and ``source`` is ``"upload"`` / ``"camera"`` / None.
    """
    tab_upload, tab_camera = st.tabs(
        [f"{icon('upload', 14)} Upload File", f"{icon('camera', 14)} Kamera"]
    )

    with tab_upload:
        uploaded_file = _render_file_tab()
        if uploaded_file is not None:
            st.caption(f"File: **{uploaded_file.name}**")
            if on_upload_callback is not None:
                on_upload_callback(uploaded_file)
            return uploaded_file, "upload"

    with tab_camera:
        camera_file = _render_camera_tab()
        if camera_file is not None:
            if on_camera_callback is not None:
                on_camera_callback(camera_file)
            return camera_file, "camera"

    return None, None
