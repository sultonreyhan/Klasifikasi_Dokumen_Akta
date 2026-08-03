"""File validation utilities for AktaSense.

Validates uploaded files before any processing begins
(Blueprint Section 16, Layer 1).

Rules (locked Blueprint):
- Single prediction accepts: PDF, PNG, JPG, JPEG.
- Batch prediction accepts: PDF only.
- Maximum file size: 10 MB per file.
- Empty files are rejected.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from streamlit.runtime.uploaded_file_manager import UploadedFile

# ── Constants ──────────────────────────────────────────────────────────────

SINGLE_ALLOWED_EXTENSIONS: set[str] = {"pdf", "png", "jpg", "jpeg"}
BATCH_ALLOWED_EXTENSIONS: set[str] = {"pdf"}

MAX_FILE_SIZE_MB: float = 10.0
MAX_FILE_SIZE_BYTES: int = int(MAX_FILE_SIZE_MB * 1024 * 1024)


# ── Internal helpers ───────────────────────────────────────────────────────

def _extension_of(filename: str) -> str:
    """Return the lowercase extension (without dot) of a filename."""
    if not filename:
        return ""
    if "." not in filename:
        return ""
    return filename.rsplit(".", 1)[-1].lower()


def _size_of(uploaded_file: UploadedFile) -> Optional[int]:
    """Return the byte size of the uploaded file, or None if unknown.

    ``UploadedFile`` exposes ``.size`` when available; otherwise the
    length of its buffered bytes is used as a fallback.
    """
    size = getattr(uploaded_file, "size", None)
    if size is not None:
        try:
            return int(size)
        except (TypeError, ValueError):
            pass
    try:
        uploaded_file.seek(0)
        data = uploaded_file.read()
        uploaded_file.seek(0)
        return len(data)
    except Exception:
        return None


# ── Public API ─────────────────────────────────────────────────────────────

def validate_single(
    uploaded_file: UploadedFile,
) -> Tuple[bool, Optional[str]]:
    """Validate a single document file.

    Args:
        uploaded_file: Streamlit UploadedFile instance.

    Returns:
        ``(valid, error)`` where ``error`` is a human-readable message
        in Indonesian, or ``None`` when the file is valid.
    """
    if uploaded_file is None:
        return False, "Tidak ada file yang dipilih."

    filename = getattr(uploaded_file, "name", "") or ""
    ext = _extension_of(filename)

    if not filename:
        return False, "Nama file tidak valid."

    if ext not in SINGLE_ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(SINGLE_ALLOWED_EXTENSIONS)).upper()
        return False, (
            f"Format file tidak didukung. Gunakan {allowed}."
        )

    size = _size_of(uploaded_file)
    if size is not None and size <= 0:
        return False, "File tidak memiliki konten atau kosong."

    if size is not None and size > MAX_FILE_SIZE_BYTES:
        return False, (
            f"Ukuran file melebihi {MAX_FILE_SIZE_MB:.0f} MB. "
            "Perkecil ukuran file atau kompres dokumen."
        )

    return True, None


def validate_batch(files: List[UploadedFile]) -> List[dict]:
    """Validate a list of uploaded PDF files for batch prediction.

    Args:
        files: List of Streamlit UploadedFile instances.

    Returns:
        A list of dicts with keys ``filename``, ``valid``, ``reason``,
        and ``size_mb``. One dict per input file, in the same order.
    """
    results: List[dict] = []
    for uploaded_file in files:
        filename = getattr(uploaded_file, "name", "") or "Tanpa nama"
        ext = _extension_of(filename)
        size = _size_of(uploaded_file)

        if ext not in BATCH_ALLOWED_EXTENSIONS:
            results.append(
                {
                    "filename": filename,
                    "valid": False,
                    "reason": "Bukan file PDF.",
                    "size_mb": _mb(size),
                }
            )
            continue

        if size is not None and size <= 0:
            results.append(
                {
                    "filename": filename,
                    "valid": False,
                    "reason": "File kosong.",
                    "size_mb": _mb(size),
                }
            )
            continue

        if size is not None and size > MAX_FILE_SIZE_BYTES:
            results.append(
                {
                    "filename": filename,
                    "valid": False,
                    "reason": f"Ukuran melebihi {MAX_FILE_SIZE_MB:.0f} MB.",
                    "size_mb": _mb(size),
                }
            )
            continue

        results.append(
            {
                "filename": filename,
                "valid": True,
                "reason": None,
                "size_mb": _mb(size),
            }
        )

    return results


def _mb(size: Optional[int]) -> Optional[float]:
    """Convert a byte count to MB (rounded to 2 decimals), or None."""
    if size is None:
        return None
    return round(size / (1024 * 1024), 2)


# ── Self-check ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("SINGLE allowed:", sorted(SINGLE_ALLOWED_EXTENSIONS))
    print("BATCH allowed:", sorted(BATCH_ALLOWED_EXTENSIONS))
    print("MAX size MB:", MAX_FILE_SIZE_MB)
