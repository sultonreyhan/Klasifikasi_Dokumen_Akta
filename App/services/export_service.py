"""Excel export service for AktaSense.

Builds a downloadable .xlsx file from batch prediction results
(Blueprint Section 22).

Default columns (confirmed C6):
    Nama File | Kelas Prediksi | Keyakinan (%) | Waktu Proses (detik) | Status

Optional columns (include_optional=True):
    Taxonomy | Versi Model | Waktu Ekspor
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import List, Optional

from App.services.prediction_service import BatchPredictionRecord

# ── Configuration ──────────────────────────────────────────────────────────

SHEET_NAME = "Hasil Prediksi"
DEFAULT_COLUMNS = [
    "Nama File",
    "Kelas Prediksi",
    "Keyakinan (%)",
    "Waktu Proses (detik)",
    "Status",
]
OPTIONAL_COLUMNS = [
    "Taxonomy",
    "Versi Model",
    "Waktu Ekspor",
]


# ── Public API ─────────────────────────────────────────────────────────────

def export_batch_to_excel(
    results: List[BatchPredictionRecord],
    model_version: Optional[str] = None,
    include_optional: bool = False,
) -> bytes:
    """Build an Excel workbook from batch results and return its bytes.

    Args:
        results: List of :class:`BatchPredictionRecord`.
        model_version: Optional version string for the "Versi Model" column.
        include_optional: Whether to append the optional columns.

    Returns:
        ``bytes`` payload ready for ``st.download_button``.

    Raises:
        ValueError: If ``results`` is empty.
    """
    if not results:
        raise ValueError("Tidak ada hasil yang dapat diekspor.")

    import pandas as pd

    export_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rows = []
    for record in results:
        row = {
            "Nama File":            record.filename,
            "Kelas Prediksi":       record.display_name,
            "Keyakinan (%)":        round(record.confidence * 100, 1),
            "Waktu Proses (detik)": round(record.processing_time, 2),
            "Status":               record.status,
        }
        if include_optional:
            row["Taxonomy"] = record.taxonomy
            row["Versi Model"] = model_version or ""
            row["Waktu Ekspor"] = export_time
        rows.append(row)

    columns = list(DEFAULT_COLUMNS)
    if include_optional:
        columns.extend(OPTIONAL_COLUMNS)

    df = pd.DataFrame(rows, columns=columns)

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=SHEET_NAME)
    return buffer.getvalue()


def build_download_name() -> str:
    """Return the suggested Excel file name (timestamped)."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"AktaSense_Batch_{stamp}.xlsx"


EXCEL_MIME = (
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
