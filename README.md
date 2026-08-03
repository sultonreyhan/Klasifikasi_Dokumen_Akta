# AktaSense

Pengembangan Sistem Klasifikasi Dokumen Akta Notaris dan PPAT Menggunakan Embedding IndoBERT dan Random Forest.

## Deskripsi

AktaSense adalah sistem klasifikasi dokumen akta Notaris dan PPAT. Dokumen direpresentasikan sebagai vektor semantik menggunakan embedding **IndoBERT** (mean pooling), kemudian diklasifikasikan menggunakan **Random Forest**, dengan interpretasi hasil melalui SHAP. Sistem diakses melalui aplikasi web Streamlit.

## Struktur Direktori

```
Klasifikasi_Dokumen_Akta (AktaSense)/
├── Dataset/        # Dataset mentah & hasil pembersihan (read-only)
├── docs/           # Dokumentasi: PRD, Blueprint, Taxonomy, Model Architecture
├── Pipeline/       # Kode sumber pipeline ML (embedding, training, dll.)
├── Models/         # Artefak model terlatih (Random Forest, LabelEncoder)
├── Evaluation/     # Hasil evaluasi model
├── Exports/        # Artefak visualisasi / ekspor
├── App/            # Aplikasi Streamlit
└── Assets/         # Aset pendukung (gambar, styling, dll.)
```

## Prasyarat Instalasi

- Python 3.12+
- GPU opsional (untuk mempercepat embedding), CPU sudah memadai

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Ringkasan Eksekusi

1. **Embedding** — `Pipeline/embedding.py` menghasilkan matriks fitur 768-dimensi dari teks dokumen via IndoBERT.
2. **Training** — `Pipeline/train.py` melatih `RandomForestClassifier` dengan fitur embedding dan menyimpan model serta LabelEncoder.
3. **Evaluasi** — mengukur performa model pada data uji (tahap lanjutan).
4. **Prediksi & SHAP** — memprediksi dokumen baru dan menjelaskan hasil (tahap lanjutan, diintegrasikan di `App/`).