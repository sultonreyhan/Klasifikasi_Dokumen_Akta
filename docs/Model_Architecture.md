
# Model Architecture

Project : AktaSense

Research Title :

Pengembangan Sistem Klasifikasi Dokumen Akta Notaris dan PPAT Menggunakan Embedding IndoBERT dan Random Forest

Version : v1.0

Status : LOCKED 🔒

---

# Purpose

Dokumen ini menjelaskan arsitektur machine learning yang digunakan pada AktaSense.

Arsitektur dirancang untuk melakukan klasifikasi dokumen akta menggunakan representasi semantik dari IndoBERT dan algoritma Random Forest.

---

# Architecture Overview

PDF Document

↓

Text Extraction

↓

Text Cleaning & Normalization

↓

IndoBERT Tokenizer

↓

IndoBERT Embedding

↓

Mean Pooling

↓

768-dimensional Embedding Vector

↓

Random Forest Classifier

↓

Prediction

↓

SHAP Explainability

---

# Component Architecture

## 1. Input Layer

Input berupa dokumen PDF.

Dokumen dapat berasal dari:

- Native PDF (text layer)
- Scanned PDF
- Foto dokumen

Apabila text layer tersedia maka dilakukan ekstraksi langsung.

Apabila tidak tersedia maka digunakan PaddleOCR sebagai fallback.

---

## 2. Text Processing Layer

Tahapan ini menghasilkan teks bersih yang siap diproses oleh model bahasa.

Tahapan:

- Text Extraction
- Minimal Cleaning
- Unicode Normalization
- Header/Footer Removal
- Paragraph Reconstruction

Tidak dilakukan:

- Stemming
- Lemmatization
- Stopword Removal

---

## 3. Embedding Layer

Model bahasa:

IndoBERT

Embedding diperoleh melalui:

- Tokenization
- Transformer Encoding
- Mean Pooling

Output:

768-dimensional dense vector.

Embedding merepresentasikan makna dokumen secara semantik.

---

## 4. Classification Layer

Algoritma:

Random Forest

Input:

Embedding Vector

Output:

Predicted Label

Model menghasilkan probabilitas untuk setiap kelas dokumen.

---

## 5. Explainability Layer

Framework:

SHAP

SHAP digunakan untuk:

- Menjelaskan prediksi model.
- Mengidentifikasi bagian teks yang paling berkontribusi terhadap prediksi.
- Mendukung interpretasi hasil bagi pengguna.

---

# Model Pipeline

Document

↓

Extract Text

↓

Clean Text

↓

IndoBERT

↓

Embedding Vector

↓

Random Forest

↓

Prediction

↓

SHAP

↓

Prediction Insight

---

# Why IndoBERT?

- Dilatih menggunakan korpus Bahasa Indonesia.
- Mampu memahami konteks hukum lebih baik dibanding metode berbasis keyword.
- Menghasilkan representasi semantik yang kaya.

---

# Why Random Forest?

- Stabil pada dataset berukuran kecil hingga menengah.
- Tidak memerlukan fine-tuning transformer.
- Mudah diinterpretasikan.
- Cepat pada proses inferensi.
- Cocok digunakan bersama embedding sebagai feature vector.

---

# Why SHAP?

- Memberikan interpretasi lokal terhadap setiap prediksi.
- Menunjukkan kontribusi fitur terhadap hasil klasifikasi.
- Mendukung transparansi sistem.

---

# Deployment Architecture

User

↓

Streamlit Application

↓

Hybrid Text Extraction

↓

Text Cleaning

↓

IndoBERT Embedding

↓

Random Forest

↓

SHAP

↓

Prediction Result

---

# Design Principles

- Modular Architecture
- Explainability First
- Human Readable Pipeline
- Research-Oriented Design
- Reproducible Workflow

---

# Status

LOCKED 🔒
