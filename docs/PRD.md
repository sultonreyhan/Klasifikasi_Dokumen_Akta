
# Product Requirements Document (PRD)

## Project Information

**Project Name**
AktaSense (Temporary Name)

**Project Title (PI)**
Pengembangan Sistem Klasifikasi Dokumen Akta Notaris dan PPAT Menggunakan Embedding IndoBERT dan Random Forest

---

# 1. Project Overview

## Background

Proses identifikasi jenis akta pada kantor Notaris dan PPAT masih banyak dilakukan secara manual dengan membaca isi dokumen satu per satu. Meskipun pegawai telah memahami jenis akta yang dikerjakan, proses identifikasi tetap memerlukan waktu, terutama ketika dokumen hasil scan memiliki kualitas yang beragam atau judul dokumen tidak dapat terbaca dengan jelas.

Penelitian ini bertujuan mengembangkan sebuah aplikasi berbasis web yang mampu membantu proses klasifikasi dokumen akta secara otomatis menggunakan Artificial Intelligence. Sistem melakukan ekstraksi teks dari dokumen, merepresentasikan isi dokumen menjadi embedding menggunakan IndoBERT, kemudian melakukan klasifikasi menggunakan Random Forest serta memberikan penjelasan terhadap hasil prediksi.

---

## Product Vision

Membangun aplikasi klasifikasi dokumen akta yang sederhana, mudah digunakan, transparan, dan mampu membantu proses identifikasi jenis dokumen berdasarkan isi dokumen.

---

## Objectives

- Membantu proses identifikasi jenis dokumen akta.
- Mengurangi proses klasifikasi manual.
- Memberikan penjelasan terhadap hasil prediksi AI.
- Menjadi fondasi pengembangan sistem digitalisasi dokumen pada penelitian selanjutnya.

---

## Value Proposition

- AI membantu proses identifikasi dokumen.
- Tidak memerlukan pengetahuan Machine Learning.
- Menampilkan alasan hasil prediksi.
- Mendukung klasifikasi satu maupun banyak dokumen.

---

# 2. Problem Statement

Permasalahan yang ingin diselesaikan:

- Identifikasi jenis dokumen masih dilakukan secara manual.
- Dokumen hasil scan memiliki kualitas yang bervariasi.
- Proses membaca keseluruhan isi dokumen membutuhkan waktu.
- Belum tersedia aplikasi sederhana yang mampu membantu klasifikasi dokumen akta berdasarkan isi dokumen.

---

# 3. Product Scope

## Included Features

- Single Prediction
- Batch Prediction
- OCR Processing
- OCR Preview
- Prediction Result
- Prediction Summary
- Prediction Insight
- Export Batch Result ke Excel

---

## Excluded Features

- Login
- Database
- Digital Archive
- Search Document
- User Management
- Dashboard Historis
- Folder Management
- Automatic File Rename

---

# 4. Target Users

## Primary Users

- Pegawai Notaris
- Pegawai PPAT

## Secondary Users

- Admin Digitalisasi Dokumen

---

# 5. Product Principles

- AI Assist, not AI Replace
- Document-Centric Workflow
- Mobile First
- Responsive Design
- User Choice
- One Workflow Across Devices
- Invisible Complexity
- Simplicity First

---

# 6. UI Flow

## First Time User

Open Application

↓

Quick Tour (Future Enhancement)

↓

Landing Page

---

## Single Prediction

Landing Page

↓

Single Prediction

↓

Document Builder

↓

Document Validation

↓

OCR Processing

↓

OCR Preview

↓

Prediction Result

↓

Prediction Summary

↓

Prediction Insight

↓

Finish

---

## Batch Prediction

Landing Page

↓

Batch Prediction

↓

Upload Multiple PDF

↓

Document Validation

↓

Processing

↓

Result Table

↓

Export Excel

↓

Finish

---

## Error Flow

Validation Failed

↓

Upload Ulang

atau

Perbaiki Dokumen

↓

Lanjutkan Proses

---

# 7. Functional Requirements

## FR-00 User Onboarding

Sistem menyediakan Quick Tour pada penggunaan pertama untuk membantu pengguna memahami fungsi utama aplikasi.

---

## FR-01 Single Prediction

Pengguna dapat melakukan klasifikasi terhadap satu dokumen.

Business Rules:

- PDF dianggap sebagai satu dokumen.
- Beberapa gambar dianggap sebagai satu dokumen.
- Seluruh halaman diproses sebagai satu kesatuan.
- Pengguna dapat memilih upload file atau kamera.

Acceptance Criteria:

- Dokumen berhasil diproses.
- Hasil klasifikasi berhasil ditampilkan.

---

## FR-02 Batch Prediction

Pengguna dapat melakukan klasifikasi terhadap banyak dokumen secara bersamaan.

Business Rules:

- Batch hanya menerima PDF.
- Satu PDF merepresentasikan satu dokumen.
- Setiap PDF diproses secara independen.

Acceptance Criteria:

- Seluruh dokumen berhasil diproses.
- Hasil dapat diekspor ke Excel.

---

## FR-03 OCR Preview

Sistem menampilkan hasil OCR sebelum proses klasifikasi.

Business Rules:

- OCR harus selesai sebelum klasifikasi.
- Pengguna dapat meninjau hasil OCR.

Acceptance Criteria:

- Hasil OCR ditampilkan.

---

## FR-04 Prediction Result

Sistem menampilkan hasil klasifikasi dokumen.

Business Rules:

- Prediction Result ditampilkan setelah inferensi selesai.

Acceptance Criteria:

- Jenis akta berhasil ditampilkan.

---

## FR-05 Prediction Summary

Sistem menampilkan ringkasan hasil prediksi.

Komponen:

- Predicted Class
- Confidence Score
- Probability Distribution
- Prediction Statement

Business Rules:

- Confidence selalu ditampilkan.
- Probability Distribution selalu ditampilkan.

Acceptance Criteria:

- Seluruh informasi ringkasan berhasil ditampilkan.

---

## FR-06 Prediction Insight

Sistem memberikan penjelasan terhadap hasil klasifikasi.

Komponen:

- Highlight Influential Text Segment
- Explanation Panel
- Document Location

Business Rules:

- Menampilkan segmen teks yang paling berkontribusi.
- Menampilkan lokasi segmen pada dokumen.
- Tidak menjelaskan proses internal model.

Acceptance Criteria:

- Prediction Insight berhasil ditampilkan.
- Minimal satu segmen teks berhasil ditampilkan.
- Lokasi segmen berhasil diidentifikasi.

---

# 8. Non Functional Requirements

- Responsive pada Desktop dan Mobile.
- Mendukung PDF, PNG, JPG, JPEG.
- Berjalan melalui browser.
- OCR menggunakan PaddleOCR.
- Antarmuka sederhana dan mudah dipahami.
- Seluruh proses berjalan secara lokal pada aplikasi.

---

# 9. Assumptions

- Dokumen menggunakan Bahasa Indonesia.
- Dokumen merupakan akta Notaris atau PPAT.
- Kualitas scan cukup baik.
- Seluruh halaman dalam satu dokumen saling berkaitan.
- Satu dokumen merepresentasikan satu jenis akta.
- OCR mampu mengekstraksi sebagian besar teks penting.

---

# 10. Constraints

- Tidak menggunakan database.
- Tidak menggunakan sistem login.
- Tidak menyimpan histori prediksi.
- Tidak melakukan manajemen arsip.
- Tidak mendukung batch gambar.
- Tidak melakukan koreksi hasil OCR.
- Explainability hanya menjelaskan hasil prediksi.
- Klasifikasi dibatasi pada delapan jenis akta.

---

# 11. Product Acceptance Checklist

Produk dianggap selesai apabila:

- Single Prediction berfungsi.
- Batch Prediction berfungsi.
- OCR Preview berfungsi.
- Prediction Result ditampilkan.
- Prediction Summary ditampilkan.
- Prediction Insight ditampilkan.
- Export Excel berhasil.
- Responsive pada Desktop dan Mobile.

---

# 12. Technical Decisions

## OCR

PaddleOCR

Alasan:
Memiliki performa yang baik untuk ekstraksi teks dokumen hasil scan dan mendukung dokumen berbahasa Indonesia.

---

## Text Representation

IndoBERT Embedding

Alasan:
Mengubah dokumen menjadi representasi numerik yang mempertahankan makna semantik.

---

## Classification Model

Random Forest

Alasan:
Stabil pada data tabular hasil embedding, robust terhadap overfitting, serta memiliki performa yang baik pada klasifikasi.

---

## Explainability

SHAP

Alasan:
Memberikan penjelasan kontribusi setiap segmen teks terhadap hasil prediksi secara lebih konsisten.

---

## Classification Strategy

Flat Classification + Taxonomy

Alasan:
Mempermudah proses klasifikasi tanpa membangun model bertingkat, namun tetap menjaga struktur kategori dokumen.

---

## Dataset

Delapan kelas dokumen akta sesuai taxonomy yang ditentukan berdasarkan praktik Notaris dan PPAT.

---

# PRD Status

Version : v1.0

Status : LOCKED 🔒
