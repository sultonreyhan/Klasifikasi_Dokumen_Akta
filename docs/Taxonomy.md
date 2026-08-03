
# Taxonomy

Project : AktaSense

Research Title :

Pengembangan Sistem Klasifikasi Dokumen Akta Notaris dan PPAT Menggunakan Embedding IndoBERT dan Random Forest

Version : v1.0

Status : LOCKED 🔒

---

# Purpose

Dokumen ini mendefinisikan struktur klasifikasi dokumen yang digunakan pada AktaSense.

Taxonomy digunakan untuk:

- Mengelompokkan dokumen ke dalam domain hukum yang sejenis.
- Mempermudah analisis dataset.
- Menjadi referensi pada proses labeling.
- Membantu interpretasi hasil klasifikasi.

Taxonomy **bukan** target prediksi model.

Model tetap melakukan **Flat Classification**.

---

# Classification Strategy

Model menggunakan dua tingkat representasi.

Level 1

Flat Classification

↓

Label dokumen secara spesifik.

Level 2

Taxonomy

↓

Kelompok dokumen berdasarkan domain.

Taxonomy hanya digunakan sebagai struktur organisasi dataset.

---

# Taxonomy Structure

## 1. Pendirian & Perubahan Badan Hukum

Deskripsi

Dokumen yang berkaitan dengan pembentukan maupun perubahan badan hukum.

Labels

- akta_pendirian
- akta_perubahan

---

## 2. Peralihan Hak

Deskripsi

Dokumen yang berkaitan dengan perpindahan hak atas objek hukum.

Labels

- ajb
- ppjb
- hibah

---

## 3. Waris & Keluarga

Deskripsi

Dokumen yang berkaitan dengan hubungan keluarga dan pewarisan.

Labels

- waris

---

## 4. Perjanjian & Pernyataan

Deskripsi

Dokumen yang berisi perjanjian, pernyataan, maupun pemberian kuasa.

Labels

- pernyataan

Catatan

Dokumen kuasa dipetakan ke kelompok ini.

---

## 5. Organisasi & Keputusan

Deskripsi

Dokumen yang berkaitan dengan keputusan organisasi maupun rapat.

Labels

- pkr

---

# Label Mapping

| Label          | Taxonomy                          |
| -------------- | --------------------------------- |
| akta_pendirian | Pendirian & Perubahan Badan Hukum |
| akta_perubahan | Pendirian & Perubahan Badan Hukum |
| ajb            | Peralihan Hak                     |
| ppjb           | Peralihan Hak                     |
| hibah          | Peralihan Hak                     |
| waris          | Waris & Keluarga                  |
| pernyataan     | Perjanjian & Pernyataan           |
| pkr            | Organisasi & Keputusan            |

---

# Design Principles

- Flat Classification sebagai target utama model.
- Taxonomy digunakan untuk organisasi dataset.
- Taxonomy tidak mempengaruhi proses training secara langsung.
- Label tetap berupa string yang mudah dibaca.
- Integer encoding dilakukan hanya pada saat proses training.

---

# Future Expansion

Taxonomy dirancang agar dapat dikembangkan tanpa mengubah struktur model.

Contoh:

- Akta Pengakuan Hutang
- Akta Fidusia
- Akta Jaminan
- Akta Wasiat

dapat ditambahkan sebagai label baru pada taxonomy yang sesuai.

---

# Status

LOCKED 🔒
