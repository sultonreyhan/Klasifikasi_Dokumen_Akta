
# System Blueprint

## Project Information

**Project Name**

AktaSense

**Project**

Pengembangan Sistem Klasifikasi Dokumen Akta Notaris dan PPAT Menggunakan Embedding IndoBERT dan Random Forest

---

# 0. Design Philosophy

## Core Principles

- Modular Architecture
- AI Pipeline First
- Explainability by Design
- User-Centered Simplicity
- Progressive Enhancement
- Device Independent
- Single Source of Truth
- Human-in-the-Loop
- Transparency over Complexity
- Research-Driven Development

---

# 1. System Overview

AktaSense merupakan aplikasi berbasis web yang membantu proses klasifikasi dokumen akta Notaris dan PPAT menggunakan Artificial Intelligence.

Pipeline utama sistem:

Document

↓

OCR

↓

Text Processing

↓

Embedding

↓

Classification

↓

Explainability

↓

Prediction

---

# 2. System Architecture

User

↓

Presentation Layer (Streamlit)

↓

Input Module

↓

Document Validation Module

↓

OCR Module

↓

NLP Module

↓

Classification Module

↓

Explainability Module

↓

Output Module

---

# 3. Technology Stack

## Frontend

- Streamlit

## Programming Language

- Python

## OCR

- PaddleOCR

## NLP

- IndoBERT

## Classification

- Random Forest

## Explainability

- SHAP

## Dataset

- Custom Dataset

---

# 4. Folder Structure

project/

│

├── app.py

├── pages/

├── assets/

├── models/

├── ocr/

├── nlp/

├── classifier/

├── explainability/

├── utils/

├── datasets/

├── exports/

└── config/

---

# 5. AI Pipeline

Document

↓

Image

↓

PaddleOCR

↓

Raw Text

↓

Text Cleaning

↓

IndoBERT Embedding

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

# 6. Dataset Design

## Classification Strategy

Flat Classification

Taxonomy

---

## Taxonomy

### 1. Akta Pendirian & Perubahan Badan Hukum

- Akta Pendirian
- Akta Perubahan
- Perubahan Anggaran Dasar

---

### 2. Akta Peralihan Hak

- AJB
- PPJB
- Hibah

---

### 3. Akta Waris & Keluarga

- Keterangan Waris
- Pernyataan Waris

---

### 4. Akta Perjanjian & Pernyataan

- Perjanjian
- Pernyataan
- Kuasa
- Pengakuan

---

### 5. Akta Organisasi & Keputusan

- Pernyataan Keputusan Rapat
- Pergantian Pengurus

---

## Final Prediction Classes

(akan disesuaikan berdasarkan dataset akhir)

Target:

8 Classes

---

# 7. OCR Pipeline

Input Document

↓

Image Extraction

↓

PaddleOCR

↓

OCR Result

↓

OCR Preview

---

# 8. NLP Pipeline

OCR Text

↓

Cleaning

↓

Normalization

↓

IndoBERT Tokenizer

↓

IndoBERT Embedding

↓

Embedding Vector

---

# 9. Classification Pipeline

Embedding Vector

↓

Random Forest

↓

Prediction

↓

Probability

↓

Confidence

---

# 10. Explainability Pipeline

Prediction

↓

SHAP

↓

Feature Contribution

↓

Influential Text Segment

↓

Document Location

↓

Prediction Insight

---

# 11. Module Design

## Presentation Layer

- Landing
- Single Prediction
- Batch Prediction
- OCR Preview
- Prediction Result
- Prediction Summary
- Prediction Insight

---

## Input Module

- Upload File
- Camera
- Batch Upload

---

## Validation Module

- File Validation
- Format Validation
- Resolution Validation

---

## OCR Module

- PaddleOCR
- OCR Processing

---

## NLP Module

- Cleaning
- Normalization
- Embedding

---

## Classification Module

- Random Forest
- Probability
- Confidence

---

## Explainability Module

- SHAP
- Highlight
- Explanation

---

## Export Module

- Excel Export

---

# 12. Data Flow

User

↓

Upload

↓

Validation

↓

OCR

↓

Cleaning

↓

Embedding

↓

Prediction

↓

Summary

↓

Insight

↓

Export

---

# 13. Deployment Architecture

Browser

↓

Streamlit

↓

Python Backend

↓

AI Models

↓

Prediction Result

---

# 14. Future Development

- Database
- Authentication
- Search Engine
- Digital Archive
- Dashboard
- Cloud Deployment
- API Service

---

# 15. Brand Guideline

## Product Name

AktaSense

---

## Design Language

Modern Legal Tech

---

## Personality

- Professional
- Trustworthy
- Transparent
- Intelligent
- Minimalist

---

## Color Palette

Primary

Royal Blue

Secondary

Slate

Accent

Gold

Background

Light Gray / White

---

## Typography

Plus Jakarta Sans

---

## Logo Concept

Huruf "A" yang dibentuk dari lembar dokumen dengan garis menyerupai teks dan satu node sebagai simbol AI.

Menghindari penggunaan ikon palu hakim atau timbangan hukum agar identitas produk tetap modern dan berorientasi pada teknologi dokumen.

---

# Blueprint Status

Version : v1.0

Status : LOCKED 🔒
