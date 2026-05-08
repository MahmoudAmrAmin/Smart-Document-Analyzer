# 🧠 Smart Document Analyzer

> An end-to-end intelligent document analysis system combining **RAG (Retrieval-Augmented Generation)**, **ML document classification**, and **DL layout understanding** — built in 9 days.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Store-blue)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-green)
![Groq](https://img.shields.io/badge/Groq-LLM%20API-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Demo](#demo)
- [Architecture](#architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [ML Model](#ml-model)
- [Configuration](#configuration)
- [Results](#results)
- [Roadmap](#roadmap)

---

## Overview

Smart Document Analyzer is a full-stack AI application that lets you upload any PDF document and ask natural language questions about it. It combines three AI components:

1. **RAG System** — retrieves the most relevant parts of your document and generates accurate answers using an LLM
2. **ML Classifier** — automatically detects the document type (contract, invoice, research) using a TF-IDF + SVC model trained on 15,000 samples
3. **DL Layout Detector** — understands document structure (titles, paragraphs, tables, lists) to produce smarter chunks

---

## Demo

```
Upload PDF → Click "Index" → Ask questions → Get instant answers
```

**Example interaction:**

```
User:  What are the payment terms in this contract?

Bot:   Payment is due within 30 days of invoice receipt.
       Late payments are subject to 1.5% monthly interest.

       📊 Quality: 🟢 Excellent  |  Confidence: 84%
       ⚡ 0.9s  |  📦 3 chunks  |  🏆 Score: 0.847
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    INDEXING PIPELINE                     │
│                                                          │
│  PDF Upload → Text Extraction → ML Classification        │
│     → Layout Detection → Smart Chunking                  │
│     → Embedding Generation → FAISS / ChromaDB            │
└───────────────────────┬──────────────────────────────────┘
                        │
┌───────────────────────▼──────────────────────────────────┐
│                    QUERY PIPELINE                         │
│                                                          │
│  User Question → Query Expansion → Semantic Retrieval    │
│     → Prompt Building → Groq LLM → Streaming Answer      │
│     → Quality Evaluation → Response Cache                │
└──────────────────────────────────────────────────────────┘
```

**Full data flow:**

```
PDF ──► Text Extraction (PyMuPDF / OCR)
             │
     ┌───────┼────────────┐
     ▼       ▼            ▼
  ML        Layout      Chunking
 Classify  Detect       (smart)
     │       │            │
     └───────┴────────────┘
                 │
            Embeddings
         (all-MiniLM-L6-v2)
                 │
         ┌───────┴──────┐
         ▼              ▼
       FAISS         ChromaDB
         └───────┬──────┘
                 │
     User Query ─┤
                 ▼
         Query Expansion
                 │
         Semantic Retrieval
                 │
         Prompt Builder
                 │
         Groq LLM (streaming)
                 │
         Answer + Eval Metrics
                 │
         Response Cache
                 │
         Streamlit UI
```

---

## Features

| Feature | Description |
|---|---|
| 📄 **PDF Ingestion** | Handles text-based and scanned PDFs with OCR fallback |
| 🧠 **RAG Q&A** | Answers questions grounded in document content only |
| 🔍 **Query Expansion** | Generates search variants for better retrieval coverage |
| 💾 **Response Cache** | Exact + semantic caching for instant repeat answers |
| ⚡ **LLM Streaming** | Tokens stream in real-time as Groq generates the answer |
| 📊 **Answer Evaluation** | Retrieval relevance, grounding score, and confidence |
| 🏷️ **ML Classification** | Auto-detects contract / invoice / research document type |
| 🗂️ **Layout Detection** | Identifies titles, paragraphs, tables, and lists |
| 🗄️ **Dual Vector Store** | FAISS (default) or ChromaDB — switchable in the UI |
| 📐 **Keyword Fallback** | Falls back to keyword search when semantic scores are low |
| 🔄 **Clean Sessions** | Every Streamlit start is fresh — no leftover files |

---

## Tech Stack

| Layer | Technology |
|---|---|
| **UI** | Streamlit |
| **LLM** | Groq API (`llama3-8b-8192`) / Ollama (local fallback) |
| **Embeddings** | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| **Vector Stores** | FAISS + ChromaDB |
| **PDF Extraction** | PyMuPDF (`fitz`) |
| **OCR** | Tesseract via `pytesseract` |
| **ML Classifier** | scikit-learn (TF-IDF + SVC) |
| **Training Data** | 20 Newsgroups + AG News (~15,000 balanced samples) |
| **Layout Detection** | Rule-based heuristics (LayoutParser optional) |
| **Caching** | Custom semantic cache (JSON + cosine similarity) |

---

## Project Structure

```
smart-doc-analyzer/
│
├── api/                              # FastAPI backend (future)
│   ├── routes/
│   ├── services/
│   └── main.py
│
├── data/
│   ├── annotations/
│   │   └── classification_labels/   # train_data.csv, test_data.csv
│   │       └── reports/             # confusion_matrix.png, etc.
│   ├── external/
│   │   ├── agnews/                  # AG News CSV files
│   │   └── squad/
│   ├── processed/                   # cleaned text, chunks
│   └── raw/documents/               # uploaded PDFs (cleared on startup)
│
├── front end/
│   ├── components/
│   │   ├── uploader.py              # PDF upload widget
│   │   ├── sidebar.py               # settings + controls
│   │   ├── chat.py                  # conversation history
│   │   ├── result_card.py           # answer + sources display
│   │   ├── doc_type_badge.py        # ML label badge
│   │   ├── layout_viewer.py         # document structure panel
│   │   └── eval_metrics.py          # quality metrics panel
│   └── app.py                       # main Streamlit entry point
│
├── models/
│   ├── dl_models/                   # reserved for DL models
│   └── ml_models/
│       └── doc_classifier.pkl       # trained SVC classifier
│
├── notebooks/
│   └── demo.ipynb                   # interactive demo notebook
│
├── src/
│   ├── config/settings.py           # all config in one place
│   ├── dl/
│   │   ├── layout_detector.py       # layout detection (DL + rules)
│   │   └── layout_chunker.py        # layout-aware chunking
│   ├── embeddings/embedder.py       # sentence-transformers wrapper
│   ├── ingestion/pdf_loader.py      # PDF text extraction + OCR
│   ├── ml/
│   │   ├── classifier.py            # TF-IDF + SVC pipeline
│   │   ├── predictor.py             # inference wrapper
│   │   ├── evaluate.py              # evaluation charts
│   │   └── train_pipeline.py        # training entry point
│   ├── pipeline/
│   │   ├── indexing_pipeline.py     # full indexing flow
│   │   └── full_pipeline.py         # single-call wrapper
│   ├── preprocessing/text_cleaner.py
│   ├── rag/
│   │   ├── generator.py             # Groq + Ollama + streaming
│   │   ├── prompt_builder.py        # prompt construction
│   │   ├── query_expander.py        # query expansion
│   │   ├── rag_pipeline.py          # ask() + ask_stream()
│   │   └── retriever.py             # FAISS / Chroma retrieval
│   ├── utils/
│   │   ├── cache.py                 # semantic response cache
│   │   └── evaluator.py             # answer quality metrics
│   └── vector_db/
│       ├── chroma_store.py
│       └── faiss_store.py
│
├── vector_db/
│   ├── chroma_db/                   # ChromaDB persistent store
│   └── faiss_index/                 # FAISS index + metadata
│
├── .env                             # secrets — NOT committed
├── .env.example                     # template for new users
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- Git
- Tesseract OCR (for scanned PDFs only)
  - **Windows:** [UB Mannheim installer](https://github.com/UB-Mannheim/tesseract/wiki)
  - **Mac:** `brew install tesseract`
  - **Ubuntu:** `sudo apt install tesseract-ocr`

### 1 — Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/smart-doc-analyzer.git
cd smart-doc-analyzer
```

### 2 — Create virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac / Linux
source .venv/bin/activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### 4 — Set up environment variables

```bash
cp .env.example .env
```

Edit `.env`:

```env
GROQ_API_KEY=gsk_your_key_here
LLM_PROVIDER=groq
```

Get a free Groq key at [console.groq.com](https://console.groq.com) → Sign up → API Keys → Create Key.

### 5 — Train the ML classifier

```bash
# Download + prepare merged dataset (auto, no account needed)
python data/annotations/classification_labels/download_dataset.py

# Train (~2 minutes)
python src/ml/train_pipeline.py

# Optional: view evaluation charts
python src/ml/evaluate.py
```

### 6 — Run the app

```bash
streamlit run "front end/app.py"
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Usage

### Step-by-step

1. **Upload PDFs** — use the sidebar file uploader (drag and drop)
2. **Index** — click **"Index / Re-index Documents"** and wait for the spinner
3. **Ask** — type any question in the chat input at the bottom
4. **Explore results:**
   - Expand **"📎 View source chunks"** to see what was retrieved
   - Expand **"📊 Answer Quality Metrics"** to see relevance + grounding scores
   - Check the right panel for **document type** and **layout structure**
5. **Switch vector store** — toggle FAISS / ChromaDB in settings and re-index
6. **Clear session** — click **"🗑️ Clear & Start Over"** to reset everything

### Tips for better answers

- Ask specific questions rather than vague ones
- If an answer seems incomplete, try rephrasing with more keywords
- Upload the full document rather than excerpts for best results
- Use FAISS for speed, ChromaDB for persistence across restarts

---

## ML Model

Trained on a merged dataset from two public sources:

| Source | Raw Samples | Label Mapping |
|---|---|---|
| 20 Newsgroups (sklearn) | ~18,000 | politics/religion → contract, sci/tech → invoice, sports/science → research |
| AG News (Kaggle) | ~120,000 | World → contract, Business/Tech → invoice, Sports → research |
| **Final balanced** | **15,000** | 5,000 per class |

**Pipeline:** TF-IDF (50k features, unigrams+bigrams) → SVC (RBF kernel)

**Results:**

| Metric | Value |
|---|---|
| Cross-validation accuracy | ~90% |
| Test set accuracy | ~88% |
| contract F1 | ~0.91 |
| invoice F1 | ~0.87 |
| research F1 | ~0.90 |

Evaluation charts saved to `data/annotations/classification_labels/reports/`.

---

## Configuration

All configuration lives in `src/config/settings.py`:

| Setting | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | `200` | Words per chunk |
| `CHUNK_OVERLAP` | `50` | Shared words between chunks |
| `TOP_K` | `5` | Chunks retrieved per query |
| `MIN_SCORE` | `0.15` | Minimum retrieval similarity |
| `VECTOR_STORE` | `"faiss"` | Default vector store |
| `EMBEDDING_MODEL` | `"all-MiniLM-L6-v2"` | Sentence transformer model |
| `GROQ_MODEL` | `"llama3-8b-8192"` | LLM (free tier) |
| `TEMPERATURE` | `0.2` | Lower = more factual answers |
| `MAX_TOKENS` | `512` | Max answer length |

---

## Results

| Metric | Value |
|---|---|
| ML classifier accuracy | 88–90% |
| Average query latency (cold) | 0.8–1.5s |
| Cache hit response time | < 0.1s |
| Chunks from a 10-page PDF | ~40–80 |
| PDF types supported | Text-based + scanned (OCR) |

---

## Roadmap

- [ ] FastAPI REST backend for programmatic access
- [ ] Multi-document cross-querying
- [ ] User authentication + document history
- [ ] Fine-tuned embedding model for domain-specific docs
- [ ] Docker containerization
- [ ] Cloud deployment (Railway / Render)
- [ ] Google Drive / S3 document source

---

## License

MIT License — free to use, modify, and distribute.

---

## Author

Built by **Mahmoud** as a 9-day end-to-end ML systems project.

Covers: RAG · ML classification · DL layout understanding · Streamlit UI · FAISS · ChromaDB · Groq LLM · streaming · caching · evaluation

> ⭐ If this project helped you, consider giving it a star!
