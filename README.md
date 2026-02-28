# Semantic Search Module — Bakery & Pastry Formulation

A RAG (Retrieval-Augmented Generation) semantic search module that queries ingredient and enzyme technical data sheets for bakery/pastry formulation. Users ask natural-language questions and receive the most relevant text fragments ranked by cosine similarity.

## Architecture

```
           PDF Tech Sheets (data/)
                    │
                    ▼
        ┌───────────────────────┐
        │   core/chunker.py     │  Golden-chunk extractor
        │   (section-aware,     │  (page 1 / page 2 / generic)
        │    product-prefixed)  │
        └───────────┬───────────┘
                    │  text chunks
                    ▼
        ┌───────────────────────┐
        │   core/embedding.py   │  all-MiniLM-L6-v2  (384 dims)
        └───────────┬───────────┘
                    │  vectors
                    ▼
        ┌───────────────────────┐
        │   core/database.py    │  PostgreSQL 14 + pgvector
        │   (IVFFlat index,     │  cosine similarity (⇐> operator)
        │    200 unique rows)   │
        └───────────┬───────────┘
                    │
       ┌────────────┼────────────┐
       ▼            ▼            ▼
   main.py       app.py      output/
   (CLI)      (Streamlit)   (JSON export)
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Embedding model | `all-MiniLM-L6-v2` (sentence-transformers, 384 dims) |
| Database | PostgreSQL 14 + pgvector 0.8.2 |
| Index | IVFFlat (cosine, lists = 14) |
| Similarity | Cosine similarity via `<=>` operator |
| PDF parsing | PyPDF2 (golden-chunk extractor) |
| Web UI | Streamlit |
| Language | Python 3.10+ |
| Package manager | uv |

## Setup

### 1. Prerequisites

- Python 3.10+
- PostgreSQL 14+ with pgvector extension
- uv package manager

### 2. Install dependencies

```bash
uv sync
```

### 3. Configure database

Copy the example and fill in your credentials:

```bash
cp .env.example .env
```

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=bakery_search
DB_USER=your_user
DB_PASSWORD=your_password
```

### 4. Create the database and table

```sql
CREATE DATABASE bakery_search;
\c bakery_search
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE embeddings (
    id              SERIAL PRIMARY KEY,
    id_document     INTEGER NOT NULL,
    texte_fragment  TEXT NOT NULL,
    vecteur         vector(384) NOT NULL
);
```

### 5. Ingest PDF data

Place your PDF technical sheets in the `data/` folder, then run:

```bash
uv run ingest.py
```

The ingestion pipeline:
1. Extracts structured chunks using the golden-chunk extractor (`core/chunker.py`) — section-aware, table-aware, product-name-prefixed
2. Deduplicates chunks across all PDFs
3. Generates 384-dim embeddings with all-MiniLM-L6-v2
4. Inserts into PostgreSQL/pgvector

### 6. Run the search

**CLI** (interactive loop):

```bash
uv run main.py
```

**Web UI** (Streamlit):

```bash
uv run streamlit run app.py
```

Opens at http://localhost:8501 with configurable Top K, minimum score filter, and color-coded results.

## Usage — CLI

```
--- BAKERY/PASTRY FORMULATION ASSISTANT ---
📊 Database contains 200 fragments.

[?] Ask about ingredients (or 'exit'): quelles sont les quantités recommandées d'alpha-amylase ?

Résultat 1
Texte: "Enzyme preparation based on Maltogenic Amylase..."
Score: 0.63

Résultat 2
Texte: "BVZyme AF220 — Fungal α-amylase..."
Score: 0.59

Résultat 3
Texte: "BVZyme AF110 — Fungal α-amylase..."
Score: 0.57
```

Results are also saved to `output/last_results.json` after each query.

## Usage — Web UI

The Streamlit interface provides:
- **Sidebar controls**: Top K slider (1–20), minimum score filter (0.0–1.0)
- **Color-coded scores**: 🟢 ≥ 0.6 | 🟡 ≥ 0.4 | 🔴 < 0.4
- **Auto-reconnection**: recovers gracefully if the DB connection drops

## Project Structure

```
semantic-search-module/
├── main.py              # Interactive CLI search
├── ingest.py            # Ingestion orchestration (dedup + batch insert)
├── app.py               # Streamlit web UI
├── pyproject.toml       # Dependencies
├── .env                 # DB credentials (git-ignored)
├── .env.example         # Template for .env
├── core/
│   ├── chunker.py       # Golden-chunk extractor (PDF → text chunks)
│   ├── embedding.py     # Embedder (all-MiniLM-L6-v2)
│   └── database.py      # VectorDB (pgvector queries + lifecycle)
├── data/                # 35 PDF technical sheets
├── output/              # Search results (JSON)
└── docs/                # Project documentation
```

## Data

The `data/` folder contains 35 PDF technical data sheets for bakery enzymes and ingredients (BVZyme series, ascorbic acid, etc.) from VTR&beyond. After ingestion, **200 unique chunks** are stored in the database (duplicates across PDFs are automatically skipped).
