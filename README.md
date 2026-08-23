# EarningsIQ

EarningsIQ is an Agentic Hybrid RAG application for analyzing quarterly financial reports of Indian listed companies. The repository currently implements **Phases 1–2**: page-aware ingestion, Nebius embeddings, Pinecone indexing, and filtered dense retrieval validation.

## Phase 1 architecture

`Quarterly PDF → PyMuPDF pages → metadata → recursive chunks → Nebius embeddings → Pinecone`

Page numbers are one-based for human-readable citations. Stable chunk IDs make later dense/BM25 fusion possible, while one Pinecone index and metadata fields support company and quarter filters.

## Setup

1. Create a Python 3.11+ virtual environment.
2. Run `pip install -r requirements.txt`.
3. Copy `.env.example` to `.env` and enter your exact Nebius embedding model and API credentials.
4. Put manually downloaded Infosys reports in `data/infosys/`.

Validate extraction without API usage:

```powershell
python ingest.py data/infosys/q1.pdf --company INFY --quarter Q1_FY27 --financial-year FY27 --dry-run
```

Index the document:

```powershell
python ingest.py data/infosys/q1.pdf --company INFY --quarter Q1_FY27 --financial-year FY27
```

Expected output resembles `Indexed 42 chunks from 12 pages into financial-rag`. Repeat for each of the 3–4 quarterly PDFs, changing quarter and financial year. The program creates the cosine Pinecone index using the embedding response dimension if it does not exist.

## Metadata

Each vector stores `company`, `quarter`, `financial_year`, `document_type`, one-based `page`, `source`, `chunk_index`, and `text`. API keys are never committed.

## Validate dense retrieval

After indexing at least one report, search all indexed Infosys quarters:

```powershell
python query.py "What risks did management mention?" --company INFY --top-k 5
```

Or restrict the search to one quarter:

```powershell
python query.py "How did revenue change?" --company INFY --quarter Q1_FY27 --top-k 5
```

Each result shows its cosine score, stable chunk ID, source, quarter, one-based page, and text. Empty results usually mean the selected metadata does not match the values used during ingestion.

## Tests

Run `pytest -q`. Tests cover deterministic chunk IDs and preservation of citation metadata. Live Nebius/Pinecone calls require user credentials and are intentionally not part of unit tests.

## Planned phases

BM25 comes next, followed by reciprocal-rank hybrid retrieval, reranking, grounded answers with citations, Streamlit, LangGraph routing, and the Management Promise Tracker. GraphRAG and automated NSE/BSE ingestion remain future enhancements and are not claimed as implemented.

