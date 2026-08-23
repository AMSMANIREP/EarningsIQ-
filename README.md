# EarningsIQ

EarningsIQ is an Agentic Hybrid RAG application for analyzing quarterly financial reports of Indian listed companies. It combines page-aware ingestion, Pinecone semantic search, local BM25, Reciprocal Rank Fusion, reranking, LangGraph routing, grounded Nebius generation, structured KPI dashboards, and a Management Promise Tracker.

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

## Complete application architecture

`Question → LangGraph router → company/quarter filter → Pinecone + BM25 → RRF → Nebius reranker → grounded answer → citations`

The dashboard reads obvious KPIs from `data/financial_metrics.json`; RAG is reserved for qualitative commentary, risks, guidance, and comparisons. This avoids asking an LLM to reproduce basic reported numbers.

## Run the complete application

Build the local corpus without API calls:

```powershell
python ingest.py data/infosys/q1.pdf --company INFY --quarter Q1_FY27 --financial-year FY27 --local-only
```

For hybrid search, run normal ingestion for every PDF so the same chunks exist in Pinecone and the local corpus. Then start the dashboard:

```powershell
streamlit run app.py
```

The four tabs provide KPI trends, grounded Q&A with diagnostics, Management Promise Tracker statuses, and an indexed source inventory. Runtime events and failures are written to `logs/earningsiq.log` with rotation; logs and extracted document text are excluded from Git.

## Retrieval controls

`RETRIEVE_K` controls candidates from each retriever, `FUSED_K` controls the RRF shortlist, and `FINAL_K` controls context supplied to the LLM. Defaults are 15, 12, and 6.

## Tests

```powershell
pytest -q
```

Tests cover metadata, stable chunk IDs, corpus persistence, BM25 filtering, RRF behavior, query routing, citations, and promise status validation.

## Limitations and future work

The repository does not include copyrighted quarterly PDFs or generated chunk text. Add the reports locally before enabling RAG. The seeded promise records are a transparent dashboard fallback and should be replaced by a live cross-quarter analysis for decisions. OCR, automatic NSE/BSE ingestion, GraphRAG/Neo4j, authentication, and deployment remain future enhancements; they are not claimed as implemented.

