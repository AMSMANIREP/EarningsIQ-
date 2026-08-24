# EarningsIQ

EarningsIQ is an Agentic Hybrid RAG application for analyzing quarterly financial reports from Infosys, Tata Consultancy Services, and HDFC Bank. It combines page-aware ingestion, Pinecone semantic search, local BM25, Reciprocal Rank Fusion, reranking, entity-aware LangGraph routing, grounded Nebius generation, industry-specific KPI dashboards, and a Management Promise Tracker.

## Phase 1 architecture

`Quarterly PDF → PyMuPDF pages → metadata → recursive chunks → Nebius embeddings → Pinecone`

Page numbers are one-based for human-readable citations. Stable chunk IDs make later dense/BM25 fusion possible, while one Pinecone index and metadata fields support company and quarter filters.

## Setup

1. Create a Python 3.11+ virtual environment.
2. Run `pip install -r requirements.txt`.
3. Copy `.env.example` to `.env` and enter your exact Nebius embedding model and API credentials.
4. Put the reports in `data/infosys/`, `data/tcs/`, and `data/hdfcbank/` using the filenames listed in `data/company_documents.json`.

Validate every available report without API usage:

```powershell
python ingest_all.py --dry-run
```

Build only the local BM25 corpus:

```powershell
python ingest_all.py --local-only
```

After approving report-content transfer to the configured services, generate embeddings and populate Pinecone:

```powershell
python ingest_all.py
```

Use `--company TCS` or repeat `--company` to process selected companies. The original `ingest.py` command remains available for indexing one document.

## Metadata

Each vector stores `company`, `quarter`, `financial_year`, `document_type`, one-based `page`, `source`, `chunk_index`, and `text`. API keys are never committed.

## Validate dense retrieval

After indexing at least one report, search all indexed quarters for one company:

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

The dashboard reads industry-aware KPIs from `data/financial_metrics.json`: revenue and margins for the IT-services companies, and total income, net profit, gross NPA, and EPS for HDFC Bank. RAG is reserved for qualitative commentary, risks, guidance, and same-company or cross-company comparisons.

## Run the complete application

Build the complete three-company local corpus without API calls:

```powershell
python ingest_all.py --local-only
```

For hybrid search, run approved normal ingestion so the same chunks exist in Pinecone and the local corpus. Then start the dashboard:

```powershell
streamlit run app.py
```

The chat-first interface opens on a persistent Q&A bot for Infosys, TCS, and HDFC Bank. A single named company keeps its metadata filter; mentioning two or more supported companies opens the comparison scope across those indexed reports. While answering, the app shows routing, entity scope, filtering, hybrid retrieval, fusion, reranking, and context-building stages, then streams generated tokens into the chat. The other tabs provide five-quarter industry-specific KPI data, guidance status, and the selected company's source inventory. Runtime events and failures are written to `logs/earningsiq.log`; logs, PDFs, and extracted text remain excluded from Git.

## Retrieval controls

`RETRIEVE_K` controls candidates from each retriever, `FUSED_K` controls the RRF shortlist, and `FINAL_K` controls context supplied to the LLM. Defaults are 15, 12, and 6.

## Tests

```powershell
pytest -q
```

Tests cover the three-company manifest and metrics, entity/quarter scope, stable chunk IDs, corpus persistence, BM25 filtering, RRF behavior, streaming, citations, and promise status validation.

## Limitations and future work

The repository does not include copyrighted quarterly PDFs or generated chunk text. Add the reports locally before enabling RAG. The seeded promise records are a transparent dashboard fallback and should be replaced by a live cross-quarter analysis for decisions. OCR, automatic NSE/BSE ingestion, GraphRAG/Neo4j, authentication, and deployment remain future enhancements; they are not claimed as implemented.

