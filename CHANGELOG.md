# Changelog

## Chat-first interface — 2026-08-24

- Made the persistent Q&A bot the first and default application view.
- Added suggested prompts, clear-chat control, retained conversation history, citations, and diagnostics.
- Fixed dark-theme text contrast in the sidebar, tabs, metrics, and chat controls.
- Removed all charts and the Plotly dependency.
- Replaced charts with a five-quarter financial performance table.
- Added Q1 FY26 structured metrics so dashboard data aligns with all indexed quarters.

## Complete application — 2026-08-23

- Added local JSONL corpus persistence with stable IDs shared with Pinecone.
- Added metadata-filtered BM25 sparse retrieval.
- Added dense + sparse Reciprocal Rank Fusion.
- Added Nebius LLM reranking with safe RRF fallback.
- Added grounded answer generation with document, quarter, and page citations.
- Added LangGraph routing for financial, document, and comparison questions.
- Added structured Infosys KPI data and official source links.
- Added a professional four-tab Streamlit financial dashboard.
- Added Management Promise Tracker with four controlled statuses.
- Added indexed-source inventory and retrieval diagnostics.
- Added rotating console/file logging for ingestion, retrieval, routing, generation, and failures.
- Added tests for corpus persistence, BM25, RRF, routing, citations, and promise validation.
- Documented GraphRAG and automated NSE/BSE ingestion as future work only.

## Phase 2 — Dense retrieval

- Added Nebius query embeddings, Pinecone filtered search, and citation-ready CLI output.

## Phase 1 — Ingestion

- Added page-aware PDF extraction, metadata enrichment, deterministic chunking, Nebius embeddings, and Pinecone upserts.
