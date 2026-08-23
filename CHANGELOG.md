# Changelog

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
