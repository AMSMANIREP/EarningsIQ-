# Changelog

## Multi-company support — 2026-08-24

- Added five-quarter structured KPI views for Infosys, TCS, and HDFC Bank.
- Added a validated 15-document manifest and batch ingestion command.
- Added company-aware source inventory, guidance seeds, and industry-specific KPI labels.
- Added entity extraction so named single-company questions remain filtered and multi-company comparisons search across companies.
- Added company tickers to citations and live retrieval-scope diagnostics.
- Built and validated a 411-chunk local BM25 corpus across all three companies.
- Added regression tests for manifests, banking KPIs, company extraction, and cross-company scope.
- Indexed and validated 325 TCS and HDFC Bank chunks in the configured Pinecone index.
- Fixed Windows UTF-8 console output for financial-report symbols during retrieval validation.

## Streaming workflow — 2026-08-24

- Replaced blocking LangGraph invocation with streamed graph updates.
- Added live status messages for routing, corpus loading, metadata filtering, query embedding, Pinecone search, BM25 search, Reciprocal Rank Fusion, reranking, and context assembly.
- Streamed Nebius answer tokens directly into the active assistant message.
- Stored the completed workflow trace with each answer's retrieval diagnostics.
- Added regression coverage for streamed route and service progress events.

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
