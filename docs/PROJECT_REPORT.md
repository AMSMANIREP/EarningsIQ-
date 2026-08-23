# EarningsIQ — Phases 1–2 Project Report

## Executive summary

EarningsIQ will provide traceable financial intelligence from quarterly Indian-company reports. Phase 1 establishes the Infosys ingestion foundation; Phase 2 adds reusable dense retrieval with company, quarter, and document-type filtering.

## Problem and objective

Quarterly disclosures contain numeric results and management commentary spread across pages and reports. The immediate objective is to turn three or four manually downloaded PDFs into consistent, searchable vector records without losing source or page provenance.

## Technology and architecture

Python, PyMuPDF, LangChain's recursive splitter, the OpenAI-compatible Nebius API, and Pinecone are used. Each PDF is extracted page by page, metadata is attached before splitting, embeddings are generated in batches, and records are upserted to one cosine index.

## Chunking and metadata design

The 1,200-character default keeps related financial commentary together without making context excessively broad. A 200-character overlap preserves concepts crossing boundaries; both values are configurable. Metadata includes company, quarter, financial year, document type, one-based page, source, and chunk position. Stable SHA-256-derived IDs will align the future BM25 corpus with Pinecone.

## Pinecone and citation design

One serverless index avoids per-company index sprawl. Metadata supports company, quarter, year, and document-type filters. Page boundaries are never crossed during chunking, so each result maps unambiguously to a document, quarter, and human-readable PDF page.

## Dense retrieval validation

The question is embedded using the same Nebius model as ingestion. Pinecone cosine search returns top-k chunks with metadata. The validation command can search across a company or narrow results by quarter and document type, then prints scores, stable IDs, source filenames, quarters, pages, and text for inspection before generation is added.

## Validation and limitations

Unit tests validate deterministic chunking and provenance. End-to-end indexing requires the user's exact Nebius embedding model, credentials, Pinecone credentials, and PDFs. Image-only scans require OCR, which Phase 1 does not implement.

## Roadmap

After live dense retrieval is validated: BM25, reciprocal rank fusion, reranking, grounded generation, Streamlit, LangGraph routing, and the Management Promise Tracker. Future GraphRAG could connect Company → Quarter → Guidance → Expected Metric → Future Actual Metric using Neo4j. Automated NSE/BSE ingestion is deferred.

## Conclusion

Phase 1 provides a reproducible, citation-safe foundation without claiming downstream features are complete.
