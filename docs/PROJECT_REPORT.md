# EarningsIQ — Agentic Hybrid RAG Project Report

## Executive summary

EarningsIQ provides traceable financial intelligence from quarterly reports for Infosys, Tata Consultancy Services, and HDFC Bank. A shared ingestion and retrieval design supports company, quarter, and document-type filtering without creating separate indexes.

## Problem and objective

Quarterly disclosures contain numeric results and management commentary spread across pages and reports. The implemented objective is to turn 15 manually supplied PDFs covering three companies and five quarters into consistent, searchable records without losing company, source, quarter, or page provenance.

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

BM25, Reciprocal Rank Fusion, reranking, grounded generation, Streamlit, entity-aware LangGraph routing, streaming, and the Management Promise Tracker are implemented. Future GraphRAG could connect Company → Quarter → Guidance → Expected Metric → Future Actual Metric using Neo4j. Automated NSE/BSE ingestion is deferred.

## Conclusion

The complete application provides a reproducible, citation-safe hybrid RAG pipeline and a professional financial dashboard while keeping advanced data acquisition and GraphRAG honestly out of scope.

## Objectives and why RAG is appropriate

The objectives are traceable report analysis, cross-quarter comparison, retrieval transparency, and separation of deterministic KPIs from qualitative evidence. RAG is appropriate because management commentary and risks are document-specific, change each quarter, and must be cited rather than recalled from model parameters.

## Dataset and documents used

The dataset covers Infosys, TCS, and HDFC Bank from Q1 FY26 through Q1 FY27. The dashboard uses revenue and margin KPIs for the IT-services companies and banking-specific total income, net profit, gross NPA, and EPS for HDFC Bank. The 15 PDFs are user-supplied and excluded from Git; a tracked manifest records their normalized metadata, and the Sources tab reports exactly what is indexed.

## Technology stack

Python, Streamlit, PyMuPDF, LangChain text splitters, Nebius OpenAI-compatible APIs, Pinecone, rank-bm25, and LangGraph. JSON/JSONL files provide structured metrics and the local sparse corpus.

## Full system architecture

A LangGraph router assigns FINANCIAL_QUERY, DOCUMENT_QUERY, or COMPARISON_QUERY and extracts supported company and quarter mentions. One named company remains filtered; multiple named companies open a cross-company scope. Metadata filters constrain Pinecone and BM25, RRF combines rankings, Nebius reranks the shortlist, and a grounded prompt generates an answer with company-aware source labels. Rotating logs capture ingestion, scope, retrieval, routing, generation, and errors.

## Dense retrieval, BM25, and Reciprocal Rank Fusion

Dense search captures semantic similarity; BM25 preserves exact finance terms, numbers, abbreviations, and guidance language. Both share deterministic chunk IDs. RRF uses `1 / (60 + rank)` from each list, avoiding incomparable raw score scales.

## Reranking

The default flow retrieves 15 candidates per retriever, fuses to 12, and asks the configured Nebius chat model to select up to 6. If reranking fails, the system logs the exception and safely falls back to RRF order.

## LangGraph agentic routing

A deliberately small graph routes three question types to a shared streamed evidence workflow. Entity extraction selects company and quarter scope before retrieval, enabling both same-company longitudinal analysis and explicit cross-company comparisons without unnecessary multi-agent complexity.

## Prompt and grounding design

The system prompt prohibits unsupported numbers and requires an explicit insufficiency response. Context excerpts receive stable labels `[S1]`, `[S2]`, and so on; every label maps to document name, quarter, and one-based PDF page.

## Management Promise Tracker

The tracker retrieves forward-looking statements and later results across quarters, then assigns Achieved, Partially Achieved, Missed, or Pending. Pending is mandatory when later evidence is unavailable. Seeded records keep the UI demonstrable but are clearly marked for revalidation.

## UI architecture

The wide Streamlit interface opens on a persistent Q&A bot with a three-company selector, suggested prompts, grounded chat history, company-aware citations, streaming, and retrieval diagnostics. Financial Snapshot renders four industry-specific KPIs and a five-quarter table without charts; Management Guidance shows status badges; Sources lists the selected company's page and chunk counts.

## Example questions

How did Infosys perform in the latest quarter? Compare Infosys, TCS, and HDFC Bank in Q1 FY27. How did HDFC Bank's gross NPA change? Why did TCS margins change? What risks did management mention? Was earlier guidance achieved?

## Screenshot placeholders

Add final screenshots of the Overview, Ask Earnings AI diagnostics, Promise Tracker, and Sources inventory after locally indexing reports and configuring credentials.

## Limitations

Scanned PDFs need OCR; the supplied image-only HDFC Bank Q4 results are therefore represented by the extractable Q4 press release. Retrieval quality depends on the selected reports. Live API indexing requires credentials and explicit approval to transfer extracted report text. Seeded tracker records are not a substitute for live analysis. The app is research support, not investment advice.

## Future GraphRAG architecture

A future Neo4j graph can model Company → Quarter → Management Guidance → Expected Metric → Compared With → Future Quarter Actual Metric. Graph retrieval would complement, not replace, evidence-bearing PDF chunks.

## Future NSE/BSE ingestion architecture

A scheduled collector can discover exchange filings, validate issuer and period, retain original artifacts, deduplicate by document hash, run OCR when required, and invoke the existing ingestion pipeline with audit logs and retry queues.

## Final conclusion

EarningsIQ demonstrates the requested hybrid and agentic RAG concepts in a compact architecture. The distinguishing feature is the Management Promise Tracker, while citations and transparent diagnostics make the output inspectable.
