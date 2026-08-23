from dataclasses import dataclass
from typing import Any

from retrieval.bm25_search import bm25_search
from retrieval.vector_search import build_metadata_filter, dense_search, embed_query
from utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class HybridResult:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    rrf_score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None

    @property
    def citation(self) -> str:
        return (
            f"{self.metadata.get('source', 'unknown document')} | "
            f"{self.metadata.get('quarter', 'unknown quarter')} | "
            f"page {self.metadata.get('page', '?')}"
        )


def reciprocal_rank_fusion(
    dense_results: list,
    sparse_results: list[dict],
    *,
    k: int = 60,
    top_k: int = 12,
) -> list[HybridResult]:
    combined: dict[str, dict] = {}
    for rank, item in enumerate(dense_results, start=1):
        combined[item.chunk_id] = {
            "chunk_id": item.chunk_id,
            "text": item.text,
            "metadata": item.metadata,
            "rrf_score": 1 / (k + rank),
            "dense_rank": rank,
            "sparse_rank": None,
        }
    for rank, item in enumerate(sparse_results, start=1):
        record = combined.setdefault(
            item["id"],
            {
                "chunk_id": item["id"],
                "text": item.get("text", ""),
                "metadata": item.get("metadata", {}),
                "rrf_score": 0.0,
                "dense_rank": None,
                "sparse_rank": None,
            },
        )
        record["rrf_score"] += 1 / (k + rank)
        record["sparse_rank"] = rank
    ranked = sorted(combined.values(), key=lambda value: value["rrf_score"], reverse=True)
    return [HybridResult(**item) for item in ranked[:top_k]]


def hybrid_search(
    query: str,
    corpus: list[dict],
    *,
    nebius_api_key: str,
    nebius_base_url: str,
    embedding_model: str,
    pinecone_api_key: str,
    index_name: str,
    company: str | None = None,
    quarter: str | None = None,
    document_type: str | None = None,
    retrieve_k: int = 15,
    fused_k: int = 12,
) -> tuple[list[HybridResult], dict]:
    metadata_filter = build_metadata_filter(
        company=company, quarter=quarter, document_type=document_type
    )
    vector = embed_query(
        query, api_key=nebius_api_key, base_url=nebius_base_url, model=embedding_model
    )
    dense = dense_search(
        vector,
        pinecone_api_key=pinecone_api_key,
        index_name=index_name,
        top_k=retrieve_k,
        metadata_filter=metadata_filter,
    )
    sparse = bm25_search(query, corpus, top_k=retrieve_k, metadata_filter=metadata_filter)
    fused = reciprocal_rank_fusion(dense, sparse, top_k=fused_k)
    diagnostics = {
        "dense_used": True,
        "bm25_used": True,
        "hybrid_fusion": "Reciprocal Rank Fusion",
        "dense_chunks": len(dense),
        "bm25_chunks": len(sparse),
        "fused_chunks": len(fused),
        "metadata_filter": metadata_filter,
    }
    logger.info("Hybrid query=%r diagnostics=%s", query, diagnostics)
    return fused, diagnostics
