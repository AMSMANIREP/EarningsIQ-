from dataclasses import dataclass
from typing import Any

from openai import OpenAI
from pinecone import Pinecone


@dataclass(frozen=True)
class DenseSearchResult:
    chunk_id: str
    score: float
    text: str
    metadata: dict[str, Any]

    @property
    def citation(self) -> str:
        return (
            f"{self.metadata.get('source', 'unknown document')} | "
            f"{self.metadata.get('quarter', 'unknown quarter')} | "
            f"page {self.metadata.get('page', '?')}"
        )


def build_metadata_filter(
    *, company: str | None = None, quarter: str | None = None,
    document_type: str | None = None,
) -> dict[str, dict[str, str]] | None:
    conditions: dict[str, dict[str, str]] = {}
    if company:
        conditions["company"] = {"$eq": company.upper()}
    if quarter:
        conditions["quarter"] = {"$eq": quarter.upper()}
    if document_type:
        conditions["document_type"] = {"$eq": document_type}
    return conditions or None


def embed_query(query: str, *, api_key: str, base_url: str, model: str) -> list[float]:
    query = query.strip()
    if not query:
        raise ValueError("query cannot be empty")
    response = OpenAI(api_key=api_key, base_url=base_url).embeddings.create(
        model=model, input=[query]
    )
    return response.data[0].embedding


def dense_search(
    query_vector: list[float], *, pinecone_api_key: str, index_name: str,
    top_k: int = 5, metadata_filter: dict | None = None,
) -> list[DenseSearchResult]:
    if not query_vector:
        raise ValueError("query_vector cannot be empty")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    response = Pinecone(api_key=pinecone_api_key).Index(index_name).query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        filter=metadata_filter,
    )
    results: list[DenseSearchResult] = []
    for match in response.matches:
        metadata = dict(match.metadata or {})
        results.append(
            DenseSearchResult(
                chunk_id=match.id,
                score=float(match.score),
                text=str(metadata.pop("text", "")),
                metadata=metadata,
            )
        )
    return results

