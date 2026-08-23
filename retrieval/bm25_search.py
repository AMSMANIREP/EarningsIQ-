import re
from typing import Any

from rank_bm25 import BM25Okapi

from utils.logging_config import get_logger

logger = get_logger(__name__)
TOKEN_PATTERN = re.compile(r"[a-zA-Z]+(?:'[a-z]+)?|\d+(?:[.,]\d+)*|₹|%")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def _matches(metadata: dict, filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True
    for field, condition in filters.items():
        expected = condition.get("$eq") if isinstance(condition, dict) else condition
        if metadata.get(field) != expected:
            return False
    return True


def bm25_search(
    query: str,
    corpus: list[dict],
    *,
    top_k: int = 10,
    metadata_filter: dict | None = None,
) -> list[dict]:
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    candidates = [item for item in corpus if _matches(item.get("metadata", {}), metadata_filter)]
    if not candidates:
        logger.info("BM25 returned no candidates filter=%s", metadata_filter)
        return []
    tokenized = [tokenize(item.get("text", "")) for item in candidates]
    query_tokens = tokenize(query)
    query_token_set = set(query_tokens)
    if not query_token_set:
        return []
    model = BM25Okapi(tokenized)
    scores = model.get_scores(query_tokens)
    ranked = sorted(
        zip(candidates, tokenized, scores, strict=True),
        key=lambda item: item[2],
        reverse=True,
    )
    results = [
        {**item, "score": float(score), "retrieval_method": "bm25"}
        for item, tokens, score in ranked
        if query_token_set.intersection(tokens)
    ][:top_k]
    logger.info(
        "BM25 query=%r candidates=%d returned=%d filter=%s",
        query,
        len(candidates),
        len(results),
        metadata_filter,
    )
    return results
