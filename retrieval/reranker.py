import json
import re

from openai import OpenAI

from retrieval.hybrid_search import HybridResult
from utils.logging_config import get_logger

logger = get_logger(__name__)


def _extract_json(text: str) -> list[str]:
    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        return []
    parsed = json.loads(match.group(0))
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def rerank_results(
    query: str,
    results: list[HybridResult],
    *,
    api_key: str,
    base_url: str,
    model: str,
    top_n: int = 6,
) -> list[HybridResult]:
    if len(results) <= top_n:
        return results
    candidates = "\n\n".join(
        f"ID: {item.chunk_id}\nTEXT: {item.text[:1800]}" for item in results
    )
    prompt = (
        "Rank the candidate chunk IDs by usefulness for answering the financial question. "
        "Prefer direct evidence, exact metrics, guidance, and risk statements. Return only a JSON "
        f"array of at most {top_n} IDs.\n\nQUESTION: {query}\n\n{candidates}"
    )
    try:
        response = OpenAI(api_key=api_key, base_url=base_url).chat.completions.create(
            model=model,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        ordered_ids = _extract_json(response.choices[0].message.content or "")
        by_id = {item.chunk_id: item for item in results}
        reranked = [by_id[item_id] for item_id in ordered_ids if item_id in by_id]
        for item in results:
            if item not in reranked and len(reranked) < top_n:
                reranked.append(item)
        logger.info("Reranked candidates=%d selected=%d", len(results), len(reranked))
        return reranked[:top_n]
    except Exception:
        logger.exception("LLM reranking failed; using RRF order")
        return results[:top_n]
