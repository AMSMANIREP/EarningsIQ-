import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from openai import OpenAI

from ingestion.corpus_store import load_corpus
from retrieval.hybrid_search import hybrid_search
from retrieval.reranker import rerank_results
from utils.config import Settings
from utils.logging_config import get_logger

logger = get_logger(__name__)
VALID_STATUSES = {"Achieved", "Partially Achieved", "Missed", "Pending"}


@dataclass(frozen=True)
class Promise:
    category: str
    promise: str
    source_quarter: str
    evaluation: str
    status: str
    citation: str


def load_seeded_promises(path: Path, company: str) -> list[Promise]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Promise(**item) for item in data.get(company, [])]


def _parse_promises(text: str) -> list[Promise]:
    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        return []
    rows = json.loads(match.group(0))
    promises = []
    for row in rows:
        if row.get("status") not in VALID_STATUSES:
            row["status"] = "Pending"
        promises.append(Promise(**{key: row.get(key, "") for key in Promise.__dataclass_fields__}))
    return promises


def analyze_promises(settings: Settings, *, company: str) -> list[Promise]:
    corpus = load_corpus(Path(settings.corpus_path))
    if not corpus:
        return []
    query = (
        "management guidance outlook commitments revenue growth margin hiring demand and whether "
        "later reported results achieved them"
    )
    results, _ = hybrid_search(
        query,
        corpus,
        nebius_api_key=settings.nebius_api_key,
        nebius_base_url=settings.nebius_base_url,
        embedding_model=settings.embedding_model,
        pinecone_api_key=settings.pinecone_api_key,
        index_name=settings.index_name,
        company=company,
        retrieve_k=settings.retrieve_k,
        fused_k=settings.fused_k,
    )
    selected = rerank_results(
        query,
        results,
        api_key=settings.nebius_api_key,
        base_url=settings.nebius_base_url,
        model=settings.chat_model,
        top_n=min(8, len(results)),
    )
    context = "\n\n".join(f"{item.citation}\n{item.text}" for item in selected)
    prompt = f"""Identify forward-looking management promises and compare each with later evidence.
Use only the excerpts. Status must be exactly Achieved, Partially Achieved, Missed, or Pending.
If later evidence is absent, use Pending. Return only a JSON array with keys: category, promise,
source_quarter, evaluation, status, citation. Keep citations as document | quarter | page.

EXCERPTS:\n{context}"""
    try:
        response = OpenAI(
            api_key=settings.nebius_api_key, base_url=settings.nebius_base_url
        ).chat.completions.create(
            model=settings.chat_model,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        promises = _parse_promises(response.choices[0].message.content or "")
        logger.info("Promise tracker company=%s promises=%d", company, len(promises))
        return promises
    except Exception:
        logger.exception("Promise tracker analysis failed company=%s", company)
        return []


def promises_as_dicts(promises: list[Promise]) -> list[dict]:
    return [asdict(item) for item in promises]
