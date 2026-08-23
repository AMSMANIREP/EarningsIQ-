from collections.abc import Iterable

from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

from utils.logging_config import get_logger

logger = get_logger(__name__)


def batched(items: list, size: int) -> Iterable[list]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def embed_chunks(chunks: list[dict], *, api_key: str, base_url: str, model: str) -> list[list[float]]:
    client = OpenAI(api_key=api_key, base_url=base_url)
    vectors: list[list[float]] = []
    for batch in batched(chunks, 64):
        response = client.embeddings.create(model=model, input=[item["text"] for item in batch])
        vectors.extend(item.embedding for item in sorted(response.data, key=lambda item: item.index))
    logger.info("Embedded chunks=%d model=%s", len(vectors), model)
    return vectors


def upsert_chunks(
    chunks: list[dict],
    embeddings: list[list[float]],
    *,
    api_key: str,
    index_name: str,
    cloud: str,
    region: str,
) -> int:
    if not chunks or len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must be non-empty and have equal lengths")
    pc = Pinecone(api_key=api_key)
    existing = {item["name"] for item in pc.list_indexes()}
    if index_name not in existing:
        logger.info("Creating Pinecone index=%s dimension=%d", index_name, len(embeddings[0]))
        pc.create_index(
            name=index_name,
            dimension=len(embeddings[0]),
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
    index = pc.Index(index_name)
    records = [
        {
            "id": chunk["id"],
            "values": vector,
            "metadata": {**chunk["metadata"], "text": chunk["text"]},
        }
        for chunk, vector in zip(chunks, embeddings, strict=True)
    ]
    for batch in batched(records, 100):
        index.upsert(vectors=batch)
    logger.info("Upserted chunks=%d index=%s", len(records), index_name)
    return len(records)

