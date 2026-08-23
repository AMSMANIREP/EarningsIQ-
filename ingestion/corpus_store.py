import json
from pathlib import Path

from utils.logging_config import get_logger

logger = get_logger(__name__)


def load_corpus(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
    return records


def save_corpus(chunks: list[dict], path: Path) -> int:
    """Merge chunks by stable ID and atomically rewrite the local BM25 corpus."""
    path.parent.mkdir(parents=True, exist_ok=True)
    merged = {record["id"]: record for record in load_corpus(path)}
    merged.update({record["id"]: record for record in chunks})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for chunk_id in sorted(merged):
            handle.write(json.dumps(merged[chunk_id], ensure_ascii=False) + "\n")
    temporary.replace(path)
    logger.info("Saved local corpus path=%s records=%d", path, len(merged))
    return len(merged)


def corpus_summary(records: list[dict]) -> list[dict]:
    grouped: dict[tuple, dict] = {}
    for record in records:
        metadata = record.get("metadata", {})
        key = (
            metadata.get("source", "unknown"),
            metadata.get("company", "unknown"),
            metadata.get("quarter", "unknown"),
        )
        item = grouped.setdefault(
            key,
            {
                "source": key[0],
                "company": key[1],
                "quarter": key[2],
                "pages": set(),
                "chunks": 0,
            },
        )
        item["chunks"] += 1
        if metadata.get("page") is not None:
            item["pages"].add(metadata["page"])
    return [
        {**item, "pages": len(item["pages"])}
        for item in sorted(grouped.values(), key=lambda value: (value["company"], value["quarter"]))
    ]
