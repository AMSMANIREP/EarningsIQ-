import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DocumentSpec:
    path: Path
    company: str
    quarter: str
    financial_year: str
    document_type: str


def load_document_manifest(path: Path) -> list[DocumentSpec]:
    if not path.exists():
        raise FileNotFoundError(f"Document manifest not found: {path}")

    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Document manifest must contain a JSON array")

    required = {"path", "company", "quarter", "financial_year"}
    specs: list[DocumentSpec] = []
    seen_paths: set[Path] = set()
    for position, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Manifest entry {position} must be an object")
        missing = sorted(required.difference(row))
        if missing:
            raise ValueError(
                f"Manifest entry {position} is missing: {', '.join(missing)}"
            )

        document_path = (path.parent / str(row["path"])).resolve()
        if document_path in seen_paths:
            raise ValueError(f"Duplicate document path in manifest: {document_path}")
        seen_paths.add(document_path)
        specs.append(
            DocumentSpec(
                path=document_path,
                company=str(row["company"]).upper(),
                quarter=str(row["quarter"]).upper(),
                financial_year=str(row["financial_year"]).upper(),
                document_type=str(row.get("document_type", "earnings_release")),
            )
        )

    return specs
