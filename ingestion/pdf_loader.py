from dataclasses import dataclass
from pathlib import Path

import pymupdf


@dataclass(frozen=True)
class PageDocument:
    text: str
    metadata: dict[str, str | int]


def load_pdf(
    path: Path,
    *,
    company: str,
    quarter: str,
    financial_year: str,
    document_type: str = "earnings_release",
) -> list[PageDocument]:
    """Extract non-empty PDF pages while retaining one-based page numbers."""
    path = path.resolve()
    pages: list[PageDocument] = []
    with pymupdf.open(path) as pdf:
        for page_index, page in enumerate(pdf):
            text = page.get_text("text").strip()
            if not text:
                continue
            pages.append(
                PageDocument(
                    text=text,
                    metadata={
                        "company": company.upper(),
                        "quarter": quarter.upper(),
                        "financial_year": financial_year.upper(),
                        "document_type": document_type,
                        "page": page_index + 1,
                        "source": path.name,
                    },
                )
            )
    return pages

