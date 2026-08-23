import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingestion.pdf_loader import PageDocument


def chunk_pages(
    pages: list[PageDocument], chunk_size: int = 1200, chunk_overlap: int = 200
) -> list[dict]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[dict] = []
    for page in pages:
        for position, text in enumerate(splitter.split_text(page.text)):
            identity = "|".join(
                [
                    str(page.metadata["company"]),
                    str(page.metadata["quarter"]),
                    str(page.metadata["source"]),
                    str(page.metadata["page"]),
                    str(position),
                ]
            )
            chunk_id = hashlib.sha256(identity.encode()).hexdigest()[:32]
            chunks.append(
                {
                    "id": chunk_id,
                    "text": text,
                    "metadata": {**page.metadata, "chunk_index": position},
                }
            )
    return chunks

