import argparse
from pathlib import Path

from ingestion.chunker import chunk_pages
from ingestion.indexer import embed_chunks, upsert_chunks
from ingestion.pdf_loader import load_pdf
from utils.config import load_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index one page-aware quarterly PDF")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--company", required=True)
    parser.add_argument("--quarter", required=True)
    parser.add_argument("--financial-year", required=True)
    parser.add_argument("--document-type", default="earnings_release")
    parser.add_argument("--dry-run", action="store_true", help="Extract and chunk without API calls")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pages = load_pdf(
        args.pdf,
        company=args.company,
        quarter=args.quarter,
        financial_year=args.financial_year,
        document_type=args.document_type,
    )
    if not pages:
        raise SystemExit("No extractable text found in the PDF")
    if args.dry_run:
        chunks = chunk_pages(pages)
        print(f"Validated {args.pdf.name}: {len(pages)} text pages, {len(chunks)} chunks")
        return
    settings = load_settings()
    chunks = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)
    embeddings = embed_chunks(
        chunks,
        api_key=settings.nebius_api_key,
        base_url=settings.nebius_base_url,
        model=settings.embedding_model,
    )
    count = upsert_chunks(
        chunks,
        embeddings,
        api_key=settings.pinecone_api_key,
        index_name=settings.index_name,
        cloud=settings.pinecone_cloud,
        region=settings.pinecone_region,
    )
    print(f"Indexed {count} chunks from {len(pages)} pages into {settings.index_name}")


if __name__ == "__main__":
    main()

