import argparse
from pathlib import Path

from ingestion.chunker import chunk_pages
from ingestion.corpus_store import save_corpus
from ingestion.indexer import embed_chunks, upsert_chunks
from ingestion.pdf_loader import load_pdf
from utils.config import load_settings
from utils.logging_config import configure_logging, get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index one page-aware quarterly PDF")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--company", required=True)
    parser.add_argument("--quarter", required=True)
    parser.add_argument("--financial-year", required=True)
    parser.add_argument("--document-type", default="earnings_release")
    parser.add_argument("--dry-run", action="store_true", help="Extract and chunk without API calls")
    parser.add_argument(
        "--local-only", action="store_true", help="Build the BM25 corpus without Nebius/Pinecone calls"
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    logger.info(
        "Starting ingestion pdf=%s company=%s quarter=%s dry_run=%s local_only=%s",
        args.pdf,
        args.company,
        args.quarter,
        args.dry_run,
        args.local_only,
    )
    pages = load_pdf(
        args.pdf,
        company=args.company,
        quarter=args.quarter,
        financial_year=args.financial_year,
        document_type=args.document_type,
    )
    if not pages:
        logger.error("No extractable text pdf=%s", args.pdf)
        raise SystemExit("No extractable text found in the PDF")
    if args.dry_run:
        chunks = chunk_pages(pages)
        print(f"Validated {args.pdf.name}: {len(pages)} text pages, {len(chunks)} chunks")
        return
    settings = load_settings(validate=not args.local_only)
    chunks = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)
    if args.local_only:
        total = save_corpus(chunks, Path(settings.corpus_path))
        print(f"Stored {len(chunks)} chunks locally; corpus now contains {total} chunks")
        return
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
    save_corpus(chunks, Path(settings.corpus_path))
    logger.info("Completed ingestion pdf=%s pages=%d chunks=%d", args.pdf, len(pages), count)
    print(f"Indexed {count} chunks from {len(pages)} pages into {settings.index_name}")


if __name__ == "__main__":
    main()

