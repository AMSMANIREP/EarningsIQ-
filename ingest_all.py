import argparse
from pathlib import Path

from ingestion.chunker import chunk_pages
from ingestion.corpus_store import save_corpus
from ingestion.indexer import embed_chunks, upsert_chunks
from ingestion.manifest import DocumentSpec, load_document_manifest
from ingestion.pdf_loader import load_pdf
from utils.config import load_settings
from utils.logging_config import configure_logging, get_logger

logger = get_logger(__name__)
DEFAULT_MANIFEST = Path(__file__).parent / "data" / "company_documents.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index all available company reports from a validated manifest"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--company",
        action="append",
        help="Only process this ticker; repeat to select multiple companies",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate extraction and chunking without writing or calling APIs",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Build the local BM25 corpus without Nebius or Pinecone calls",
    )
    return parser.parse_args()


def process_document(spec: DocumentSpec, settings, *, dry_run: bool, local_only: bool) -> int:
    if not spec.path.exists():
        logger.warning("Skipping missing report path=%s", spec.path)
        print(f"Skipped missing file: {spec.path}")
        return 0

    logger.info(
        "Starting batch document path=%s company=%s quarter=%s type=%s",
        spec.path,
        spec.company,
        spec.quarter,
        spec.document_type,
    )
    pages = load_pdf(
        spec.path,
        company=spec.company,
        quarter=spec.quarter,
        financial_year=spec.financial_year,
        document_type=spec.document_type,
    )
    if not pages:
        logger.warning("Skipping report with no extractable text path=%s", spec.path)
        print(f"Skipped report with no extractable text: {spec.path.name}")
        return 0

    chunks = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)
    if dry_run:
        print(
            f"Validated {spec.company} {spec.quarter}: "
            f"{len(pages)} pages, {len(chunks)} chunks"
        )
        return len(chunks)

    if not local_only:
        embeddings = embed_chunks(
            chunks,
            api_key=settings.nebius_api_key,
            base_url=settings.nebius_base_url,
            model=settings.embedding_model,
        )
        upsert_chunks(
            chunks,
            embeddings,
            api_key=settings.pinecone_api_key,
            index_name=settings.index_name,
            cloud=settings.pinecone_cloud,
            region=settings.pinecone_region,
        )

    corpus_total = save_corpus(chunks, Path(settings.corpus_path))
    logger.info(
        "Completed batch document company=%s quarter=%s chunks=%d corpus_total=%d",
        spec.company,
        spec.quarter,
        len(chunks),
        corpus_total,
    )
    action = "Stored locally" if local_only else "Indexed"
    print(f"{action} {spec.company} {spec.quarter}: {len(chunks)} chunks")
    return len(chunks)


def main() -> None:
    configure_logging()
    args = parse_args()
    settings = load_settings(validate=not (args.local_only or args.dry_run))
    selected = {value.upper() for value in args.company or []}
    specs = [
        spec
        for spec in load_document_manifest(args.manifest)
        if not selected or spec.company in selected
    ]
    if not specs:
        raise SystemExit("No manifest documents matched the selected companies")

    total_chunks = 0
    completed = 0
    for spec in specs:
        chunks = process_document(
            spec,
            settings,
            dry_run=args.dry_run,
            local_only=args.local_only,
        )
        if chunks:
            completed += 1
            total_chunks += chunks

    mode = "validated" if args.dry_run else "processed"
    logger.info(
        "Batch ingestion complete mode=%s documents=%d chunks=%d",
        mode,
        completed,
        total_chunks,
    )
    print(f"Batch complete: {completed} documents {mode}, {total_chunks} chunks")


if __name__ == "__main__":
    main()
