import argparse
import sys

from retrieval.vector_search import build_metadata_filter, dense_search, embed_query
from utils.config import load_settings
from utils.logging_config import configure_logging, get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate dense Pinecone retrieval")
    parser.add_argument("question")
    parser.add_argument("--company", default="INFY")
    parser.add_argument("--quarter")
    parser.add_argument("--document-type")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    configure_logging()
    args = parse_args()
    logger.info("Dense validation question=%r company=%s quarter=%s", args.question, args.company, args.quarter)
    settings = load_settings()
    metadata_filter = build_metadata_filter(
        company=args.company,
        quarter=args.quarter,
        document_type=args.document_type,
    )
    vector = embed_query(
        args.question,
        api_key=settings.nebius_api_key,
        base_url=settings.nebius_base_url,
        model=settings.embedding_model,
    )
    matches = dense_search(
        vector,
        pinecone_api_key=settings.pinecone_api_key,
        index_name=settings.index_name,
        top_k=args.top_k,
        metadata_filter=metadata_filter,
    )
    print(f"Dense retrieval returned {len(matches)} chunk(s). Filter: {metadata_filter}")
    if not matches:
        print("No matches. Check the index and the selected metadata filters.")
    for rank, match in enumerate(matches, start=1):
        print(f"\n[{rank}] score={match.score:.4f} | {match.citation} | chunk={match.chunk_id}")
        print(match.text.strip() or "[No chunk text stored]")


if __name__ == "__main__":
    main()

