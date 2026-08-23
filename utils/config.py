import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    nebius_api_key: str
    nebius_base_url: str
    embedding_model: str
    chat_model: str
    pinecone_api_key: str
    index_name: str
    pinecone_cloud: str
    pinecone_region: str
    chunk_size: int
    chunk_overlap: int
    corpus_path: str
    retrieve_k: int
    fused_k: int
    final_k: int


def load_settings(*, validate: bool = True) -> Settings:
    load_dotenv()
    settings = Settings(
        nebius_api_key=os.getenv("NEBIUS_API_KEY", ""),
        nebius_base_url=os.getenv("NEBIUS_BASE_URL", "https://api.studio.nebius.ai/v1/"),
        embedding_model=os.getenv("NEBIUS_EMBEDDING_MODEL", ""),
        chat_model=os.getenv("NEBIUS_CHAT_MODEL", ""),
        pinecone_api_key=os.getenv("PINECONE_API_KEY", ""),
        index_name=os.getenv("PINECONE_INDEX_NAME", "financial-rag"),
        pinecone_cloud=os.getenv("PINECONE_CLOUD", "aws"),
        pinecone_region=os.getenv("PINECONE_REGION", "us-east-1"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1200")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
        corpus_path=os.getenv("CORPUS_PATH", "data/chunks.jsonl"),
        retrieve_k=int(os.getenv("RETRIEVE_K", "15")),
        fused_k=int(os.getenv("FUSED_K", "12")),
        final_k=int(os.getenv("FINAL_K", "6")),
    )
    missing = [
        name
        for name, value in {
            "NEBIUS_API_KEY": settings.nebius_api_key,
            "NEBIUS_EMBEDDING_MODEL": settings.embedding_model,
            "NEBIUS_CHAT_MODEL": settings.chat_model,
            "PINECONE_API_KEY": settings.pinecone_api_key,
        }.items()
        if not value
    ]
    if validate and missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
    return settings

