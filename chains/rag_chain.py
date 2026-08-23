from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI

from chains.prompts import GROUNDED_SYSTEM_PROMPT, build_grounded_prompt
from ingestion.corpus_store import load_corpus
from retrieval.hybrid_search import HybridResult, hybrid_search
from retrieval.reranker import rerank_results
from utils.config import Settings
from utils.logging_config import get_logger

logger = get_logger(__name__)
ProgressCallback = Callable[[str, dict[str, Any]], None]


def _emit(
    callback: ProgressCallback | None,
    stage: str,
    message: str,
    **details: Any,
) -> None:
    if stage != "token":
        logger.info("Progress stage=%s message=%s details=%s", stage, message, details)
    if callback:
        callback(stage, {"message": message, **details})


@dataclass
class RAGAnswer:
    answer: str
    sources: list[HybridResult] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    route: str = "DOCUMENT_QUERY"


def format_context(results: list[HybridResult]) -> str:
    return "\n\n".join(
        f"[S{position}] {item.citation}\n{item.text}"
        for position, item in enumerate(results, start=1)
    )


class RAGService:
    def __init__(self, settings: Settings):
        self.settings = settings

    def answer(
        self,
        question: str,
        route: str,
        *,
        company: str | None = None,
        quarter: str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> RAGAnswer:
        _emit(
            progress_callback,
            "corpus",
            "Loading the locally indexed document chunks.",
        )
        corpus = load_corpus(Path(self.settings.corpus_path))
        _emit(
            progress_callback,
            "corpus",
            f"Loaded {len(corpus)} indexed chunks.",
            corpus_chunks=len(corpus),
        )
        if not corpus:
            return RAGAnswer(
                answer="No indexed local corpus is available. Ingest at least one quarterly PDF first.",
                diagnostics={"dense_used": False, "bm25_used": False, "reason": "empty_corpus"},
                route=route,
            )

        fused, diagnostics = hybrid_search(
            question,
            corpus,
            nebius_api_key=self.settings.nebius_api_key,
            nebius_base_url=self.settings.nebius_base_url,
            embedding_model=self.settings.embedding_model,
            pinecone_api_key=self.settings.pinecone_api_key,
            index_name=self.settings.index_name,
            company=company,
            quarter=quarter,
            retrieve_k=self.settings.retrieve_k,
            fused_k=self.settings.fused_k,
            progress_callback=progress_callback,
        )

        _emit(
            progress_callback,
            "rerank",
            f"Reranking {len(fused)} fused candidates with the language model.",
            candidates=len(fused),
        )
        final_chunks = rerank_results(
            question,
            fused,
            api_key=self.settings.nebius_api_key,
            base_url=self.settings.nebius_base_url,
            model=self.settings.chat_model,
            top_n=self.settings.final_k,
        )
        _emit(
            progress_callback,
            "rerank",
            f"Selected the {len(final_chunks)} strongest evidence chunks.",
            selected_chunks=len(final_chunks),
        )

        diagnostics["chunks_supplied_to_llm"] = len(final_chunks)
        diagnostics["reranker_used"] = len(fused) > self.settings.final_k
        if not final_chunks:
            return RAGAnswer(
                answer="The indexed documents do not contain enough evidence to answer this question.",
                diagnostics=diagnostics,
                route=route,
            )

        _emit(
            progress_callback,
            "context",
            f"Building a grounded prompt from {len(final_chunks)} cited evidence chunks.",
            sources=len(final_chunks),
        )
        _emit(
            progress_callback,
            "generation",
            f"Generating the grounded answer with {self.settings.chat_model}.",
            model=self.settings.chat_model,
        )
        stream = OpenAI(
            api_key=self.settings.nebius_api_key,
            base_url=self.settings.nebius_base_url,
        ).chat.completions.create(
            model=self.settings.chat_model,
            temperature=0,
            messages=[
                {"role": "system", "content": GROUNDED_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_grounded_prompt(question, route, format_context(final_chunks)),
                },
            ],
            stream=True,
        )
        answer_parts: list[str] = []
        for chunk in stream:
            if not chunk.choices:
                continue
            token = chunk.choices[0].delta.content or ""
            if not token:
                continue
            answer_parts.append(token)
            _emit(progress_callback, "token", "", token=token)

        answer = "".join(answer_parts).strip() or "No answer was generated."
        diagnostics["generation_streamed"] = True
        _emit(
            progress_callback,
            "complete",
            f"Answer complete with {len(final_chunks)} cited sources.",
            sources=len(final_chunks),
        )
        logger.info(
            "Generated answer route=%s company=%s quarter=%s sources=%d",
            route,
            company,
            quarter,
            len(final_chunks),
        )
        return RAGAnswer(answer=answer, sources=final_chunks, diagnostics=diagnostics, route=route)
