from types import SimpleNamespace
from unittest.mock import MagicMock

from chains.rag_chain import RAGService
from retrieval.hybrid_search import HybridResult


def test_rag_service_streams_model_tokens(monkeypatch):
    evidence = HybridResult(
        chunk_id="chunk-1",
        text="Revenue increased during the quarter.",
        metadata={"source": "report.pdf", "quarter": "Q1_FY27", "page": 4},
        rrf_score=1.0,
    )
    monkeypatch.setattr("chains.rag_chain.load_corpus", lambda _path: [{"id": "chunk-1"}])

    def fake_hybrid_search(*_args, progress_callback=None, **_kwargs):
        if progress_callback:
            progress_callback("fusion", {"message": "Fusion complete."})
        return [evidence], {"fused_chunks": 1}

    monkeypatch.setattr("chains.rag_chain.hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(
        "chains.rag_chain.rerank_results",
        lambda _question, results, **_kwargs: results,
    )

    client = MagicMock()
    client.chat.completions.create.return_value = [
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="Revenue "))]
        ),
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="increased."))]
        ),
    ]
    monkeypatch.setattr("chains.rag_chain.OpenAI", lambda **_kwargs: client)

    settings = SimpleNamespace(
        corpus_path="unused.jsonl",
        nebius_api_key="test",
        nebius_base_url="https://example.invalid/v1",
        embedding_model="embedding-model",
        pinecone_api_key="test",
        index_name="test-index",
        retrieve_k=15,
        fused_k=12,
        chat_model="chat-model",
        final_k=6,
    )
    events = []

    result = RAGService(settings).answer(
        "How did revenue change?",
        "FINANCIAL_QUERY",
        company="INFY",
        quarter="Q1_FY27",
        progress_callback=lambda stage, payload: events.append((stage, payload)),
    )

    assert result.answer == "Revenue increased."
    assert result.diagnostics["generation_streamed"] is True
    assert client.chat.completions.create.call_args.kwargs["stream"] is True
    assert "".join(
        payload["token"] for stage, payload in events if stage == "token"
    ) == "Revenue increased."
