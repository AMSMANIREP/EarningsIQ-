from retrieval.hybrid_search import reciprocal_rank_fusion
from retrieval.vector_search import DenseSearchResult


def dense(chunk_id: str) -> DenseSearchResult:
    return DenseSearchResult(chunk_id, 0.9, f"text {chunk_id}", {"source": "q.pdf"})


def sparse(chunk_id: str) -> dict:
    return {"id": chunk_id, "text": f"text {chunk_id}", "metadata": {"source": "q.pdf"}}


def test_rrf_rewards_results_found_by_both_retrievers():
    fused = reciprocal_rank_fusion(
        [dense("dense-only"), dense("shared")],
        [sparse("shared"), sparse("sparse-only")],
    )
    assert fused[0].chunk_id == "shared"
    assert fused[0].dense_rank == 2
    assert fused[0].sparse_rank == 1
