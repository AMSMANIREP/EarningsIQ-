from ingestion.corpus_store import corpus_summary, load_corpus, save_corpus


def make_chunk(chunk_id: str, quarter: str, page: int) -> dict:
    return {
        "id": chunk_id,
        "text": f"Revenue evidence {chunk_id}",
        "metadata": {
            "company": "INFY",
            "quarter": quarter,
            "source": f"{quarter}.pdf",
            "page": page,
        },
    }


def test_corpus_merge_is_idempotent(tmp_path):
    path = tmp_path / "chunks.jsonl"
    assert save_corpus([make_chunk("a", "Q1_FY27", 1)], path) == 1
    assert save_corpus(
        [make_chunk("a", "Q1_FY27", 1), make_chunk("b", "Q1_FY27", 2)], path
    ) == 2
    assert [item["id"] for item in load_corpus(path)] == ["a", "b"]


def test_corpus_summary_counts_unique_pages():
    records = [
        make_chunk("a", "Q1_FY27", 1),
        make_chunk("b", "Q1_FY27", 1),
        make_chunk("c", "Q1_FY27", 2),
    ]
    summary = corpus_summary(records)
    assert summary[0]["chunks"] == 3
    assert summary[0]["pages"] == 2
