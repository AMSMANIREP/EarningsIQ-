from retrieval.bm25_search import bm25_search, tokenize


CORPUS = [
    {
        "id": "revenue",
        "text": "Revenue increased to ₹48,211 crore with 14.0% year-on-year growth.",
        "metadata": {"company": "INFY", "quarter": "Q1_FY27"},
    },
    {
        "id": "risk",
        "text": "Management discussed demand uncertainty and discretionary spending risk.",
        "metadata": {"company": "INFY", "quarter": "Q1_FY27"},
    },
    {
        "id": "other",
        "text": "Revenue from another company.",
        "metadata": {"company": "TCS", "quarter": "Q1_FY27"},
    },
]


def test_tokenizer_preserves_financial_tokens():
    assert "48,211" in tokenize("₹48,211 crore")
    assert "₹" in tokenize("₹48,211 crore")


def test_bm25_ranks_and_filters():
    results = bm25_search(
        "discretionary spending risk",
        CORPUS,
        top_k=2,
        metadata_filter={"company": {"$eq": "INFY"}},
    )
    assert results[0]["id"] == "risk"
    assert all(item["metadata"]["company"] == "INFY" for item in results)
