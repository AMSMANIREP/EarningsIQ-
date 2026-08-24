import pytest

from retrieval.vector_search import DenseSearchResult, build_metadata_filter, embed_query


def test_metadata_filter_normalizes_company_and_quarter():
    assert build_metadata_filter(
        company="infy", quarter="q1_fy27", document_type="earnings_release"
    ) == {
        "company": {"$eq": "INFY"},
        "quarter": {"$eq": "Q1_FY27"},
        "document_type": {"$eq": "earnings_release"},
    }


def test_empty_filter_is_omitted():
    assert build_metadata_filter() is None


def test_citation_contains_provenance():
    result = DenseSearchResult(
        "id", 0.9, "Revenue grew.",
        {"source": "q1.pdf", "quarter": "Q1_FY27", "page": 5},
    )
    assert result.citation == "q1.pdf | Q1_FY27 | page 5"



def test_citation_includes_company_when_available():
    result = DenseSearchResult(
        "id",
        0.9,
        "Revenue grew.",
        {"company": "TCS", "source": "q1.pdf", "quarter": "Q1_FY27", "page": 5},
    )

    assert result.citation == "TCS | q1.pdf | Q1_FY27 | page 5"
def test_empty_query_is_rejected_before_api_call():
    with pytest.raises(ValueError, match="cannot be empty"):
        embed_query(" ", api_key="unused", base_url="https://example.com/v1", model="unused")
