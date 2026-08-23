from ingestion.chunker import chunk_pages
from ingestion.pdf_loader import PageDocument


def test_chunk_metadata_and_ids_are_stable():
    page = PageDocument(
        text="Revenue increased. " * 50,
        metadata={
            "company": "INFY",
            "quarter": "Q1_FY27",
            "financial_year": "FY27",
            "document_type": "earnings_release",
            "page": 5,
            "source": "q1.pdf",
        },
    )
    first = chunk_pages([page], chunk_size=150, chunk_overlap=20)
    second = chunk_pages([page], chunk_size=150, chunk_overlap=20)
    assert first == second
    assert len(first) > 1
    assert all(item["metadata"]["page"] == 5 for item in first)
    assert all(item["metadata"]["source"] == "q1.pdf" for item in first)


def test_overlap_must_be_smaller_than_chunk_size():
    try:
        chunk_pages([], chunk_size=100, chunk_overlap=100)
    except ValueError as error:
        assert "smaller" in str(error)
    else:
        raise AssertionError("Expected ValueError")

