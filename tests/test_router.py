from agents.router import classify_query, extract_quarters


def test_router_classifies_three_workflows():
    assert classify_query("What was revenue?") == "FINANCIAL_QUERY"
    assert classify_query("What risks did management mention?") == "DOCUMENT_QUERY"
    assert classify_query("Compare Q3 and Q4") == "COMPARISON_QUERY"


def test_quarter_extraction_normalizes_labels():
    assert extract_quarters("Compare Q3 FY26 vs Q4 2026") == ["Q3_FY26", "Q4_FY26"]
