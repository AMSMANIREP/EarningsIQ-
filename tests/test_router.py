from agents.router import classify_query, extract_companies, extract_quarters


def test_router_classifies_three_workflows():
    assert classify_query("What was revenue?") == "FINANCIAL_QUERY"
    assert classify_query("What risks did management mention?") == "DOCUMENT_QUERY"
    assert classify_query("Compare Q3 and Q4") == "COMPARISON_QUERY"
    assert classify_query("What was HDFC Bank's gross NPA?") == "FINANCIAL_QUERY"
    assert classify_query("How did deposits and NPAs change?") == "COMPARISON_QUERY"


def test_quarter_extraction_normalizes_labels():
    assert extract_quarters("Compare Q3 FY26 vs Q4 2026") == ["Q3_FY26", "Q4_FY26"]


def test_company_extraction_preserves_mention_order():
    assert extract_companies("Compare HDFC Bank, Infosys, and TCS") == [
        "HDFCBANK",
        "INFY",
        "TCS",
    ]
