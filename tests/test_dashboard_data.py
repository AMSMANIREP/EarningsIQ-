import json
from pathlib import Path


def test_dashboard_metrics_cover_all_indexed_quarters():
    data = json.loads(Path("data/financial_metrics.json").read_text(encoding="utf-8"))
    assert list(data["INFY"]["quarters"]) == [
        "Q1_FY26",
        "Q2_FY26",
        "Q3_FY26",
        "Q4_FY26",
        "Q1_FY27",
    ]
