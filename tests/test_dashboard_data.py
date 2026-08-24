import json
from pathlib import Path


EXPECTED_QUARTERS = [
    "Q1_FY26",
    "Q2_FY26",
    "Q3_FY26",
    "Q4_FY26",
    "Q1_FY27",
]


def test_dashboard_metrics_cover_three_companies_and_all_quarters():
    data = json.loads(Path("data/financial_metrics.json").read_text(encoding="utf-8"))

    assert set(data) == {"INFY", "TCS", "HDFCBANK"}
    for company in data.values():
        assert list(company["quarters"]) == EXPECTED_QUARTERS
        assert len(company["kpis"]) == 4
        for quarter in company["quarters"].values():
            assert all(kpi["key"] in quarter for kpi in company["kpis"])


def test_hdfc_bank_uses_banking_specific_kpis():
    data = json.loads(Path("data/financial_metrics.json").read_text(encoding="utf-8"))
    labels = [kpi["label"] for kpi in data["HDFCBANK"]["kpis"]]

    assert labels == ["Total income", "Net profit", "Gross NPA", "Basic EPS"]
