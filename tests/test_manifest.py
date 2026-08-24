from pathlib import Path

from ingestion.manifest import load_document_manifest


def test_company_manifest_covers_three_companies_and_five_quarters():
    specs = load_document_manifest(Path("data/company_documents.json"))

    assert {spec.company for spec in specs} == {"INFY", "TCS", "HDFCBANK"}
    for company in ("INFY", "TCS", "HDFCBANK"):
        quarters = {spec.quarter for spec in specs if spec.company == company}
        assert quarters == {
            "Q1_FY26",
            "Q2_FY26",
            "Q3_FY26",
            "Q4_FY26",
            "Q1_FY27",
        }


def test_company_manifest_paths_are_unique_and_resolved():
    specs = load_document_manifest(Path("data/company_documents.json"))

    assert len({spec.path for spec in specs}) == len(specs)
    assert all(spec.path.exists() for spec in specs)
