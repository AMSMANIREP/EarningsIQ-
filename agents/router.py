import re

COMPARISON_TERMS = re.compile(r"\b(compare|comparison|versus|vs\.?|between|qoq|yoy|changed?)\b", re.I)
FINANCIAL_TERMS = re.compile(
    r"\b(revenue|profit|margin|eps|growth|income|cash|deal value|headcount|attrition|"
    r"deposit|advances?|npa|interest|assets?)\b",
    re.I,
)
COMPANY_PATTERNS = {
    "INFY": re.compile(r"\b(infosys|infy)\b", re.I),
    "TCS": re.compile(r"\b(tcs|tata consultancy services)\b", re.I),
    "HDFCBANK": re.compile(r"\b(hdfc\s*bank|hdfcbank)\b", re.I),
}


def extract_companies(question: str) -> list[str]:
    matches = []
    for company, pattern in COMPANY_PATTERNS.items():
        match = pattern.search(question)
        if match:
            matches.append((match.start(), company))
    return [company for _, company in sorted(matches)]


def classify_query(question: str) -> str:
    if COMPARISON_TERMS.search(question):
        return "COMPARISON_QUERY"
    if FINANCIAL_TERMS.search(question):
        return "FINANCIAL_QUERY"
    return "DOCUMENT_QUERY"


def extract_quarters(question: str) -> list[str]:
    matches = re.findall(r"\bQ([1-4])\s*(?:FY)?\s*(\d{2,4})\b", question, re.I)
    quarters = []
    for number, year in matches:
        normalized_year = year[-2:]
        value = f"Q{number}_FY{normalized_year}"
        if value not in quarters:
            quarters.append(value)
    return quarters
