ROUTE_INSTRUCTIONS = {
    "FINANCIAL_QUERY": "Focus on reported metrics, units, period labels, and changes.",
    "DOCUMENT_QUERY": "Focus on management commentary, risks, outlook, and qualitative evidence.",
    "COMPARISON_QUERY": "Compare periods explicitly and distinguish each quarter's evidence.",
}

GROUNDED_SYSTEM_PROMPT = """You are EarningsIQ, a careful financial research assistant.
Answer only from the supplied source excerpts. Never invent numbers, causes, guidance, or risks.
If evidence is insufficient, say exactly what is unavailable. Use INR crore unless a source states
another unit. Cite factual claims inline with the supplied source labels such as [S1]. End with a
brief Sources section listing only labels actually used. This is document analysis, not investment advice.
"""


def build_grounded_prompt(question: str, route: str, context: str) -> str:
    instruction = ROUTE_INSTRUCTIONS.get(route, ROUTE_INSTRUCTIONS["DOCUMENT_QUERY"])
    return f"{instruction}\n\nQuestion: {question}\n\nSource excerpts:\n{context}"
