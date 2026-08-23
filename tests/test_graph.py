from agents.graph import build_query_graph


def test_comparison_route_searches_across_quarters():
    calls = []

    def executor(question, route, *, company, quarter):
        calls.append({"question": question, "route": route, "company": company, "quarter": quarter})
        return "done"

    graph = build_query_graph(executor)
    result = graph.invoke(
        {
            "question": "Compare Q3 and Q4 revenue",
            "company": "INFY",
            "quarter": "Q1_FY27",
        }
    )
    assert result["result"] == "done"
    assert calls[0]["route"] == "COMPARISON_QUERY"
    assert calls[0]["quarter"] is None
