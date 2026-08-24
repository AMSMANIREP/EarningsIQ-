from typing import Any, Callable, TypedDict

from langgraph.graph import END, StateGraph

from agents.router import classify_query, extract_companies, extract_quarters
from utils.logging_config import get_logger

logger = get_logger(__name__)


class QueryState(TypedDict, total=False):
    question: str
    company: str
    quarter: str | None
    route: str
    result: Any


def build_query_graph(
    executor: Callable[..., Any],
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
):
    graph = StateGraph(QueryState)

    def route_node(state: QueryState) -> QueryState:
        route = classify_query(state["question"])
        logger.info("Query routed route=%s question=%r", route, state["question"])
        if progress_callback:
            progress_callback(
                "route",
                {
                    "message": f"Classified the question as {route.replace('_', ' ').lower()}.",
                    "route": route,
                },
            )
        return {**state, "route": route}

    def execute_node(state: QueryState) -> QueryState:
        mentioned_companies = extract_companies(state["question"])
        mentioned_quarters = extract_quarters(state["question"])

        if len(mentioned_companies) == 1:
            company_scope = mentioned_companies[0]
        elif len(mentioned_companies) > 1:
            company_scope = None
        else:
            company_scope = state.get("company")

        if len(mentioned_quarters) == 1:
            quarter_scope = mentioned_quarters[0]
        elif state["route"] == "COMPARISON_QUERY" or len(mentioned_quarters) > 1:
            quarter_scope = None
        else:
            quarter_scope = state.get("quarter")

        if progress_callback:
            company_label = company_scope or "all indexed companies"
            quarter_label = quarter_scope or "all indexed quarters"
            progress_callback(
                "scope",
                {
                    "message": (
                        f"Retrieval scope: {company_label}, {quarter_label}."
                    ),
                    "company": company_scope,
                    "quarter": quarter_scope,
                    "mentioned_companies": mentioned_companies,
                },
            )

        kwargs = {"company": company_scope, "quarter": quarter_scope}
        if progress_callback:
            kwargs["progress_callback"] = progress_callback
        result = executor(state["question"], state["route"], **kwargs)
        return {**state, "result": result}

    graph.add_node("router", route_node)
    for route in ("financial", "document", "comparison"):
        graph.add_node(route, execute_node)
        graph.add_edge(route, END)
    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        lambda state: {
            "FINANCIAL_QUERY": "financial",
            "DOCUMENT_QUERY": "document",
            "COMPARISON_QUERY": "comparison",
        }[state["route"]],
    )
    return graph.compile()


def run_query(
    service,
    question: str,
    *,
    company: str,
    quarter: str | None = None,
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
):
    graph = build_query_graph(service.answer, progress_callback=progress_callback)
    result = None
    initial_state = {
        "question": question,
        "company": company,
        "quarter": quarter,
    }
    for update in graph.stream(initial_state, stream_mode="updates"):
        for node_state in update.values():
            if isinstance(node_state, dict) and "result" in node_state:
                result = node_state["result"]

    if result is None:
        raise RuntimeError("The query workflow completed without producing an answer.")
    return result
