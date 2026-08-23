from typing import Any, Callable, TypedDict

from langgraph.graph import END, StateGraph

from agents.router import classify_query
from utils.logging_config import get_logger

logger = get_logger(__name__)


class QueryState(TypedDict, total=False):
    question: str
    company: str
    quarter: str | None
    route: str
    result: Any


def build_query_graph(executor: Callable[..., Any]):
    graph = StateGraph(QueryState)

    def route_node(state: QueryState) -> QueryState:
        route = classify_query(state["question"])
        logger.info("Query routed route=%s question=%r", route, state["question"])
        return {**state, "route": route}

    def execute_node(state: QueryState) -> QueryState:
        result = executor(
            state["question"],
            state["route"],
            company=state.get("company"),
            quarter=None if state["route"] == "COMPARISON_QUERY" else state.get("quarter"),
        )
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


def run_query(service, question: str, *, company: str, quarter: str | None = None):
    graph = build_query_graph(service.answer)
    state = graph.invoke({"question": question, "company": company, "quarter": quarter})
    return state["result"]
