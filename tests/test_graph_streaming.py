from agents.graph import run_query


class FakeService:
    def __init__(self):
        self.calls = []

    def answer(
        self,
        question,
        route,
        *,
        company,
        quarter,
        progress_callback=None,
    ):
        self.calls.append(
            {
                "question": question,
                "route": route,
                "company": company,
                "quarter": quarter,
            }
        )
        if progress_callback:
            progress_callback("corpus", {"message": "Loaded the test corpus."})
        return "streamed answer"


def test_run_query_streams_graph_progress_and_result():
    service = FakeService()
    events = []

    result = run_query(
        service,
        "Compare Q3 and Q4 revenue",
        company="INFY",
        quarter="Q1_FY27",
        progress_callback=lambda stage, payload: events.append((stage, payload)),
    )

    assert result == "streamed answer"
    assert service.calls[0]["route"] == "COMPARISON_QUERY"
    assert service.calls[0]["quarter"] is None
    assert [stage for stage, _ in events] == ["route", "corpus"]
