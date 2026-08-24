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
    assert [stage for stage, _ in events] == ["route", "scope", "corpus"]


def test_cross_company_question_removes_company_filter_but_keeps_single_quarter():
    service = FakeService()

    result = run_query(
        service,
        "Compare Infosys, TCS, and HDFC Bank in Q1 FY27",
        company="INFY",
        quarter="Q4_FY26",
    )

    assert result == "streamed answer"
    assert service.calls[0] == {
        "question": "Compare Infosys, TCS, and HDFC Bank in Q1 FY27",
        "route": "COMPARISON_QUERY",
        "company": None,
        "quarter": "Q1_FY27",
    }

def test_named_company_overrides_sidebar_company_for_single_company_question():
    service = FakeService()

    run_query(
        service,
        "What was TCS revenue?",
        company="INFY",
        quarter="Q1_FY27",
    )

    assert service.calls[0]["company"] == "TCS"
    assert service.calls[0]["quarter"] == "Q1_FY27"
