import json
from pathlib import Path

import pandas as pd
import streamlit as st

from agents.graph import run_query
from chains.rag_chain import RAGService
from guidance.tracker import analyze_promises, load_seeded_promises
from ingestion.corpus_store import corpus_summary, load_corpus
from utils.config import load_settings
from utils.logging_config import get_logger

logger = get_logger("app")
ROOT = Path(__file__).parent
METRICS_PATH = ROOT / "data" / "financial_metrics.json"
PROMISES_PATH = ROOT / "data" / "management_promises.json"

st.set_page_config(page_title="EarningsIQ", page_icon="💬", layout="wide")
st.markdown(
    """
    <style>
    :root {color-scheme: dark;}
    html, body, .stApp, [data-testid="stAppViewContainer"] {
        background: #07111f; color: #e8eef7;
    }
    [data-testid="stHeader"] {background: rgba(7,17,31,.92);}
    [data-testid="stSidebar"] {
        background: #0b1728; border-right: 1px solid #263d56;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] small,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #dbe9f7 !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] * {
        color: #0a1726 !important;
    }
    [data-testid="stSidebar"] hr {border-color: #263d56;}
    .hero {
        padding: 1.05rem 1.3rem; border: 1px solid #294560; border-radius: 18px;
        background: linear-gradient(130deg,#0e2238,#103139); margin-bottom: 1rem;
        box-shadow: 0 12px 34px rgba(0,0,0,.2);
    }
    .hero h1 {margin: 0; font-size: 2.05rem; color: #f7fbff;}
    .hero p {margin: .3rem 0 0; color: #b8cce0;}
    button[data-baseweb="tab"] p {color: #a9bdd2 !important; font-weight: 650;}
    button[data-baseweb="tab"][aria-selected="true"] p {color: #63e6b5 !important;}
    div[data-testid="stMetric"] {
        background:#0e1d2e; border:1px solid #294560; border-radius:14px;
        padding:16px; box-shadow:0 8px 24px rgba(0,0,0,.18);
    }
    div[data-testid="stMetricLabel"] p {color:#b8cce0 !important;}
    div[data-testid="stMetricValue"] {color:#62e6b5;}
    [data-testid="stChatMessage"] {
        background: #0d1b2a; border: 1px solid #233d57; border-radius: 16px;
        padding: .35rem .65rem; margin-bottom: .7rem;
    }
    [data-testid="stChatInput"] {border-color:#31506e; background:#0d1b2a;}
    .chat-intro {
        background:#0d1b2a; border:1px solid #294560; border-radius:16px;
        padding:1rem 1.15rem; margin:.25rem 0 1rem;
    }
    .chat-intro h3 {color:#f2f7fc; margin:0 0 .25rem;}
    .chat-intro p {color:#adc3d8; margin:0;}
    .badge {display:inline-block; padding:.25rem .6rem; border-radius:999px;
        font-size:.8rem; font-weight:700; margin-right:.35rem;}
    .ok {background:#123c32;color:#78efc1}.warn {background:#4a3614;color:#ffd47a}
    .bad {background:#4a2029;color:#ff9aaa}.pending {background:#23334c;color:#a9c9ff}
    .source-card {background:#0d1b2a;border:1px solid #294560;border-radius:12px;padding:12px;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))

def format_kpi_value(kpi: dict, value: float | int) -> str:
    decimals = int(kpi.get("decimals", 0))
    formatted = f"{value:,.{decimals}f}"
    unit = kpi.get("unit")
    if unit == "currency":
        return f"₹{formatted} cr"
    if unit == "percent":
        return f"{formatted}%"
    if unit == "rupees":
        return f"₹{formatted}"
    return formatted


def kpi_table_label(kpi: dict) -> str:
    suffix = {
        "currency": " (₹ cr)",
        "percent": " (%)",
        "rupees": " (₹)",
    }.get(kpi.get("unit"), "")
    return f"{kpi['label']}{suffix}"


def build_company_summary(company_data: dict, quarter: str, current: dict) -> str:
    kpis = company_data["kpis"]
    primary = kpis[0]
    growth_metric = company_data["growth_metric"]
    qoq = current.get(f"{growth_metric}_qoq_growth")
    yoy = current.get(f"{growth_metric}_yoy_growth")
    growth_parts = []
    if qoq is not None:
        growth_parts.append(f"{qoq:+.1f}% sequentially")
    if yoy is not None:
        growth_parts.append(f"{yoy:+.1f}% year over year")
    growth_text = f", {' and '.join(growth_parts)}" if growth_parts else ""

    other_metrics = ", ".join(
        f"{kpi['label'].lower()} was {format_kpi_value(kpi, current[kpi['key']])}"
        for kpi in kpis[1:]
    )
    return (
        f"{company_data['company_name']} reported {primary['label'].lower()} of "
        f"{format_kpi_value(primary, current[primary['key']])} in {quarter}{growth_text}. "
        f"{other_metrics.capitalize()}."
    )


def performance_rows(company_data: dict) -> list[dict]:
    growth_metric = company_data["growth_metric"]
    growth_label = company_data["growth_label"]
    rows = []
    for quarter_name, values in company_data["quarters"].items():
        row = {"Quarter": quarter_name, "Period ended": values["period_end"]}
        for kpi in company_data["kpis"]:
            row[kpi_table_label(kpi)] = values[kpi["key"]]
        row[f"{growth_label} QoQ (%)"] = values.get(
            f"{growth_metric}_qoq_growth"
        )
        row[f"{growth_label} YoY (%)"] = values.get(
            f"{growth_metric}_yoy_growth"
        )
        rows.append(row)
    return rows

def configured(settings) -> bool:
    return all(
        [settings.nebius_api_key, settings.embedding_model, settings.chat_model, settings.pinecone_api_key]
    )


def status_badge(status: str) -> str:
    css = {
        "Achieved": "ok",
        "Partially Achieved": "warn",
        "Missed": "bad",
        "Pending": "pending",
    }.get(status, "pending")
    return f'<span class="badge {css}">{status}</span>'


def answer_question(question: str, company: str, quarter: str, settings, corpus: list[dict]) -> dict:
    if not configured(settings):
        return {
            "role": "assistant",
            "content": "The API configuration is incomplete. Restart Streamlit after saving all Nebius and Pinecone values in `.env`.",
            "diagnostics": {"status": "configuration_required"},
            "sources": [],
        }
    if not corpus:
        return {
            "role": "assistant",
            "content": "No indexed report corpus is available. Run the ingestion commands and reload this page.",
            "diagnostics": {"status": "indexing_required"},
            "sources": [],
        }

    workflow: list[dict[str, str]] = []
    streamed_parts: list[str] = []
    status_widget = st.status("Starting the retrieval workflow…", expanded=True)
    answer_placeholder = st.empty()

    def report_progress(stage: str, payload: dict) -> None:
        if stage == "token":
            token = str(payload.get("token", ""))
            if token:
                streamed_parts.append(token)
                answer_placeholder.markdown("".join(streamed_parts) + "▌")
            return

        message = str(payload.get("message", "")).strip()
        if message:
            workflow.append({"stage": stage, "message": message})
            status_widget.update(label=message, state="running", expanded=True)
            status_widget.write(message)

    try:
        result = run_query(
            RAGService(settings),
            question,
            company=company,
            quarter=quarter,
            progress_callback=report_progress,
        )
        answer_placeholder.markdown(result.answer)
        status_widget.update(label="Answer ready", state="complete", expanded=False)
        return {
            "role": "assistant",
            "content": result.answer,
            "diagnostics": {
                "route": result.route,
                "workflow": workflow,
                **result.diagnostics,
            },
            "sources": [
                {"citation": source.citation, "text": source.text[:900]}
                for source in result.sources
            ],
        }
    except Exception as error:
        logger.exception("Question failed")
        status_widget.update(label="The workflow stopped with an error", state="error")
        answer_placeholder.error(f"I could not complete that request: {error}")
        return {
            "role": "assistant",
            "content": f"I could not complete that request: {error}",
            "diagnostics": {"status": "error", "workflow": workflow},
            "sources": [],
        }


metrics = load_metrics()
settings = load_settings(validate=False)
corpus = load_corpus(Path(settings.corpus_path))
companies = list(metrics)

all_documents = corpus_summary(corpus)

with st.sidebar:
    st.markdown("## EarningsIQ")
    st.caption("Quarterly financial intelligence")
    company = st.selectbox(
        "Company",
        companies,
        format_func=lambda key: metrics[key]["company_name"],
    )
    quarters = list(metrics[company]["quarters"])
    quarter = st.selectbox("Quarter context", quarters, index=len(quarters) - 1)
    documents = [
        document for document in all_documents if document["company"] == company
    ]
    company_chunks = sum(document["chunks"] for document in documents)
    ready = bool(configured(settings) and documents)
    st.divider()
    st.markdown("### System status")
    st.markdown(
        status_badge("Achieved" if ready else "Pending")
        + (
            " Ready for selected company"
            if ready
            else " Selected company needs indexing"
        ),
        unsafe_allow_html=True,
    )
    st.caption(f"Selected-company chunks: {company_chunks:,}")
    st.caption(f"All local chunks: {len(corpus):,}")
    st.caption(f"Pinecone index: {settings.index_name}")
    with st.expander(f"Indexed documents ({len(documents)})", expanded=True):
        if documents:
            for document in documents:
                st.markdown(f"✓ **{document['quarter']}**  \n{document['source']}")
        else:
            st.caption("No PDFs indexed locally for this company")

company_data = metrics[company]
current = company_data["quarters"][quarter]
st.markdown(
    f'<div class="hero"><h1>💬 EarningsIQ Assistant</h1><p>{company_data["company_name"]} · {quarter} · Answers grounded in indexed quarterly reports</p></div>',
    unsafe_allow_html=True,
)

tab_chat, tab_overview, tab_guidance, tab_sources = st.tabs(
    ["Q&A Bot", "Financial Snapshot", "Management Guidance", "Sources"]
)

with tab_chat:
    st.markdown(
        """<div class="chat-intro"><h3>Ask the earnings assistant</h3>
        <p>Ask about performance, margins, risks, guidance, or changes between quarters. Answers include document and page citations.</p></div>""",
        unsafe_allow_html=True,
    )
    suggestions = [
        f"How did {company_data['company_name']} perform in {quarter}?",
        "Compare Infosys, TCS, and HDFC Bank in Q1 FY27.",
        f"What risks did {company_data['company_name']} mention?",
        f"Compare the last three quarters for {company_data['company_name']}.",
    ]
    suggestion_columns = st.columns(4)
    selected_question = None
    for index, suggestion in enumerate(suggestions):
        if suggestion_columns[index].button(suggestion, key=f"suggestion_{index}", width="stretch"):
            selected_question = suggestion

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    clear_column, context_column = st.columns([1, 5])
    if clear_column.button("Clear chat", width="stretch"):
        st.session_state.chat_messages = []
        st.rerun()
    context_column.caption(f"Current retrieval context: {company} · {quarter}")

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("diagnostics"):
                with st.expander("Retrieval details"):
                    st.json(message["diagnostics"])
            if message.get("sources"):
                with st.expander(f"Sources ({len(message['sources'])})"):
                    for source_index, source in enumerate(message["sources"], start=1):
                        st.markdown(f"**S{source_index}: {source['citation']}**")
                        st.caption(source["text"])

    typed_question = st.chat_input("Ask about this quarter or compare multiple quarters…")
    question = typed_question or selected_question
    if question:
        user_message = {
            "role": "user",
            "content": question,
            "context": f"{company} · {quarter}",
        }
        st.session_state.chat_messages.append(user_message)
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            assistant_message = answer_question(
                question, company, quarter, settings, corpus
            )
        st.session_state.chat_messages.append(assistant_message)
        st.rerun()

with tab_overview:
    kpi_columns = st.columns(len(company_data["kpis"]))
    growth_key = f"{company_data['growth_metric']}_qoq_growth"
    for index, (column, kpi) in enumerate(
        zip(kpi_columns, company_data["kpis"], strict=True)
    ):
        delta_value = current.get(growth_key) if index == 0 else None
        delta = f"{delta_value:+.1f}% QoQ" if delta_value is not None else None
        column.metric(
            kpi["label"],
            format_kpi_value(kpi, current[kpi["key"]]),
            delta,
        )

    st.info(build_company_summary(company_data, quarter, current))
    if current.get("note"):
        st.caption(current["note"])
    st.markdown("### Quarterly performance")
    performance = pd.DataFrame(performance_rows(company_data))
    st.dataframe(performance, width="stretch", hide_index=True)

with tab_guidance:
    st.markdown("### Management Promise Tracker")
    st.caption("Compares forward-looking statements with evidence from later indexed quarters.")
    promises = load_seeded_promises(PROMISES_PATH, company)
    if st.button("Run AI promise review", disabled=not ready):
        with st.spinner("Reviewing guidance across quarters…"):
            generated = analyze_promises(settings, company=company)
            if generated:
                promises = generated
    if not ready:
        st.info("Showing seeded demonstration records. Restart Streamlit after credentials and indexing are complete.")
    if not promises:
        st.info("No seeded management promises are available for this company yet.")
    for promise in promises:
        with st.container(border=True):
            columns = st.columns([3, 1])
            columns[0].markdown(f"**{promise.category}** · {promise.source_quarter}")
            columns[1].markdown(status_badge(promise.status), unsafe_allow_html=True)
            st.write(promise.promise)
            st.caption(promise.evaluation)
            st.caption(f"Source: {promise.citation}")

with tab_sources:
    st.markdown("### Indexed source inventory")
    if documents:
        st.dataframe(pd.DataFrame(documents), width="stretch", hide_index=True)
    else:
        st.warning(
            f"No local source corpus exists for {company_data['company_name']}. "
            "Run ingest_all.py with --local-only and reload the page."
        )
    st.markdown("### Structured KPI sources")
    for quarter_name, values in company_data["quarters"].items():
        source_file = values.get("source_file", "Source document")
        source_url = values.get("source_url")
        source_reference = (
            f'<a href="{source_url}">Official company source</a> · {source_file}'
            if source_url
            else source_file
        )
        st.markdown(
            f'<div class="source-card"><b>{quarter_name}</b> · '
            f'period ended {values["period_end"]}<br>{source_reference}</div><br>',
            unsafe_allow_html=True,
        )
