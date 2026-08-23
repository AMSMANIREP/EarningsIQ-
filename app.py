import json
from pathlib import Path

import pandas as pd
import plotly.express as px
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

st.set_page_config(page_title="EarningsIQ", page_icon="📈", layout="wide")
st.markdown(
    """
    <style>
    .stApp {background: #07111f; color: #e8eef7;}
    [data-testid="stSidebar"] {background: #0b1728; border-right: 1px solid #1f3148;}
    .hero {padding: 1.2rem 1.4rem; border: 1px solid #20364f; border-radius: 18px;
           background: linear-gradient(130deg,#0e2238,#102c33); margin-bottom: 1rem;}
    .hero h1 {margin: 0; font-size: 2.2rem; color: #f7fbff;}
    .hero p {margin: .35rem 0 0; color: #a9bdd2;}
    div[data-testid="stMetric"] {background:#0e1d2e; border:1px solid #21374f;
        border-radius:14px; padding:16px; box-shadow:0 8px 24px rgba(0,0,0,.18);}
    div[data-testid="stMetricValue"] {color:#62e6b5;}
    .badge {display:inline-block; padding:.25rem .6rem; border-radius:999px;
        font-size:.8rem; font-weight:700; margin-right:.35rem;}
    .ok {background:#123c32;color:#78efc1}.warn {background:#4a3614;color:#ffd47a}
    .bad {background:#4a2029;color:#ff9aaa}.pending {background:#23334c;color:#a9c9ff}
    .source-card {background:#0d1b2a;border:1px solid #21374f;border-radius:12px;padding:12px;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


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


metrics = load_metrics()
settings = load_settings(validate=False)
corpus = load_corpus(Path(settings.corpus_path))
companies = sorted(metrics)

with st.sidebar:
    st.markdown("## EarningsIQ")
    company = st.selectbox("Company", companies, format_func=lambda key: metrics[key]["company_name"])
    quarters = list(metrics[company]["quarters"])
    quarter = st.selectbox("Quarter", quarters, index=len(quarters) - 1)
    st.markdown("### Available documents")
    documents = corpus_summary(corpus)
    if documents:
        for document in documents:
            st.caption(f"✓ {document['source']} · {document['quarter']}")
    else:
        st.caption("No PDFs indexed locally")
    st.markdown("### RAG system status")
    st.markdown(
        status_badge("Achieved" if configured(settings) and corpus else "Pending")
        + (" Ready" if configured(settings) and corpus else " Needs credentials/indexed PDFs"),
        unsafe_allow_html=True,
    )
    st.caption(f"Local chunks: {len(corpus):,}")
    st.caption(f"Pinecone index: {settings.index_name}")

company_data = metrics[company]
current = company_data["quarters"][quarter]
st.markdown(
    f'<div class="hero"><h1>EarningsIQ</h1><p>{company_data["company_name"]} · {quarter} · Agentic Hybrid RAG financial intelligence</p></div>',
    unsafe_allow_html=True,
)

tab_overview, tab_ask, tab_guidance, tab_sources = st.tabs(
    ["Overview", "Ask Earnings AI", "Management Guidance", "Sources"]
)

with tab_overview:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Revenue", f"₹{current['revenue']:,} cr", f"{current['revenue_qoq_growth']:.1f}% QoQ")
    col2.metric("Net profit", f"₹{current['net_profit']:,} cr")
    col3.metric("Operating margin", f"{current['operating_margin']:.1f}%")
    col4.metric("Basic EPS", f"₹{current['eps']:.2f}")

    frame = pd.DataFrame(
        [{"Quarter": key, **value} for key, value in company_data["quarters"].items()]
    )
    left, right = st.columns(2)
    with left:
        revenue_fig = px.line(frame, x="Quarter", y="revenue", markers=True, title="Revenue trend (₹ crore)")
        revenue_fig.update_layout(template="plotly_dark", paper_bgcolor="#07111f", plot_bgcolor="#07111f")
        st.plotly_chart(revenue_fig, width="stretch")
    with right:
        profit_fig = px.line(frame, x="Quarter", y="net_profit", markers=True, title="Net profit trend (₹ crore)")
        profit_fig.update_layout(template="plotly_dark", paper_bgcolor="#07111f", plot_bgcolor="#07111f")
        st.plotly_chart(profit_fig, width="stretch")
    margin_fig = px.bar(frame, x="Quarter", y="operating_margin", title="Operating margin trend (%)")
    margin_fig.update_layout(template="plotly_dark", paper_bgcolor="#07111f", plot_bgcolor="#07111f")
    st.plotly_chart(margin_fig, width="stretch")
    summary = (
        f"{company_data['company_name']} reported revenue of ₹{current['revenue']:,} crore in {quarter}, "
        f"up {current['revenue_qoq_growth']:.1f}% sequentially. Net profit was ₹{current['net_profit']:,} "
        f"crore and operating margin was {current['operating_margin']:.1f}%."
    )
    st.info(summary)
    if current.get("note"):
        st.caption(current["note"])

with tab_ask:
    st.markdown("### Ask questions grounded in quarterly reports")
    example = st.selectbox(
        "Try an example",
        [
            "How did Infosys perform in the latest quarter?",
            "Why did operating margins change?",
            "What risks did management mention?",
            "Compare the last three quarters.",
            "What guidance did management provide?",
        ],
    )
    question = st.chat_input("Ask EarningsIQ", key="earnings_chat")
    if st.button("Ask selected example", width="content"):
        question = example
    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            if not configured(settings):
                st.warning("Add Nebius and Pinecone credentials/model names to .env to enable live answers.")
            elif not corpus:
                st.warning("Index at least one PDF before asking document-grounded questions.")
            else:
                try:
                    with st.spinner("Running hybrid retrieval and grounded generation…"):
                        result = run_query(
                            RAGService(settings), question, company=company, quarter=quarter
                        )
                    st.markdown(result.answer)
                    with st.expander("Retrieval diagnostics"):
                        st.json({"route": result.route, **result.diagnostics})
                    with st.expander("Retrieved sources"):
                        for index, source in enumerate(result.sources, start=1):
                            st.markdown(f"**S{index}: {source.citation}**")
                            st.caption(source.text[:700])
                except Exception as error:
                    logger.exception("Question failed")
                    st.error(f"The query could not be completed: {error}")

with tab_guidance:
    st.markdown("### Management Promise Tracker")
    st.caption("Compares forward-looking statements with evidence from later indexed quarters.")
    promises = load_seeded_promises(PROMISES_PATH, company)
    if st.button("Run AI promise review", disabled=not (configured(settings) and corpus)):
        with st.spinner("Reviewing guidance across quarters…"):
            generated = analyze_promises(settings, company=company)
            if generated:
                promises = generated
    if not configured(settings) or not corpus:
        st.info("Showing seeded demonstration records. Live statuses require indexed reports and API credentials.")
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
        st.warning("No local source corpus exists yet. Add PDFs under data/infosys and run ingest.py.")
    st.markdown("### Structured KPI sources")
    for quarter_name, values in company_data["quarters"].items():
        st.markdown(
            f'<div class="source-card"><b>{quarter_name}</b> · period ended {values["period_end"]}<br>'
            f'<a href="{values["source_url"]}">Official Infosys source</a></div><br>',
            unsafe_allow_html=True,
        )
