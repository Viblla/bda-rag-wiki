# app.py
import time
import json
import pandas as pd
import streamlit as st
import altair as alt

from src.retrieval import (
    bm25_retrieve,
    vector_retrieve,
    hybrid_retrieve,
    rerank,
    load_bm25,
    load_faiss,
    load_embedder,
    load_reranker,
    get_device_info,
)
from src.rag_answer import rag_answer_timed, rag_answer_iterative_timed
from src.eval_compare import run_eval
from src.confidence import compute_confidence, source_coverage
from src.hallucination import run_hallucination_suite, DEFAULT_TESTS
from src.cache_benchmark import benchmark_cache

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Wiki whatiz", layout="wide")

# ----------------------------
# Theme constants (dark grey background everywhere)
# ----------------------------
BG = "#2b2b2b"           # background
CARD = "#333333"         # sidebar/panels
BORDER = "#444444"
ORANGE = "#ff8a00"       # all font colors

# Chart dark theme
CHART_BG = "#161616"
CHART_GRID = "#2a2a2a"

# ----------------------------
# Global CSS (UI polish)
# ----------------------------
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');

html, body, [class*="css"]  {{
    background: {BG} !important;
    color: {ORANGE} !important;
}}

.stApp {{
    background: {BG} !important;
    color: {ORANGE} !important;
}}

h1, h2, h3, h4, h5, h6 {{
    color: {ORANGE} !important;
}}

p, span, div {{
    color: {ORANGE} !important;
}}

a, a:visited {{
    color: {ORANGE} !important;
}}

[data-testid="stSidebar"] {{
    background: {CARD} !important;
    border-right: 1px solid {BORDER} !important;
}}

[data-testid="stSidebar"] * {{
    color: {ORANGE} !important;
}}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {{
    font-family: 'Press Start 2P', monospace !important;
}}

.title-pixel {{
    font-family: 'Press Start 2P', monospace !important;
    font-size: 54px !important;
    text-align: center !important;
    margin-top: 10px !important;
    margin-bottom: 6px !important;
    letter-spacing: 1px;
    color: {ORANGE} !important;
}}

.subcap {{
    text-align: center !important;
    margin-top: 0px !important;
    margin-bottom: 22px !important;
    color: {ORANGE} !important;
    opacity: 0.9;
}}

div[data-baseweb="tab-border"] {{
    display: none !important;   /* removes underline under active tab */
}}

button, .stButton>button {{
    background: #1f1f1f !important;
    color: {ORANGE} !important;
    border: 1px solid #1f1f1f !important;
    border-radius: 12px !important;
    padding: 10px 18px !important;
    transition: transform 0.08s ease-in-out, opacity 0.1s ease-in-out;
}}

button:hover, .stButton>button:hover {{
    opacity: 0.92 !important;
    transform: translateY(-1px);
}}

input, textarea {{
    border-radius: 14px !important;
    border: 1px solid {BORDER} !important;
    background: #1f1f1f !important;
    color: {ORANGE} !important;
}}

[data-testid="stTextInput"] input {{
    padding: 14px 14px !important;
    font-size: 16px !important;
    background: #1f1f1f !important;
    color: {ORANGE} !important;
}}

label {{
    color: {ORANGE} !important;
}}

[data-testid="stSlider"] > div {{
    padding-top: 6px !important;
    padding-bottom: 6px !important;
}}

code, pre {{
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace !important;
}}

pre {{
    background: #121212 !important;
    color: {ORANGE} !important;
    border: 1px solid #0b0b0b !important;
    border-radius: 14px !important;
    padding: 12px 14px !important;
}}

pre code {{
    color: {ORANGE} !important;
}}

[data-testid="stCodeBlock"] {{
    border-radius: 14px !important;
}}

div[data-testid="stJson"] {{
    background: #121212 !important;
    color: {ORANGE} !important;
    border-radius: 14px !important;
    border: 1px solid #0b0b0b !important;
}}

div[data-testid="stMetric"] {{
    background: {CARD} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 14px !important;
    padding: 10px !important;
    color: {ORANGE} !important;
}}

hr {{
    border: none !important;
    border-top: 1px solid transparent !important;
    margin: 0 !important;
}}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------
# Title
# ----------------------------
st.markdown('<div class="title-pixel">Wiki whatiz</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subcap">GPT use karlo... Sab ko sab nahi milta...</div>',
    unsafe_allow_html=True,
)

# ----------------------------
# Sidebar controls
# ----------------------------
with st.sidebar:
    st.header("Retrieval Settings")
    bm25_k = st.slider("BM25 candidates (Hybrid)", 10, 200, 60, 10)
    vec_k = st.slider("Vector candidates (Hybrid)", 10, 200, 60, 10)
    merge_k = st.slider("Candidates passed forward", 5, 100, 30, 5)
    rerank_k = st.slider("Top-K after rerank", 1, 20, 5, 1)

    st.header("Modes")
    retriever_mode = st.selectbox(
        "Retriever",
        ["Hybrid (no rerank)", "Hybrid + Re-rank", "BM25", "Vector"],
        index=1,
    )
    output_mode = st.radio("Output", ["Full RAG", "Retrieval Debug"], index=0)

    sarcastic_mode = st.checkbox(
        "Sarcastic out-of-scope replies",
        value=False,
        help="If evidence is weak / out-of-scope, reply with a safe humorous refusal (no hallucinations).",
    )

    st.divider()
    use_iterative = st.checkbox("Enable Iterative RAG (Query Refinement)", value=True)
    refine_n = st.slider("Refined queries count", 1, 5, 3, 1)

    st.divider()
    if st.button("Warm-up Cache (recommended)"):
        with st.spinner("Loading BM25, FAISS, embedder, reranker into memory..."):
            load_bm25()
            load_faiss()
            load_embedder()
            load_reranker()
        st.success("Cache warmed up! Next queries should be faster.")

    # GPU Status display
    st.divider()
    st.subheader("🖥️ GPU Status")
    device_info = get_device_info()
    if device_info["cuda_available"]:
        st.success(f"✅ {device_info['gpu_name']}")
        st.caption(f"VRAM: {device_info['gpu_memory_gb']} GB")
    else:
        st.warning("⚠️ Running on CPU")

# Helper: map UI mode -> base retriever + rerank toggle
def _mode_to_flags(mode: str):
    if mode == "Hybrid (no rerank)":
        return "Hybrid", False
    if mode == "Hybrid + Re-rank":
        return "Hybrid", True
    if mode == "BM25":
        return "BM25", True
    if mode == "Vector":
        return "Vector", True
    return "Hybrid", True

# ----------------------------
# Altair chart helpers (FIXED: no LayerChart config conflict)
# ----------------------------
def _apply_dark_config(chart: alt.Chart) -> alt.Chart:
    return (
        chart.configure_view(strokeOpacity=0, fill=CHART_BG)
        .configure_axis(
            grid=True,
            gridColor=CHART_GRID,
            labelColor=ORANGE,
            titleColor=ORANGE,
            domainColor=CHART_GRID,
            tickColor=CHART_GRID,
        )
        .configure_title(color=ORANGE)
        .configure_legend(labelColor=ORANGE, titleColor=ORANGE)
    )

def _alt_bar_from_series(series: pd.Series, title: str = "", x_label="Category", y_label="Value"):
    df = series.reset_index()
    df.columns = [x_label, y_label]
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(f"{x_label}:N", sort="-y"),
            y=alt.Y(f"{y_label}:Q"),
            tooltip=[x_label, y_label],
        )
        .properties(title=title, height=260)
    )
    st.altair_chart(_apply_dark_config(chart), use_container_width=True)

def _alt_bar_from_dict(d: dict, title: str = "", x_label="Stage", y_label="Seconds"):
    df = pd.DataFrame({"Stage": list(d.keys()), "Seconds": list(d.values())})
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Stage:N", sort=None),
            y=alt.Y("Seconds:Q"),
            tooltip=["Stage", "Seconds"],
        )
        .properties(title=title, height=260)
    )
    st.altair_chart(_apply_dark_config(chart), use_container_width=True)

def _alt_bar_df(df: pd.DataFrame, x_col: str, y_col: str, title: str = ""):
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(f"{x_col}:N", sort="-y"),
            y=alt.Y(f"{y_col}:Q"),
            tooltip=[x_col, y_col],
        )
        .properties(title=title, height=260)
    )
    st.altair_chart(_apply_dark_config(chart), use_container_width=True)

# ----------------------------
# Tabs
# ----------------------------
tab1, tab2, tab3 = st.tabs(["Ask", "Evaluation", "Hallucination Test + Speed"])

# ----------------------------
# Tab 1: Ask
# ----------------------------
with tab1:
    q = st.text_input("", placeholder="Type your question…")

    colA, colB = st.columns([1.15, 0.85])

    if st.button("Run"):
        if not q.strip():
            st.warning("Please enter a question.")
            st.stop()

        base_mode, use_rerank = _mode_to_flags(retriever_mode)

        # Retrieval Debug mode
        if output_mode == "Retrieval Debug":
            start = time.time()

            if base_mode == "BM25":
                cands = bm25_retrieve(q, k=merge_k)
            elif base_mode == "Vector":
                cands = vector_retrieve(q, k=merge_k)
            else:
                cands = hybrid_retrieve(q, bm25_k=bm25_k, vec_k=vec_k, final_k=merge_k)

            top = rerank(q, cands, top_k=rerank_k) if use_rerank else cands[:rerank_k]

            with colA:
                st.subheader("Top Results")
                st.caption("After re-ranking" if use_rerank else "Without re-ranking (raw top-k)")
                for i, r in enumerate(top, 1):
                    st.markdown(
                        f"**#{i}** | `{r['source']}` | "
                        + (f"rerank `{r.get('rerank_score', 0):.4f}` | " if use_rerank else "")
                        + f"chunk_id={r['chunk_id']}"
                    )
                    st.code(r["text"][:900])

            with colB:
                st.subheader("Raw Retrieved Candidates (first 10)")
                for i, r in enumerate(cands[:10], 1):
                    st.markdown(
                        f"**#{i}** | `{r['source']}` | score `{r['score']:.4f}` | chunk_id={r['chunk_id']}"
                    )
                    st.code(r["text"][:500])

                st.subheader("Debug Stats")
                st.write(
                    {
                        "retriever_mode": retriever_mode,
                        "base_mode": base_mode,
                        "use_rerank": use_rerank,
                        "bm25_k": bm25_k,
                        "vec_k": vec_k,
                        "merge_k": merge_k,
                        "rerank_k": rerank_k,
                        "time_sec": round(time.time() - start, 2),
                    }
                )

        # Full RAG mode
        else:
            refined = None
            timings = None

            with st.spinner("Running retrieval → (optional refinement) → (optional re-rank) → LLM..."):
                if use_iterative and base_mode == "Hybrid":
                    answer, sources, refined, timings = rag_answer_iterative_timed(
                        q,
                        bm25_k=bm25_k,
                        vec_k=vec_k,
                        merge_k=merge_k,
                        rerank_k=rerank_k,
                        refine_n=refine_n,
                        use_rerank=use_rerank,
                        sarcastic_mode=sarcastic_mode,
                    )
                else:
                    answer, sources, timings = rag_answer_timed(
                        q,
                        retriever_mode=base_mode,
                        bm25_k=bm25_k,
                        vec_k=vec_k,
                        merge_k=merge_k,
                        rerank_k=rerank_k,
                        use_rerank=use_rerank,
                        sarcastic_mode=sarcastic_mode,
                    )

            conf = compute_confidence(sources)
            cov = source_coverage(sources)

            with colA:
                st.subheader("Answer")
                st.write(answer)

                st.subheader("Sources Used")
                for i, s in enumerate(sources, 1):
                    st.markdown(
                        f"**Source {i}** | chunk_id={s['chunk_id']} | doc_id={s['doc_id']} | `{s['source']}`"
                    )
                    st.code(s["text"][:850])

            with colB:
                if refined:
                    st.subheader("Refined Queries Used")
                    st.code(json.dumps(refined, indent=2), language="json")

                st.subheader("Confidence")
                st.write(f"**{conf['label']}**  (score: `{conf['confidence']}`)")
                st.caption(conf["reason"])
                st.progress(min(1.0, max(0.0, conf["confidence"])) )

                st.subheader("Source Coverage")
                _alt_bar_from_series(pd.Series(cov), title="Source Coverage", x_label="Retriever", y_label="Count")

                st.subheader("Latency Breakdown (seconds)")
                if timings:
                    st.code(json.dumps(timings, indent=2), language="json")
                    st.subheader("Latency Chart")
                    _alt_bar_from_dict(timings, title="Latency (sec)")

                st.info("Tuning tips: Reduce candidates for speed. Increase candidates / rerank_k for quality.")

# ----------------------------
# Tab 2: Evaluation
# ----------------------------
with tab2:
    st.subheader("Retriever Comparison (Hit@K, MRR)")
    st.write("Includes BM25, Vector, Hybrid, and Hybrid + Re-rank.")

    if st.button("Run Evaluation"):
        with st.spinner("Evaluating..."):
            df = run_eval()

        st.dataframe(df, use_container_width=True)

        hit_cols = [c for c in df.columns if c.startswith("Hit@")]
        if hit_cols:
            st.markdown("### Hit@K Comparison")
            df_hit = df.melt(id_vars=["Retriever"], value_vars=hit_cols, var_name="K", value_name="Hit")
            chart = (
                alt.Chart(df_hit)
                .mark_bar()
                .encode(
                    x=alt.X("K:N"),
                    y=alt.Y("Hit:Q"),
                    column=alt.Column("Retriever:N"),
                    tooltip=["Retriever", "K", "Hit"],
                )
                .properties(height=240)
            )
            st.altair_chart(_apply_dark_config(chart), use_container_width=True)

        st.markdown("### MRR Comparison")
        df_mrr = df[["Retriever", "MRR"]].copy()
        _alt_bar_df(df_mrr, "Retriever", "MRR", title="MRR")

    st.info("Tip: Expand `data/eval/questions.jsonl` to 30–100 questions for stable metrics.")

# ----------------------------
# Tab 3: Hallucination Test + Cache Speed
# ----------------------------
with tab3:
    st.subheader("Hallucination Test Dashboard")
    st.write(
        "This runs a small test suite containing answerable + out-of-scope questions. "
        "We expect the system to refuse when sources don't support an answer."
    )

    with st.expander("See test questions"):
        for i, tcase in enumerate(DEFAULT_TESTS, 1):
            st.markdown(f"**{i}.** {tcase.question}  \nExpected: `{tcase.expected}`  \n_{tcase.notes}_")

    colX, colY = st.columns([0.65, 0.35])
    with colX:
        run_iterative = st.checkbox("Run Iterative RAG for the suite", value=True)
    with colY:
        suite_refine_n = st.number_input("Refine N (suite)", min_value=1, max_value=5, value=3, step=1)

    if st.button("Run Hallucination Suite"):
        with st.spinner("Running suite... this may take time because it calls the LLM multiple times."):
            df, summary = run_hallucination_suite(
                rag_fn=rag_answer_iterative_timed if run_iterative else rag_answer_timed,
                tests=DEFAULT_TESTS,
                bm25_k=bm25_k,
                vec_k=vec_k,
                merge_k=merge_k,
                rerank_k=rerank_k,
                iterative=run_iterative,
                refine_n=int(suite_refine_n),
            )

        st.success("Suite completed.")
        st.markdown("### Summary")
        st.write(summary)

        st.markdown("### Results Table")
        st.dataframe(df, use_container_width=True)

        st.markdown("### Charts")
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("Passed vs Failed")
            _alt_bar_from_series(df["Passed"].value_counts(), title="Passed vs Failed", x_label="Passed", y_label="Count")

        with c2:
            st.markdown("Likely Hallucinations")
            _alt_bar_from_series(
                df["LikelyHallucination"].value_counts(),
                title="Likely Hallucinations",
                x_label="LikelyHallucination",
                y_label="Count",
            )

        c3, c4 = st.columns(2)

        with c3:
            st.markdown("Source Overlap per Question")
            if "SourceOverlap" in df.columns:
                df_so = df[["Question", "SourceOverlap"]].copy()
                df_so["Question"] = df_so["Question"].astype(str).str.slice(0, 40) + "…"
                _alt_bar_df(df_so, "Question", "SourceOverlap", title="Source Overlap")

        with c4:
            st.markdown("Latency per Question (sec)")
            if df.get("TotalLatencySec") is not None and df["TotalLatencySec"].notna().any():
                df_lat = df[["Question", "TotalLatencySec"]].copy()
                df_lat["Question"] = df_lat["Question"].astype(str).str.slice(0, 40) + "…"
                _alt_bar_df(df_lat, "Question", "TotalLatencySec", title="Latency (sec)")
            else:
                st.info("Latency values not available for some runs.")

        st.info(
            "Interpretation: A likely hallucination is flagged when the question was expected to be REFUSED, "
            "but the model did NOT refuse AND evidence overlap is low."
        )

    st.divider()
    st.subheader("Cache Speedup Benchmark (Cold vs Warm)")

    bench_q = st.text_input(
        "Benchmark Query",
        value="Explain Rayleigh scattering and why the sky appears blue.",
        help="This query will run twice: cold (cache cleared) then warm (cached).",
        key="bench_query",
    )

    bench_iter = st.checkbox("Use Iterative RAG for benchmark", value=True, key="bench_iter")
    bench_refine = st.slider("Refine N (benchmark)", 1, 5, 3, 1, key="bench_refine")

    if st.button("Benchmark Cache Speedup"):
        if not bench_q.strip():
            st.warning("Please enter a benchmark query.")
            st.stop()

        with st.spinner("Running cold → warm benchmark..."):
            cold, warm = benchmark_cache(
                rag_fn=rag_answer_iterative_timed if bench_iter else rag_answer_timed,
                question=bench_q,
                bm25_k=bm25_k,
                vec_k=vec_k,
                merge_k=merge_k,
                rerank_k=rerank_k,
                iterative=bench_iter,
                refine_n=int(bench_refine),
            )

        st.success("Benchmark completed.")

        df_bench = pd.DataFrame([cold, warm])
        st.dataframe(df_bench, use_container_width=True)

        st.markdown("Speed Comparison")
        if {"Run", "WallClockSec", "PipelineLatencySec"}.issubset(df_bench.columns):
            df_plot = df_bench[["Run", "WallClockSec", "PipelineLatencySec"]].copy()
            df_plot = df_plot.melt(id_vars=["Run"], var_name="Metric", value_name="Seconds")
            chart = (
                alt.Chart(df_plot)
                .mark_bar()
                .encode(
                    x=alt.X("Metric:N"),
                    y=alt.Y("Seconds:Q"),
                    column=alt.Column("Run:N"),
                    tooltip=["Run", "Metric", "Seconds"],
                )
                .properties(height=240)
            )
            st.altair_chart(_apply_dark_config(chart), use_container_width=True)

        if isinstance(warm, dict) and "Speedup_%" in warm:
            st.metric("Speedup (%)", f"{warm['Speedup_%']}%")

        st.info("Cold run includes index/model loading. Warm run uses cached resources and should be faster.")
