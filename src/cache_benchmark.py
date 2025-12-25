# src/cache_benchmark.py
import time
import streamlit as st
from typing import Dict, Tuple


def _sum_timings(timings: Dict) -> float:
    return round(sum(v for v in timings.values() if isinstance(v, (int, float))), 3)


def benchmark_cache(
    rag_fn,
    question: str,
    *,
    bm25_k: int,
    vec_k: int,
    merge_k: int,
    rerank_k: int,
    iterative: bool = True,
    refine_n: int = 3,
) -> Tuple[Dict, Dict]:
    """
    Runs the same query twice:
    1) Cold run (cache cleared)
    2) Warm run (cache populated)

    Returns:
        cold_stats, warm_stats
    """

    # -------- Cold run --------
    st.cache_resource.clear()
    st.cache_data.clear()

    t0 = time.time()

    if iterative:
        _, _, _, cold_timings = rag_fn(
            question,
            bm25_k=bm25_k,
            vec_k=vec_k,
            merge_k=merge_k,
            rerank_k=rerank_k,
            refine_n=refine_n,
        )
    else:
        _, _, cold_timings = rag_fn(
            question,
            retriever_mode="Hybrid",
            bm25_k=bm25_k,
            vec_k=vec_k,
            merge_k=merge_k,
            rerank_k=rerank_k,
        )

    cold_total = _sum_timings(cold_timings)
    cold_wall = round(time.time() - t0, 3)

    # -------- Warm run --------
    t1 = time.time()

    if iterative:
        _, _, _, warm_timings = rag_fn(
            question,
            bm25_k=bm25_k,
            vec_k=vec_k,
            merge_k=merge_k,
            rerank_k=rerank_k,
            refine_n=refine_n,
        )
    else:
        _, _, warm_timings = rag_fn(
            question,
            retriever_mode="Hybrid",
            bm25_k=bm25_k,
            vec_k=vec_k,
            merge_k=merge_k,
            rerank_k=rerank_k,
        )

    warm_total = _sum_timings(warm_timings)
    warm_wall = round(time.time() - t1, 3)

    speedup = round((cold_wall - warm_wall) / max(cold_wall, 1e-6) * 100, 2)

    cold_stats = {
        "Run": "Cold (No Cache)",
        "PipelineLatencySec": cold_total,
        "WallClockSec": cold_wall,
    }

    warm_stats = {
        "Run": "Warm (Cached)",
        "PipelineLatencySec": warm_total,
        "WallClockSec": warm_wall,
        "Speedup_%": speedup,
    }

    return cold_stats, warm_stats
