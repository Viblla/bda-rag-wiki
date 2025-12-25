# src/rag_answer.py
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor
import re

from src.llm import call_ollama
from src.timing import Timer
from src.query_refine import refine_queries
from src.retrieval import bm25_retrieve, vector_retrieve, hybrid_retrieve, rerank

from src.refusal import refusal_message  # NEW


def build_prompt(query: str, sources: List[Dict], sarcastic: bool = False) -> str:
    """
    Build a grounded prompt: answer only using the provided sources.
    """
    context_blocks = []
    for i, s in enumerate(sources, 1):
        text = (s.get("text") or "").strip()
        meta = f"[Source {i} | chunk_id={s.get('chunk_id')} | doc_id={s.get('doc_id')} | {s.get('source')}]"
        context_blocks.append(f"{meta}\n{text}")

    context = "\n\n".join(context_blocks)

    if sarcastic:
        prompt = f"""
You are a sarcastic, witty Wikipedia expert. Answer accurately but with dry humor.

FORMATTING RULES:
1. Use emojis for sarcasm: 😒 🙄 😤 🤦 💀 😑 (NOT *sigh* or *eye roll*)
2. Put the KEY FACTUAL ANSWER on its own line, wrapped in [[double brackets]] like this:
   [[The actual factual answer goes here]]
3. Structure: Sarcastic intro → [[KEY ANSWER]] → Snarky outro

CONTENT RULES:
- Use ONLY the provided sources
- Be concise and entertaining
- If sources lack info: "😒 Even my sources don't have this."

Question: {query}

Sources:
{context}

Answer:
""".strip()
    else:
        prompt = f"""
You are a retrieval-augmented assistant. Answer the user's question using ONLY the provided sources.

Rules:
- If the sources do not contain enough information, say: "I don't know based on the provided sources."
- Do not invent facts or citations.
- Keep the answer concise and clear.
- If helpful, use bullet points.

Question: {query}

Sources:
{context}

Answer:
""".strip()

    return prompt


def _should_refuse(sources: List[Dict], min_chars: int = 120) -> bool:
    """
    Very safe refusal check:
    - refuse if no sources
    - refuse if all source texts are tiny/empty (often means retrieval failed)
    """
    if not sources:
        return True

    total = 0
    for s in sources:
        total += len((s.get("text") or "").strip())

    return total < min_chars


def _is_out_of_scope_query(query: str) -> bool:
    """
    Heuristic detector for questions that typically require live/real-time data
    or are not answerable from static Wikipedia chunks (our corpus).
    This is intentionally conservative (low risk of false positives).
    """
    q = query.strip().lower()

    patterns = [
        r"\b(current|today|right now|latest|live|real[- ]?time|as of now)\b",
        r"\b(price|btc|bitcoin|ethereum|stock|share price|exchange rate|rate)\b",
        r"\b(score|match|who won|yesterday|last night|this morning)\b",
        r"\bweather|temperature\b",
        r"\btrending\b",
    ]
    return any(re.search(p, q) for p in patterns)


def rag_answer_timed(
    query: str,
    retriever_mode: str = "Hybrid",
    bm25_k: int = 60,
    vec_k: int = 60,
    merge_k: int = 30,
    rerank_k: int = 5,
    use_rerank: bool = True,
    sarcastic_mode: bool = False,
) -> Tuple[str, List[Dict], Dict[str, float]]:
    """
    Non-iterative RAG with timing.
    retriever_mode: "Hybrid" | "BM25" | "Vector"
    Returns: (answer, top_sources, timings_dict)
    """
    timer = Timer()

    # ✅ NEW: if sarcastic_mode ON and query is clearly out-of-scope, refuse immediately
    if sarcastic_mode and _is_out_of_scope_query(query):
        timer.start("ollama_llm")
        answer = refusal_message(sarcastic=True)
        timer.stop("ollama_llm")
        return answer, [], timer.summary()

    # ---- Retrieve ----
    if retriever_mode == "BM25":
        timer.start("bm25_retrieve")
        cands = bm25_retrieve(query, k=merge_k)
        timer.stop("bm25_retrieve")

    elif retriever_mode == "Vector":
        timer.start("vector_retrieve")
        cands = vector_retrieve(query, k=merge_k)
        timer.stop("vector_retrieve")

    else:
        timer.start("hybrid_retrieve")
        cands = hybrid_retrieve(query, bm25_k=bm25_k, vec_k=vec_k, final_k=merge_k)
        timer.stop("hybrid_retrieve")

    # ---- Re-rank (optional) ----
    if use_rerank:
        timer.start("rerank")
        top = rerank(query, cands, top_k=rerank_k)
        timer.stop("rerank")
    else:
        timer.start("rerank")
        top = cands[:rerank_k]
        timer.stop("rerank")

    # refusal gate BEFORE LLM (plain or sarcastic depending on toggle)
    if _should_refuse(top):
        timer.start("ollama_llm")
        answer = refusal_message(sarcastic=sarcastic_mode)
        timer.stop("ollama_llm")
        return answer, [], timer.summary()

    # ---- LLM ----
    timer.start("ollama_llm")
    prompt = build_prompt(query, top, sarcastic=sarcastic_mode)
    answer = call_ollama(prompt)
    timer.stop("ollama_llm")

    return answer, top, timer.summary()


def rag_answer_iterative_timed(
    query: str,
    bm25_k: int = 60,
    vec_k: int = 60,
    merge_k: int = 30,
    rerank_k: int = 5,
    refine_n: int = 3,
    use_rerank: bool = True,
    sarcastic_mode: bool = False,
) -> Tuple[str, List[Dict], List[str], Dict[str, float]]:
    """
    Iterative RAG (Query Refinement) with full timing.
    Returns: (answer, top_sources, refined_queries, timings_dict)
    """
    timer = Timer()

    # ✅ NEW: if sarcastic_mode ON and query is clearly out-of-scope, refuse immediately
    if sarcastic_mode and _is_out_of_scope_query(query):
        timer.start("ollama_llm")
        answer = refusal_message(sarcastic=True)
        timer.stop("ollama_llm")
        return answer, [], [], timer.summary()

    # 1) Base retrieval
    timer.start("hybrid_retrieve_0")
    base_cands = hybrid_retrieve(query, bm25_k=bm25_k, vec_k=vec_k, final_k=merge_k)
    timer.stop("hybrid_retrieve_0")

    # 2) Query refinement
    timer.start("query_refine")
    refined = refine_queries(query, n=refine_n)
    timer.stop("query_refine")

    # 3) Retrieval on refined queries - PARALLELIZED for speed
    all_cands = list(base_cands)
    timer.start("hybrid_retrieve_refined_total")
    
    # Use ThreadPoolExecutor to parallelize retrieval across refined queries
    def _retrieve_one(rq: str) -> List[Dict]:
        return hybrid_retrieve(rq, bm25_k=bm25_k, vec_k=vec_k, final_k=merge_k)
    
    if refined:
        with ThreadPoolExecutor(max_workers=min(len(refined), 4)) as executor:
            results = list(executor.map(_retrieve_one, refined))
            for res in results:
                all_cands.extend(res)
    
    timer.stop("hybrid_retrieve_refined_total")

    # 4) Deduplicate by chunk_id
    timer.start("dedup_merge")
    seen = {}
    for c in all_cands:
        seen[c["chunk_id"]] = c
    merged = list(seen.values())
    timer.stop("dedup_merge")

    # 5) Re-rank (optional)
    if use_rerank:
        timer.start("rerank")
        top = rerank(query, merged, top_k=rerank_k)
        timer.stop("rerank")
    else:
        timer.start("rerank")
        top = merged[:rerank_k]
        timer.stop("rerank")

    # refusal gate BEFORE LLM
    if _should_refuse(top):
        timer.start("ollama_llm")
        answer = refusal_message(sarcastic=sarcastic_mode)
        timer.stop("ollama_llm")
        return answer, [], refined, timer.summary()

    # 6) LLM answer
    timer.start("ollama_llm")
    prompt = build_prompt(query, top, sarcastic=sarcastic_mode)
    answer = call_ollama(prompt)
    timer.stop("ollama_llm")

    return answer, top, refined, timer.summary()
