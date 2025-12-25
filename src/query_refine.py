# src/query_refine.py
import json
import re
from typing import List

from src.llm import call_ollama


def refine_queries(user_query: str, n: int = 3) -> List[str]:
    """
    Uses the LLM to rewrite the user's question into short keyword-style queries
    better suited for Wikipedia retrieval.

    Returns a list of refined queries (length <= n).
    """
    prompt = f"""
You are helping improve retrieval for a Wikipedia RAG system.
Rewrite the user's question into {n} short keyword-style search queries.

Rules:
- Keep each query <= 10 words.
- Use Wikipedia-style keywords (proper nouns, scientific terms).
- No explanations.
- Output ONLY a JSON list of strings.

User question: {user_query}
""".strip()

    out = call_ollama(prompt)

    # First try JSON list
    try:
        qs = json.loads(out)
        if isinstance(qs, list):
            qs = [str(q).strip() for q in qs if str(q).strip()]
            return qs[:n]
    except Exception:
        pass

    # Fallback: parse lines
    lines = [re.sub(r"^[\-\*\d\.\)]\s*", "", l).strip() for l in out.splitlines()]
    lines = [l for l in lines if l]
    return lines[:n]
