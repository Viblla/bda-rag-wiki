# src/confidence.py
from typing import Dict, List
import math


def compute_confidence(sources: List[Dict]) -> Dict:
    """
    Computes a confidence score and qualitative label
    based ONLY on retrieved evidence (no LLM guessing).
    """

    if not sources:
        return {
            "confidence": 0.0,
            "label": "Very Low",
            "reason": "No sources retrieved",
        }

    # ---- 1) Rerank score signal ----
    rerank_scores = [s.get("rerank_score", 0.0) for s in sources]
    max_rerank = max(rerank_scores) if rerank_scores else 0.0
    avg_rerank = sum(rerank_scores) / max(1, len(rerank_scores))

    # Normalize roughly (cross-encoders often ~[-10, +10])
    rerank_signal = 1 / (1 + math.exp(-avg_rerank))

    # ---- 2) Evidence count signal ----
    evidence_count = len(sources)
    evidence_signal = min(1.0, evidence_count / 5.0)

    # ---- 3) Source diversity signal ----
    unique_docs = len(set(s.get("doc_id") for s in sources))
    diversity_signal = min(1.0, unique_docs / max(1, evidence_count))

    # ---- Final confidence ----
    confidence = (
        0.5 * rerank_signal +
        0.3 * evidence_signal +
        0.2 * diversity_signal
    )

    confidence = round(float(confidence), 3)

    if confidence >= 0.75:
        label = "High"
    elif confidence >= 0.5:
        label = "Medium"
    elif confidence >= 0.25:
        label = "Low"
    else:
        label = "Very Low"

    return {
        "confidence": confidence,
        "label": label,
        "reason": (
            f"avg rerank={avg_rerank:.2f}, "
            f"sources={evidence_count}, "
            f"unique docs={unique_docs}"
        ),
    }


def source_coverage(sources: List[Dict]) -> Dict[str, int]:
    """
    Counts how many retrieved chunks came from each retriever.
    Expects `source` field to contain 'bm25' or 'vector'.
    """
    coverage = {
        "bm25": 0,
        "vector": 0,
        "other": 0,
    }

    for s in sources:
        src = (s.get("source") or "").lower()
        if "bm25" in src:
            coverage["bm25"] += 1
        elif "vector" in src:
            coverage["vector"] += 1
        else:
            coverage["other"] += 1

    return coverage
