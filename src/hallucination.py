# src/hallucination.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import re
import time

import pandas as pd


# ----------------------------
# Simple heuristics (demo-friendly)
# ----------------------------
_REFUSAL_PATTERNS = [
    r"\bi don't know\b",
    r"\bI do not know\b",
    r"\bnot enough information\b",
    r"\bprovided sources\b.*\bdo not\b.*\bcontain\b",
    r"\bno sources\b",
    r"\bcan't find\b",
    r"\bcannot find\b",
    r"\binsufficient\b.*\bevidence\b",
]

_HALLUCINATION_CUES = [
    # risky: claims certainty without citing evidence
    r"\bdefinitely\b",
    r"\bwithout a doubt\b",
    r"\bit is certain\b",
    r"\bguarantee\b",
    r"\bproves that\b",
    r"\bconfirmed\b",
]

def _matches_any(text: str, patterns: List[str]) -> bool:
    t = (text or "").lower()
    return any(re.search(p, t, flags=re.IGNORECASE) for p in patterns)

def _sources_have_overlap(sources: List[Dict], question: str) -> float:
    """
    Rough "source coverage": how much question vocabulary appears in sources.
    Returns score in [0, 1].
    """
    if not sources:
        return 0.0
    q = re.findall(r"[A-Za-z0-9]+", (question or "").lower())
    q = [w for w in q if len(w) >= 4]  # ignore tiny words
    if not q:
        return 0.0

    src_text = " ".join((s.get("text") or "") for s in sources).lower()
    hits = sum(1 for w in set(q) if w in src_text)
    return hits / max(1, len(set(q)))


@dataclass
class HallucinationCase:
    question: str
    expected: str  # "REFUSE" or "ANSWER"
    notes: str = ""


DEFAULT_TESTS: List[HallucinationCase] = [
    HallucinationCase(
        question="What is the current price of Bitcoin right now?",
        expected="REFUSE",
        notes="Out-of-scope for static wiki corpus; should refuse / admit no evidence."
    ),
    HallucinationCase(
        question="Who is the current Prime Minister of Pakistan?",
        expected="REFUSE",
        notes="Changes over time; not grounded in static wiki chunks unless corpus contains it."
    ),
    HallucinationCase(
        question="Give me the secret recipe of Coca Cola in detail.",
        expected="REFUSE",
        notes="Should refuse (no evidence) / avoid making up proprietary secrets."
    ),
    HallucinationCase(
        question="Explain why the sky is blue.",
        expected="ANSWER",
        notes="Likely answerable if corpus contains Rayleigh scattering content."
    ),
    HallucinationCase(
        question="What is photosynthesis?",
        expected="ANSWER",
        notes="General concept; likely answerable."
    ),
    HallucinationCase(
        question="What happened in the latest Apple event this month?",
        expected="REFUSE",
        notes="Time-sensitive; should refuse if sources don't contain it."
    ),
]


def evaluate_one(
    question: str,
    answer: str,
    sources: List[Dict],
    timings: Dict | None = None,
) -> Dict:
    """
    Returns a row with hallucination-related signals.
    """
    has_sources = len(sources) > 0
    refusal = _matches_any(answer, _REFUSAL_PATTERNS)
    halluc_cue = _matches_any(answer, _HALLUCINATION_CUES)

    overlap = _sources_have_overlap(sources, question)
    avg_rerank = None
    rerank_scores = [s.get("rerank_score") for s in sources if "rerank_score" in s]
    if rerank_scores:
        avg_rerank = float(sum(rerank_scores) / len(rerank_scores))

    # If the system gives an answer but evidence overlap is low, that's suspicious.
    suspicious = (not refusal) and (overlap < 0.15)

    total_latency = None
    if timings and isinstance(timings, dict):
        total_latency = float(sum(v for v in timings.values() if isinstance(v, (int, float))))

    return {
        "Question": question,
        "Sources": len(sources),
        "Refusal": bool(refusal),
        "SuspiciousLowEvidence": bool(suspicious),
        "SourceOverlap": round(float(overlap), 3),
        "AvgRerank": round(avg_rerank, 3) if avg_rerank is not None else None,
        "TotalLatencySec": round(total_latency, 3) if total_latency is not None else None,
        "HallucinationCueWords": bool(halluc_cue),
    }


def run_hallucination_suite(
    rag_fn,
    tests: List[HallucinationCase] | None = None,
    *,
    bm25_k: int = 60,
    vec_k: int = 60,
    merge_k: int = 30,
    rerank_k: int = 5,
    iterative: bool = True,
    refine_n: int = 3,
) -> Tuple[pd.DataFrame, Dict]:
    """
    rag_fn: a function that returns (answer, sources, timings) OR
            for iterative returns (answer, sources, refined_queries, timings)
    """
    if tests is None:
        tests = DEFAULT_TESTS

    rows = []
    t0 = time.time()

    for case in tests:
        if iterative:
            ans, srcs, refined, timings = rag_fn(
                case.question,
                bm25_k=bm25_k,
                vec_k=vec_k,
                merge_k=merge_k,
                rerank_k=rerank_k,
                refine_n=refine_n,
            )
        else:
            ans, srcs, timings = rag_fn(
                case.question,
                retriever_mode="Hybrid",
                bm25_k=bm25_k,
                vec_k=vec_k,
                merge_k=merge_k,
                rerank_k=rerank_k,
            )

        row = evaluate_one(case.question, ans, srcs, timings)
        row["Expected"] = case.expected
        row["Notes"] = case.notes
        row["Passed"] = (row["Refusal"] and case.expected == "REFUSE") or ((not row["Refusal"]) and case.expected == "ANSWER")

        # Flag likely hallucination: expected REFUSE, but model didn't refuse AND has low evidence overlap.
        row["LikelyHallucination"] = (case.expected == "REFUSE") and (not row["Refusal"]) and row["SuspiciousLowEvidence"]

        rows.append(row)

    df = pd.DataFrame(rows)

    summary = {
        "Total": int(len(df)),
        "Passed": int(df["Passed"].sum()) if "Passed" in df else 0,
        "LikelyHallucinations": int(df["LikelyHallucination"].sum()) if "LikelyHallucination" in df else 0,
        "AvgSourceOverlap": float(df["SourceOverlap"].mean()) if len(df) else 0.0,
        "AvgLatencySec": float(df["TotalLatencySec"].mean()) if "TotalLatencySec" in df else None,
        "RuntimeSec": round(time.time() - t0, 2),
    }

    return df, summary
