# src/eval_compare.py
import json
from pathlib import Path
from typing import Callable, Dict, List

import pandas as pd

from src.retrieval import bm25_retrieve, vector_retrieve, hybrid_retrieve, rerank

EVAL_PATH = Path("data/eval/questions.jsonl")


def _hit_at_k(texts: List[str], keywords: List[str], k: int) -> bool:
    joined = " ".join(texts[:k]).lower()
    return any(kw.lower() in joined for kw in keywords)


def _reciprocal_rank(texts: List[str], keywords: List[str]) -> float:
    for i, t in enumerate(texts, start=1):
        tl = t.lower()
        for kw in keywords:
            if kw.lower() in tl:
                return 1.0 / i
    return 0.0


def _eval_retriever(
    name: str,
    retrieve_fn: Callable[[str, int], List[Dict]],
    k_values=(1, 3, 5, 10),
    topn: int = 20,
) -> Dict:
    hits = {k: [] for k in k_values}
    mrrs = []

    if not EVAL_PATH.exists():
        raise FileNotFoundError(
            f"Missing {EVAL_PATH}. Create it as jsonl with fields: "
            f'{{"question": "...", "answer_keywords": ["..."]}}'
        )

    with EVAL_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            q = rec["question"]
            kws = rec["answer_keywords"]

            results = retrieve_fn(q, topn)
            texts = [r["text"] for r in results]

            for k in k_values:
                hits[k].append(_hit_at_k(texts, kws, k))
            mrrs.append(_reciprocal_rank(texts, kws))

    row = {"Retriever": name}
    for k in k_values:
        row[f"Hit@{k}"] = sum(hits[k]) / max(1, len(hits[k]))
    row["MRR"] = sum(mrrs) / max(1, len(mrrs))
    return row


# ---- wrappers ----
def _bm25_fn(q: str, topn: int) -> List[Dict]:
    return bm25_retrieve(q, k=topn)


def _vec_fn(q: str, topn: int) -> List[Dict]:
    return vector_retrieve(q, k=topn)


def _hyb_fn(q: str, topn: int) -> List[Dict]:
    return hybrid_retrieve(q, bm25_k=60, vec_k=60, final_k=topn)


def _hyb_rerank_fn(q: str, topn: int) -> List[Dict]:
    # retrieve more, rerank, then keep topn
    cands = hybrid_retrieve(q, bm25_k=60, vec_k=60, final_k=max(50, topn))
    top = rerank(q, cands, top_k=topn)
    return top


def run_eval() -> pd.DataFrame:
    rows = [
        _eval_retriever("BM25", _bm25_fn),
        _eval_retriever("Vector", _vec_fn),
        _eval_retriever("Hybrid", _hyb_fn),
        _eval_retriever("Hybrid + Re-rank", _hyb_rerank_fn),
    ]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = run_eval()
    print(df.to_string(index=False))
