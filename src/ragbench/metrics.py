"""检索指标：Recall@k / MRR@k / nDCG@k / HitRate@k。

全部按「qrels 里 score>0 即正例」的二值相关性计算，与 C-MTEB 的口径一致，
这样结论可以和公开榜单横向对照。多正例样本一律取平均，不做特殊加权。
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

import numpy as np


def recall_at_k(ranked: Sequence[str], gold: set[str], k: int) -> float:
    """命中正例数 / 正例总数。"""
    if not gold:
        return 0.0
    top = ranked[:k]
    return len(set(top) & gold) / len(gold)


def precision_at_k(ranked: Sequence[str], gold: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(set(ranked[:k]) & gold) / k


def hit_rate_at_k(ranked: Sequence[str], gold: set[str], k: int) -> float:
    """top-k 里至少命中一个正例即为 1（对应「能不能答」）。"""
    return 1.0 if set(ranked[:k]) & gold else 0.0


def mrr_at_k(ranked: Sequence[str], gold: set[str], k: int) -> float:
    """第一个正例的倒数排名。"""
    for i, pid in enumerate(ranked[:k], start=1):
        if pid in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], gold: set[str], k: int) -> float:
    """二值相关性的 nDCG@k。"""
    if not gold:
        return 0.0
    dcg = 0.0
    for i, pid in enumerate(ranked[:k], start=1):
        if pid in gold:
            dcg += 1.0 / math.log2(i + 1)
    ideal_n = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_n + 1))
    return dcg / idcg if idcg > 0 else 0.0


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    return float(np.mean(vals)) if vals else 0.0


def evaluate_rankings(
    rankings: Mapping[str, Sequence[str]],
    gold: Mapping[str, set[str]],
    ks: Sequence[int] = (1, 5, 10),
) -> dict[str, float]:
    """对一组查询的排名结果算指标均值。

    ``rankings`` 只应包含有正例的 qid，否则空正例会拉低指标（口径须一致）。
    """
    missing = [q for q in rankings if q not in gold]
    if missing:
        raise KeyError(f"这些 qid 没有 qrels：{missing[:5]}")

    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"recall@{k}"] = _mean(recall_at_k(r, gold[q], k) for q, r in rankings.items())
        metrics[f"mrr@{k}"] = _mean(mrr_at_k(r, gold[q], k) for q, r in rankings.items())
        metrics[f"ndcg@{k}"] = _mean(ndcg_at_k(r, gold[q], k) for q, r in rankings.items())
        metrics[f"hit@{k}"] = _mean(hit_rate_at_k(r, gold[q], k) for q, r in rankings.items())
    return metrics


def evaluate_with_details(
    rankings: Mapping[str, Sequence[str]],
    gold: Mapping[str, set[str]],
    ks: Sequence[int] = (1, 5, 10),
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """返回总体指标 + 每条查询的明细（用于分析失败案例）。"""
    overall = evaluate_rankings(rankings, gold, ks)
    detail: dict[str, dict[str, float]] = {}
    for qid, ranked in rankings.items():
        g = gold[qid]
        detail[qid] = {f"recall@{k}": recall_at_k(ranked, g, k) for k in ks} | {
            f"mrr@{k}": mrr_at_k(ranked, g, k) for k in ks
        }
    return overall, detail
