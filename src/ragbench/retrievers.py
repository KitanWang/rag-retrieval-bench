"""五种检索策略，统一实现 :class:`Retriever` 协议。

| 方案 | 说明 |
| :--- | :--- |
| ``dense`` | 只用稠密向量（BGE-M3 → Milvus） |
| ``bm25`` | 只用稀疏词法（jieba + BM25） |
| ``hybrid`` | 稠密 + 稀疏，RRF 融合 |
| ``hybrid_rerank`` | hybrid 召回 top-N 后用 cross-encoder 重排 |
| ``router`` | 按查询特征在 bm25 / hybrid / hybrid_rerank 之间选路 |

所有实现共享同一份语料与同一份评测集，且**都返回文档 id**，
这样指标计算与生成评测可以完全复用。
"""

from __future__ import annotations

import logging
import time
from typing import Protocol, Sequence

import numpy as np

from ragbench.config import Config
from ragbench.embedding import Embedder, Reranker
from ragbench.index import VectorIndex
from ragbench.sparse import BM25Index
from ragbench.types import Hit, RetrievalResult

logger = logging.getLogger(__name__)


class Retriever(Protocol):
    name: str

    def search(self, query: str, top_k: int = 10) -> list[Hit]: ...
    def search_batch(self, queries: Sequence[str], top_k: int = 10) -> list[RetrievalResult]: ...


# --------------------------------------------------------------------------- #
# 融合函数（纯函数，方便单测）
# --------------------------------------------------------------------------- #
def reciprocal_rank_fusion(
    rankings: list[list[Hit]], k: int = 60, weights: list[float] | None = None
) -> list[Hit]:
    """RRF：score = Σ w_i / (k + rank_i)。对分数尺度不敏感，天然适合异构检索融合。"""
    weights = weights or [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights 与 rankings 数量不一致")
    fused: dict[str, float] = {}
    for hits, w in zip(rankings, weights):
        for rank, hit in enumerate(hits, start=1):
            fused[hit.pid] = fused.get(hit.pid, 0.0) + w / (k + rank)
    return sorted(
        (Hit(pid, score) for pid, score in fused.items()), key=lambda h: -h.score
    )


def min_max_normalize(hits: list[Hit]) -> list[Hit]:
    """把一路检索的分数缩放到 [0,1]，用于加权融合。"""
    if not hits:
        return []
    scores = np.array([h.score for h in hits], dtype=np.float32)
    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo < 1e-9:
        return [Hit(h.pid, 1.0) for h in hits]
    return [Hit(h.pid, float((h.score - lo) / (hi - lo))) for h in hits]


def weighted_fusion(
    rankings: list[list[Hit]], weights: list[float], top_k: int
) -> list[Hit]:
    """归一化加权融合：score = Σ w_i * norm(score_i)。缺失的一路按 0 计。"""
    table: dict[str, float] = {}
    for hits, w in zip(rankings, weights):
        for h in min_max_normalize(hits):
            table[h.pid] = table.get(h.pid, 0.0) + w * h.score
    merged = sorted((Hit(pid, s) for pid, s in table.items()), key=lambda h: -h.score)
    return merged[:top_k]


# --------------------------------------------------------------------------- #
# 检索器实现
# --------------------------------------------------------------------------- #
class _BaseRetriever:
    name = "base"

    def __init__(self, cfg: Config, corpus_ids: list[str], corpus_texts: list[str]) -> None:
        self.cfg = cfg
        self.corpus_ids = corpus_ids
        self.corpus_texts = corpus_texts
        self.id_to_idx = {pid: i for i, pid in enumerate(corpus_ids)}

    @property
    def rcfg(self):
        return self.cfg.retriever

    def _finalize(self, scored: list[Hit], top_k: int) -> list[Hit]:
        return scored[:top_k]

    def search_batch(self, queries: Sequence[str], top_k: int = 10) -> list[RetrievalResult]:
        results = []
        for qid, q in enumerate(queries):
            t0 = time.perf_counter()
            hits = self.search(q, top_k=top_k)
            dt = (time.perf_counter() - t0) * 1000
            results.append(RetrievalResult(qid=str(qid), hits=hits, latency_ms=dt))
        return results


class DenseRetriever(_BaseRetriever):
    """朴素向量检索。"""

    name = "dense"

    def __init__(self, cfg: Config, corpus_ids, corpus_texts, embedder: Embedder, index: VectorIndex):
        super().__init__(cfg, corpus_ids, corpus_texts)
        self.embedder = embedder
        self.index = index
        self._qcache: dict[str, np.ndarray] = {}

    def _encode_query(self, query: str) -> np.ndarray:
        if query not in self._qcache:
            self._qcache[query] = self.embedder.encode_queries([query])[0]
        return self._qcache[query]

    def search(self, query: str, top_k: int = 10) -> list[Hit]:
        vec = self._encode_query(query)
        raw = self.index.search(vec, top_k=top_k)
        return [Hit(h.idx, h.score) for h in raw]


class BM25Retriever(_BaseRetriever):
    """纯稀疏词法检索。"""

    name = "bm25"

    def __init__(self, cfg: Config, corpus_ids, corpus_texts, bm25: BM25Index):
        super().__init__(cfg, corpus_ids, corpus_texts)
        self.bm25 = bm25

    def search(self, query: str, top_k: int = 10) -> list[Hit]:
        hits = self.bm25.search(query, top_k=top_k)
        return [Hit(self.corpus_ids[h.idx], h.score) for h in hits]


class HybridRetriever(_BaseRetriever):
    """稠密 + 稀疏融合。"""

    name = "hybrid"

    def __init__(
        self,
        cfg: Config,
        corpus_ids,
        corpus_texts,
        dense: DenseRetriever,
        sparse: BM25Retriever,
    ):
        super().__init__(cfg, corpus_ids, corpus_texts)
        self.dense = dense
        self.sparse = sparse

    def search(self, query: str, top_k: int = 10) -> list[Hit]:
        pool = max(top_k * 10, self.rcfg.rerank_top_k)
        d_hits = self.dense.search(query, top_k=pool)
        s_hits = self.sparse.search(query, top_k=pool)
        if self.rcfg.fusion == "weighted":
            return weighted_fusion(
                [d_hits, s_hits], [self.rcfg.hybrid_alpha, 1 - self.rcfg.hybrid_alpha], top_k
            )
        return reciprocal_rank_fusion([d_hits, s_hits], k=self.rcfg.rrf_k)[:top_k]


class HybridRerankRetriever(_BaseRetriever):
    """混合召回 + cross-encoder 重排。"""

    name = "hybrid_rerank"

    def __init__(
        self,
        cfg: Config,
        corpus_ids,
        corpus_texts,
        hybrid: HybridRetriever,
        reranker: Reranker,
    ):
        super().__init__(cfg, corpus_ids, corpus_texts)
        self.hybrid = hybrid
        self.reranker = reranker

    def search(self, query: str, top_k: int = 10) -> list[Hit]:
        recall_k = max(top_k, self.rcfg.rerank_top_k)
        cand = self.hybrid.search(query, top_k=recall_k)
        if not cand:
            return []
        docs = [self.corpus_texts[self.id_to_idx[h.pid]] for h in cand]
        scores = self.reranker.rerank(query, docs)
        rescored = [Hit(h.pid, float(s)) for h, s in zip(cand, scores)]
        rescored.sort(key=lambda h: -h.score)
        return rescored[:top_k]


# --------------------------------------------------------------------------- #
# 策略路由
# --------------------------------------------------------------------------- #
class QueryRouter:
    """按查询的**可观测特征**选路，不训练分类器。

    三条规则（都能在报告里用一句话解释清楚）：

    1. 含型号/编号/数字 → 稀疏检索更强（精确匹配），直接走 ``bm25``；
    2. 长句、口语化描述 → 稠密语义更强，但为避免漏掉关键词，
       走 ``hybrid`` 更稳；
    3. 其余（短关键词查询）→ 稠密与稀疏都不可靠，走 ``hybrid_rerank``
       用 cross-encoder 做精排，代价是延迟更高。
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    @staticmethod
    def features(query: str) -> dict[str, float]:
        import re

        n = max(1, len(query))
        code_chars = len(re.findall(r"[A-Za-z0-9\-_./]", query))
        chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", query))
        return {
            "length": float(n),
            "code_ratio": code_chars / n,
            "chinese_ratio": chinese_chars / n,
            "has_digits": float(bool(re.search(r"\d", query))),
        }

    def route(self, query: str) -> str:
        f = self.features(query)
        rcfg = self.cfg.router
        if f["code_ratio"] >= rcfg.code_like_ratio and f["has_digits"]:
            return "bm25"
        if f["length"] >= rcfg.long_query_chars:
            return "hybrid"
        return "hybrid_rerank"


class RouterRetriever(_BaseRetriever):
    """策略路由检索器：把查询分派给不同检索路径。"""

    name = "router"

    def __init__(
        self,
        cfg: Config,
        corpus_ids,
        corpus_texts,
        sparse: BM25Retriever,
        hybrid: HybridRetriever,
        hybrid_rerank: HybridRerankRetriever,
    ):
        super().__init__(cfg, corpus_ids, corpus_texts)
        self.sparse = sparse
        self.hybrid = hybrid
        self.hybrid_rerank = hybrid_rerank
        self.router = QueryRouter(cfg)
        self._routes: dict[str, str] = {}

    def search(self, query: str, top_k: int = 10) -> list[Hit]:
        route = self.router.route(query)
        self._routes[query] = route
        table = {
            "bm25": self.sparse,
            "hybrid": self.hybrid,
            "hybrid_rerank": self.hybrid_rerank,
        }
        return table[route].search(query, top_k=top_k)

    def search_batch(self, queries: Sequence[str], top_k: int = 10) -> list[RetrievalResult]:
        out = []
        for qid, q in enumerate(queries):
            route = self.router.route(q)
            t0 = time.perf_counter()
            hits = self.search(q, top_k=top_k)
            dt = (time.perf_counter() - t0) * 1000
            out.append(RetrievalResult(qid=str(qid), hits=hits, latency_ms=dt, route=route))
        return out

    @property
    def routes(self) -> dict[str, str]:
        return dict(self._routes)
