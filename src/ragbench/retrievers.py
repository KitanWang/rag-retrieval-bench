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
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from ragbench.config import Config
from ragbench.embedding import Embedder, Reranker
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
    for hits, w in zip(rankings, weights, strict=False):
        for rank, hit in enumerate(hits, start=1):
            fused[hit.pid] = fused.get(hit.pid, 0.0) + w / (k + rank)
    return sorted((Hit(pid, score) for pid, score in fused.items()), key=lambda h: -h.score)


def min_max_normalize(hits: list[Hit]) -> list[Hit]:
    """把一路检索的分数缩放到 [0,1]，用于加权融合。"""
    if not hits:
        return []
    scores = np.array([h.score for h in hits], dtype=np.float32)
    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo < 1e-9:
        return [Hit(h.pid, 1.0) for h in hits]
    return [Hit(h.pid, float((h.score - lo) / (hi - lo))) for h in hits]


def weighted_fusion(rankings: list[list[Hit]], weights: list[float], top_k: int) -> list[Hit]:
    """归一化加权融合：score = Σ w_i * norm(score_i)。缺失的一路按 0 计。"""
    table: dict[str, float] = {}
    for hits, w in zip(rankings, weights, strict=False):
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

    def __init__(self, cfg: Config, corpus_ids, corpus_texts, embedder: Embedder, index):
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
        return [Hit(pid, score) for pid, score in raw]


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
        return self.rerank_hits(query, cand, top_k)

    def rerank_hits(self, query: str, cand: list[Hit], top_k: int) -> list[Hit]:
        """对给定候选做 cross-encoder 重排（路由升级路径直接复用）。"""
        if not cand:
            return []
        # 去重后再重排：融合后的候选里常出现同一文档的两条记录
        seen: set[str] = set()
        uniq: list[Hit] = []
        for h in cand:
            if h.pid not in seen:
                seen.add(h.pid)
                uniq.append(h)
        uniq = uniq[: max(top_k, self.rcfg.rerank_top_k)]
        docs = [self.corpus_texts[self.id_to_idx[h.pid]] for h in uniq]
        scores = self.reranker.rerank(query, docs)
        rescored = sorted(
            (Hit(h.pid, float(s)) for h, s in zip(uniq, scores, strict=False)),
            key=lambda h: -h.score,
        )
        return rescored[:top_k]


# --------------------------------------------------------------------------- #
# 策略路由
# --------------------------------------------------------------------------- #
class QueryRouter:
    """级联路由：先用**廉价通道**拿信号，只在必要时升级到重排。

    设计动机（也是实测结论驱动的）：
    cross-encoder 重排是这条链路上最贵的一步（CPU 上约 7 pair/s），
    但它只对「稀疏与稠密给出不同答案」的查询才有价值。
    所以这里不做「按查询长度猜」这种拍脑袋规则，而是**先看两个廉价通道是否一致**：

    * 含型号/编号/数字 → 精确匹配场景，直接走 ``bm25``；
    * 稀疏与稠密 top-k 重合度高、且 top1 分数接近 → 两路一致，走 ``hybrid``；
    * 两路分歧（重合度低或 top1 差距大）→ 说明该查询本身有歧义，
      升级到 ``hybrid_rerank`` 用 cross-encoder 仲裁。

    这样重排只在少数查询上付费，且规则可解释、可单测。
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

    def is_code_like(self, query: str) -> bool:
        f = self.features(query)
        return f["code_ratio"] >= self.cfg.router.code_like_ratio and f["has_digits"] > 0

    @staticmethod
    def agreement(
        dense: list[Hit], sparse: list[Hit], overlap_k: int = 3, gap: float = 0.6
    ) -> float:
        """两路的一致性，取值 [0,1]：top-k 重合度与 top1 得分一致度的加权。"""
        if not dense or not sparse:
            return 0.0
        d_ids = {h.pid for h in dense[:overlap_k]}
        s_ids = {h.pid for h in sparse[:overlap_k]}
        union = d_ids | s_ids
        overlap = len(d_ids & s_ids) / len(union) if union else 0.0

        # top1 是否同一文档；是则再看次席分数是否拉开
        same_top1 = 1.0 if dense[0].pid == sparse[0].pid else 0.0
        d_gap = 1.0
        if len(dense) > 1 and abs(dense[0].score) > 1e-6:
            d_gap = (dense[0].score - dense[1].score) / abs(dense[0].score)
        close = 1.0 if d_gap < gap else 0.5
        return 0.6 * overlap + 0.4 * same_top1 * close

    def needs_rerank(self, dense: list[Hit], sparse: list[Hit]) -> bool:
        return self.agreement(dense, sparse) < self.cfg.router.rerank_gap


class RouterRetriever(_BaseRetriever):
    """级联路由检索器：廉价通道先试，分歧时才升级重排。"""

    name = "router"

    def __init__(
        self,
        cfg: Config,
        corpus_ids,
        corpus_texts,
        dense: DenseRetriever,
        sparse: BM25Retriever,
        hybrid: HybridRetriever,
        hybrid_rerank: HybridRerankRetriever,
    ):
        super().__init__(cfg, corpus_ids, corpus_texts)
        self.dense = dense
        self.sparse = sparse
        self.hybrid = hybrid
        self.hybrid_rerank = hybrid_rerank
        self.router = QueryRouter(cfg)
        self._routes: dict[str, str] = {}

    def decide(self, query: str, top_k: int) -> tuple[str, list[Hit]]:
        """先跑两路廉价召回，再决定是否升级。返回 (路由名, 结果)。"""
        if self.router.is_code_like(query):
            return "bm25", self.sparse.search(query, top_k=top_k)

        pool = max(top_k * 10, self.rcfg.rerank_top_k)
        d_hits = self.dense.search(query, top_k=pool)
        s_hits = self.sparse.search(query, top_k=pool)
        agree = self.router.agreement(d_hits, s_hits)

        # 无论走哪条分支，都先用 RRF 融合成一路候选。
        # 早期版本这里直接把 dense + sparse 两个列表首尾相接再截断，
        # 结果稠密分支占满名额、稀疏分支的候选全被丢掉，
        # 等于「升级重排」其实只重排了向量结果，路由也就失去了意义。
        fused_all = reciprocal_rank_fusion([d_hits, s_hits], k=self.rcfg.rrf_k)

        if agree >= self.cfg.router.rerank_gap:
            return "hybrid", fused_all[:top_k]

        # 升级分支只重排更少的候选：这是路由能省钱的关键
        n_cand = max(top_k, self.cfg.router.rerank_candidates)
        reranked = self.hybrid_rerank.rerank_hits(query, fused_all[:n_cand], top_k)
        return "hybrid_rerank", reranked

    def search(self, query: str, top_k: int = 10) -> list[Hit]:
        route, hits = self.decide(query, top_k)
        self._routes[query] = route
        return hits

    def search_batch(self, queries: Sequence[str], top_k: int = 10) -> list[RetrievalResult]:
        out = []
        for qid, q in enumerate(queries):
            t0 = time.perf_counter()
            route, hits = self.decide(q, top_k)
            dt = (time.perf_counter() - t0) * 1000
            self._routes[q] = route
            out.append(RetrievalResult(qid=str(qid), hits=hits, latency_ms=dt, route=route))
        return out

    @property
    def routes(self) -> dict[str, str]:
        return dict(self._routes)
