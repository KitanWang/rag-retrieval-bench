"""检索评测流水线：建索引 → 跑五种方案 → 算指标 → 落盘。

一个刻意的设计：**指标计算与检索解耦**。
检索器只负责返回有序文档 id，指标/耗时统计在这里统一做，
这样新增一种检索方案不需要改评测代码，也不会出现「各算各的」口径漂移。
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from ragbench import data
from ragbench.config import Config
from ragbench.embedding import Embedder, Reranker
from ragbench.index import build_vector_index
from ragbench.metrics import evaluate_with_details
from ragbench.retrievers import (
    BM25Retriever,
    DenseRetriever,
    HybridRerankRetriever,
    HybridRetriever,
    RouterRetriever,
)
from ragbench.sparse import BM25Index

logger = logging.getLogger(__name__)

ALL_METHODS = ("dense", "bm25", "hybrid", "hybrid_rerank", "router")


class RetrieverBundle:
    """一次性构建所有检索器，共享嵌入与索引，避免重复计算。"""

    def __init__(self, cfg: Config, corpus_ids: list[str], corpus_texts: list[str]) -> None:
        self.cfg = cfg
        self.corpus_ids = corpus_ids
        self.corpus_texts = corpus_texts
        rcfg = cfg.retriever

        logger.info("构建 BM25 索引（%d 篇文档）…", len(corpus_texts))
        t0 = time.perf_counter()
        self.bm25 = BM25Index(k1=rcfg.bm25_k1, b=rcfg.bm25_b).fit(corpus_texts)
        logger.info(
            "BM25 完成，用时 %.1fs，词表 %d", time.perf_counter() - t0, len(self.bm25.vocab)
        )

        self.embedder = Embedder(
            rcfg.embed_model,
            batch_size=rcfg.embed_batch_size,
            cache_dir=rcfg.cache_dir,
            query_instruction=rcfg.embed_query_instruction,
        )
        self.reranker = Reranker(rcfg.rerank_model, batch_size=rcfg.rerank_batch_size)

        logger.info("计算文档向量（%d 篇，model=%s）…", len(corpus_texts), rcfg.embed_model)
        t0 = time.perf_counter()
        embeddings = self.embedder.encode_documents(corpus_texts, show_progress=True)
        logger.info(
            "文档向量完成，用时 %.1fs，shape=%s", time.perf_counter() - t0, embeddings.shape
        )

        self.index = build_vector_index(cfg, embeddings, corpus_ids)

        self._build_retrievers()

    def _build_retrievers(self) -> None:
        cfg, cids, texts = self.cfg, self.corpus_ids, self.corpus_texts
        self.sparse = BM25Retriever(cfg, cids, texts, self.bm25)
        self.dense = DenseRetriever(cfg, cids, texts, self.embedder, self.index)
        self.hybrid = HybridRetriever(cfg, cids, texts, self.dense, self.sparse)
        self.hybrid_rerank = HybridRerankRetriever(cfg, cids, texts, self.hybrid, self.reranker)
        self.router = RouterRetriever(
            cfg, cids, texts, self.dense, self.sparse, self.hybrid, self.hybrid_rerank
        )

    def get(self, name: str):
        table = {
            "dense": self.dense,
            "bm25": self.sparse,
            "hybrid": self.hybrid,
            "hybrid_rerank": self.hybrid_rerank,
            "router": self.router,
        }
        if name not in table:
            raise KeyError(f"未知检索方案 {name}，可选：{sorted(table)}")
        return table[name]

    def close(self) -> None:
        close = getattr(self.index, "close", None)
        if callable(close):
            close()


def run_retrieval_eval(
    cfg: Config,
    methods: list[str] | None = None,
    reuse: RetrieverBundle | None = None,
) -> dict:
    """跑检索评测，返回并落盘结果。"""
    methods = list(methods or ALL_METHODS)
    corpus, queries, qrels = data.load_processed(cfg.data)
    corpus_ids = corpus["pid"].tolist()
    corpus_texts = corpus["text"].tolist()
    gold = data.gold_map(qrels)
    qids = [q for q in queries["qid"].tolist() if q in gold]
    qtexts = dict(zip(queries["qid"], queries["text"], strict=False))

    logger.info("评测集：语料 %d / 查询 %d", len(corpus_ids), len(qids))

    bundle = reuse or RetrieverBundle(cfg, corpus_ids, corpus_texts)
    ks = list(cfg.eval.top_ks)
    max_k = max(ks)
    results: dict[str, dict] = {}

    try:
        for name in methods:
            retriever = bundle.get(name)
            logger.info("== 方案 %s", name)
            rankings: dict[str, list[str]] = {}
            latencies: list[float] = []
            routes: dict[str, str] = {}
            for qid in qids:
                t0 = time.perf_counter()
                hits = retriever.search(qtexts[qid], top_k=max_k)
                latencies.append((time.perf_counter() - t0) * 1000)
                rankings[qid] = [h.pid for h in hits]
                if name == "router":
                    routes[qid] = retriever.routes.get(qtexts[qid], "")

            overall, detail = evaluate_with_details(rankings, gold, ks)
            results[name] = {
                "metrics": overall,
                "latency_ms": {
                    "mean": round(float(np.mean(latencies)), 2),
                    "p50": round(float(np.median(latencies)), 2),
                    "p95": round(float(np.percentile(latencies, 95)), 2),
                },
                "detail": detail,
                "rankings": rankings,
            }
            if name == "router":
                from collections import Counter

                results[name]["routes"] = dict(Counter(routes.values()))
                # 按路由分组统计延迟——这是回答「路由到底省了多少」的关键数据。
                #
                # 只看整体 p50 会得出错误结论：本数据集上 54% 的查询仍升级到重排，
                # 因此整体 p50 落在 4.8s，看起来和全量重排（5.8s）差不多，
                # 会让人以为「路由没用」。真实情况是：
                #   · 走 hybrid 的 111 条查询，延迟是毫秒级；
                #   · 走 rerank 的 163 条查询，延迟是秒级。
                # 两者是**双峰分布**，用单一中位数描述它是误导。
                by_route: dict[str, list[float]] = {}
                for qid, lat in zip(qids, latencies, strict=False):
                    by_route.setdefault(routes[qid] or "unknown", []).append(lat)
                results[name]["latency_by_route"] = {
                    route: {
                        "n": len(vals),
                        "ratio": round(len(vals) / len(latencies), 4),
                        "mean": round(float(np.mean(vals)), 2),
                        "p50": round(float(np.median(vals)), 2),
                        "p95": round(float(np.percentile(vals, 95)), 2),
                    }
                    for route, vals in sorted(by_route.items())
                }
                # 理论节省：若全部走 rerank，总耗时是多少；实际是多少。
                rerank_route = results[name]["latency_by_route"].get("hybrid_rerank")
                if rerank_route:
                    all_rerank = rerank_route["mean"] * len(latencies)
                    actual = sum(latencies)
                    results[name]["cost_saving"] = {
                        "actual_total_ms": round(actual, 1),
                        "all_rerank_total_ms": round(all_rerank, 1),
                        "saved_pct": round((1 - actual / all_rerank) * 100, 1)
                        if all_rerank
                        else 0.0,
                    }
            logger.info(
                "   Recall@5=%.4f  MRR@5=%.4f  nDCG@5=%.4f  p50=%.1fms",
                overall["recall@5"],
                overall["mrr@5"],
                overall["ndcg@5"],
                results[name]["latency_ms"]["p50"],
            )
    finally:
        if reuse is None:
            bundle.close()

    out = {
        "dataset": {
            "name": cfg.data.hf_dataset,
            "corpus": len(corpus_ids),
            "queries": len(qids),
            "seed": cfg.data.seed,
        },
        "config": cfg.to_dict(),
        "results": results,
    }
    save_results(cfg, out, "retrieval")
    return out


def save_results(cfg: Config, payload: dict, tag: str) -> Path:
    path = cfg.output_dir / f"{tag}_results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info("结果已写入 %s", path)
    return path
