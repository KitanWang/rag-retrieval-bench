#!/usr/bin/env python
"""融合方式消融实验：RRF vs 归一化加权融合（扫描稠密分支权重 α）。

## 为什么单独做这个实验

主实验里 `hybrid`（RRF）的 Recall@5 低于纯向量检索（72.6% vs 76.5%）。
如果就此收尾，结论会是「混合检索没用」——但这是一个**被融合方式污染的结论**。

RRF 只用排名、不看分数：当两路检索质量差距很大时（本数据集 BM25 61.5% vs 稠密 76.5%），
它仍会给弱分支同等权重，噪声就被带进融合结果。
加权融合则可以把权重压向强分支。

这个脚本把「融合方式」这一变量单独拎出来扫一遍，用来判断：
**混合检索输给单路，是方法本身的问题，还是融合策略没配对？**

    PYTHONPATH=src venv/bin/python scripts/ablation_fusion.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ragbench import data  # noqa: E402
from ragbench.config import Config  # noqa: E402
from ragbench.metrics import evaluate_rankings  # noqa: E402
from ragbench.pipeline import RetrieverBundle  # noqa: E402
from ragbench.retrievers import reciprocal_rank_fusion, weighted_fusion  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
for noisy in ("httpx", "urllib3", "huggingface_hub", "sentence_transformers", "faiss"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logger = logging.getLogger("ablation")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--pool", type=int, default=100, help="每路召回候选数")
    ap.add_argument("--alphas", nargs="*", type=float,
                    default=[0.0, 0.3, 0.5, 0.7, 0.85, 0.9, 0.95, 1.0])
    ap.add_argument("--rrf-ks", nargs="*", type=int, default=[10, 60, 100])
    args = ap.parse_args()

    cfg = Config.load(args.config)
    corpus, queries, qrels = data.load_processed(cfg.data)
    gold = data.gold_map(qrels)
    qids = [q for q in queries["qid"].tolist() if q in gold]
    qtext = dict(zip(queries["qid"], queries["text"], strict=False))
    logger.info("评测集：语料 %d / 查询 %d / k=%d", len(corpus), len(qids), args.k)

    bundle = RetrieverBundle(cfg, corpus["pid"].tolist(), corpus["text"].tolist())

    def evaluate(make_ranker) -> dict:
        """make_ranker 接收 qid，返回该查询的有序 Hit 列表。"""
        rankings = {q: [h.pid for h in make_ranker(q)] for q in qids}
        return evaluate_rankings(rankings, gold, [args.k])

    results: dict[str, dict] = {}
    try:
        dense_hits = {q: bundle.dense.search(qtext[q], top_k=args.pool) for q in qids}
        sparse_hits = {q: bundle.sparse.search(qtext[q], top_k=args.pool) for q in qids}

        # 注意：dense_hits / sparse_hits 的键是 **qid**，不是查询文本。
        # 早先这里写成 dense_hits[qtext[q]]（用文本索引），一旦真正跑就会 KeyError——
        # 因为脚本没被列入 CI 的冒烟测试，这个错误一直没暴露。
        single_dense = evaluate(lambda q: dense_hits[q][: args.k])
        single_sparse = evaluate(lambda q: sparse_hits[q][: args.k])
        results["dense_only"] = single_dense
        results["bm25_only"] = single_sparse
        logger.info(
            "单路：dense R@%d=%.4f / bm25 R@%d=%.4f",
            args.k, single_dense[f"recall@{args.k}"], args.k, single_sparse[f"recall@{args.k}"],
        )

        for rrf_k in args.rrf_ks:
            m = evaluate(
                lambda q, rk=rrf_k: reciprocal_rank_fusion(
                    [dense_hits[q], sparse_hits[q]], k=rk
                )[: args.k]
            )
            results[f"rrf@{rrf_k}"] = m
            logger.info("RRF k=%-3d R@%d=%.4f", rrf_k, args.k, m[f"recall@{args.k}"])

        best_alpha, best_recall = None, -1.0
        for alpha in args.alphas:
            m = evaluate(
                lambda q, a=alpha: weighted_fusion(
                    [dense_hits[q], sparse_hits[q]], [a, 1 - a], args.k
                )
            )
            results[f"weighted@{alpha}"] = m
            r = m[f"recall@{args.k}"]
            logger.info("加权 α=%.2f R@%d=%.4f", alpha, args.k, r)
            if r > best_recall:
                best_alpha, best_recall = alpha, r

        logger.info(
            "\n最佳加权 α=%.2f（R@%d=%.4f）vs 纯向量 %.4f → %+.2f pp",
            best_alpha, args.k, best_recall,
            single_dense[f"recall@{args.k}"],
            (best_recall - single_dense[f"recall@{args.k}"]) * 100,
        )
    finally:
        bundle.close()

    out = {
        "k": args.k,
        "pool": args.pool,
        "n_queries": len(qids),
        "embed_model": cfg.retriever.embed_model,
        "results": {
            name: {
                "recall": m[f"recall@{args.k}"],
                "mrr": m[f"mrr@{args.k}"],
                "ndcg": m[f"ndcg@{args.k}"],
                "hit": m[f"hit@{args.k}"],
            }
            for name, m in results.items()
        },
    }
    path = cfg.output_dir / "ablation_fusion.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("已写入 %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
