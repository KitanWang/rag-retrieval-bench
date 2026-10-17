#!/usr/bin/env python
"""融合策略与权重扫描：回答「混合检索到底该不该等权」。

## 要回答的问题

主实验里出现了一个**反直觉**的结果：RRF 等权融合（Recall@5 = 0.726）
反而**低于**纯向量检索（0.765）。常见说法是「混合检索一定更好」，但这份数据不支持。

怀疑的原因是：在这个数据集上两路检索器的强弱差距很大
（dense 0.765 vs bm25 0.615），等权融合等于把弱的一路按同样的票数掺进来，
把强的一路已经排对的结果挤下去了。

这个脚本把「融合方式」和「权重」两个变量拆开扫一遍：

* `rrf`：倒数排名融合，只看名次，不看分数
* `weighted`：归一化加权融合，alpha 控制稠密分支权重（1.0 = 纯向量，0.0 = 纯 BM25）

从而回答三个问题：
1. 双路融合能不能超过单路最优？（不能的话，「混合一定更好」就是伪命题）
2. 权重需要调吗？最优 alpha 是多少？
3. RRF 和加权融合谁更稳？

    PYTHONPATH=src venv/bin/python scripts/tune_fusion.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ragbench.config import Config  # noqa: E402
from ragbench.data import gold_map, load_processed  # noqa: E402
from ragbench.metrics import evaluate  # noqa: E402
from ragbench.pipeline import RetrieverBundle  # noqa: E402
from ragbench.retrievers import reciprocal_rank_fusion, weighted_fusion  # noqa: E402


def run(cfg_path: str, out_path: Path, max_queries: int | None) -> int:
    cfg = Config.load(ROOT / cfg_path)
    corpus, queries, qrels = load_processed(cfg.data)
    ids = corpus["pid"].tolist()
    texts = corpus["text"].tolist()
    gold = gold_map(qrels)
    qids = [q for q in queries["qid"].tolist() if q in gold]
    if max_queries:
        qids = qids[:max_queries]
    qtexts = dict(zip(queries["qid"], queries["text"]))
    print(f"语料 {len(ids)} / 查询 {len(qids)}", flush=True)

    bundle = RetrieverBundle(cfg, ids, texts)
    pool = max(cfg.retriever.rerank_top_k, 10) * 3  # 融合候选池给大一些，避免截断影响结论
    ks = list(cfg.eval.top_ks)
    main_k = cfg.eval.main_k

    try:
        dense = bundle.get("dense")
        sparse = bundle.get("bm25")

        # 先把两路的完整候选取出来，后续所有融合都在内存里算，避免重复检索
        d_ranks: dict[str, list] = {}
        s_ranks: dict[str, list] = {}
        for qid in qids:
            d_ranks[qid] = dense.search(qtexts[qid], top_k=pool)
            s_ranks[qid] = sparse.search(qtexts[qid], top_k=pool)
        print("候选已缓存", flush=True)

        def score(fuse) -> dict:
            rankings = {}
            for qid in qids:
                rankings[qid] = [h.pid for h in fuse(d_ranks[qid], s_ranks[qid])]
            return evaluate_rankings(rankings, gold, ks)

        rows: list[dict] = []

        # 单路基线
        rows.append({"name": "dense（纯向量）", **_pick(score(
            lambda d, s: d[:main_k]), "recall@5", "mrr@5", "ndcg@5")})
        rows.append({"name": "bm25（纯稀疏）", **_pick(score(
            lambda d, s: s[:main_k]), "recall@5", "mrr@5", "ndcg@5")})

        # RRF：k 值也影响融合，扫几个
        for rrf_k in (10, 60, 100):
            rows.append({
                "name": f"RRF（k={rrf_k}）",
                **_pick(score(lambda d, s, k=rrf_k: reciprocal_rank_fusion([d, s], k=k)[:main_k]),
                        "recall@5", "mrr@5", "ndcg@5"),
            })

        # 加权融合：alpha = 稠密权重
        for alpha in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0):
            rows.append({
                "name": f"加权融合（alpha={alpha:.1f}）",
                **_pick(score(
                    lambda d, s, a=alpha: weighted_fusion([d, s], [a, 1 - a], main_k)),
                    "recall@5", "mrr@5", "ndcg@5"),
            })

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "queries": len(qids), "corpus": len(ids), "main_k": main_k,
            "rows": rows,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"\n| 融合策略 | Recall@{main_k} | MRR@{main_k} | nDCG@{main_k} |")
        print("| :--- | ---: | ---: | ---: |")
        for r in rows:
            print(f"| {r['name']} | {r['recall'] * 100:.1f}% | "
                  f"{r['mrr'] * 100:.1f}% | {r['ndcg'] * 100:.1f}% |")

        best = max(rows, key=lambda r: r["recall"])
        print(f"\n最优：{best['name']}  Recall@{main_k}={best['recall'] * 100:.1f}%")
        dense_row = rows[0]
        if best["recall"] <= dense_row["recall"] + 1e-9:
            print("**结论**：没有任何双路融合超过纯向量检索的单项最优。")
            print("在这个数据集上「混合一定更好」不成立——两路强弱差距太大时，等权融合会把弱项引入。")
        else:
            gain = (best["recall"] - dense_row["recall"]) * 100
            print(f"**结论**：最优融合比纯向量高 {gain:.1f} pp，但需要调权重（非等权）。")
        print(f"\n已写入 {out_path}")
    finally:
        bundle.close()
    return 0


def _pick(metrics: dict, *keys: str) -> dict:
    return {"recall": metrics[keys[0]], "mrr": metrics[keys[1]], "ndcg": metrics[keys[2]]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--max-queries", type=int, default=None,
                    help="只用前 N 条查询（调参时加速）")
    ap.add_argument("--out", default="artifacts/fusion_tuning.json")
    args = ap.parse_args()
    return run(args.config, ROOT / args.out, args.max_queries)


if __name__ == "__main__":
    raise SystemExit(main())
