#!/usr/bin/env python
"""向量后端基准：在同一份向量、同一批查询上对比 FAISS 与 Milvus Lite 的检索耗时。

## 为什么必须有这个脚本

`index.py` 的模块注释里写着「Milvus Lite 单次检索 ~1700 ms，FAISS ~0.14 ms」。
这组数字差异高达**四个数量级**，如果只是随手测一次就写进报告，是无法被信任的——
要么是本机环境特例，要么是测法有问题（比如没预热、把建模时间算进去了）。

所以把它做成脚本：**任何看到结论的人都能自己跑一遍**，用同一份数据复现同一个量级。

    PYTHONPATH=src venv/bin/python scripts/bench_vector_backends.py
    PYTHONPATH=src venv/bin/python scripts/bench_vector_backends.py --docs 20000 --queries 100

脚本做的事：
1. 用真实语料的前 N 篇编码出向量（复用 embedding 层，不另写一套）；
2. 分别建 FAISS HNSW / FAISS FlatIP / Milvus Lite 三个索引；
3. **先预热**再计时（预热是排除冷启动的关键，第一版没做导致数字虚高）；
4. 报告 p50 / p95 / mean，并校验三个后端返回的 top-1 是否一致
   （性能差不能以牺牲正确性为代价）。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ragbench.config import Config  # noqa: E402
from ragbench.data import load_processed  # noqa: E402
from ragbench.embedding import Embedder  # noqa: E402


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round((p / 100) * (len(s) - 1)))))
    return s[idx]


def bench_faiss(embeddings: np.ndarray, queries: np.ndarray, index_type: str, k: int,
                repeats: int) -> dict:
    import faiss

    dim = embeddings.shape[1]
    t0 = time.perf_counter()
    if index_type == "hnsw":
        index = faiss.IndexHNSWFlat(dim, 16, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 128
        index.hnsw.efSearch = 64
    else:
        index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    build_s = time.perf_counter() - t0

    # 预热：不计入统计。不做这一步会把首次查询的页错误/分支预测失败算进去。
    for q in queries[: min(5, len(queries))]:
        index.search(q.reshape(1, -1), k)

    lat: list[float] = []
    top1: list[int] = []
    for _ in range(repeats):
        for q in queries:
            t0 = time.perf_counter()
            _, ids = index.search(q.reshape(1, -1), k)
            lat.append((time.perf_counter() - t0) * 1000)
            if not top1:
                top1 = [int(i) for i in ids[0]]
    return {
        "backend": f"faiss-{index_type}",
        "build_s": round(build_s, 2),
        "p50_ms": round(percentile(lat, 50), 3),
        "p95_ms": round(percentile(lat, 95), 3),
        "mean_ms": round(statistics.fmean(lat), 3),
        "n_queries": len(lat),
        "_top1": top1,
    }


def bench_milvus(embeddings: np.ndarray, queries: np.ndarray, k: int, repeats: int,
                 db_path: Path, collection: str) -> dict:
    from pymilvus import MilvusClient

    if db_path.exists():
        import shutil

        shutil.rmtree(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    client = MilvusClient(str(db_path))
    dim = embeddings.shape[1]
    client.create_collection(
        collection_name=collection, dimension=dim,
        metric_type="IP", auto_id=False, id_type="int",
    )
    t0 = time.perf_counter()
    rows = [{"id": i, "vector": emb.tolist()} for i, emb in enumerate(embeddings)]
    for i in range(0, len(rows), 1000):
        client.insert(collection, rows[i:i + 1000])
    client.load_collection(collection)
    build_s = time.perf_counter() - t0

    for q in queries[: min(5, len(queries))]:
        client.search(collection, data=[q.tolist()], limit=k)

    lat: list[float] = []
    top1: list[int] = []
    for _ in range(repeats):
        for q in queries:
            t0 = time.perf_counter()
            res = client.search(collection, data=[q.tolist()], limit=k)
            lat.append((time.perf_counter() - t0) * 1000)
            if not top1 and res:
                top1 = [int(h["id"]) for h in res[0]]
    client.close()
    return {
        "backend": "milvus-lite",
        "build_s": round(build_s, 2),
        "p50_ms": round(percentile(lat, 50), 3),
        "p95_ms": round(percentile(lat, 95), 3),
        "mean_ms": round(statistics.fmean(lat), 3),
        "n_queries": len(lat),
        "_top1": top1,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--docs", type=int, default=20000, help="参与建索引的文档数")
    ap.add_argument("--queries", type=int, default=100, help="计时的查询数")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--skip-milvus", action="store_true", help="Milvus 很慢，调试时可跳过")
    ap.add_argument("--out", default="artifacts/backend_bench.json")
    args = ap.parse_args()

    cfg = Config.load(ROOT / args.config)
    corpus, queries_df, _ = load_processed(cfg.data)
    n_docs = min(args.docs, len(corpus))
    texts = corpus["text"].tolist()[:n_docs]
    qtexts = queries_df["text"].tolist()[: args.queries]
    print(f"语料 {n_docs} 篇 / 计时查询 {len(qtexts)} 条 / 后端 k={args.k}", flush=True)

    emb = Embedder(cfg.retriever.embed_model, cfg.retriever.embed_batch_size)
    t0 = time.perf_counter()
    doc_vecs = emb.encode_documents(texts)
    print(f"文档向量编码：{time.perf_counter() - t0:.1f}s  shape={doc_vecs.shape}", flush=True)
    q_vecs = emb.encode_queries(qtexts)

    results: list[dict] = []
    reports = [
        ("faiss-hnsw", lambda: bench_faiss(doc_vecs, q_vecs, "hnsw", args.k, args.repeats)),
        ("faiss-flat", lambda: bench_faiss(doc_vecs, q_vecs, "flat", args.k, args.repeats)),
    ]
    if not args.skip_milvus:
        reports.append((
            "milvus-lite",
            lambda: bench_milvus(doc_vecs, q_vecs, args.k, args.repeats,
                                 ROOT / "data" / "milvus" / "bench_bench.db", "bench_docs"),
        ))

    for name, fn in reports:
        print(f"-- 测试 {name} …", flush=True)
        try:
            r = fn()
        except Exception as e:  # noqa: BLE001
            print(f"   {name} 失败：{e}", flush=True)
            continue
        results.append(r)
        print(f"   p50={r['p50_ms']}ms  p95={r['p95_ms']}ms  build={r['build_s']}s", flush=True)

    # 正确性校验：以最快的 FAISS HNSW 为基准，比较各后端 top-1 是否一致。
    # 性能差异必须建立在「结果相同」的前提下，否则快也没意义。
    baseline = next((r for r in results if r["backend"] == "faiss-hnsw"), None)
    if baseline:
        for r in results:
            if r is baseline:
                continue
            same = r.get("_top1")[:1] == baseline.get("_top1")[:1]
            r["top1_match"] = bool(same)

    for r in results:
        r.pop("_top1", None)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "docs": n_docs, "queries": len(qtexts), "k": args.k,
        "embed_model": cfg.retriever.embed_model,
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {out}")

    print("\n| 后端 | 建索引 | p50 | p95 | top1 一致 |")
    print("| :--- | ---: | ---: | ---: | :--- |")
    for r in results:
        match = "-" if "top1_match" not in r else ("是" if r["top1_match"] else "**否**")
        print(f"| {r['backend']} | {r['build_s']}s | {r['p50_ms']} ms | "
              f"{r['p95_ms']} ms | {match} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
