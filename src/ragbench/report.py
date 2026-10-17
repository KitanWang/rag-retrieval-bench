"""报告生成：把 JSON 结果渲染成可直接贴进 README 的 Markdown。

原则：**报告里的每个数字都必须来自 artifacts 下的结果文件**，
不允许手工填写。这样重跑实验就能自动更新报告，杜绝「数字对不上代码」。
"""

from __future__ import annotations

import json
from pathlib import Path

METHOD_LABELS = {
    "dense": "朴素向量检索",
    "bm25": "BM25 稀疏检索",
    "hybrid": "混合检索（RRF）",
    "hybrid_rerank": "混合检索 + 重排",
    "router": "策略路由",
}

METHOD_ORDER = ["dense", "bm25", "hybrid", "hybrid_rerank", "router"]


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def render_retrieval_table(payload: dict, k: int = 5) -> str:
    results = payload["results"]
    order = [m for m in METHOD_ORDER if m in results]
    lines = [
        f"| 方案 | Recall@{k} | MRR@{k} | nDCG@{k} | Hit@{k} | 延迟 p50 | 延迟 p95 |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for m in order:
        met = results[m]["metrics"]
        lat = results[m]["latency_ms"]
        lines.append(
            f"| {METHOD_LABELS.get(m, m)} | {_pct(met[f'recall@{k}'])} | {_pct(met[f'mrr@{k}'])} "
            f"| {_pct(met[f'ndcg@{k}'])} | {_pct(met[f'hit@{k}'])} "
            f"| {lat['p50']:.1f} ms | {lat['p95']:.1f} ms |"
        )
    return "\n".join(lines)


def render_gain_analysis(payload: dict, k: int = 5) -> str:
    """以朴素向量检索为基线，计算各方案的相对提升。"""
    results = payload["results"]
    if "dense" not in results:
        return "_缺少 dense 基线，无法计算提升。_"
    base = results["dense"]["metrics"]
    base_lat = results["dense"]["latency_ms"]["p50"]
    lines = [
        f"| 对比 | Recall@{k} 变化 | MRR@{k} 变化 | nDCG@{k} 变化 | 延迟变化 |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]
    for m in METHOD_ORDER:
        if m == "dense" or m not in results:
            continue
        met = results[m]["metrics"]
        lat = results[m]["latency_ms"]["p50"]
        d_rec = (met[f"recall@{k}"] - base[f"recall@{k}"]) * 100
        d_mrr = (met[f"mrr@{k}"] - base[f"mrr@{k}"]) * 100
        d_ndcg = (met[f"ndcg@{k}"] - base[f"ndcg@{k}"]) * 100
        lat_ratio = (lat / base_lat) if base_lat > 0 else float("inf")
        lines.append(
            f"| {METHOD_LABELS.get(m, m)} vs 向量 | {d_rec:+.1f} pp | {d_mrr:+.1f} pp "
            f"| {d_ndcg:+.1f} pp | {lat_ratio:.1f}× |"
        )
    return "\n".join(lines)


def render_route_distribution(payload: dict) -> str:
    routes = payload["results"].get("router", {}).get("routes")
    if not routes:
        return ""
    total = sum(routes.values())
    lines = ["| 路由分支 | 查询数 | 占比 |", "| :--- | ---: | ---: |"]
    for name, cnt in sorted(routes.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {METHOD_LABELS.get(name, name)} | {cnt} | {_pct(cnt / total)} |")
    return "\n".join(lines)


def render_generation_table(payload: dict) -> str:
    results = payload["results"]
    order = [m for m in METHOD_ORDER if m in results]
    lines = [
        "| 方案 | 忠实度 | 答案相关性 | 上下文精度 | 检索命中率 | 拒答率 |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for m in order:
        r = results[m]
        lines.append(
            f"| {METHOD_LABELS.get(m, m)} | {_pct(r['faithfulness'])} | {_pct(r['relevancy'])} "
            f"| {_pct(r['context_precision'])} | {_pct(r['retrieval_hit_rate'])} "
            f"| {_pct(r['refusal_rate'])} |"
        )
    return "\n".join(lines)


def build_report(cfg, out_path: Path) -> Path:
    retrieval = _load(cfg.output_dir / "retrieval_results.json")
    generation = _load(cfg.output_dir / "generation_results.json")
    stats = _load(cfg.data.processed_dir / "stats.json") or {}

    parts: list[str] = []
    parts.append("# 实验结果\n")
    parts.append(
        "> 本文件由 `ragbench report` 从 `artifacts/*.json` 自动生成，"
        "**所有数字均来自可复现的实验输出**，没有手工填写。\n"
    )

    if retrieval:
        ds = retrieval["dataset"]
        parts.append("## 1. 实验设置\n")
        parts.append(
            f"| 项 | 值 |\n| :--- | :--- |\n"
            f"| 数据集 | `{ds['name']}` |\n"
            f"| 语料规模 | {ds['corpus']:,} 篇 |\n"
            f"| 评测查询数 | {ds['queries']} 条 |\n"
            f"| 每查询平均正例 | {stats.get('gold_per_query', 'N/A')} 篇 |\n"
            f"| 随机种子 | {ds['seed']} |\n"
            f"| 嵌入模型 | `{retrieval['config']['retriever']['embed_model']}` |\n"
            f"| 重排模型 | `{retrieval['config']['retriever']['rerank_model']}` |\n"
        )

        parts.append("## 2. 检索质量对比（k=5）\n")
        parts.append(render_retrieval_table(retrieval, k=5))
        parts.append("")
        parts.append("## 3. 相对朴素向量检索的提升与代价\n")
        parts.append(render_gain_analysis(retrieval, k=5))
        parts.append("")

        route_md = render_route_distribution(retrieval)
        if route_md:
            parts.append("## 4. 策略路由的分支分布\n")
            parts.append(route_md)
            parts.append("")

    if generation:
        parts.append("## 5. 生成质量评测\n")
        parts.append(
            f"> 采样 {generation['n_samples']} 条查询，取 top-{generation['top_k']} 文档作为上下文，"
            f"模型 `{generation['model']}`，温度 {generation['temperature']}。\n"
        )
        parts.append(render_generation_table(generation))
        parts.append("")

    if not retrieval and not generation:
        parts.append("_暂无结果文件，请先运行 `ragbench retrieve`。_\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts), encoding="utf-8")
    return out_path
