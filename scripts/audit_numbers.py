#!/usr/bin/env python
"""数字溯源审计：检查 Markdown 里的每个数字是否都能在实验输出中找到来源。

## 为什么需要这个工具

本项目的每条结论都依赖具体数字。最危险的情况不是「算错」，
而是**文档里的数字与实验输出脱节**——改了配置重跑、或手工微调了文档，
数字就对不上了，而读者无从察觉。

用法：

    # 审计仓库内的 README
    PYTHONPATH=src venv/bin/python scripts/audit_numbers.py README.md

    # 审计简历等其他文档（指定实验输出目录）
    PYTHONPATH=src venv/bin/python scripts/audit_numbers.py ../../resume/项目C/*.md

输出无法溯源的数字清单；有未溯源项时退出码为 1，可直接接进 CI。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

NUMBER_RE = re.compile(r"\d+\.?\d*(?:%| ms)")


def collect_allowed(artifact_dir: Path) -> set[str]:
    """从所有实验结果 JSON 汇总允许出现的数字形态。"""
    allowed: set[str] = set()

    def add_pct(v: float) -> None:
        allowed.update({f"{v * 100:.1f}%", f"{v * 100:.0f}%", f"{round(v * 100)}%"})

    def add_ms(v: float) -> None:
        allowed.update({f"{v:.1f} ms", f"{v:.0f} ms", f"{v:.3f} ms"})

    # 优先读 artifacts/（完整实验输出）；CI 里只有 reports/ 下的归档快照，
    # 两者的结构一致（summary_* 是去掉逐条明细后的精简版）
    candidates = {
        "retrieval": ["retrieval_results.json", "summary_retrieval.json"],
        "fusion": ["fusion_tuning.json"],
        "generation": ["generation_results.json", "summary_generation.json"],
        "backend": ["backend_bench.json"],
    }
    search_dirs = [artifact_dir, artifact_dir.parent / "reports"]

    resolved: dict[str, Path] = {}
    for key, names in candidates.items():
        for name in names:
            for d in search_dirs:
                cand = d / name
                if cand.exists():
                    resolved[key] = cand
                    break
            if key in resolved:
                break

    for key, path in resolved.items():
        data = json.loads(path.read_text(encoding="utf-8"))
        name = key

        if name == "retrieval":
            # 兼容两种结构：完整输出有 results 外壳，归档快照把方法直接放顶层
            methods = data.get("results") or {
                k: v for k, v in data.items()
                if not k.startswith("_") and isinstance(v, dict)
            }
            for method in methods.values():
                for v in method.get("metrics", {}).values():
                    add_pct(v)
                for v in method.get("latency_ms", {}).values():
                    add_ms(v)
                for d in (method.get("latency_by_route") or {}).values():
                    add_pct(d["ratio"])
                    add_ms(d["mean"])
                    add_ms(d["p50"])
                saving = method.get("cost_saving")
                if saving:
                    add_pct(saving["saved_pct"] / 100)
                    # p50 相对节省 = 1 - router_p50 / 全量重排的均值
                    p50 = method.get("latency_ms", {}).get("p50")
                    total = saving.get("all_rerank_total_ms")
                    ds = data.get("dataset") or data.get("_dataset") or {}
                    n = ds.get("queries") or 300
                    if p50 and total:
                        add_pct(round(1 - p50 / (total / n), 4))

        elif name == "fusion":
            for row in data.get("rows", []):
                add_pct(row["recall"])
                add_pct(row["ndcg"])
                add_pct(row["mrr"])

        elif name == "generation":
            methods = data.get("results") or {
                k: v for k, v in data.items()
                if not k.startswith("_") and isinstance(v, dict)
            }
            for method in methods.values():
                for key in ("faithfulness", "relevancy", "context_precision",
                            "retrieval_hit_rate", "refusal_rate", "error_rate"):
                    add_pct(method[key])

        elif name == "backend":
            for row in data.get("results", []):
                for key in ("p50_ms", "p95_ms", "mean_ms"):
                    add_ms(row[key])

    return allowed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="待审计的 Markdown 文件")
    ap.add_argument("--artifacts", default=None, help="实验输出目录（默认 ROOT/artifacts）")
    ap.add_argument("--allow", nargs="*", default=[], help="额外允许的数字形态，如 11%")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    allowed = collect_allowed(Path(args.artifacts) if args.artifacts else root / "artifacts")
    allowed.update(args.allow)

    failed = False
    for f in args.files:
        path = Path(f)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.exists():
            print(f"跳过（不存在）：{f}")
            continue
        text = path.read_text(encoding="utf-8")
        found = set(NUMBER_RE.findall(text))
        unknown = sorted(x for x in found if x not in allowed)
        status = "✅" if not unknown else "❌"
        print(f"{status} {path.name}：{len(found)} 个数字，未溯源 {len(unknown)} 个")
        for u in unknown:
            print(f"      {u}")
        failed = failed or bool(unknown)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
