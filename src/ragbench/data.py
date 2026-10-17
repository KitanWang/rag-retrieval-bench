"""数据层：从公开中文检索数据集构建可复现的评测集。

设计要点（也是为了面试能讲清楚）：

* **语料子采样与评测集解耦**：先按 gold 文档闭包收敛语料规模，再稳定随机补齐。
  这样即使把语料缩到两万条，评测集里的每一条 query 仍有完整的正例集合，
  指标不会因为「正例被采样掉」而虚高。
* **固定随机种子**：同一条脚本重跑，切分完全一致。
* **落盘为 parquet**：训练/评测之间只通过文件交互，任何一步都能单独重跑。
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from ragbench.config import DataConfig


def _hf_download(dataset: str, filename_pattern: str, out_dir: Path) -> list[Path]:
    """从 HuggingFace 拉取数据集文件（带本地缓存，重复调用不会重复下载）。"""
    from huggingface_hub import hf_hub_download, list_repo_files

    out_dir.mkdir(parents=True, exist_ok=True)
    files = [f for f in list_repo_files(dataset, repo_type="dataset") if f.startswith("data/")]
    if filename_pattern not in ("", None):
        files = [f for f in files if filename_pattern in f]
    paths = []
    for name in files:
        local = hf_hub_download(
            repo_id=dataset, filename=name, repo_type="dataset",
            local_dir=str(out_dir / dataset.replace("/", "__")),
        )
        paths.append(Path(local))
    return paths


def load_raw(cfg: DataConfig, force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """下载并读取原始 corpus / queries / qrels（列名统一为 id/text）。"""
    cache = cfg.raw_dir / "duretrival"
    corpus_f = cache / "corpus.parquet"
    queries_f = cache / "queries.parquet"
    qrels_f = cache / "qrels.parquet"

    if force or not (corpus_f.exists() and queries_f.exists() and qrels_f.exists()):
        cache.mkdir(parents=True, exist_ok=True)
        _fetch_parquet(cfg.hf_dataset, "corpus", corpus_f, force)
        _fetch_parquet(cfg.hf_dataset, "queries", queries_f, force)
        _fetch_parquet(cfg.hf_qrels_dataset, "", qrels_f, force)

    corpus = pd.read_parquet(corpus_f)
    queries = pd.read_parquet(queries_f)
    qrels = pd.read_parquet(qrels_f)
    return corpus, queries, qrels


def _fetch_parquet(repo: str, prefix: str, dest: Path, force: bool) -> None:
    """下载 repo 下的 parquet 并另存为稳定的文件名。"""
    from huggingface_hub import hf_hub_download, list_repo_files

    if dest.exists() and not force:
        return
    names = [
        n for n in list_repo_files(repo, repo_type="dataset")
        if n.startswith("data/") and n.endswith(".parquet") and prefix in n
    ]
    if not names:
        raise FileNotFoundError(f"{repo} 中没有找到 {prefix}*.parquet")
    frames = [
        pd.read_parquet(
            hf_hub_download(repo_id=repo, filename=n, repo_type="dataset")
        )
        for n in names
    ]
    pd.concat(frames, ignore_index=True).to_parquet(dest, index=False)


def normalize_qrels(qrels: pd.DataFrame) -> pd.DataFrame:
    """统一 qrels 列名，并只保留 score>0 的正例。"""
    df = qrels.rename(
        columns={"query-id": "qid", "corpus-id": "pid", "query_id": "qid", "doc_id": "pid"}
    )
    need = {"qid", "pid", "score"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"qrels 缺少列：{sorted(missing)}；实际列：{list(df.columns)}")
    df = df[df["score"] > 0][["qid", "pid", "score"]].copy()
    df["qid"] = df["qid"].astype(str)
    df["pid"] = df["pid"].astype(str)
    return df.reset_index(drop=True)


def build_eval_set(
    corpus: pd.DataFrame,
    queries: pd.DataFrame,
    qrels: pd.DataFrame,
    cfg: DataConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """构造子采样后的语料 + 评测集。

    步骤：
    1. 抽 ``query_size`` 条 query（种子固定）；
    2. 取这些 query 的全部 gold 文档，构成**闭包**，保证正例不被采样掉；
    3. 用剩余额度的随机文档补齐 ``corpus_size``，作为检索干扰项。

    这样得到的 Recall@k 不会因为「正例不在库里」而被人为压低或抬高。
    """
    rng = random.Random(cfg.seed)
    qrels = normalize_qrels(qrels)
    corpus = corpus.rename(columns={"doc_id": "pid", "text_id": "text"})
    queries = queries.rename(columns={"query_id": "qid", "text": "text"})

    corpus["pid"] = corpus["pid"].astype(str)
    queries["qid"] = queries["qid"].astype(str)

    # 只用有正例、且正例都在语料里的 query
    corpus_ids = set(corpus["pid"])
    gold_by_q: dict[str, set[str]] = defaultdict(set)
    for qid, pid in zip(qrels["qid"], qrels["pid"]):
        if pid in corpus_ids:
            gold_by_q[qid].add(pid)
    usable = [q for q in queries["qid"] if q in gold_by_q]
    usable.sort()  # 排序后再采样，避免依赖原始行序
    rng.shuffle(usable)
    picked = usable[: cfg.query_size]
    picked_set = set(picked)

    gold_pids = {p for q in picked for p in gold_by_q[q]}
    others = sorted(corpus_ids - gold_pids)
    rng.shuffle(others)
    fill = max(0, cfg.corpus_size - len(gold_pids))
    keep = gold_pids | set(others[:fill])

    sub_corpus = corpus[corpus["pid"].isin(keep)].drop_duplicates("pid").reset_index(drop=True)
    sub_queries = queries[queries["qid"].isin(picked_set)].drop_duplicates("qid").reset_index(drop=True)
    sub_qrels = qrels[qrels["qid"].isin(picked_set) & qrels["pid"].isin(keep)].reset_index(drop=True)
    return sub_corpus, sub_queries, sub_qrels


def prepare(cfg: DataConfig, force: bool = False) -> dict[str, int]:
    """一键构建评测集并落盘，返回规模统计。"""
    corpus, queries, qrels = load_raw(cfg, force=force)
    sub_corpus, sub_queries, sub_qrels = build_eval_set(corpus, queries, qrels, cfg)
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    sub_corpus.to_parquet(cfg.corpus_path, index=False)
    sub_queries.to_parquet(cfg.queries_path, index=False)
    sub_qrels.to_parquet(cfg.qrels_path, index=False)

    stats = {
        "corpus": len(sub_corpus),
        "queries": len(sub_queries),
        "qrels": len(sub_qrels),
        "gold_per_query": round(len(sub_qrels) / max(1, len(sub_queries)), 2),
        "corpus_chars_mean": round(float(sub_corpus["text"].str.len().mean()), 1),
        "corpus_chars_p95": int(sub_corpus["text"].str.len().quantile(0.95)),
    }
    with open(cfg.processed_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    return stats


def load_processed(cfg: DataConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """读取已构建好的评测集。"""
    missing = [p for p in (cfg.corpus_path, cfg.queries_path, cfg.qrels_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"评测集不存在：{missing}；请先运行 `ragbench prepare`"
        )
    return (
        pd.read_parquet(cfg.corpus_path),
        pd.read_parquet(cfg.queries_path),
        pd.read_parquet(cfg.qrels_path),
    )


def gold_map(qrels: pd.DataFrame) -> dict[str, set[str]]:
    """qrel 表 → {qid: {pid...}}。"""
    out: dict[str, set[str]] = defaultdict(set)
    for qid, pid in zip(qrels["qid"], qrels["pid"]):
        out[qid].add(pid)
    return dict(out)


def iter_batches(items: Sequence, size: int) -> Iterable[Sequence]:
    for i in range(0, len(items), size):
        yield items[i : i + size]
