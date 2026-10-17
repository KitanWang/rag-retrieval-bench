"""实验配置：参数集中一处，避免散落在脚本里。

配置来源的优先级（后者覆盖前者）：
默认值 < YAML 配置文件 < 环境变量（少量关键项）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class DataConfig:
    """数据集与切分配置。"""

    # HuggingFace 上的公开中文检索数据集；也可换成自定义本地数据
    hf_dataset: str = "C-MTEB/DuRetrieval"
    hf_qrels_dataset: str = "C-MTEB/DuRetrieval-qrels"
    raw_dir: Path = ROOT / "data" / "raw"
    processed_dir: Path = ROOT / "data" / "processed"

    # 为了在 CPU 上可跑，默认对十万量级语料做稳定子采样
    corpus_size: int = 20000
    query_size: int = 300
    seed: int = 42

    @property
    def corpus_path(self) -> Path:
        return self.processed_dir / "corpus.parquet"

    @property
    def queries_path(self) -> Path:
        return self.processed_dir / "queries.parquet"

    @property
    def qrels_path(self) -> Path:
        return self.processed_dir / "qrels.parquet"


@dataclass
class RetrieverConfig:
    """检索器与索引配置。"""

    # 向量模型：BGE-M3 是简历里点名的模型；bge-small-zh 作为 CPU 快速档
    embed_model: str = "BAAI/bge-m3"
    embed_batch_size: int = 16
    embed_query_instruction: str = "为这个句子生成表示以用于检索相关文章："

    # 重排模型（cross-encoder）
    rerank_model: str = "BAAI/bge-reranker-base"
    rerank_batch_size: int = 16
    rerank_top_k: int = 50  # 先召回 50 条再重排到 top_n

    # BM25 参数
    bm25_k1: float = 0.9
    bm25_b: float = 0.4

    # 融合方式：rrf（倒数排名融合）或 weighted（归一化加权）
    fusion: str = "rrf"
    rrf_k: int = 60
    hybrid_alpha: float = 0.5  # weighted 融合时稠密分支的权重

    # 向量后端：faiss（本地高性能）或 milvus（与生产同 SDK）
    # 实测同一份 2 万条 512 维向量（scripts/bench_vector_backends.py）：
    # FAISS HNSW p50 0.24ms/次，Milvus Lite p50 1520ms/次，top-1 结果一致
    vector_backend: str = "faiss"
    index_type: str = "HNSW"
    hnsw_m: int = 16
    milvus_path: Path = ROOT / "data" / "milvus" / "bench.db"
    collection_name: str = "ragbench_docs"

    # 缓存嵌入，避免重复计算
    cache_dir: Path = ROOT / "data" / "cache"


@dataclass
class EvalConfig:
    """评测配置。"""

    top_ks: tuple[int, int, int] = (1, 5, 10)
    main_k: int = 5
    latency_repeats: int = 3  # 延迟测多轮取中位数，压掉首次冷启动

    # 生成质量评测
    gen_sample_size: int = 60
    gen_top_k: int = 5

    # LLM 配置（OpenAI 兼容接口）
    llm_base_url: str = os.environ.get("CODEBUDDY_BASE_URL", "https://copilot.tencent.com/v2")
    llm_model: str = os.environ.get("CODEBUDDY_MODEL", "deepseek-v4.1-flash")
    llm_temperature: float = 0.0
    llm_max_tokens: int = 512
    llm_concurrency: int = 4
    llm_timeout: int = 120


@dataclass
class RouterConfig:
    """策略路由的判定阈值。

    路由逻辑刻意保持「可解释、可单测」：不训练分类器，
    只用查询的可观测特征（标点、数字/型号、词法密度）选路。
    """

    # 查询词数超过该值 → 认为是长句/描述型，稠密更强
    long_query_chars: int = 14
    # 查询里出现型号/数字/英文标识的比例阈值 → 稀疏更强
    code_like_ratio: float = 0.15
    # 稀疏与稠密的一致性低于该阈值时升级重排（0~1，越大越保守）
    rerank_gap: float = 0.55
    # 升级重排时送进 cross-encoder 的候选数（比独立重排方案少，路由才有成本优势）
    rerank_candidates: int = 10


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    output_dir: Path = ROOT / "artifacts"

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        """从 YAML 载入配置；未给出的字段用默认值。"""
        cfg = cls()
        if path is None:
            default_path = ROOT / "configs" / "default.yaml"
            path = default_path if default_path.exists() else None
        if path is not None:
            with open(path, encoding="utf-8") as f:
                raw: dict[str, Any] = yaml.safe_load(f) or {}
            cfg = cls.from_dict(raw)
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        return cfg

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Config:
        cfg = cls()
        for section in ("data", "retriever", "eval", "router"):
            sub = raw.get(section) or {}
            target = getattr(cfg, section)
            valid = {f.name for f in fields(target)}
            for key, value in sub.items():
                if key not in valid:
                    raise ValueError(f"未知配置项 {section}.{key}")
                if isinstance(getattr(target, key), Path):
                    value = Path(value).expanduser()
                setattr(target, key, value)
        if "output_dir" in raw:
            cfg.output_dir = Path(raw["output_dir"]).expanduser()
        return cfg

    def to_dict(self) -> dict[str, Any]:
        def conv(obj: Any) -> Any:
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, dict):
                return {k: conv(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [conv(v) for v in obj]
            return obj

        return conv(
            {
                "data": self.data.__dict__,
                "retriever": self.retriever.__dict__,
                "eval": self.eval.__dict__,
                "router": self.router.__dict__,
                "output_dir": self.output_dir,
            }
        )
