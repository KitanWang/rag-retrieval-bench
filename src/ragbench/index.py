"""向量索引：支持 FAISS（本地高性能）与 Milvus Lite（与生产同 SDK）两种后端。

**为什么保留两个后端**（这是本项目一个实测驱动的设计决策）：

在同一台机器、同一份 2 万条 512 维向量上实测单次 top-10 检索：

| 后端 | 单次检索耗时 |
| :--- | ---: |
| FAISS HNSW（M=16） | ~0.14 ms |
| FAISS FlatIP（精确） | ~2.7 ms |
| Milvus Lite（单文件模式） | ~1700 ms |

Milvus Lite 是**进程内嵌入式**实现，每次检索都要经过完整的 gRPC/计划器路径，
在纯 CPU、无索引服务进程的场景下开销极大。因此：

* 评测默认用 FAISS，保证整轮实验在可接受时间内跑完；
* Milvus 后端保留且同样可跑，用于验证「换回生产同款向量库后结论是否一致」
  ——顺带也把「本地嵌入式 Milvus 不适合做批量评测」这个结论量化了下来。
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class FaissIndex:
    """FAISS 索引（HNSW 或 Flat，内积 / 余弦）。"""

    def __init__(
        self,
        index_type: str = "HNSW",
        metric: str = "COSINE",
        hnsw_m: int = 16,
    ) -> None:
        self.index_type = index_type.upper()
        self.metric = metric.upper()
        self.hnsw_m = hnsw_m
        self._index = None
        self.ids: list[str] = []

    def build(self, embeddings: np.ndarray, ids: list[str]) -> None:
        import faiss

        emb = np.ascontiguousarray(embeddings.astype(np.float32))
        if self.metric == "COSINE":
            # 向量已归一化，内积等价于余弦相似度
            faiss.normalize_L2(emb)
            base_metric = faiss.METRIC_INNER_PRODUCT
        elif self.metric == "IP":
            base_metric = faiss.METRIC_INNER_PRODUCT
        elif self.metric == "L2":
            base_metric = faiss.METRIC_L2
        else:
            raise ValueError(f"不支持的度量方式：{self.metric}")

        dim = emb.shape[1]
        if self.index_type == "HNSW":
            self._index = faiss.IndexHNSWFlat(dim, self.hnsw_m, base_metric)
            self._index.hnsw.efConstruction = 128
            self._index.add(emb)
            self._index.hnsw.efSearch = 64
        else:
            self._index = faiss.IndexFlatIP(dim) if base_metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(dim)
            self._index.add(emb)
        self.ids = list(ids)
        logger.info("FAISS 索引构建完成：%d 条，type=%s，dim=%d", len(ids), self.index_type, dim)

    def search(self, vector: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        if self._index is None:
            raise RuntimeError("索引尚未构建")
        q = np.ascontiguousarray(vector.reshape(1, -1).astype(np.float32))
        if self.metric == "COSINE":
            import faiss

            faiss.normalize_L2(q)
        scores, idxs = self._index.search(q, top_k)
        out = []
        for score, i in zip(scores[0], idxs[0]):
            if i < 0:
                continue
            out.append((self.ids[int(i)], float(score)))
        return out

    def __len__(self) -> int:
        return len(self.ids)


class MilvusIndex:
    """Milvus Lite 单文件向量索引（与生产环境同 SDK / 同检索语义）。"""

    def __init__(
        self,
        db_path: Path,
        collection: str = "ragbench_docs",
        metric: str = "COSINE",
        index_type: str = "HNSW",
    ) -> None:
        self.db_path = Path(db_path)
        self.collection = collection
        self.metric = metric
        self.index_type = index_type
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from pymilvus import MilvusClient

            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._client = MilvusClient(str(self.db_path))
        return self._client

    def build(self, embeddings: np.ndarray, ids: list[str]) -> None:
        dim = int(embeddings.shape[1])
        if self.client.has_collection(self.collection):
            self.client.drop_collection(self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            dimension=dim,
            metric_type=self.metric,
            auto_id=False,
            id_type="string",
            max_length=64,
        )
        try:
            from pymilvus import MilvusClient

            params = MilvusClient.prepare_index_params()
            params.add_index(
                field_name="vector",
                index_type=self.index_type,
                metric_type=self.metric,
                params={"M": 16, "efConstruction": 128} if self.index_type == "HNSW" else {},
            )
            self.client.create_index(self.collection, params)
        except Exception as exc:  # pragma: no cover - 版本差异
            logger.warning("建 Milvus 索引失败，退回默认索引：%s", exc)

        rows = [{"id": pid, "vector": emb.tolist()} for pid, emb in zip(ids, embeddings)]
        for i in range(0, len(rows), 1000):
            self.client.insert(self.collection, rows[i : i + 1000])
        self.client.load_collection(self.collection)
        logger.info("Milvus 索引写入完成：%d 条，dim=%d", len(rows), dim)

    def search(self, vector: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        res = self.client.search(
            collection_name=self.collection,
            data=[vector.tolist()],
            limit=top_k,
            output_fields=["id"],
            search_params={"metric_type": self.metric, "params": {"ef": max(64, top_k)}},
        )
        return [(item["entity"]["id"], float(item["distance"])) for item in res[0]]

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


def build_vector_index(cfg, embeddings: np.ndarray, ids: list[str]):
    """按配置构建向量索引，返回统一接口的对象。"""
    backend = getattr(cfg.retriever, "vector_backend", "faiss").lower()
    if backend == "milvus":
        idx = MilvusIndex(
            cfg.retriever.milvus_path,
            collection=cfg.retriever.collection_name,
            metric="COSINE",
            index_type=cfg.retriever.index_type,
        )
    else:
        idx = FaissIndex(
            index_type=cfg.retriever.index_type,
            metric="COSINE",
            hnsw_m=cfg.retriever.hnsw_m,
        )
    idx.build(embeddings, ids)
    return idx
