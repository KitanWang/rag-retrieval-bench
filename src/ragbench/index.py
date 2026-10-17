"""向量索引：Milvus Lite（单文件、免部署、可随仓库脚本重建）。

选 Milvus 而不是 faiss/nano 的原因：简历里点名的就是 Milvus，
而 Milvus Lite 与生产版共用同一套 SDK 与检索语义（HNSW/IVF 参数、度量方式），
本地实验结论可以直接迁移到线上，不需要重写代码。
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ragbench.sparse import SparseHit

logger = logging.getLogger(__name__)


class VectorIndex:
    """基于 Milvus Lite 的稠密向量索引。"""

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

    def build(self, embeddings: np.ndarray, ids: list[str], overwrite: bool = True) -> None:
        """建集合、插数据、建索引。"""
        dim = int(embeddings.shape[1])
        if overwrite and self.client.has_collection(self.collection):
            self.client.drop_collection(self.collection)
        if not self.client.has_collection(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                dimension=dim,
                metric_type=self.metric,
                auto_id=False,
                id_type="string",
                max_length=64,
            )
            if self.index_type:
                idx_params = {"index_type": self.index_type, "metric_type": self.metric, "params": {}}
                if self.index_type == "HNSW":
                    idx_params["params"] = {"M": 16, "efConstruction": 128}
                try:
                    self.client.create_index(self.collection, idx_params)
                except Exception as exc:  # pragma: no cover - 不同版本 API 差异
                    logger.warning("建向量索引失败，使用默认索引：%s", exc)
        rows = [
            {"id": pid, "vector": emb.tolist()}
            for pid, emb in zip(ids, embeddings)
        ]
        # 分批插入，Milvus 单次请求有大小限制
        for i in range(0, len(rows), 1000):
            self.client.insert(self.collection, rows[i : i + 1000])
        logger.info("向量索引写入完成：%d 条，dim=%d", len(rows), dim)

    def search(self, vector: np.ndarray, top_k: int = 100) -> list[SparseHit]:
        """返回 [(语料内下标缺失)] —— 这里返回的是 Milvus 内的 id 与分数。"""
        res = self.client.search(
            collection_name=self.collection,
            data=[vector.tolist()],
            limit=top_k,
            output_fields=["id"],
            search_params={"metric_type": self.metric, "params": {"ef": max(64, top_k)}},
        )
        hits = []
        for item in res[0]:
            hits.append(SparseHit(idx=item["entity"]["id"], score=float(item["distance"])))
        return hits

    def count(self) -> int:
        stats = self.client.get_collection_stats(self.collection)
        return int(stats.get("row_count", 0))

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
