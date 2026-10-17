"""嵌入与重排模型封装。

单独一层的原因（面试常问的点）：

* **设备与批大小只在这里决定**，上层检索器不关心模型怎么跑；
* **向量缓存**：同一模型 + 同一文本只算一次，改成多组实验时省下大量时间；
* **查询侧指令前缀**：BGE 系列对短查询建议加指令前缀，这里统一处理，
  避免不同检索分支的查询表示不一致导致融合失真。
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class Embedder:
    """句向量编码器（CPU/GPU 自适应）。"""

    def __init__(
        self,
        model_name: str,
        batch_size: int = 16,
        cache_dir: Path | None = None,
        device: str | None = None,
        query_instruction: str = "",
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.device = device or self._auto_device()
        self.query_instruction = query_instruction
        self._model = None

    @staticmethod
    def _auto_device() -> str:
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:  # pragma: no cover - torch 是可选依赖
            return "cpu"

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("加载嵌入模型 %s（device=%s）", self.model_name, self.device)
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._model.eval()
        return self._model

    def encode_documents(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        return self._encode(texts, prefix="", show_progress=show_progress)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        prefix = self.query_instruction
        return self._encode(texts, prefix=prefix, show_progress=False)

    def _encode(self, texts: list[str], prefix: str, show_progress: bool) -> np.ndarray:
        cache_file = self._cache_path(texts, prefix)
        if cache_file is not None and cache_file.exists():
            return np.load(cache_file)
        payload = [prefix + t for t in texts] if prefix else list(texts)
        emb = self.model.encode(
            payload,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        ).astype(np.float32)
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_file, emb)
        return emb

    def _cache_path(self, texts: list[str], prefix: str) -> Path | None:
        if self.cache_dir is None:
            return None
        h = hashlib.sha256()
        h.update(self.model_name.encode())
        h.update(prefix.encode())
        for t in texts:
            h.update(t.encode())
            h.update(b"\x00")
        safe_model = self.model_name.replace("/", "__")
        return self.cache_dir / f"{safe_model}_{h.hexdigest()[:16]}.npy"


class Reranker:
    """cross-encoder 重排器。"""

    def __init__(
        self,
        model_name: str,
        batch_size: int = 16,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device or Embedder._auto_device()
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("加载重排模型 %s（device=%s）", self.model_name, self.device)
            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def rerank(self, query: str, docs: list[str]) -> np.ndarray:
        """返回与 docs 同序的打分（越大越相关）。"""
        if not docs:
            return np.zeros(0, dtype=np.float32)
        pairs = [(query, d) for d in docs]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        return np.asarray(scores, dtype=np.float32).reshape(-1)
