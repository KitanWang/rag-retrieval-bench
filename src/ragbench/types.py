"""跨模块共用的数据类型。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Hit:
    """一条检索结果。``pid`` 是文档 id，``score`` 越大越相关。"""

    pid: str
    score: float

    def __repr__(self) -> str:  # pragma: no cover - 调试方便
        return f"Hit({self.pid!r}, {self.score:.4f})"


@dataclass(frozen=True)
class RetrievalResult:
    """单条查询的检索结果 + 耗时。"""

    qid: str
    hits: list[Hit]
    latency_ms: float
    route: str = ""
