"""LLM 客户端：OpenAI 兼容的流式 / 非流式调用。

* 只依赖 requests，不引入额外 SDK（评测脚本要能在最小环境里跑起来）；
* 支持并发调用 + 指数退避重试，生成评测动辄上百次请求，必须能扛住偶发失败；
* 强制 ``temperature=0``，保证同一评测可复现。
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import requests

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


@dataclass
class LLMConfig:
    base_url: str
    model: str
    api_key: str
    temperature: float = 0.0
    max_tokens: int = 512
    timeout: int = 120
    max_retries: int = 3
    concurrency: int = 4

    @classmethod
    def from_env(cls, base_url: str, model: str, **kwargs: Any) -> LLMConfig:
        key = os.environ.get("CODEBUDDY_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise LLMError(
                "缺少 API Key：请设置环境变量 CODEBUDDY_API_KEY（或 OPENAI_API_KEY），"
                "可参考 .env.example / scripts/run_generation.sh"
            )
        return cls(base_url=base_url, model=model, api_key=key, **kwargs)


class LLMClient:
    """极简 OpenAI 兼容客户端。"""

    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self._session = requests.Session()
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def chat(self, messages: list[dict], *, stream: bool = True) -> str:
        url = f"{self.cfg.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "stream": stream,
        }
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "text/event-stream",
            # 显式要求 UTF-8：服务端默认回 ISO-8859-1，会导致中文乱码
            "Accept-Charset": "utf-8",
        }

        last_err: Exception | None = None
        for attempt in range(self.cfg.max_retries):
            try:
                resp = self._session.post(
                    url, headers=headers, json=payload, timeout=self.cfg.timeout, stream=stream
                )
                if resp.status_code != 200:
                    raise LLMError(f"HTTP {resp.status_code}: {resp.text[:300]}")
                text = self._parse_stream(resp) if stream else self._parse_json(resp)
                self.calls += 1
                return text
            except Exception as exc:  # noqa: BLE001 - 统一重试
                last_err = exc
                wait = 1.5 * (2**attempt)
                logger.warning("LLM 调用失败（第 %d 次）：%s，%.1fs 后重试", attempt + 1, exc, wait)
                time.sleep(wait)
        raise LLMError(f"LLM 调用重试耗尽：{last_err}")

    @staticmethod
    def _parse_json(resp: requests.Response) -> str:
        data = resp.json()
        return data["choices"][0]["message"].get("content", "")

    @staticmethod
    def _parse_stream(resp: requests.Response) -> str:
        """解析 SSE 流。

        **必须显式按 UTF-8 解码**：服务端在 `text/event-stream` 上返回的
        `charset` 是 `ISO-8859-1`（HTTP/1.1 的默认值），requests 会照此解码，
        导致中文全部变成乱码（"你好" → "ä½  å¥½"）。
        这不会报错，只会让下游的忠实度判定悄悄失效——
        属于典型的「不报错但结果全错」缺陷。
        """
        chunks: list[str] = []
        for raw in resp.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace")
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                obj = json.loads(body)
            except json.JSONDecodeError:
                continue
            for choice in obj.get("choices", []):
                delta = choice.get("delta") or {}
                piece = delta.get("content")
                if piece:
                    chunks.append(piece)
        return "".join(chunks)

    def map(self, fn: Callable[[str], Any], items: Sequence[str]) -> list[Any]:
        """并发处理一批独立请求，保持输入顺序。"""
        results: list[Any] = [None] * len(items)
        with ThreadPoolExecutor(max_workers=self.cfg.concurrency) as pool:
            futures = {pool.submit(fn, item): i for i, item in enumerate(items)}
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results[idx] = fut.result()
                except Exception as exc:  # noqa: BLE001
                    logger.error("第 %d 项生成失败：%s", idx, exc)
                    results[idx] = {"error": str(exc)}
        return results
