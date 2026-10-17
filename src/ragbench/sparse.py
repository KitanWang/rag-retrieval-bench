"""中文分词与 BM25 稀疏检索。

* 分词用 jieba + 简单的标点归一，并对数字与英文标识单独切分，
  这样型号（``HX-1234``）与纯数字也能落在同一个词表里；
* BM25 用**倒排索引**实现：打分时只遍历查询词对应的 posting list，
  复杂度是 O(查询词数 × 平均文档频次)，而不是 O(语料规模)。
  这一点在十万级语料上是能不能跑完实验的分水岭。
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

import jieba
import numpy as np

# 关掉 jieba 的启动日志，避免污染评测输出
jieba.setLogLevel(60)

_PUNCT = re.compile(r"[\s\u3000-\u303f\uff00-\uff0f\uff1a-\uff20\u2018-\u201d]+")

# 常见中文停用词（检索场景里几乎全是噪声）
STOPWORDS = frozenset(
    """
    的 了 和 与 及 或 是 在 有 为 对 也 就 都 而 被 把 让 从 到 中 上 下
    请问 如何 怎么 怎样 什么 哪些 多少 为什么 可以 能否 是否 需要
    一个 一下 这个 那个 这些 那些 我们 你们 他们 自己 现在 已经
    """.split()
)


def _segment_text(text: str) -> list[str]:
    """中英混合切分：jieba 分词 + 数字/英文单独拆出。"""
    text = text.lower()
    text = text.replace("，", ",").replace("。", ".")
    tokens: list[str] = []
    for chunk in re.split(r"([\u4e00-\u9fff]+|[a-z0-9][a-z0-9\-_.]*)", text):
        if not chunk or _PUNCT.fullmatch(chunk):
            continue
        if re.fullmatch(r"[\u4e00-\u9fff]+", chunk):
            tokens.extend(jieba.lcut_for_search(chunk))
        else:
            token = chunk.strip("-_.")
            if token:
                tokens.append(token)
    out = []
    for t in tokens:
        t = t.strip()
        if not t or t in STOPWORDS:
            continue
        if not re.search(r"[\u4e00-\u9fff0-9a-z]", t):
            continue
        out.append(t)
    return out


def tokenize(text: str) -> list[str]:
    """对外暴露的分词入口（测试会用它钉住行为）。"""
    return _segment_text(text)


def tokenize_batch(texts: Iterable[str]) -> list[list[str]]:
    return [tokenize(t) for t in texts]


@dataclass
class SparseHit:
    idx: int
    score: float


class BM25Index:
    """BM25 稀疏索引（倒排表实现）。

    * ``postings[term_id]`` = (doc_ids, term_freqs)
    * 打分：对每个查询词，把 ``idf * tf_norm`` 累加到命中文档上
    """

    def __init__(self, k1: float = 0.9, b: float = 0.4) -> None:
        self.k1 = k1
        self.b = b
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None
        self.doc_len: np.ndarray = np.zeros(0, dtype=np.float32)
        self.avgdl: float = 0.0
        self._postings: list[tuple[np.ndarray, np.ndarray]] = []
        self._n_docs = 0

    def fit(self, docs: Sequence[str]) -> BM25Index:
        tokenized = tokenize_batch(docs)
        self._n_docs = len(tokenized)

        # 先建词表与倒排的中间结构
        term_docs: dict[str, list[tuple[int, int]]] = defaultdict(list)
        lengths = np.zeros(self._n_docs, dtype=np.float32)
        for i, tokens in enumerate(tokenized):
            lengths[i] = max(1, len(tokens))
            tf: dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            for t, c in tf.items():
                term_docs[t].append((i, c))

        self.vocab = {t: vid for vid, t in enumerate(term_docs)}
        n = self._n_docs
        idf = np.zeros(len(self.vocab), dtype=np.float32)
        postings: list[tuple[np.ndarray, np.ndarray]] = [None] * len(self.vocab)  # type: ignore[list-item]
        for term, vid in self.vocab.items():
            pairs = term_docs[term]
            d = len(pairs)
            postings[vid] = (
                np.fromiter((p[0] for p in pairs), dtype=np.int32, count=d),
                np.fromiter((p[1] for p in pairs), dtype=np.float32, count=d),
            )
            # BM25+ 的 idf 变体，保证非负，避免短文档堆里出现负分
            idf[vid] = float(np.log(1 + (n - d + 0.5) / (d + 0.5)))

        self.doc_len = lengths
        self.avgdl = float(lengths.mean()) if n else 0.0
        self._postings = postings
        self.idf = idf
        return self

    def get_scores(self, query: str) -> np.ndarray:
        """全量打分（小语料 / 单元测试用）。"""
        scores = np.zeros(self._n_docs, dtype=np.float32)
        q_ids = [self.vocab[t] for t in tokenize(query) if t in self.vocab]
        if not q_ids or self.idf is None:
            return scores
        # 查询词去重（BM25 里同一个词重复出现按一次算，避免长查询刷分）
        for vid in dict.fromkeys(q_ids):
            doc_ids, freqs = self._postings[vid]
            norm = self.k1 * (1 - self.b + self.b * self.doc_len[doc_ids] / max(self.avgdl, 1e-6))
            contribution = freqs / (freqs + norm) * (self.k1 + 1) * self.idf[vid]
            np.add.at(scores, doc_ids, contribution)
        return scores

    def search(self, query: str, top_k: int = 100) -> list[SparseHit]:
        scores = self.get_scores(query)
        return top_from_scores(scores, top_k)

    def __len__(self) -> int:
        return self._n_docs


def top_from_scores(scores: np.ndarray, top_k: int) -> list[SparseHit]:
    """从打分数组取 top_k（用 argpartition 避免全量排序）。"""
    if scores.size == 0 or top_k <= 0:
        return []
    if top_k >= scores.size:
        order = np.argsort(-scores)
    else:
        part = np.argpartition(-scores, top_k - 1)[:top_k]
        order = part[np.argsort(-scores[part])]
    return [SparseHit(int(i), float(scores[i])) for i in order if scores[i] > 0][:top_k]
