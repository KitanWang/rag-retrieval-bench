"""分词与 BM25 的单元测试。

BM25 里最容易出错的两处：
1. 倒排表打分与「逐个文档重算」是否等价（重构后极易产生偏差）；
2. 文档长度归一化的方向是否正确（长文档该被惩罚）。
"""

from __future__ import annotations

import numpy as np
import pytest

from ragbench.sparse import BM25Index, tokenize, top_from_scores


class TestTokenize:
    def test_chinese_is_segmented(self):
        toks = tokenize("国家法定节假日共多少天")
        assert len(toks) >= 2
        assert all(isinstance(t, str) for t in toks)

    def test_lowercase_english(self):
        assert "abc" in tokenize("ABC技术")

    def test_model_number_kept(self):
        assert "hx-1234" in tokenize("型号 HX-1234 的报价")

    def test_punctuation_dropped(self):
        toks = tokenize("你好，世界。")
        assert "，" not in toks and "。" not in toks

    def test_stopwords_removed(self):
        toks = tokenize("请问这个怎么用")
        assert "请问" not in toks and "怎么" not in toks

    def test_empty(self):
        assert tokenize("") == []


class TestBM25:
    @pytest.fixture
    def index(self):
        docs = [
            "退款流程：在订单页面点击申请退款，三个工作日内到账",
            "退款多久到账？一般三个工作日左右",
            "如何修改收货地址",
            "会员权益说明与积分兑换规则",
        ]
        return BM25Index(k1=1.2, b=0.75).fit(docs), docs

    def test_relevant_doc_scores_higher(self, index):
        idx, docs = index
        scores = idx.get_scores("退款到账时间")
        assert scores[0] > scores[2]
        assert scores[1] > scores[2]

    def test_unseen_query_returns_zeros(self, index):
        idx, _ = index
        scores = idx.get_scores("完全不相关的词汇集合")
        assert np.all(scores == 0)

    def test_scores_non_negative(self, index):
        idx, _ = index
        assert (idx.get_scores("退款") >= 0).all()

    def test_search_returns_sorted_and_bounded(self, index):
        idx, _ = index
        hits = idx.search("退款", top_k=2)
        assert len(hits) <= 2
        assert all(h.score > 0 for h in hits)
        assert hits == sorted(hits, key=lambda h: -h.score)

    def test_inverted_index_matches_naive(self, index):
        """倒排表打分必须与「逐文档暴力计算」完全一致。"""
        idx, docs = index
        query = "退款 到账"
        got = idx.get_scores(query)

        # 朴素实现
        q_ids = [idx.vocab[t] for t in tokenize(query) if t in idx.vocab]
        expected = np.zeros(len(docs), dtype=np.float32)
        for vid in dict.fromkeys(q_ids):
            doc_ids, freqs = idx._postings[vid]
            for d, f in zip(doc_ids, freqs, strict=False):
                norm = idx.k1 * (1 - idx.b + idx.b * idx.doc_len[d] / idx.avgdl)
                expected[d] += float(f / (f + norm) * (idx.k1 + 1) * idx.idf[vid])
        assert np.allclose(got, expected, atol=1e-5)

    def test_long_document_penalized(self):
        """同样命中一次，长文档得分应低于短文档（b>0 时）。"""
        idx = BM25Index(k1=1.2, b=0.75).fit(["退款", "退款" + "无关内容" * 50])
        scores = idx.get_scores("退款")
        assert scores[0] > scores[1]

    def test_empty_corpus(self):
        idx = BM25Index().fit([])
        assert len(idx) == 0
        assert idx.search("退款", top_k=5) == []

    def test_top_from_scores_filters_zero(self):
        hits = top_from_scores(np.array([0.0, 1.0, 0.5], dtype=np.float32), 10)
        assert len(hits) == 2
        assert hits[0].score == 1.0

    def test_top_from_scores_respects_k(self):
        scores = np.arange(10, dtype=np.float32)
        hits = top_from_scores(scores, 3)
        assert len(hits) == 3
        assert [h.score for h in hits] == [9.0, 8.0, 7.0]
