"""端到端流程测试：数据切分、融合函数、路由判定。

这些是「结论依赖的逻辑」，比单点函数更需要钉住：
切分错了指标就失真，融合错了方案对比就没有意义。
"""

from __future__ import annotations

import pandas as pd
import pytest

from ragbench.config import Config, DataConfig
from ragbench.data import build_eval_set, gold_map, normalize_qrels
from ragbench.retrievers import (
    QueryRouter,
    min_max_normalize,
    reciprocal_rank_fusion,
    weighted_fusion,
)
from ragbench.types import Hit


def _fake_data(n_corpus: int = 200, n_queries: int = 20):
    corpus = pd.DataFrame(
        {
            "id": [f"d{i}" for i in range(n_corpus)],
            "text": [f"文档内容 {i} 关于退款与订单" for i in range(n_corpus)],
        }
    )
    queries = pd.DataFrame(
        {"id": [f"q{i}" for i in range(n_queries)], "text": [f"问题 {i}" for i in range(n_queries)]}
    )
    rows = []
    for i in range(n_queries):
        for j in range(3):
            rows.append({"qid": f"q{i}", "pid": f"d{(i * 3 + j) % n_corpus}", "score": 1})
    qrels = pd.DataFrame(rows)
    return corpus, queries, qrels


class TestBuildEvalSet:
    def test_gold_closure_kept(self):
        """子采样后，所有被选中 query 的正例文档必须都在语料里。"""
        corpus, queries, qrels = _fake_data()
        cfg = DataConfig(corpus_size=50, query_size=10, seed=1)
        sub_c, sub_q, sub_r = build_eval_set(corpus, queries, qrels, cfg)
        assert set(sub_r["pid"]) <= set(sub_c["pid"])
        assert set(sub_r["qid"]) <= set(sub_q["qid"])

    def test_corpus_size_bounded(self):
        corpus, queries, qrels = _fake_data()
        cfg = DataConfig(corpus_size=50, query_size=10, seed=1)
        sub_c, _, _ = build_eval_set(corpus, queries, qrels, cfg)
        assert len(sub_c) <= 50

    def test_same_seed_is_deterministic(self):
        corpus, queries, qrels = _fake_data()
        a = build_eval_set(corpus, queries, qrels, DataConfig(corpus_size=50, query_size=10, seed=7))
        b = build_eval_set(corpus, queries, qrels, DataConfig(corpus_size=50, query_size=10, seed=7))
        assert list(a[2]["qid"]) == list(b[2]["qid"])

    def test_different_seed_differs(self):
        corpus, queries, qrels = _fake_data(n_queries=40)
        a = build_eval_set(corpus, queries, qrels, DataConfig(corpus_size=80, query_size=10, seed=1))
        b = build_eval_set(corpus, queries, qrels, DataConfig(corpus_size=80, query_size=10, seed=2))
        assert set(a[1]["qid"]) != set(b[1]["qid"])

    def test_only_queries_with_positive_qrels(self):
        corpus, queries, qrels = _fake_data()
        qrels = qrels[qrels["qid"] != "q0"]  # q0 没有正例
        cfg = DataConfig(corpus_size=100, query_size=30, seed=3)
        _, sub_q, _ = build_eval_set(corpus, queries, qrels, cfg)
        assert "q0" not in set(sub_q["qid"])

    def test_drops_qrels_pointing_to_missing_docs(self):
        corpus, queries, qrels = _fake_data()
        qrels.loc[0, "pid"] = "not-exists"
        _, _, sub_r = build_eval_set(corpus, queries, qrels, DataConfig(corpus_size=100, query_size=30))
        assert "not-exists" not in set(sub_r["pid"])


class TestQrelsNormalization:
    def test_accepts_alternate_column_names(self):
        df = pd.DataFrame({"query-id": ["a"], "corpus-id": ["b"], "score": [1]})
        out = normalize_qrels(df)
        assert list(out.columns) == ["qid", "pid", "score"]

    def test_filters_non_positive(self):
        df = pd.DataFrame({"qid": ["a", "a"], "pid": ["b", "c"], "score": [1, 0]})
        assert len(normalize_qrels(df)) == 1

    def test_missing_column_raises(self):
        with pytest.raises(ValueError):
            normalize_qrels(pd.DataFrame({"qid": ["a"]}))

    def test_gold_map(self):
        df = pd.DataFrame({"qid": ["a", "a", "b"], "pid": ["x", "y", "z"], "score": [1, 1, 1]})
        g = gold_map(df)
        assert g["a"] == {"x", "y"} and g["b"] == {"z"}


class TestFusion:
    def test_rrf_orders_by_rank_agreement(self):
        a = [Hit("x", 0.9), Hit("y", 0.8)]
        b = [Hit("x", 0.7), Hit("z", 0.6)]
        fused = reciprocal_rank_fusion([a, b], k=60)
        assert fused[0].pid == "x"

    def test_rrf_score_formula(self):
        fused = reciprocal_rank_fusion([[Hit("x", 1.0)]], k=60)
        assert fused[0].score == pytest.approx(1 / 61)

    def test_rrf_weights(self):
        a = [Hit("x", 1.0)]
        b = [Hit("y", 1.0)]
        fused = reciprocal_rank_fusion([a, b], k=60, weights=[3.0, 1.0])
        assert fused[0].pid == "x"

    def test_rrf_rejects_mismatched_weights(self):
        with pytest.raises(ValueError):
            reciprocal_rank_fusion([[Hit("x", 1.0)]], k=60, weights=[1.0, 2.0])

    def test_rrf_handles_empty(self):
        assert reciprocal_rank_fusion([[], []], k=60) == []

    def test_min_max_normalize(self):
        out = min_max_normalize([Hit("a", 2.0), Hit("b", 4.0)])
        assert out[0].score == 0.0 and out[1].score == 1.0

    def test_min_max_all_equal(self):
        out = min_max_normalize([Hit("a", 3.0), Hit("b", 3.0)])
        assert all(h.score == 1.0 for h in out)

    def test_weighted_fusion_prefers_heavier_branch(self):
        a = [Hit("x", 1.0)]
        b = [Hit("y", 1.0)]
        out = weighted_fusion([a, b], weights=[0.9, 0.1], top_k=2)
        assert out[0].pid == "x"

    def test_weighted_fusion_respects_top_k(self):
        a = [Hit(f"d{i}", float(10 - i)) for i in range(10)]
        out = weighted_fusion([a], weights=[1.0], top_k=3)
        assert len(out) == 3


class TestRouter:
    @pytest.fixture
    def router(self):
        return QueryRouter(Config())

    def test_code_like_query_detected(self, router):
        assert router.is_code_like("HX-1234 多少钱")
        assert router.is_code_like("iPhone 15 Pro 价格")

    def test_natural_query_not_code_like(self, router):
        assert not router.is_code_like("如何修改收货地址")

    def test_agreement_high_when_identical(self, router):
        hits = [Hit("a", 0.9), Hit("b", 0.8), Hit("c", 0.7)]
        assert router.agreement(hits, hits) > 0.8

    def test_agreement_low_when_disjoint(self, router):
        a = [Hit("a", 0.9), Hit("b", 0.8), Hit("c", 0.7)]
        b = [Hit("x", 5.0), Hit("y", 4.0), Hit("z", 3.0)]
        assert router.agreement(a, b) < 0.3

    def test_needs_rerank_on_disagreement(self, router):
        a = [Hit("a", 0.9), Hit("b", 0.8), Hit("c", 0.7)]
        b = [Hit("x", 5.0), Hit("y", 4.0), Hit("z", 3.0)]
        assert router.needs_rerank(a, b)

    def test_no_rerank_on_agreement(self, router):
        hits = [Hit("a", 0.9), Hit("b", 0.8), Hit("c", 0.7)]
        assert not router.needs_rerank(hits, hits)

    def test_agreement_with_empty_branch_is_zero(self, router):
        assert router.agreement([Hit("a", 1.0)], []) == 0.0


class TestConfig:
    def test_default_config_builds(self):
        cfg = Config()
        assert cfg.data.corpus_size > 0
        assert cfg.eval.top_ks == (1, 5, 10)

    def test_from_dict_overrides(self):
        cfg = Config.from_dict({"data": {"corpus_size": 123}})
        assert cfg.data.corpus_size == 123

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="未知配置项"):
            Config.from_dict({"data": {"nonexistent": 1}})

    def test_to_dict_is_json_safe(self):
        import json

        json.dumps(Config().to_dict())  # Path 已转 str，不应抛异常
