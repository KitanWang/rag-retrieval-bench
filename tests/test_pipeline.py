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
        a = build_eval_set(
            corpus, queries, qrels, DataConfig(corpus_size=50, query_size=10, seed=7)
        )
        b = build_eval_set(
            corpus, queries, qrels, DataConfig(corpus_size=50, query_size=10, seed=7)
        )
        assert list(a[2]["qid"]) == list(b[2]["qid"])

    def test_different_seed_differs(self):
        corpus, queries, qrels = _fake_data(n_queries=40)
        a = build_eval_set(
            corpus, queries, qrels, DataConfig(corpus_size=80, query_size=10, seed=1)
        )
        b = build_eval_set(
            corpus, queries, qrels, DataConfig(corpus_size=80, query_size=10, seed=2)
        )
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
        _, _, sub_r = build_eval_set(
            corpus, queries, qrels, DataConfig(corpus_size=100, query_size=30)
        )
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


class TestRouterCascade:
    """路由升级分支的两个关键不变量。

    这两条都是实测暴露出来的缺陷，必须用测试钉住，
    否则后人改路由时很容易又退化成「等于没路由」。
    """

    @pytest.fixture
    def stub_router(self):
        from ragbench.retrievers import (
            BM25Retriever,
            DenseRetriever,
            HybridRerankRetriever,
            HybridRetriever,
            RouterRetriever,
        )

        cfg = Config()
        ids = [f"d{i}" for i in range(20)]
        texts = [f"文档 {i}" for i in range(20)]

        class _StubDense(DenseRetriever):
            def __init__(self):  # noqa: D107 - 跳过真实模型加载
                self.name = "dense"

            def search(self, query, top_k=10):
                # 稠密分支只返回 d0..d9
                return [Hit(f"d{i}", 1.0 - i * 0.01) for i in range(min(10, top_k))]

        class _StubSparse(BM25Retriever):
            def __init__(self):  # noqa: D107
                self.name = "bm25"
                self.seen: list[list[Hit]] = []

            def search(self, query, top_k=10):
                # 稀疏分支只返回完全不同的 d10..d19
                return [Hit(f"d{10 + i}", 1.0 - i * 0.01) for i in range(min(10, top_k))]

        class _StubRerank(HybridRerankRetriever):
            def __init__(self):  # noqa: D107
                self.calls: list[int] = []

            def rerank_hits(self, query, cand, top_k):
                self.calls.append(len(cand))
                return cand[:top_k]

        dense = _StubDense()
        sparse = _StubSparse()
        rerank = _StubRerank()
        hybrid = HybridRetriever(cfg, ids, texts, dense, sparse)
        router = RouterRetriever(cfg, ids, texts, dense, sparse, hybrid, rerank)
        return router, rerank, cfg

    def test_upgrade_keeps_both_branches(self, stub_router):
        """升级重排的候选必须同时包含稠密与稀疏两路的文档。

        回归用例：早期实现直接把两个列表首尾相接再截断，
        稀疏分支（排在后面）会被整段丢掉。
        """
        router, rerank, _ = stub_router
        route, hits = router.decide("任意查询", top_k=10)
        assert route == "hybrid_rerank"
        pids = {h.pid for h in hits}
        assert any(p.startswith("d0") or p in {f"d{i}" for i in range(10)} for p in pids)
        assert any(p in {f"d{i}" for i in range(10, 20)} for p in pids)

    def test_upgrade_reranks_same_candidate_count_as_full_rerank(self, stub_router):
        """路由升级送进重排的候选数，必须与独立重排方案相同。

        ## 这条测试改过一次，原因值得记下来

        原来的断言是 `rerank_candidates < rerank_top_k`——即「路由升级应当少重排一些」。
        听起来合理，实际却制造了一个**不公平的对比**：
        `hybrid_rerank` 重排 50 篇、`router` 只重排 10 篇，于是 router 的延迟天然低 5 倍。
        这个「节省」完全来自少干活，不是路由策略的功劳，质量下降（Recall 0.74 vs 0.81）
        也被掩盖成了「用一点质量换大量延迟」。

        正确的对比条件：**同样的候选规模**下，只在「哪些查询需要升级」这一维度上比较。
        候选数不同时，任何延迟/质量的差异都不可归因。
        """
        router, rerank, cfg = stub_router
        router.decide("任意查询", top_k=10)
        assert rerank.calls, "升级分支未被触发，测试前提不成立"
        # stub 语料只有 20 篇，两路各返回 10 篇且互不重叠，故融合后上限是 20。
        # 断言「等于 min(配置, 可用候选数)」而不是硬编码，既能钉住「用满候选池」，
        # 又不会因为 stub 数据规模而与配置脱钩。
        expected = min(cfg.router.rerank_candidates, 20)
        assert rerank.calls[0] == expected, (
            f"升级时应当把所有可用候选（{expected}）都送去重排，实际只送了 {rerank.calls[0]}"
        )
        assert cfg.router.rerank_candidates == cfg.retriever.rerank_top_k

    def test_agreement_path_skips_rerank(self):
        """两路完全一致时不应触发重排。"""
        from ragbench.retrievers import HybridRerankRetriever, HybridRetriever, RouterRetriever

        cfg = Config()
        ids = [f"d{i}" for i in range(20)]
        texts = [f"文档 {i}" for i in range(20)]

        class _Same:
            name = "same"

            def search(self, query, top_k=10):
                return [Hit(f"d{i}", 1.0 - i * 0.01) for i in range(min(10, top_k))]

        class _Rerank(HybridRerankRetriever):
            def __init__(self):  # noqa: D107
                self.calls = 0

            def rerank_hits(self, query, cand, top_k):
                self.calls += 1
                return cand[:top_k]

        rerank = _Rerank()
        hybrid = HybridRetriever(cfg, ids, texts, _Same(), _Same())
        router = RouterRetriever(cfg, ids, texts, _Same(), _Same(), hybrid, rerank)
        route, _ = router.decide("任意查询", top_k=10)
        assert route == "hybrid"
        assert rerank.calls == 0


def test_router_rerank_candidates_matches_rerank_top_k() -> None:
    """router 的升级候选数必须与独立重排方案一致。

    这条测试来自一次**实验设计缺陷**：`router.rerank_candidates` 默认是 10，
    而 `retriever.rerank_top_k` 是 50。于是 router 每次只重排 10 篇，
    延迟看起来比 hybrid_rerank 低 5 倍（1.6s vs 6.9s），质量也更差
    （Recall@5 0.7425 vs 0.8095）。

    当时那份报告会得出「路由省了 72% 延迟」的错误结论——实际上省下来的延迟
    完全来自「少重排 40 篇文档」，和路由策略本身无关。
    这类「对比条件不一致」的坑在调参实验里极常见，必须用测试钉死。
    """
    from ragbench.config import Config

    cfg = Config.load("configs/default.yaml")
    assert cfg.router.rerank_candidates == cfg.retriever.rerank_top_k, (
        f"router 的重排候选数（{cfg.router.rerank_candidates}）"
        f"必须等于 rerank_top_k（{cfg.retriever.rerank_top_k}），否则两种方案的对比不公平"
    )


def test_router_upgrade_uses_full_candidate_pool() -> None:
    """router 升级时，实际送进 cross-encoder 的候选数应当等于 rerank_candidates。"""
    import inspect

    from ragbench import retrievers

    src = inspect.getsource(retrievers.RouterRetriever.decide)
    assert "rerank_candidates" in src, "router 的升级路径应当读取 rerank_candidates"
    # 回归点：不能出现「首尾相接后截断」这种让稠密分支占满名额的写法
    assert "d_hits + s_hits" not in src, (
        "升级路径不能直接把两路候选首尾相接——那会让稠密分支占满名额，"
        "稀疏分支的候选全被丢掉，等于没做混合"
    )
