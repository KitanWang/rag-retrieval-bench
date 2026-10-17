"""指标函数的单元测试：这些函数决定了全部实验结论，必须钉死。"""

from __future__ import annotations

import math

import pytest

from ragbench.metrics import (
    evaluate_rankings,
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


class TestRecallAtK:
    def test_all_gold_found(self):
        assert recall_at_k(["a", "b"], {"a", "b"}, 2) == 1.0

    def test_partial(self):
        assert recall_at_k(["a", "x"], {"a", "b"}, 2) == 0.5

    def test_respects_k(self):
        # b 排在第 3 位，k=2 时不该被算进去
        assert recall_at_k(["a", "x", "b"], {"a", "b"}, 2) == 0.5

    def test_empty_gold_is_zero_not_error(self):
        assert recall_at_k(["a"], set(), 5) == 0.0

    def test_duplicates_in_ranking_count_once(self):
        # 同一文档重复出现不能把 recall 刷过 1
        assert recall_at_k(["a", "a", "a"], {"a", "b"}, 3) == 0.5


class TestPrecisionHitMrr:
    def test_precision_denominator_is_k(self):
        assert precision_at_k(["a"], {"a"}, 4) == 0.25

    def test_hit_rate_any_match(self):
        assert hit_rate_at_k(["x", "y", "a"], {"a"}, 3) == 1.0
        assert hit_rate_at_k(["x", "y", "a"], {"a"}, 2) == 0.0

    def test_mrr_takes_first_hit(self):
        assert mrr_at_k(["x", "a", "b"], {"a", "b"}, 5) == 0.5

    def test_mrr_zero_when_no_hit(self):
        assert mrr_at_k(["x"], {"a"}, 5) == 0.0


class TestNdcg:
    def test_perfect_ranking_is_one(self):
        assert ndcg_at_k(["a", "b"], {"a", "b"}, 2) == pytest.approx(1.0)

    def test_reversed_ranking_lower(self):
        """只有 1 个正例时，位置越靠前 nDCG 越高。

        注意：二值相关性下，若 top-k 内已包含**全部**正例，
        则任何排列的 nDCG 都是 1.0（这是 nDCG 的定义，不是缺陷），
        所以这里用一个正例 + 干扰项来验证"位置"确实起作用。
        """
        first = ndcg_at_k(["a", "x"], {"a"}, 2)
        second = ndcg_at_k(["x", "a"], {"a"}, 2)
        assert first > second

    def test_all_gold_in_topk_gives_one(self):
        # 两个正例都进 top-2，互换顺序不改变 nDCG
        assert ndcg_at_k(["a", "b"], {"a", "b"}, 2) == pytest.approx(1.0)
        assert ndcg_at_k(["b", "a"], {"a", "b"}, 2) == pytest.approx(1.0)

    def test_idcg_uses_min_gold_and_k(self):
        # 1 个正例排在首位 → 无论 k 多大都是 1.0
        assert ndcg_at_k(["a", "x", "y"], {"a"}, 10) == pytest.approx(1.0)

    def test_known_value(self):
        # 正例在位置 2：dcg = 1/log2(3)，idcg = 1 → 0.6309
        assert ndcg_at_k(["x", "a"], {"a"}, 2) == pytest.approx(1 / math.log2(3))


class TestEvaluateRankings:
    def test_aggregates_mean(self):
        rankings = {"q1": ["a"], "q2": ["x"]}
        gold = {"q1": {"a"}, "q2": {"a"}}
        out = evaluate_rankings(rankings, gold, ks=(1,))
        assert out["recall@1"] == pytest.approx(0.5)
        assert out["hit@1"] == pytest.approx(0.5)

    def test_missing_qrel_raises(self):
        with pytest.raises(KeyError):
            evaluate_rankings({"q1": ["a"]}, {}, ks=(1,))

    def test_all_k_present(self):
        out = evaluate_rankings({"q": ["a"]}, {"q": {"a"}}, ks=(1, 5, 10))
        assert set(out) == {
            "recall@1", "recall@5", "recall@10",
            "mrr@1", "mrr@5", "mrr@10",
            "ndcg@1", "ndcg@5", "ndcg@10",
            "hit@1", "hit@5", "hit@10",
        }
