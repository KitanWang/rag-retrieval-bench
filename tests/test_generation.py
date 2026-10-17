"""生成评测相关逻辑的单元测试。

重点测「不依赖 LLM 的部分」：JSON 抽取、拒答判定、上下文拼接、
启发式兜底——这些决定了评测在 LLM 不稳定时是否还能给出可用结果。
"""

from __future__ import annotations

import pytest

from ragbench.generation import (
    _extract_json,
    _heuristic_faithfulness,
    _heuristic_relevancy,
    build_context,
    context_precision,
    is_refusal,
)


class TestExtractJson:
    def test_plain_json(self):
        assert _extract_json('{"a": 1}') == {"a": 1}

    def test_markdown_fenced(self):
        text = '```json\n{"claims": [{"text": "x", "supported": true}]}\n```'
        assert _extract_json(text)["claims"][0]["supported"] is True

    def test_json_with_surrounding_prose(self):
        text = '好的，结果如下：{"relevant": true, "reason": "切题"} 以上。'
        assert _extract_json(text)["relevant"] is True

    def test_invalid_returns_none(self):
        assert _extract_json("完全不是 JSON") is None
        assert _extract_json("") is None
        assert _extract_json("{broken") is None

    def test_nested(self):
        text = '{"a": {"b": [1, 2]}}'
        assert _extract_json(text) == {"a": {"b": [1, 2]}}


class TestRefusal:
    @pytest.mark.parametrize(
        "answer",
        ["资料不足，无法回答", "抱歉，资料中没有提到这个", "无法确定", "没有找到相关信息"],
    )
    def test_refusals_detected(self, answer):
        assert is_refusal(answer)

    @pytest.mark.parametrize(
        "answer",
        ["退款一般三个工作日到账", "可以在设置里关闭自动续费", ""],
    )
    def test_normal_answers_not_refusal(self, answer):
        assert not is_refusal(answer)


class TestBuildContext:
    def test_numbers_documents(self):
        ctx = build_context(["第一段", "第二段"])
        assert "[1] 第一段" in ctx and "[2] 第二段" in ctx

    def test_respects_char_budget(self):
        docs = ["很长的内容" * 100 for _ in range(10)]
        ctx = build_context(docs, max_chars=500)
        assert len(ctx) <= 600  # 允许编号与换行的少量开销

    def test_empty(self):
        assert build_context([]) == ""


class TestContextPrecision:
    def test_all_relevant(self):
        assert context_precision({"a", "b"}, ["a", "b"]) == 1.0

    def test_half_relevant(self):
        assert context_precision({"a"}, ["a", "x"]) == 0.5

    def test_none_relevant(self):
        assert context_precision({"a"}, ["x", "y"]) == 0.0

    def test_empty_retrieved(self):
        assert context_precision({"a"}, []) == 0.0


class TestHeuristicFallback:
    def test_faithful_answer_scores_high(self):
        docs = ["退款流程：在订单页面点击申请退款，三个工作日内到账。"]
        answer = "在订单页面点击申请退款，三个工作日内到账。"
        assert _heuristic_faithfulness(answer, docs) >= 0.9

    def test_unrelated_answer_scores_low(self):
        docs = ["退款流程：在订单页面点击申请退款。"]
        answer = "今天天气很好，适合出门旅游散心。"
        assert _heuristic_faithfulness(answer, docs) <= 0.5

    def test_empty_docs(self):
        assert _heuristic_faithfulness("任意回答", []) == 0.0

    def test_relevancy_overlapping(self):
        assert _heuristic_relevancy("如何修改收货地址", "修改收货地址的方法如下") == 1.0

    def test_relevancy_unrelated(self):
        assert _heuristic_relevancy("如何修改收货地址", "今天天气不错") == 0.0


class TestStreamDecoding:
    """SSE 流的编码回归测试。

    真实缺陷：服务端在 text/event-stream 上返回 charset=ISO-8859-1，
    requests 会照此解码，中文变成 "ä½ å¥½" 这类乱码。
    不会报错，但会让忠实度判定全盘失真。
    """

    def test_utf8_chunks_decoded_correctly(self):
        from ragbench.llm import LLMClient

        class _FakeResp:
            encoding = "ISO-8859-1"  # 故意模拟服务端的错误声明

            def iter_lines(self, *a, **kw):
                payload = '{"choices":[{"delta":{"content":"你好世界"}}]}'
                return [
                    ("data: " + payload).encode("utf-8"),
                    b"",
                    b"data: [DONE]",
                ]

        out = LLMClient._parse_stream(_FakeResp())
        assert out == "你好世界"

    def test_ignores_non_data_lines(self):
        from ragbench.llm import LLMClient

        class _FakeResp:
            def iter_lines(self, *a, **kw):
                chunk = '{"choices":[{"delta":{"content":"答"}}]}'
                return [b"event: ping", ("data: " + chunk).encode("utf-8"), b"data: [DONE]"]

        assert LLMClient._parse_stream(_FakeResp()) == "答"

    def test_multiple_chunks_concatenated(self):
        from ragbench.llm import LLMClient

        class _FakeResp:
            def iter_lines(self, *a, **kw):
                lines: list[bytes] = []
                for piece in ("一", "二", "三"):
                    body = f'{{"choices":[{{"delta":{{"content":"{piece}"}}}}]}}'
                    lines.append(("data: " + body).encode("utf-8"))
                lines.append(b"data: [DONE]")
                return lines

        assert LLMClient._parse_stream(_FakeResp()) == "一二三"
