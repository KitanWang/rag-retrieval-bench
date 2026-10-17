"""生成质量评测：忠实度（faithfulness）/ 答案相关性 / 上下文精度。

为什么不用 RAGAS 库直接跑：

* RAGAS 的默认实现会并发发起大量 LLM 调用且指标定义版本间有漂移，
  复现实验时经常出现「同一份数据两次跑出不同结论」；
* 更关键的是——**我们要能解释每一个分数是怎么来的**。

所以这里的做法是：**自己实现可解释的评测口径，并用固定 prompt 约束 LLM 只做判别**
（输出结构化判定，不让它自由发挥），失败时降级为基于字符串/引用的启发式打分。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from ragbench.llm import LLMClient

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Prompt 模板：固定、可审计
# --------------------------------------------------------------------------- #
ANSWER_PROMPT = """你是一个严谨的中文问答助手。请仅依据下面提供的资料回答问题。
如果资料不足以回答，请直接回答「资料不足，无法回答」，不要编造。

【资料】
{context}

【问题】
{question}

【回答要求】
1. 先给出直接答案，再补充必要的依据；
2. 不要输出「根据资料」这类套话；
3. 控制在 200 字以内。"""

FAITHFULNESS_PROMPT = """你是一个严格的答案核查员。给定【资料】和【回答】，
请把回答拆成若干条独立的陈述（claim），逐条判断能否在资料中找到依据。

只输出 JSON，不要任何其他文字，格式如下：
{{"claims": [{{"text": "陈述内容", "supported": true}}]}}

判定标准：
- 资料中明确写出或有直接可推出的依据 → supported = true
- 资料中没有、或只是常识性推断但与资料无关 → supported = false
- 回答若为「资料不足，无法回答」这类拒答，claims 输出空数组

【资料】
{context}

【回答】
{answer}"""

RELEVANCY_PROMPT = """你是一个答案相关性评估员。给定【问题】和该问题的一个【回答】，
判断这个回答是否切题（是否真的在回答被问的问题）。

注意：「资料不足，无法回答」不算切题；答非所问、只重复问题也不算切题。

只输出 JSON，不要任何其他文字：
{{"relevant": true, "reason": "一句话理由"}}

【问题】
{question}

【回答】
{answer}"""


def _extract_json(text: str) -> dict | None:
    """从可能带 markdown 围栏的回复里抠出第一个 JSON 对象。"""
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


@dataclass
class GenSample:
    qid: str
    question: str
    contexts: list[str]
    answer: str = ""
    claims: list[dict] = field(default_factory=list)
    faithful_score: float = 0.0
    relevant: float = 0.0
    context_precision: float = 0.0
    error: str = ""


def build_context(docs: list[str], max_chars: int = 3000) -> str:
    """把检索到的文档拼成上下文，按顺序编号并截断。"""
    parts, used = [], 0
    for i, d in enumerate(docs, 1):
        piece = f"[{i}] {d.strip()}"
        if used + len(piece) > max_chars:
            break
        parts.append(piece)
        used += len(piece)
    return "\n\n".join(parts)


def answer_question(llm: LLMClient, question: str, docs: list[str]) -> str:
    prompt = ANSWER_PROMPT.format(context=build_context(docs), question=question)
    return llm.chat([{"role": "user", "content": prompt}]).strip()


def score_faithfulness(llm: LLMClient, answer: str, docs: list[str]) -> tuple[float, list[dict]]:
    """忠实度 = 被支撑的陈述数 / 总陈述数。"""
    if not answer:
        return 0.0, []
    if is_refusal(answer):
        return 1.0, []  # 拒答不产生幻觉，按满分算（另用拒答率单独统计）
    prompt = FAITHFULNESS_PROMPT.format(context=build_context(docs), answer=answer)
    raw = llm.chat([{"role": "user", "content": prompt}])
    data = _extract_json(raw)
    if not data or "claims" not in data:
        return _heuristic_faithfulness(answer, docs), []
    claims = [c for c in data["claims"] if isinstance(c, dict) and c.get("text")]
    if not claims:
        return 1.0, []
    supported = sum(1 for c in claims if c.get("supported") is True)
    return supported / len(claims), claims


def score_relevancy(llm: LLMClient, question: str, answer: str) -> float:
    if not answer:
        return 0.0
    prompt = RELEVANCY_PROMPT.format(question=question, answer=answer)
    raw = llm.chat([{"role": "user", "content": prompt}])
    data = _extract_json(raw)
    if not data or "relevant" not in data:
        return _heuristic_relevancy(question, answer)
    return 1.0 if data.get("relevant") is True else 0.0


def is_refusal(answer: str) -> bool:
    patterns = ("资料不足", "无法回答", "没有找到", "无法确定", "资料中没有", "抱歉")
    return any(p in answer for p in patterns)


def context_precision(gold_ids: set[str], retrieved: list[str]) -> float:
    """检索到的文档里有多少是真正相关的（对应 RAGAS 的 context precision 简化口径）。"""
    if not retrieved:
        return 0.0
    return len(set(retrieved) & gold_ids) / len(retrieved)


# --------------------------------------------------------------------------- #
# 降级启发式：LLM 判定失败时的兜底，保证整批评测不会因为个别请求失败而空缺
# --------------------------------------------------------------------------- #
_STOP = set("的了和与及或是在有为对也就都而被把让从到中上下这那一个我们你们他们".split())


def _bigrams(text: str) -> set[str]:
    text = re.sub(r"[^\u4e00-\u9fffa-zA-Z0-9]", "", text)
    return {text[i : i + 2] for i in range(len(text) - 1)}


def _heuristic_faithfulness(answer: str, docs: list[str]) -> float:
    """句级二元组覆盖率：回答里的每个句子有多少能被上下文覆盖。"""
    ctx = _bigrams(" ".join(docs))
    if not ctx:
        return 0.0
    sents = [s for s in re.split(r"[。！？；\n]", answer) if len(s.strip()) > 4]
    if not sents:
        return 0.0
    covered = 0
    for s in sents:
        grams = _bigrams(s)
        if grams and len(grams & ctx) / len(grams) >= 0.5:
            covered += 1
    return covered / len(sents)


def _heuristic_relevancy(question: str, answer: str) -> float:
    q_grams = _bigrams(question) - _STOP
    a_grams = _bigrams(answer)
    if not q_grams:
        return 0.0
    return 1.0 if len(q_grams & a_grams) / len(q_grams) >= 0.25 else 0.0
