"""生成质量评测流水线。

对每个方案，用**该方案检索到的文档**作为上下文，让 LLM 回答问题，
再用固定的核查 prompt 打分。这样测的才是「检索 → 生成」的端到端质量，
而不是「拿到 gold 文档能不能答对」。
"""

from __future__ import annotations

import json
import logging
import random

from ragbench import data
from ragbench.config import Config
from ragbench.generation import (
    GenSample,
    answer_question,
    context_precision,
    is_refusal,
    score_faithfulness,
    score_relevancy,
)
from ragbench.llm import LLMClient, LLMConfig
from ragbench.pipeline import ALL_METHODS, RetrieverBundle

logger = logging.getLogger(__name__)


def run_generation_eval(
    cfg: Config,
    methods: list[str] | None = None,
    sample_size: int | None = None,
) -> dict:
    methods = list(methods or ALL_METHODS)
    sample_size = sample_size or cfg.eval.gen_sample_size

    corpus, queries, qrels = data.load_processed(cfg.data)
    gold = data.gold_map(qrels)
    usable = [q for q in queries["qid"].tolist() if q in gold]
    rng = random.Random(cfg.data.seed)
    rng.shuffle(usable)
    picked = sorted(usable[:sample_size])
    qtext = dict(zip(queries["qid"], queries["text"], strict=False))

    llm_cfg = LLMConfig.from_env(
        base_url=cfg.eval.llm_base_url,
        model=cfg.eval.llm_model,
        temperature=cfg.eval.llm_temperature,
        max_tokens=cfg.eval.llm_max_tokens,
        timeout=cfg.eval.llm_timeout,
        concurrency=cfg.eval.llm_concurrency,
    )
    llm = LLMClient(llm_cfg)
    logger.info(
        "生成评测：%d 条查询 × %d 种方案，模型 %s", len(picked), len(methods), llm_cfg.model
    )

    bundle = RetrieverBundle(cfg, corpus["pid"].tolist(), corpus["text"].tolist())
    pid2text = dict(zip(corpus["pid"], corpus["text"], strict=False))
    results: dict[str, dict] = {}
    retrieved_ids: dict[str, list[str]] = {}
    try:
        for name in methods:
            retriever = bundle.get(name)
            logger.info("== 方案 %s", name)
            samples: list[GenSample] = []
            for qid in picked:
                hits = retriever.search(qtext[qid], top_k=cfg.eval.gen_top_k)
                doc_ids = [h.pid for h in hits]
                docs = [pid2text[p] for p in doc_ids if p in pid2text]
                s = GenSample(
                    qid=qid,
                    question=qtext[qid],
                    contexts=docs,
                    context_precision=context_precision(gold[qid], doc_ids),
                )
                # 记录检索 id 与命中情况：后面算 hit rate 要用，不能事后反查
                retrieved_ids[s.qid] = doc_ids
                samples.append(s)

            # 生成与打分都走并发，顺序由 llm.map 保证
            answers = llm.map(lambda s: answer_question(llm, s.question, s.contexts), samples)
            for s, a in zip(samples, answers, strict=False):
                if isinstance(a, dict):
                    s.error = a.get("error", "")
                else:
                    s.answer = a

            faith = llm.map(lambda s: score_faithfulness(llm, s.answer, s.contexts), samples)
            rel = llm.map(lambda s: score_relevancy(llm, s.question, s.answer), samples)

            for s, f, r in zip(samples, faith, rel, strict=False):
                if isinstance(f, tuple):
                    s.faithful_score, s.claims = f
                if isinstance(r, float):
                    s.relevant = r

            results[name] = _aggregate(name, samples, gold, dict(retrieved_ids))
            logger.info(
                "   忠实度=%.3f 相关性=%.3f 上下文精度=%.3f",
                results[name]["faithfulness"],
                results[name]["relevancy"],
                results[name]["context_precision"],
            )
    finally:
        bundle.close()

    payload = {
        "model": llm_cfg.model,
        "temperature": llm_cfg.temperature,
        "n_samples": len(picked),
        "top_k": cfg.eval.gen_top_k,
        "results": results,
    }
    out = cfg.output_dir / "generation_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("生成评测结果已写入 %s", out)
    return payload


def _aggregate(
    name: str,
    samples: list[GenSample],
    gold: dict[str, set[str]],
    retrieved: dict[str, list[str]],
) -> dict:
    n = max(1, len(samples))
    errors = [s for s in samples if s.error]
    hit = sum(1 for s in samples if set(retrieved.get(s.qid, [])) & gold.get(s.qid, set()))
    return {
        "n": len(samples),
        "faithfulness": sum(s.faithful_score for s in samples) / n,
        "relevancy": sum(s.relevant for s in samples) / n,
        "context_precision": sum(s.context_precision for s in samples) / n,
        "retrieval_hit_rate": hit / n,
        "refusal_rate": sum(1 for s in samples if is_refusal(s.answer)) / n,
        "error_rate": len(errors) / n,
        "samples": [
            {
                "qid": s.qid,
                "question": s.question,
                "answer": s.answer,
                "faithfulness": s.faithful_score,
                "relevant": s.relevant,
                "context_precision": s.context_precision,
                "error": s.error,
            }
            for s in samples
        ],
    }
