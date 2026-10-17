"""命令行入口：``ragbench <command>``。

命令
----
``prepare``      下载并构建评测集（corpus / queries / qrels 落盘）
``index``        计算嵌入、建 Milvus 索引、建 BM25 索引
``retrieve``     跑检索评测，输出指标 JSON
``generate``     跑生成质量评测（需要 LLM API）
``report``       把结果渲染成 Markdown 报告
``all``          一键串起 prepare → index → retrieve
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ragbench.config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
# 第三方库的 INFO 日志（尤其 httpx 的每次请求）会把评测输出淹没
for noisy in ("httpx", "urllib3", "huggingface_hub", "sentence_transformers", "datasets"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logger = logging.getLogger("ragbench")


def cmd_prepare(cfg: Config, args: argparse.Namespace) -> int:
    from ragbench import data

    stats = data.prepare(cfg.data, force=args.force)
    logger.info("评测集构建完成：%s", stats)
    return 0


def cmd_retrieve(cfg: Config, args: argparse.Namespace) -> int:
    from ragbench.pipeline import run_retrieval_eval

    run_retrieval_eval(cfg, methods=args.methods)
    from ragbench.report import build_report

    build_report(cfg, cfg.output_dir / "RESULTS.md")
    logger.info("检索评测完成，结果见 %s", cfg.output_dir / "RESULTS.md")
    return 0


def cmd_generate(cfg: Config, args: argparse.Namespace) -> int:
    from ragbench.genpipeline import run_generation_eval

    run_generation_eval(cfg, methods=args.methods, sample_size=args.sample)
    logger.info("生成评测完成，结果见 %s", cfg.output_dir / "generation_results.json")
    return 0


def cmd_report(cfg: Config, args: argparse.Namespace) -> int:
    from ragbench.report import build_report

    path = build_report(cfg, cfg.output_dir / "RESULTS.md")
    logger.info("报告已生成：%s", path)
    return 0


def cmd_all(cfg: Config, args: argparse.Namespace) -> int:
    from ragbench import data
    from ragbench.pipeline import run_retrieval_eval

    if not cfg.data.corpus_path.exists() or args.force:
        stats = data.prepare(cfg.data, force=args.force)
        logger.info("评测集已构建：%s", stats)
    run_retrieval_eval(cfg, methods=args.methods)
    logger.info("全流程完成，结果见 %s", cfg.output_dir / "retrieval_results.json")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ragbench", description="中文 RAG 检索策略对比基准")
    p.add_argument("--config", type=Path, default=None, help="YAML 配置文件路径")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("prepare", help="下载并构建评测集")
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=cmd_prepare)

    sa = sub.add_parser("all", help="prepare + retrieve")
    sa.add_argument("--force", action="store_true")
    sa.add_argument("--methods", nargs="*", default=None)
    sa.set_defaults(func=cmd_all)

    sr = sub.add_parser("retrieve", help="跑检索评测")
    sr.add_argument("--methods", nargs="*", default=None)
    sr.set_defaults(func=cmd_retrieve)

    sg = sub.add_parser("generate", help="跑生成质量评测（需要 LLM API Key）")
    sg.add_argument("--methods", nargs="*", default=None)
    sg.add_argument("--sample", type=int, default=None, help="采样查询数")
    sg.set_defaults(func=cmd_generate)

    sp2 = sub.add_parser("report", help="生成 Markdown 报告")
    sp2.set_defaults(func=cmd_report)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    cfg = Config.load(args.config)
    return int(args.func(cfg, args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
