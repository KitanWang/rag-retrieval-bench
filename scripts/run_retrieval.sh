#!/usr/bin/env bash
# 端到端跑一遍检索评测并生成报告。
#
# 为什么单独放一个脚本：
#   1. 统一去掉 NO_PROXY（本机代理与 httpx 的 URL 解析不兼容，会直接抛异常）；
#   2. 固定 HF 离线模式，避免评测过程中因为联网检查卡住。
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${1:-configs/default.yaml}"

echo "== 构建评测集 =="
env -u NO_PROXY -u no_proxy \
  PYTHONPATH=src HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" \
  venv/bin/python -m ragbench.cli --config "$CONFIG" prepare

echo "== 检索评测 =="
env -u NO_PROXY -u no_proxy \
  PYTHONPATH=src HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" \
  venv/bin/python -m ragbench.cli --config "$CONFIG" retrieve

echo "== 生成报告 =="
env -u NO_PROXY -u no_proxy \
  PYTHONPATH=src \
  venv/bin/python -m ragbench.cli --config "$CONFIG" report

echo "完成：artifacts/RESULTS.md"
