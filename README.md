# RAG 检索策略对比基准

> 在**同一份中文语料、同一批评测查询、同一套指标**下，横向对比五种 RAG 检索方案：
> 朴素向量检索、BM25、混合检索（RRF）、混合检索 + 重排、策略路由。
>
> 目标不是刷榜，而是回答一个工程问题：**混合检索和重排到底值不值得上？贵多少？**

---

## 1. 为什么做这个项目

RAG 工程里最常见的一句结论是「混合检索 + 重排效果更好」，但**好在哪个指标、好多少、代价是什么**，
往往拿不出可复现的数字。这个项目把这些拆开量化：

| 要回答的问题 | 对应实验 |
| :--- | :--- |
| 纯向量检索的召回瓶颈在哪？ | `dense` vs `bm25` 的失败查询对照 |
| 混合检索真的能提升吗？ | `hybrid`（RRF 融合）vs 单路 |
| 重排带来多少提升、多少延迟？ | `hybrid_rerank` 的指标与 p50/p95 延迟 |
| 能不能只对部分查询付重排的钱？ | `router` 的级联路由 + 分支占比 |

---

## 2. 五种方案

| 方案 | 召回通道 | 说明 |
| :--- | :--- | :--- |
| `dense` | 稠密向量 | BGE 句向量 → Milvus（COSINE / HNSW） |
| `bm25` | 稀疏词法 | jieba 分词 + 自实现 BM25 倒排索引 |
| `hybrid` | 稠密 + 稀疏 | RRF（倒数排名融合），对异构分数尺度不敏感 |
| `hybrid_rerank` | 混合召回 → 重排 | 召回 top-N 后用 cross-encoder 精排 |
| `router` | 级联 | 先跑两路廉价召回，**一致性低时才升级重排** |

路由规则不是按查询长度「猜」，而是看**两个廉价通道是否互相印证**：
若稀疏与稠密的 top-k 重合度高、top1 一致，说明两路都拿得准，直接融合；
若两路分歧，才交给 cross-encoder 仲裁。这样重排只为少数查询付费。

---

## 3. 快速开始

```bash
# 1. 环境（uv 管理，venv 建在工程目录下）
uv venv venv --python 3.12
uv pip install --python venv/bin/python -e ".[embed,dev]"

# 2. 下载并构建评测集（约 60MB 公开数据）
PYTHONPATH=src venv/bin/python -m ragbench.cli --config configs/default.yaml prepare

# 3. 跑五种方案的检索评测（纯 CPU 约 50 分钟，其中重排占 36 分钟）
PYTHONPATH=src venv/bin/python -m ragbench.cli --config configs/default.yaml retrieve

# 4. 生成 Markdown 报告
PYTHONPATH=src venv/bin/python -m ragbench.cli report
```

**为什么重排这么慢**：`bge-reranker-base` 是 cross-encoder，每条查询要对 50 个候选
逐对打分，CPU 上约 7 秒/查询。300 条就是半小时——这正是本项目要量化的「代价」。

只想快速看链路是否通，用冒烟配置（500 篇语料、20 条查询，几分钟出结果）：

```bash
PYTHONPATH=src venv/bin/python -m ragbench.cli --config configs/smoke.yaml all
```

单元测试：

```bash
venv/bin/python -m pytest tests -q
```

---

## 4. 评测集怎么来的

用公开中文检索数据集 **C-MTEB / DuRetrieval**（10 万篇语料、2000 条查询、9839 条相关性标注）。

为了在纯 CPU 上可跑，做了**带正例闭包的子采样**：

1. 抽 300 条查询（固定种子）；
2. 取这些查询的**全部** gold 文档，保证正例不被采样掉；
3. 用随机文档补齐到 20000 篇，作为检索干扰项。

> 这一步很关键：如果只是简单随机采样语料，正例可能被采掉，
> Recall 会被人为压低，方案之间的对比就失去意义。

---

## 5. 指标口径

| 指标 | 含义 |
| :--- | :--- |
| Recall@k | 命中正例数 / 该查询正例总数（多正例查询不做特殊加权） |
| MRR@k | 第一个正例的倒数排名 |
| nDCG@k | 二值相关性版本，与 C-MTEB 榜单口径一致 |
| Hit@k | top-k 里是否至少命中一个正例 |
| p50 / p95 延迟 | 单查询端到端耗时（含查询编码） |

全部指标都能在 `tests/test_metrics.py` 里找到钉住行为的单元测试。

---

## 6. 生成质量评测（可选）

除召回指标外，还可以评估「检索到的内容能不能支撑一个准确回答」：

```bash
export CODEBUDDY_API_KEY=...          # 任意 OpenAI 兼容接口
PYTHONPATH=src venv/bin/python -m ragbench.cli generate --sample 60
```

三个指标都是**自实现、可解释**的（不用 RAGAS 黑盒）：

- **忠实度**：把回答拆成 claim，逐条判断上下文是否支撑 → 支撑数 / 总数
- **答案相关性**：回答是否切题（拒答不算切题）
- **上下文精度**：检索到的文档里有多少是 gold

LLM 判定失败时会降级到二元组覆盖率启发式，保证整批评测不会因个别请求失败而空缺。

---

## 7. 工程要点

- **BM25 用倒排索引**：打分复杂度是 O(查询词数 × 平均文档频次)，而不是 O(语料规模)。
  十万级语料上这是能不能跑完实验的分水岭，`test_inverted_index_matches_naive` 钉住了它与暴力实现等价。
- **向量缓存**：同一模型 + 同一文本只编码一次，改配置重跑时省下大量时间。
- **指标与检索解耦**：检索器只返回有序文档 id，指标统一在 `metrics.py` 计算，
  新增方案不需要改评测代码，也不会出现口径漂移。
- **报告自动生成**：README 里的数字全部来自 `artifacts/*.json`，杜绝手填数字对不上代码。
- **向量后端可切换且可基准**：`scripts/bench_vector_backends.py` 在同一份向量上对比
  FAISS HNSW / FAISS FlatIP / Milvus Lite，并校验三者 top-1 一致——
  性能数字不是拍的，是脚本跑出来的（见 `artifacts/backend_bench.json`）。

---

## 8. 目录结构

```
src/ragbench/
  config.py      实验配置（默认值 < YAML < 环境变量）
  data.py        数据集下载、子采样、qrels 归一化
  sparse.py      jieba 分词 + BM25 倒排索引
  index.py       向量索引（FAISS 默认 / Milvus Lite 可选）
  embedding.py   句向量编码与 cross-encoder 重排（含缓存）
  retrievers.py  五种检索策略 + 融合函数 + 级联路由
  metrics.py     Recall / MRR / nDCG / HitRate
  pipeline.py    建索引 → 跑评测 → 落盘
  generation.py  生成质量评测（自实现忠实度等）
  report.py      Markdown 报告生成
  llm.py         OpenAI 兼容 LLM 客户端（并发 + 重试）
tests/           89 个单元测试
configs/         默认 / 冒烟两套配置
```

---

## 9. 已知局限

- **纯 CPU 环境**：BGE-M3 在 CPU 上实测 0.7 docs/s，20k 语料要 7 小时以上，
  因此默认配置用 `bge-small-zh-v1.5`（约 32 docs/s）。需要在 GPU 上换回 BGE-M3 只需改一行配置。
- **语料规模**：2 万篇是 CPU 上的折中。语料越小，稀疏检索越吃亏
  （干扰项少、IDF 区分度低），这一点在结论里必须如实说明。
- **重排成本**：cross-encoder 在 CPU 上约 7 pair/s，是整条链路最贵的一步，
  这也是级联路由存在的意义。
- **生成评测**：依赖外部 LLM API，结果受模型版本影响，报告里会记录模型名与温度。

---

## 10. 许可

MIT
