# Theory-Guided Agent（检索匹配优先）

Training-free 个体认知建模：GenMinds 做生成骨干，**Theory 库只做检索匹配的详细理论支撑**（不是把论文摘要塞进 prompt）。

- **生成**：GenMinds memory banks（大用户 `1989660417` 上 opinion **0.5982** 量级）
- **Theory 库**：`canonical_theories.json` 详卡 + 爬取论文经 `enrich-theory` 蒸馏成 summary / mechanism / propositions / constructs
- **匹配**：按刺激稀疏检索详卡 → 注入机制先验与可解释痕迹 → GenMinds 生成观点

C–U–V 形式化可暂缓；当前工程重心是 **可检索、有机制正文的理论卡**。
框架定位（C=Harness Loop / U=理论库+记忆 / V=个体认知图谱）见 `docs/FRAMEWORK.md`。

旧版「整库理论 dump 进 prompt」会伤大用户分数（约 −6~−10）。本系统只稀疏取 top-k 详卡。

> 不可能爬「全网社科论文」。做法是：canonical 详卡打底 + OpenAlex/Crossref 按坐标扩库 + enrich 蒸馏结构化字段。

## Quick start

```powershell
cd d:\UserAgent\theory_guided_agent
# keys: ..\agentic-harness-engineering\.env

# 1) 扩库（可选）
python -m tg_agent.cli bootstrap-theory --per-query 40 --pages 2

# 2) 只从论文摘要蒸馏详卡（无摘要不蒸馏；evidence_quotes 必须能在摘要中找回）
python -m tg_agent.cli enrich-theory --reset-ungrounded --limit 40 --min-citations 80

# 2b) 无摘要卡用 DeepSeek 训练知识回填（标 model_knowledge，匹配降权 ×0.7）
python -m tg_agent.enrich_theories --knowledge --limit 2000 --min-citations 80

# 3) Smoke：预测 + 理论匹配痕迹
python -m tg_agent.cli smoke --stimulus "某外媒称中国经济即将崩溃，请预测该用户会如何回应并解释机制"

# 4) Loop
python -m tg_agent.cli loop --user 1989660417 --stimulus "某热点事件发酵，预测该用户会如何回应并解释机制"
```

## Benchmark (DeepSeek predict · Qwen3.7-Plus judge)

```powershell
$env:QWEN_API_KEY='...'
$env:QWEN_BASE_URL='https://dashscope.aliyuncs.com/compatible-mode/v1'

python -m tg_agent.run_benchmark --methods GenMinds,CUV-TG --limit 0
```

## Layout

```
theory_guided_agent/
  docs/FRAMEWORK.md             # 框架说明（含 graph pathway + failure repair）
  tg_agent/
    graph_engine.py             # 轻量 StateGraph（typed state / 条件边 / route history）
    agent_state.py              # PathAgent 共享状态 schema
    path_workflow.py            # PathAgent 工作流节点与路由
    failure_memory.py           # 失败结构 + 条件化修复（禁止完整任务回放）
    causal_graph.py / path_agent.py / ...
```
