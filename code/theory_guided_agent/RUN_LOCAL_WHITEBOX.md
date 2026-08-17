# 本地 DeepSeek-V4-Flash + RTWI 白盒思维链（UserAgent）

按 [Reliable Thinking with Images](https://arxiv.org/abs/2602.12916) 本地化：
- 思维链拆成 **mining（信息采集）** / **reasoning（推理判断）**
- `stage_reliability` + `error_stage` 归因 → 知道错在哪一段、如何纠正
- 本地 Flash 开启 thinking 后，`message.reasoning` / `reasoning_content` 写入 `c_trace.model_reasoning`

## 硬件与模型

| 模型 | 状态 | 说明 |
|------|------|------|
| **DeepSeek-V4-Flash** | **已拉起** | `http://127.0.0.1:8001/v1`，2×H800，TP=2+EP，`max_model_len=4096` |
| Qwen3-VL-8B-Thinking | 备选 | 单卡时可改回 `.env` / `config.yaml` |

## 1) 启动本地 Flash

```bash
source /root/autodl-tmp/env.sh
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda_envs/rtwi

setsid env MAX_MODEL_LEN=4096 GPU_UTIL=0.98 \
  bash /root/autodl-tmp/models/serve_deepseek_v4_flash.sh \
  >/root/autodl-tmp/logs/v4_flash_serve_final.log 2>&1 </dev/null &
```

健康检查：

```bash
curl -sS http://127.0.0.1:8001/v1/models
```

## 2) 跑 UserAgent smoke（白盒）

```bash
cd /root/autodl-tmp/UserAgent/theory_guided_agent
source /root/autodl-tmp/env.sh
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda_envs/rtwi

python -m tg_agent.cli smoke --user 1989660417 \
  --stimulus "外媒称中国经济即将崩溃并配了夸张图表。请预测该用户会如何回应，并解释机制。"
```

产物：`data/users/1989660417_smoke.json`

关注字段：
- `predicted_opinion`：最终短评
- `c_trace.model_reasoning`：完整思维链（白盒）
- `c_trace.white_box.mining_excerpt` / `reasoning_excerpt`：采集段 / 判断段
- 顺序基准 `seq-CUV-Agent/Path` 会再写入 `stage_reliability`、`error_stage`（错题本 / skills 蒸馏入口）

## 3) 短顺序基准（可选，含归因 loop）

```bash
python -m tg_agent.run_sequential --methods seq-CUV-Agent --user 1989660417 --max-steps 3
```

这对应会议里的 **loop / graph engineering**：预测 → 验证 → 错误归因（mining/reasoning）→ 图/认知更新 → 错题本沉淀。

## 配置

- `config.yaml` → `llm.model: DeepSeek-V4-Flash`，`enable_thinking: true`
- `.env` → `LLM_BASE_URL=http://127.0.0.1:8001/v1`，`LLM_ENABLE_THINKING=true`
- 本地 ctx 护栏：`LLM_MAX_TOKENS=768`，`LLM_MAX_MODEL_LEN=4096`，`agent.context_char_budget=5500`

## 会议要点如何落到代码

| 帽子 / 理念 | 落地位置 |
|---|---|
| Theory-guided + Agentic RAG | `match_theories` / tool 检索 + 高置信理论筛选 |
| Graph / Loop engineering | `agent_graph.py`：预测→工具→finalize→校准→吸收 |
| 白盒：为何错、如何改 | RTWI `mining`/`reasoning` 拆分 + `error_stage` 归因 |
| 错题本 / FLARE 飞轮 | `failure_memory` + `skills_distiller`（按阶段分文件夹防干扰） |
| 自进化（小代价） | 失败经验进 graph/认知更新，而非丢弃 |
| 时间窗 / 分层采样 | `temporal_windows` + `--window-sample-ratio` |
