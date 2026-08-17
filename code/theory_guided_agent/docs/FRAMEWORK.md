# Theory-Guided 个体对齐框架

> 会议落地文档（0720 → 0728）。主线：**C–U–V 帽子 × Graph/Loop Engineering × Theory-Guided Agentic RAG**。  
> 目标：**小样本、自进化、可解释**；更新机制是**错题本式结构修复**，不是背完整任务。

---

## 0. 一句话

我们不是「把理论塞进 prompt」，而是用 **有向图工作流** 把  
**情境路由 → 高置信理论检索 → 个体证据路径 → 预测 → 验证 → 错误归因 → 图/认知更新**  
框成可审计、可进化的系统；帽子（C–U–V）决定 Memory / Skill / Graph 怎么分工。

---

## 1. 为什么要用 Theory-Guided（应用方式）

| 问题 | 回答 |
|---|---|
| **为什么要用** | 群体理论给「人会怎么想」的**可言化坐标与搜索方向**；个体史 alone 小样本不够稳，纯 LLM 模拟不可解释、不可局部更新。 |
| **怎么用** | **路由 + 条件匹配**：遇到什么情境 → 激活哪类坐标 → 只取高置信详卡 → 注入路径推理，不是全库 dump。 |
| **创新点** | ① 理论是 **带触发条件的检索单元**；② 预测理由=因果路径（非事后合理化）；③ 失败经验蒸馏成 **结构+条件修复** 可组合复用；④ Graph 工作流把 Loop 变成显式路由。 |
| **实验已验证** | 顺序小用户 seq-CUV-TG 相对 GenMinds 近翻倍；大用户 +0.06；trigger 回填后静态首次超过基线。瓶颈已定位到「单理论事后合理化」→ v2 路径推理。 |

### 需不需要路由？

**需要。** 否则 skills/理论互相干扰，context 被低置信卡占满。  
路由回答会议原话：**「遇到什么情况，需要用到什么样的理论，来帮助预测」**。

实现：`tg_agent/theory_router.py`  
`factor_type + 词汇线索 → preferred coordinates → match → confidence gate`。

---

## 2. 大帽子 C–U–V 与 Element 怎么结合

| 帽子 | 含义 | Element 结合 |
|---|---|---|
| **C 认知架构** | 如何组织信息、形成判断 | **Graph workflow**（nodes/edges）+ Loop（predict→judge→evolve） |
| **U 知识能力** | 知道什么、会用什么 | **Theory 库（群体先验）** + **GenMinds 记忆（个体证据）** + **Failure 错题本（可迁移修复）** |
| **V 价值** | 身份、立场、偏好 | **因果图坐标权重** + **persona 置信度** + **situational env** |

结合方式（不是并列堆砌）：

```
刺激
 → C: 工作流路由（是否 repair / 是否用理论）
 → U: Agentic RAG 检索（理论详卡 + 个体证据 + 错题修复）
 → V: 坐标/画像加权，决定路径走向
 → 输出：stance + 路径 verbalization（可解释）
 → Loop: Judge → 归因 → 更新 U/V 图结构（上下文学习，不微调基座）
```

---

## 3. Agentic RAG：检索更准、理论更高置信

会议要求：**筛选更高置信度的理论 / 怎么去用理论**。

| 步骤 | 行为 |
|---|---|
| Route | 因素类型 + 线索 → 坐标先验（如 identity 威胁 → identity_threat） |
| Retrieve | `TheoryLibrary.match` + env/user 权重 |
| Grade | `theory_confidence`（richness×grounded×conditions×score）低于阈值则剔除 |
| Repair | evidence_grade 弱 → 加宽检索（不回放旧任务） |
| Use | 只把过门的卡写入路径合成；`why` 带 `route=` / `cond=` / `conf=` |

弱匹配 / 历史不足 → **保守策略**（uncertain + low_evidence 警告），避免硬套理论。

---

## 3b. v1+v2 融合：图推理本身即理由（0728）

会议发现：「强制要求模型遵循特定理论路径（如因果图）进行推导，反而导致整体效果不如自由发挥的初版模型」。
设计结论：**图推理产物本身可以作为原因，没有必要强制结构化因果链**。

| | strict（seq-CUV-Path, v2） | **fusion（seq-CUV-Fusion, v1+v2）** |
|---|---|---|
| 图工作流（路由/检索/分级/repair/吸收/错题本） | ✓ | ✓（完全保留） |
| 合成约束 | 必须输出类型化边因果路径，stance 从路径聚合 | 自由判断 stance；`reason` 自然语言说明实际依据，`used` 宽松引用材料 |
| 理由（verbalization） | 路径渲染 | **reason（图推理说明）+【图推理引用】渲染** |
| 图吸收 | 强制路径边 | used 引用转 supports 边（幻觉坐标剔除） |

实现：`PathAgent(path_mode="fusion")`（`_FUSION_SYSTEM` + `_validate_used`），
`run_sequential --methods seq-CUV-Fusion`。fast_path / 归因缓存 / 错题本对两种模式一致生效。

---

## 3c. Graph Agent / Loop Agent：彻底 Agent 化（0730）

此前「Graph/Loop」是工程形态：9 节点固定流水线（`path_workflow.py`）+ 脚本式 chrono
循环（`run_sequential.py`）+ bat 看门狗，LLM 只是被函数调用，路由/检索/修复全是硬编码。
本次把主线落成**真 Agent 架构**（`seq-CUV-Agent`）：

| 层 | 实现 | 职责 |
|---|---|---|
| **Graph Agent**（内层，每刺激一次） | `tg_agent/agent_graph.py`（LangGraph StateGraph）+ `tg_agent/agent_tools.py` | LLM 持 8 个工具（decompose/retrieve_theory/retrieve_memory/query_causal_graph/read_failure_notes/read_situational/skeptic_check/finalize_prediction）**自主决定**检索什么、检索多少、何时收工 |
| **Loop Agent**（外层，chrono 协议） | `tg_agent/loop_agent.py` | 循环契约（goal=judge OA / act=predict→judge→ingest→evolve / verify=judge_one / stop=n_steps）；CheckpointStore 断点续跑 + 进程内看门狗（吸收 bat 职责） |

要点：
- 确定性书挡保留：resolve_context（u/v 快照、情境、错题本**权重组合**与基线一致）→
  fast/slow 门控（`novelty.py` 不变）→ calibrate → absorb_finalize；中间全部交给 LLM 工具循环
  （`agent.max_tool_rounds` 硬上界 + forced finalize + fast_fallback 三级防失控）。
- LLM 传输仍是 raw-urllib `DeepSeekClient.chat_completion`（OpenAI tools/tool_calls + 重试）；
  端点不支持 tool_calls 或连续两轮解析失败时**自动降级 xml 协议**（`agent.tool_protocol`）。
- evolve_attributed 零改动（输出形状与图工作流一致）；归因输入补一行【工具轨迹】。
- fusion-only：strict 模式不移植，旧 `seq-CUV-Path/Fusion/TG` 全部留作基线，结果与旧数字不可比。
- 成本：慢帖 3–7 次调用（实测小用户 mean 5.0/帖），`metrics.json` 新增 `llm_calls` 聚合块。
- 运行：`run_agent_big_watchdog.bat` 或 `--methods seq-CUV-Agent`（默认开进程内看门狗，
  `--no-self-restart` 可关）；`--max-steps N`（warmup 后切片）用于便宜 smoke。

---

## 3d. RTWI 式阶段分解与可靠性跃升（0731）

此前错误归因是 6 个 flat cause（factor_extraction / retrieval / theory_prior / profile / short_term_state / context_shift），
只知道「什么错了」不知道「错在哪一段」。RTWI（Reliable Thinking with Images, ICML 2026）的核心洞见是把
思维链拆成 **信息采集（mining）** 和 **推理判断（reasoning）** 两阶段，用 token 熵做可靠性估计，区分 mining error vs reasoning error。

我们做了三件事（本地化 RTWI，不依赖 logprobs）：

| RTWI 做法 | 我们的落地 |
|---|---|
| Token 熵 → 阶段可靠性 | **结构性代理指标**（coverage/richness/diversity/confidence/reason_len）→ `stage_reliability` 写入 c_trace |
| Dual-stage filtering | **`error_stage: mining|reasoning`** → `FailureStructure` 按阶段标签检索修复（`CAUSE_TO_STAGE` 映射 + LLM 直接判断） |
| Reliability Leap（正确线索提升置信） | **`_reliability_leap_check`**：证据弱+自信高 → suspicious_overconfident → 强制 retry；证据强+自信低 → genuine_dilemma → 接受 uncertain；替代原二值证据门控 |

效果：
- 归因提示词从「6 选 1」升级为「先判断阶段 → 再定位 cause」
- 错题修复带 `error_stage` 标签，检索时 stage_bonus 提升匹配精度
- 证据门控从 50% 硬底线升级为连续可靠性信号 + 方向判断

模块：`failure_memory.py`（`CAUSE_TO_STAGE` / `error_stage` 字段）、`path_agent.py`（`_ATTRIB_SYSTEM` / `_compute_stage_reliability_hint`）、`agent_graph.py`（`_compute_stage_reliability` / `_reliability_leap_check`）。

---

## 3e. 自适应百分位阈值（0731）

原所有超参固定（min_confidence=0.35 / evidence_gate=50% / surprise_threshold=0.80），困难用户和简单用户用同一套阈值。
RTWI 的做法：对每个问题用当前批次 trace 的百分位计算过滤阈值。

落地：`tg_agent/adaptive_thresholds.py`
- 滚动窗口（默认 50 步）追踪 coverage / mining_score / synthesis_score / surprise / theory_confidence / repair_overlap
- α=0.30 分位数作为自适应阈值（过滤底部 30%）
- 样本不足 10 步时退回 fallback 默认值（小样本不瞎调）
- 按用户独立追踪（困难用户阈值自动调低，简单用户阈值自动调高）

集成：evolve 后记录 stage_reliability + surprise → `adaptive_thresholds.record_from_trace()`

---

## 3f. Skills 蒸馏（0731）

会议要求：「在较小的训练代价得到在全样本上训练的结果」「skills 冲突会不会影响整个工作」。

落地：`tg_agent/skills_distiller.py`
- 从 failure_memory 的 admitted repairs 蒸馏 `DistilledRule`（紧凑自然语言规则 ≤120 字）
- 去重：token Jaccard > 0.7 的规则合并（保留高分、叠加 support_count）
- 冲突检测：关键词对比（如「信任理论」vs「以个体为准」）→ 高分规则 supersedes 低分
- 组织：按 error_stage 分 mining / reasoning / cross_cutting
- 注入：蒸馏规则作为 prompt block 注入 agent context（优先级低于 strategy_notes，高于 raw 检索结果）
- 持久化：`data/users/{uid}_distilled_skills.json`

与错题本的关系：错题本是「源数据库」（精细、可组合修复），蒸馏技能是「缓存层」（紧凑、即时注入）。
蒸馏层定期从错题本刷新（每步 evolve 后检查 admitted_repairs ≥ 3 则更新）。

---

## 4. Graph Engineering：我们的图 vs 别人的图

**一句话区分：别人的 graph 是流程编排工具，我们的是可进化的个体认知数字孪生。**

| | 通用 KG / LangGraph 业务图 | **我们的 User Cognitive Graph** |
|---|---|---|
| 节点 | 实体/文档/工单步骤 | factor / coordinate / evidence / stance / **failure 结构（带 error_stage）** |
| 边 | 引用、流转 | supports / contradicts / triggers / moderates / updates（带时间衰减、contested、**stage 标签**） |
| 更新 | 追加日志或微调模型 | **错题结构 → 条件修复 → 组合**（ICL 式局部更新，不重训；mining/reasoning 分阶段修复） |
| 目的 | 流程编排或事实检索 | **个体认知数字孪生**：可解释预测 + 小样本自进化 + **知道为什么错 + 错在哪一段** |
| 与 Theory | 常无 | Theory 是 **带触发条件的先验边候选**，经路由+**自适应阈值**才进入图 |
| 可靠性 | 无内置机制 | **RTWI 式阶段可靠性估计**（mining/retrieval/synthesis 三阶段得分 + leap 方向判断） |

核心区别：别人的图在**编排**（把固定步骤画成 DAG），我们的图在**学习**（每次错误都变成可检索的结构修复，
下次同类情境自动 compose——不是背任务，是学方法）。加上 RTWI 的阶段拆解后，不仅知道错了，
还知道**错在信息采集还是推理判断**，修复更精准。

---

## 5. 自进化 = 错题本数据飞轮（FLARE 理念本地化）

```
预测 → 验证(Judge) → 错误归因 → 识别失败结构 → 学条件化修复
      → 写入 failure_memory + 因果图 → 新任务检索组合修复
```

| 会议点 | 落地 |
|---|---|
| 哪里弱练哪里 | 归因到 retrieval/theory_prior/… 只改对应边/权重 |
| 每次练的是新题目 | 不存完整任务；只存结构签名 |
| 更新了什么 / 为何有用 | 降权害人的坐标、加宽检索、重置短期情绪；compose 后有 success/fail 反馈 |
| Memory token 有限 | `prune_to_budget`：按频次×修复成功率×新近性保留 |
| 观点变化导致错误 | `recency_boost` 调高近期发帖权重 |
| Skills 互扰 | 理论路由 + 置信门控；工程 skills 分目录（harness / loop / theory-guided） |

持久化：`data/users/{uid}_failure_memory.json`  
契约：**structure + conditional repair + compose；never full-task replay**。

### 5b. ASPIRE 式错误经验积累（0728，arXiv 2607.00272 本地化）

| ASPIRE 机制 | 我们的落地 |
|---|---|
| 一次性失败是噪声；≥2 trials 同 (symptom, applicability) 才成库 | **准入闸门**：同 cause 失败总频次 ≥ `failure_memory.admission_min_cause_freq`(2) 才 `admitted`，未准入修复不参与 compose（小样本防噪声修复带歪预测） |
| skill = failure signature + when-to-apply guard + validated repair + origin | `FailureStructure.when_to_apply`（因素/坐标/错因条件描述）+ `exemplar`（一行刺激摘要，仍非完整任务） |
| 异构修复知识（不只权重 delta） | 归因 LLM 产出 `transferable_strategy` → `strategy_note` 修复，准入后以【错题本策略】块**注入合成 prompt**（带 guard 与失败次数），小样本下比 ±0.15 权重影响直接得多 |
| 细粒度 traces 支撑归因 | c_trace + verbalization + 检索分数已在归因输入中 |

配套：fusion prompt 改为**个体证据优先**（历史原话 > 群体理论，冲突以个体为准；reason 先引证据再引理论）；
`evidence_grade=="fail"` 时合成材料标注「证据薄弱，以画像/身份为准」。

---

## 6. Loop 为什么成环（工作流）

```
resolve_context → decompose → retrieve_compose(+router+repairs)
  → grade_evidence → (repair_retrieve ↺) → synthesize
  → skeptic → calibrate → absorb_finalize
                 ↓
            Judge / evolve_attributed
                 ↓
         failure_memory + causal_graph + weights + memory_layers
                 ↓
              下一刺激（chrono，防泄漏）
```

这是 **Loop Engineering 的有向图扩展**：每一步是产品事件，可审计、可分支、可修复。

---

## 7. 数据与评测效率（小代价）

- **时间窗**：同话题间隔 >1h（`config temporal_windows.gap_hours`）切开 → data point；
  `run_sequential --aggregate-windows` 后同窗连续帖共享一次预测/评判（0728 已接入主循环）
- **分层采样**：大窗/小窗按比例抽，未抽中的步仅 ingest 不预测：
  `run_sequential --window-sample-ratio 0.2`；静态 benchmark 用 `run_benchmark --stratify-ratio`
- **顺序协议**：warmup → predict-with-history-only → ingest GT → evolve
- **评测加权**：`config scoring`：结果四维 `result_weights` + `reason_weight`
  （民调 w=0 重结果；深访 w≈0.4 → composite=(1-w)·result+w·reason_correctness）

### 7b. 大用户推理效率：快慢通路（D-MEM 本地化，0728）

问题：大用户 ~2700 帖 × 每帖 6-8 次 LLM 调用 ≈ 6-7 小时。但时间线里大部分是 routine 帖。
论文依据：D-MEM（RPE 门控，>80% token 削减）、EM-LLM（training-free surprise）、
Titans/Nested Learning（prediction-error gating）、Instance-Adaptive Scaling、AgentDiet。

| 机制 | 实现 | 省的调用 |
|---|---|---|
| **surprise 门控快慢通路** | `novelty.py`：话题新奇度×词面惊讶×先验路径强度（全程无 LLM）；`config fast_path` 阈值路由。routine 帖 → `PathAgent._fast_predict` 单调用直觉预测 | 6-8 → 1 次/帖 |
| **归因缓存** | 失败结构与错题本重叠 ≥ `attribution_cache_threshold`(0.6) → 复用历史归因；fast_path 失败不调归因 LLM | 1 次/错误帖 |
| **自适应 skeptic** | 证据 strong + 立场 support/oppose + 无低证据因素 → 跳过质疑（`skeptic_adaptive`） | 1 次/易帖 |
| **GT 情绪缓存** | `detect_emotions` 按 GT 文本 sha1 缓存（`gt_emotion_cache.json`） | 重跑/重复文本省 1 次 |

语义：agent 学得越多（先验路径越强、话题越熟），surprise 越低 → 越多帖走快速通道——
**自进化本身降低推理成本**，与错题本飞轮同向。fast_path 占比写入 metrics `fast_path.rate`。

---

## 8. Skills / 文件夹边界（防冲突）

| 目录 | 职责 | 不放什么 |
|---|---|---|
| `.cursor/skills/harness/*` | 工程编排 build/eval/ship | 用户认知理论 |
| `.cursor/skills/loop-engineering/` | 通用 write-verify 循环 | 微博任务细节 |
| `.cursor/skills/theory-guided/` | **本任务**帽子、路由、错题本约定 | 通用 git/PR 流程 |
| `theory_guided_agent/tg_agent/` | 运行时实现 | Cursor skill 文档重复堆砌 |
| `theory_guided_agent/data/` | theory 库 + 用户图状态 | 代码逻辑 |

若干 skill 冲突时：**以 FRAMEWORK 帽子为准**；运行时只加载本任务路由，不把全部 skill 正文塞进 context。

---

## 9. 模块地图（技术更新）

| 模块 | 作用 |
|---|---|
| `agent_graph.py` / `agent_tools.py` | **Graph Agent**：LangGraph 工具循环 + 8 工具定义/执行 + RTWI 可靠性门控（0730→0731） |
| `loop_agent.py` | **Loop Agent**：CheckpointStore 断点续跑 + 进程内看门狗（0730） |
| `theory_router.py` | 情境→理论路由 + 置信筛选 |
| `path_workflow.py` / `graph_engine.py` | Graph 工作流 |
| `novelty.py` | surprise 门控（快慢通路路由，无 LLM） |
| `failure_memory.py` | 错题本 + prune + 归因结构缓存 + **RTWI error_stage 字段 + repair_effectiveness 报告**（0731） |
| `adaptive_thresholds.py` | **RTWI 式自适应百分位阈值**：滚动窗口追踪 → α 分位数校准（0731 新增） |
| `skills_distiller.py` | **Skills 蒸馏**：repair→compact rule + 去重 + 冲突检测 + prompt 注入（0731 新增） |
| `causal_graph.py` / `path_agent.py` | 个体认知图与路径预测（含 `_fast_predict` / **双阶段错误归因 `_compute_stage_reliability_hint`**） |
| `temporal_windows.py` | 话题时间窗 / 分层采样 |
| `theory_lib.py` + enrich | 详卡检索语料 |
| `genminds.py` | 个体证据；`recency_boost` |

---

## 10. 已有证据（摘要）

| 实验 | 结果 |
|---|---|
| 静态大用户 n=676 | CUV-TG 0.6452 > GenMinds 0.6331（trigger 回填后） |
| 顺序小用户 n=34 | **0.6588 vs 0.3279** |
| 顺序大用户 n=2647 | **0.7651 vs 0.7047**（last10 0.876 vs 0.705） |
| reason correctness | ~0.12 → 需 v2 Path 大样本验证（判据） |

---

## 11. 下一步

- [x] **RTWI 阶段分解**（0731：`error_stage` 字段 + `_ATTRIB_SYSTEM` 升级 + `_reliability_leap_check`）
- [x] **自适应百分位阈值**（0731：`adaptive_thresholds.py` + 集成 evolve）
- [x] **Skills 蒸馏**（0731：`skills_distiller.py` + 集成 prompt 注入）
- [x] **修复有效性追踪**（0731：`repair_effectiveness()` 报告 + c_trace 富信息）
- [ ] 大用户全量 2656 步 Agent 跑完 + reason_correctness 判据
- [ ] 修复有效性 ablation：Agent with/without 错题本 / with/without 蒸馏
- [ ] 多用户验证（当前 2 用户，目标 ≥5）
- [ ] 自适应阈值在长序列上的效果验证
- [x] 时间窗聚合 + 分层采样接入 `run_sequential`（0728）
- [ ] 错题本 prune 在长序列上的 ablation
- [x] Agentic RAG repair 轮查询改写（0728）
- [x] 超参收进 config（0728：五段；0731 +adaptive_window/alpha）

## 12. 产物索引

- 顺序：`outputs/benchmark_sequential_weibo_ai*/`
- 静态：`outputs/benchmark_cuv_tg/`
- 可视化：`viz/verbalizable_path.html`、`viz/live_graph.html`
