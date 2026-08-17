"""路径推理 Agent（蓝图阶段四核心 + 阶段五输出）。

与 CUVAgent 的结构性区别：预测不是「生成评论 + 附带理由」，而是
    事件 → 因素分解(f1..fk) → 每因素激活理论坐标 + 检索历史证据
    → 显式因果路径(类型化边) → 由路径推出 stance/情绪/评论。
verbalization = 路径渲染本身（理由即路径，不是事后合理化）。

附加能力：
- 反证质疑步（轻量 skeptic，蓝图多智能体之 5）
- 置信度校准 + 低证据警告（蓝图阶段五·4，多智能体之 6）
- 情绪多标签概率输出（蓝图阶段五·2）
- 因果记忆网络读写（蓝图阶段三）
- 错误归因驱动的定向进化（蓝图·在线演化：因素抽取/检索/画像/短期状态/理论先验）
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .adaptive_thresholds import AdaptiveThresholds
from .agent import CUVAgent, _as_str_list, _parse_json, _regex_field
from .causal_graph import CausalGraph, _nid
from .factors import EventFactor, decompose_factors
from .failure_memory import CAUSE_TO_STAGE
from .failure_memory import FailureMemory
from .memory_layers import (
    EMOTIONS,
    MemoryLayers,
    emotion_alignment,
    load_layers,
    save_layers,
)
from .models import AgentOutput
from .situational_env import (
    format_situational_block,
    resolve_situational,
    situational_env_weights,
)
from .user_actions import format_source_block

PATH_EDGE_TYPES = ["supports", "contradicts", "triggers", "moderates"]

_SYNTH_SYSTEM = """你是一个 Theory-Guided 因果路径推理体，扮演指定微博用户本人。
须遵守 Identity/人设表达特征（任意大V/KOL 皆适用）：用其本人声口与惯用自称，勿写成旁观者点评。
工作方式（必须严格遵守）：
1. 对每个【事件因素】，从给定的【匹配理论】【历史证据】【先验路径】中选择能解释「该用户会如何反应」的链条；
2. 每条路径声明边类型：supports(证据支持该推导) | contradicts(证据与该因素方向相反) | triggers(因素触发坐标/价值) | moderates(因素调节强度)；
3. stance 必须从路径聚合推出（先路径、后立场），不允许先定立场再补路径；
4. 情绪分布反映该用户看到此事件时的情绪（结合其短期情绪状态与历史风格）；
5. predicted_opinion 用该用户本人的声音写，<=120 字。
只输出 JSON，不要 markdown：
{"paths":[{"factor_id":"f1","coordinate":"坐标id","edge_type":"supports","evidence_idx":[0],"reasoning":"<=60字","stance_implication":"support|oppose|mixed|uncertain"}],
 "stance":"support|oppose|mixed|uncertain",
 "emotion_probs":{"anger":0.0,"joy":0.0,"sadness":0.0,"fear":0.0,"disgust":0.0,"surprise":0.0,"neutral":1.0},
 "predicted_opinion":"...",
 "low_evidence_factors":["f2"]}"""

_SKEPTIC_SYSTEM = """你是反证质疑员。给定一个用户的「事件因素→理论坐标→历史证据→态度」推理路径和结论，
你的任务是找茬：哪条路径最弱？证据是否真的支持 stance_implication？有没有同样合理但方向相反的解读？
如果反对理由足以推翻结论，给出修正；否则确认。
只输出 JSON：{"challenge":"<=80字","overturn":false,"revised_stance":"support|oppose|mixed|uncertain","revised_opinion":"(仅 overturn=true 时给，<=120字)"}"""

_ATTRIB_SYSTEM = """你是预测错误归因员。模型对某微博用户的评论预测与真实评论有偏差。
错误分两大阶段（RTWI：拆链归因）：
- mining（信息采集层）：因素没拆对或证据没查全 → 调检索/因素分解
- reasoning（推理判断层）：因素和证据都对，但推理/立场判断出错 → 调理论权重/策略

错误来源只能是以下之一：
- factor_extraction: 事件因素分解错（抓错/漏掉了真正驱动该用户的因素）→ mining
- retrieval: 检索错（没检索到真正相关的历史证据，或证据不相关）→ mining
- theory_prior: 理论先验错（理论方向与该用户个体规律相反，理论帮了倒忙）→ reasoning
- profile: 用户画像错（画像中某条属性与该用户实际不符）→ reasoning
- short_term_state: 短期状态错（用户近期情绪/话题状态变化导致，非长期规律问题）→ reasoning
- context_shift: 立场/语境漂移（该用户观点随时间已变，旧历史证据不再代表其当前立场——与短期情绪不同，这是长期转向）→ reasoning
- none: 无法归因/检测模型误差

另外给出 transferable_strategy：一句 <=60 字的可迁移策略——下次遇到同类情境时应该怎么做
（例如"涉及X类议题时以该用户近期原话为准，不套理论"）。要具体、可执行，不写空话。

只输出 JSON：{"primary_cause":"retrieval","error_stage":"mining","detail":"<=60字","implicated_profile_attr":"(profile 错时给出画像属性原文，否则空)","transferable_strategy":"<=60字"}"""

_FAST_SYSTEM = """你扮演指定微博用户本人，对事件直接发表一条短评（快速直觉通道，无需显式推理过程）。
严格贴合 Identity/人设表达特征、历史立场与风格；你就是该账号本人，不是旁观者。只输出 JSON，不要 markdown：
{"stance":"support|oppose|mixed|uncertain",
 "emotion_probs":{"anger":0.0,"joy":0.0,"sadness":0.0,"fear":0.0,"disgust":0.0,"surprise":0.0,"neutral":1.0},
 "predicted_opinion":"<=120字"}"""

# v1+v2 融合：保留图工作流的检索/路由/分级/吸收，但不强制结构化因果链。
# 图推理产物（召回的理论/证据/先验路径 + 实际引用）本身就是理由。
_FUSION_SYSTEM = """你是 Theory-Guided 图推理体，扮演指定微博用户本人。
须遵守 Identity/人设表达特征（任意大V/KOL）：用其本人声口与惯用自称，勿写成旁观者点评。
图工作流已为每个【事件因素】召回【匹配理论】【历史证据】【先验路径】——这就是图推理的结果。
工作方式：
1. 自由判断 stance，不强制构建结构化因果链，也不要求先定路径后定立场；
2. 个体证据优先：该用户的历史原话/行为 > 群体理论。两者冲突时以个体证据为准，
   理论只用于解释机制，不得压过该用户表现出的一贯立场；
3. reason（<=150字）用自然语言写出判断依据——这段图推理说明本身就是理由，不是事后补写。
   引用顺序：先点个体证据（如 证据[f1e0]、先验路径），再点理论坐标；没用到的不许提；
4. used 只列出真正影响判断的材料 [{"factor_id":"f1","coordinate":"坐标id或空","evidence_idx":[0]}]；
5. predicted_opinion 用该用户本人的声音写，<=120 字；
6. 情绪分布反映该用户看到此事件时的情绪（结合其短期情绪状态与历史风格）。
只输出 JSON，不要 markdown：
{"stance":"support|oppose|mixed|uncertain",
 "emotion_probs":{"anger":0.0,"joy":0.0,"sadness":0.0,"fear":0.0,"disgust":0.0,"surprise":0.0,"neutral":1.0},
 "predicted_opinion":"...",
 "reason":"...",
 "used":[{"factor_id":"f1","coordinate":"","evidence_idx":[0]}],
 "low_evidence_factors":["f2"]}"""


@dataclass
class PathOutput(AgentOutput):
    factors: list[dict[str, Any]] = field(default_factory=list)
    paths: list[dict[str, Any]] = field(default_factory=list)
    emotion_probs: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    low_evidence: bool = False
    skeptic: dict[str, Any] = field(default_factory=dict)


class PathAgent(CUVAgent):
    def __init__(
        self,
        *args: Any,
        use_skeptic: bool = True,
        source_profile: dict[str, Any] | None = None,
        tuning: dict[str, Any] | None = None,
        failure_memory_budget: dict[str, Any] | None = None,
        path_mode: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.graph = CausalGraph(Path(self.state_dir) / f"{self.user_id}_causal_graph.json")
        self.layers: MemoryLayers = load_layers(self.state_dir, self.user_id)
        self.layers.seed_persona(self.memory.beliefs[:8] + self.memory.values[:6])
        self.use_skeptic = use_skeptic
        self.source_profile = source_profile or {"available": False}
        self.failure_memory = FailureMemory.load(self.state_dir, self.user_id)
        # 图工作流超参（config.yaml path_agent 段；缺省值保持原硬编码行为）
        self.tuning: dict[str, Any] = {
            "min_confidence": 0.35,
            "min_richness": 0.35,
            "repair_budget": 1,
            "skeptic_overturn_min_conf": 0.75,
            "query_reformulation": True,
        } | dict(tuning or {})
        self.fm_budget: dict[str, Any] = {
            "max_structures": 80,
            "max_repairs": 120,
        } | dict(failure_memory_budget or {})
        # ASPIRE 准入闸门阈值（同 cause 复发次数）
        self.failure_memory.admission_min_cause_freq = int(
            self.fm_budget.get("admission_min_cause_freq", 2)
        )
        # ablation / 负迁移防线开关（0729；config failure_memory 段，--fm-mode 覆盖）
        self.failure_memory.enabled = bool(self.fm_budget.get("enabled", True))
        self.failure_memory.weight_repairs_enabled = bool(
            self.fm_budget.get("weight_repairs_enabled", True)
        )
        self.failure_memory.revoke_on_fail = bool(self.fm_budget.get("revoke_on_fail", False))
        self.failure_memory.revoke_min_feedback = int(self.fm_budget.get("revoke_min_feedback", 3))
        self.failure_memory.revoke_max_score = float(self.fm_budget.get("revoke_max_score", 0.4))
        # RTWI-style adaptive thresholds (per-user percentile calibration)
        self.adaptive_thresholds = AdaptiveThresholds(
            window_size=int(self.tuning.get("adaptive_window", 50)),
            alpha=float(self.tuning.get("adaptive_alpha", 0.30)),
        )
        # strict: 强制类型化因果路径（v2）；fusion: 图推理产物即理由（v1 式自由生成 + v2 图工作流）
        self.path_mode = path_mode or str(self.tuning.get("path_mode") or "strict")
        if self.path_mode not in {"strict", "fusion"}:
            self.path_mode = "strict"

    # ---------------- predict (LangGraph-style pathway) ----------------
    def predict(
        self,
        stimulus: str,
        *,
        post_id: str = "",
        bid: str = "",
        date: str = "",
        topic: str = "",
    ) -> PathOutput:
        """Typed-state graph: resolve→decompose→retrieve+compose repairs→
        grade→(repair loop)→synthesize→skeptic→calibrate→absorb.

        Learning policy: compose conditional repairs from failure structures,
        never replay a full past task episode.

        fast_path（D-MEM 式 RPE 门控）：surprise 低于阈值的 routine 帖走
        单调用直觉预测，跳过 decompose/skeptic/repair，大用户时间线省调用。
        """
        fp = self.tuning.get("fast_path") or {}
        if fp.get("enabled"):
            from .novelty import compute_surprise

            gate = compute_surprise(
                self,
                stimulus=stimulus,
                topic=topic,
                recent_window=int(fp.get("recent_window", 50)),
                w_topic=float(fp.get("w_topic", 0.35)),
                w_lexical=float(fp.get("w_lexical", 0.35)),
                w_prior=float(fp.get("w_prior", 0.30)),
            )
            threshold = float(fp.get("surprise_threshold", 0.35))
            gate["threshold"] = threshold
            gate["route"] = "fast" if gate["surprise"] < threshold else "slow"
            if gate["route"] == "fast":
                return self._fast_predict(
                    stimulus, post_id=post_id, date=date, topic=topic, gate=gate
                )

        from .path_workflow import run_path_predict

        return run_path_predict(
            self,
            stimulus=stimulus,
            post_id=post_id,
            bid=bid,
            date=date,
            topic=topic,
            repair_budget=int(self.tuning.get("repair_budget", 1)),
        )

    # ---------------- 快速通道（routine 帖，单 LLM 调用） ----------------
    def _fast_predict(
        self,
        stimulus: str,
        *,
        post_id: str = "",
        date: str = "",
        topic: str = "",
        gate: dict[str, Any],
    ) -> PathOutput:
        u = self.memory.u_snapshot(max_motifs=self.max_motifs)
        sit = resolve_situational(
            self.situational_store,
            post_id=post_id or None,
            date=date or None,
            topic=topic or None,
            text=stimulus,
        )
        events = self.memory.retrieve(stimulus, top_k=4)
        ev_block = "\n".join(
            f"- {e.event_title}: {(e.user_opinion or e.text)[:120]}" for e in events
        ) or "- （无直接相关历史）"
        dom = self.layers.dominant_emotions()
        user_msg = f"""## Identity
{self.memory.identity_block()}

## 用户画像（带置信度）
{self.layers.persona_block()}

## 短期情绪状态
{', '.join(f'{e}={v_}' for e, v_ in dom) or '(平静/无显著情绪)'}

## 信源画像
{format_source_block(self.source_profile)}

## 情境环境
{format_situational_block(sit)}

## 相关历史证据
{ev_block}

## 事件
{stimulus}

请以该用户身份直接发表一条简短原创微博评论（JSON）。"""
        raw = self.llm.chat(
            [{"role": "system", "content": _FAST_SYSTEM},
             {"role": "user", "content": user_msg}],
            temperature=0.4, max_tokens=500, disable_thinking=True,
        )
        parsed = _parse_json(raw)
        opinion = (
            str(parsed.get("predicted_opinion") or "").strip()
            or _regex_field(raw, "predicted_opinion")
            or raw.strip()[:200]
        )
        stance = str(parsed.get("stance") or "uncertain")
        if stance not in {"support", "oppose", "mixed", "uncertain"}:
            stance = "uncertain"
        surprise = float(gate.get("surprise") or 0.0)
        return PathOutput(
            user_id=self.user_id,
            stimulus=stimulus,
            predicted_opinion=opinion,
            stance=stance,
            activated_coordinates=[],
            matched_theories=[],
            evidence_events=[
                {
                    "map_id": e.map_id,
                    "title": e.event_title,
                    "score": round(e.score, 4),
                    "opinion": (e.user_opinion or "")[:200],
                }
                for e in events
            ],
            verbalization=(
                f"fast_path 直觉预测（surprise={surprise:.2f} < "
                f"{float(gate.get('threshold') or 0):.2f}，routine 帖无显式路径）"
            ),
            c_trace={
                "mode": "fast_path",
                "gate": gate,
                "num_events": len(getattr(self.memory, "events", [])),
                "post_id": post_id,
            },
            u_snapshot=u,
            v_snapshot={},
            caveats=["fast_path: routine 帖单调用预测，未做显式路径推理"],
            emotion_probs=self._validate_emotions(parsed.get("emotion_probs")),
            confidence=round(1.0 - 0.5 * surprise, 3),
            low_evidence=False,
        )

    # ---------------- 内部步骤 ----------------
    def _synthesize(
        self,
        stimulus: str,
        u: dict[str, Any],
        v: dict[str, Any],
        sit: dict[str, Any] | None,
        factors: list[EventFactor],
        per_factor: list[dict[str, Any]],
        fusion: bool = False,
        strategy_notes: list[str] | None = None,
        weak_evidence: bool = False,
    ) -> dict[str, Any]:
        blocks: list[str] = []
        for pf in per_factor:
            f: EventFactor = pf["factor"]
            lines = [f"### {f.id} [{f.type}] {f.text} (salience={f.salience})"]
            for i, m in enumerate(pf["matched"]):
                lines.append(
                    f"  理论: [{m.card.coordinate}] {m.card.name} "
                    f"mechanism={m.card.mechanism[:140]} (score={m.score:.3f})"
                )
            for i, e in enumerate(pf["events"]):
                lines.append(
                    f"  证据[{f.id}e{i}]: {e.event_title}: {(e.user_opinion or e.text)[:140]} "
                    f"(score={e.score:.3f})"
                )
            for pr in pf["priors"]:
                lines.append(
                    f"  先验路径: {pr['factor']} --{pr['edge1']}--> {pr['via']} "
                    f"--{pr['edge2']}--> {pr['to']} (score={pr['score']:.3f})"
                )
            blocks.append("\n".join(lines))
        dom = self.layers.dominant_emotions()
        strategy_block = ""
        if strategy_notes:
            strategy_block = (
                "## 错题本策略（该用户历史失败中学到的可迁移修正，优先遵守）\n"
                + "\n".join(f"- {n}" for n in strategy_notes[:4])
                + "\n\n"
            )
        weak_note = (
            "【注意】本轮检索证据薄弱/相关性低：材料仅供参考，请以用户画像、身份与一贯立场为准。\n"
            if weak_evidence else ""
        )
        user_msg = f"""## Identity
{self.memory.identity_block()}

## 事件
{stimulus}

## 用户画像（带置信度）
{self.layers.persona_block()}

## 短期情绪状态（近窗滚动）
{', '.join(f'{e}={v_}' for e, v_ in dom) or '(平静/无显著情绪)'}

## 信源画像
{format_source_block(self.source_profile)}

## 情境环境
{format_situational_block(sit)}

## 表达风格
{'; '.join(u.get('communication', [])[:4])}

{strategy_block}## 事件因素与可用材料
{weak_note}{chr(10).join(blocks)}

请按系统指令的流程输出 JSON。"""
        raw = self.llm.chat(
            [{"role": "system", "content": _FUSION_SYSTEM if fusion else _SYNTH_SYSTEM},
             {"role": "user", "content": user_msg}],
            temperature=0.35, max_tokens=2200, disable_thinking=True,
        )
        parsed = _parse_json(raw)
        if not parsed.get("predicted_opinion"):
            raw = self.llm.chat(
                [{"role": "system", "content": "Output one compact JSON object only."},
                 {"role": "user", "content": user_msg + "\n\n上次输出无效，请重新输出合法 JSON。"}],
                temperature=0.2, max_tokens=1600, disable_thinking=True,
            )
            parsed = _parse_json(raw)
        if not parsed.get("predicted_opinion"):
            parsed["predicted_opinion"] = _regex_field(raw, "predicted_opinion") or raw.strip()[:300]
        return parsed

    def _skeptic_check(
        self,
        stimulus: str,
        factors: list[EventFactor],
        paths: list[dict[str, Any]],
        stance: str,
        opinion: str,
    ) -> dict[str, Any]:
        try:
            raw = self.llm.chat(
                [{"role": "system", "content": _SKEPTIC_SYSTEM},
                 {"role": "user", "content": (
                     f"【事件】{stimulus[:400]}\n"
                     f"【路径】{self._render_paths(factors, paths, stance)}\n"
                     f"【结论】stance={stance}\n【评论】{opinion}\n请质疑。"
                 )}],
                temperature=0.2, max_tokens=500, disable_thinking=True,
            )
            obj = _parse_json(raw)
            return {
                "challenge": str(obj.get("challenge") or "")[:200],
                "overturn": bool(obj.get("overturn")),
                "revised_stance": str(obj.get("revised_stance") or ""),
                "revised_opinion": str(obj.get("revised_opinion") or ""),
            }
        except Exception as e:  # noqa: BLE001
            return {"challenge": f"skeptic_error:{e}", "overturn": False}

    def _calibrate(
        self,
        per_factor: list[dict[str, Any]],
        paths: list[dict[str, Any]],
        low_evidence_factors: list[str],
        skeptic: dict[str, Any],
    ) -> tuple[float, bool]:
        """置信度 = 证据覆盖 × 匹配强度 × 路径一致性 × 质疑惩罚；并给出低证据警告。"""
        n = max(1, len(per_factor))
        covered = sum(1 for pf in per_factor if pf["events"] or pf["matched"])
        coverage = covered / n
        tops = [max([m.score for m in pf["matched"]], default=0.0) for pf in per_factor]
        score_term = min(1.0, (sum(tops) / n) / 0.10)  # 新分尺度 0.046–0.107
        if paths:
            constructive = sum(1 for p in paths if p.get("edge_type") in {"supports", "triggers"})
            path_term = constructive / len(paths)
        else:
            path_term = 0.0
        skeptic_term = 0.7 if skeptic.get("overturn") else 1.0
        confidence = round(
            (0.45 * coverage + 0.25 * score_term + 0.20 * path_term + 0.10) * skeptic_term, 3
        )
        low_ev = (
            coverage < 0.5
            or score_term < 0.5
            or len(low_evidence_factors) >= max(1, n // 2)
        )
        return confidence, low_ev

    # ---------------- 路径工具 ----------------
    def _validate_paths(
        self,
        raw_paths: Any,
        factors: list[EventFactor],
        per_factor: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        valid_ids = {f.id for f in factors}
        # 每因素实际匹配到的理论坐标（防止模型把 factor_id 填进 coordinate 字段）
        coord_map = {
            pf["factor"].id: [m.card.coordinate for m in pf["matched"] if m.card.coordinate]
            for pf in per_factor
        }
        out: list[dict[str, Any]] = []
        for p in raw_paths or []:
            if not isinstance(p, dict):
                continue
            fid = str(p.get("factor_id") or "")
            if fid not in valid_ids:
                continue
            et = str(p.get("edge_type") or "supports")
            if et not in PATH_EDGE_TYPES:
                et = "supports"
            si = str(p.get("stance_implication") or "uncertain")
            if si not in {"support", "oppose", "mixed", "uncertain"}:
                si = "uncertain"
            ev = p.get("evidence_idx")
            coord = str(p.get("coordinate") or "")
            allowed = coord_map.get(fid) or []
            if coord not in allowed:
                coord = allowed[0] if allowed else ""
            out.append({
                "factor_id": fid,
                "coordinate": coord,
                "edge_type": et,
                "evidence_idx": [int(x) for x in ev if str(x).lstrip("-").isdigit()] if isinstance(ev, list) else [],
                "reasoning": str(p.get("reasoning") or "")[:160],
                "stance_implication": si,
            })
        return out

    def _validate_used(
        self,
        raw_used: Any,
        factors: list[EventFactor],
        per_factor: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """fusion 模式：把「实际引用的材料」宽松转成图可吸收的条目。

        不强制类型化因果链：edge_type 固定 supports，坐标只允许引用该因素
        实际匹配到的理论（防止幻觉坐标），引不到合法坐标就记为个体记忆边。
        """
        valid_ids = {f.id for f in factors}
        coord_map = {
            pf["factor"].id: [m.card.coordinate for m in pf["matched"] if m.card.coordinate]
            for pf in per_factor
        }
        out: list[dict[str, Any]] = []
        for u_ in raw_used or []:
            if not isinstance(u_, dict):
                continue
            fid = str(u_.get("factor_id") or "")
            if fid not in valid_ids:
                continue
            coord = str(u_.get("coordinate") or "")
            allowed = coord_map.get(fid) or []
            if coord not in allowed:
                coord = ""
            ev = u_.get("evidence_idx")
            out.append({
                "factor_id": fid,
                "coordinate": coord,
                "edge_type": "supports",
                "evidence_idx": [int(x) for x in ev if str(x).lstrip("-").isdigit()] if isinstance(ev, list) else [],
                "reasoning": "",
                "stance_implication": "uncertain",
            })
        return out

    @staticmethod
    def _validate_emotions(raw: Any) -> dict[str, float]:
        out = {e: 0.0 for e in EMOTIONS}
        if isinstance(raw, dict):
            for e in EMOTIONS:
                try:
                    out[e] = max(0.0, float(raw.get(e, 0.0)))
                except (TypeError, ValueError):
                    pass
        total = sum(out.values())
        if total <= 0:
            out["neutral"] = 1.0
            return out
        return {e: round(v / total, 4) for e, v in out.items()}

    @staticmethod
    def _render_paths(factors: list[EventFactor], paths: list[dict[str, Any]], stance: str) -> str:
        fmap = {f.id: f for f in factors}
        lines = []
        for p in paths:
            f = fmap.get(p["factor_id"])
            ftxt = f.text[:40] if f else p["factor_id"]
            lines.append(
                f"{ftxt} --{p['edge_type']}--> [{p.get('coordinate') or '个体记忆'}] "
                f"→ {p['stance_implication']}（{p.get('reasoning', '')[:60]}）"
            )
        lines.append(f"⇒ 聚合立场：{stance}")
        return "\n".join(lines)

    def _graph_absorb(
        self,
        factors: list[EventFactor],
        per_factor: list[dict[str, Any]],
        paths: list[dict[str, Any]],
        stance: str,
    ) -> None:
        pf_map = {pf["factor"].id: pf for pf in per_factor}
        for p in paths:
            f = next((x for x in factors if x.id == p["factor_id"]), None)
            if not f:
                continue
            coord = p.get("coordinate") or ""
            conf = 0.55 + 0.15 * min(1.0, f.salience * 2)
            if coord:
                self.graph.add_edge("factor", f.text, "triggers", "coordinate", coord,
                                    weight=f.salience, confidence=conf)
                self.graph.add_edge("coordinate", coord, p["edge_type"], "stance", stance,
                                    weight=1.0, confidence=conf)
            pf = pf_map.get(f.id)
            if pf:
                for idx in p.get("evidence_idx") or []:
                    if 0 <= idx < len(pf["events"]):
                        ev = pf["events"][idx]
                        self.graph.add_edge(
                            "evidence", f"{ev.event_title or ev.map_id}",
                            "supports" if p["edge_type"] != "contradicts" else "contradicts",
                            "stance", stance,
                            weight=min(1.0, ev.score * 3), confidence=0.5,
                        )
        self.graph.save()

    # ---------------- 错误归因进化（结构修复，非完整任务记忆） ----------------
    def evolve_attributed(
        self,
        output: PathOutput,
        *,
        gt: str,
        oa: float,
        gt_emotion_probs: dict[str, float] | None = None,
        topic: str = "",
        threshold: float = 0.75,
        lr: float | None = None,
    ) -> dict[str, Any]:
        """识别失败结构 → 学条件化修复 → 反馈已组合修复；不存完整任务副本。"""
        if lr is None:
            lr = self.evolve_lr
        helpful = oa >= threshold
        log: dict[str, Any] = {
            "user_id": self.user_id,
            "oa": oa,
            "helpful": helpful,
            "update_policy": "failure_structure+conditional_repair+compose",
        }

        # 图边按结果反馈
        fmap = {f["id"]: f for f in output.factors}
        for p in output.paths:
            f = fmap.get(p["factor_id"])
            if not f:
                continue
            coord = p.get("coordinate") or ""
            if coord:
                self.graph.reinforce(
                    _nid("factor", f["text"]), "triggers", _nid("coordinate", coord),
                    good=helpful, lr=lr,
                )
                self.graph.reinforce(
                    _nid("coordinate", coord), p["edge_type"], _nid("stance", output.stance),
                    good=helpful, lr=lr,
                )

        # feedback on repairs that were composed into this prediction
        applied_ids = list((output.c_trace or {}).get("repairs_applied") or [])
        if applied_ids:
            self.failure_memory.feedback(applied_ids, helpful=helpful)
            log["repair_feedback"] = {"ids": applied_ids, "helpful": helpful}

        if helpful:
            for t in output.matched_theories:
                tid = t["id"]
                self.weights[tid] = max(0.2, min(2.5, self.weights.get(tid, 1.0) + lr * 0.5))
            log["attribution"] = {"primary_cause": "none", "note": "prediction accepted"}
        else:
            kinds = [str(f.get("type") or "") for f in output.factors]
            coords = [
                str(p.get("coordinate") or "")
                for p in output.paths
                if p.get("coordinate")
            ] or [str(t.get("coordinate") or "") for t in output.matched_theories]

            # 归因缓存（D-MEM fast buffer 理念）：失败结构命中错题本 → 复用历史归因，
            # 省一次归因 LLM 调用；fast_path 失败一律不调归因 LLM
            attrib = None
            cache_thr = float(self.tuning.get("attribution_cache_threshold", 0.6))
            cached_fs = None
            if kinds or coords:
                cached_fs = self.failure_memory.match_structure(
                    factor_kinds=kinds, coordinates=coords, min_overlap=cache_thr
                )
            if cached_fs is not None:
                attrib = {
                    "primary_cause": cached_fs.primary_cause,
                    "detail": f"cached:{cached_fs.id}",
                    "implicated_profile_attr": "",
                }
                log["attribution_cached"] = cached_fs.id
            elif (output.c_trace or {}).get("mode") == "fast_path":
                attrib = {
                    "primary_cause": "none",
                    "detail": "fast_path: skip LLM attribution",
                    "implicated_profile_attr": "",
                }
            else:
                attrib = self._attribute_error(output, gt)
            log["attribution"] = attrib
            cause = str(attrib.get("primary_cause") or "none")
            error_stage = str(attrib.get("error_stage") or "")

            fs, repair = self.failure_memory.observe_failure(
                primary_cause=cause,
                factor_kinds=kinds,
                coordinates=coords,
                oa=oa,
                detail=str(attrib.get("detail") or ""),
                implicated_profile_attr=str(attrib.get("implicated_profile_attr") or ""),
                strategy=str(attrib.get("transferable_strategy") or ""),
                exemplar=output.stimulus[:100],
                error_stage=error_stage,
            )
            log["failure_structure"] = {
                "id": fs.id,
                "cause": fs.primary_cause,
                "error_stage": fs.error_stage,
                "freq": fs.freq,
                "kinds": fs.factor_kinds,
                "coords": fs.coordinates,
            }
            log["learned_repair"] = {
                "id": repair.id,
                "action": repair.action,
                "payload": repair.payload,
            }
            # write structure node into user KG (signature only)
            self.graph.touch_node("failure", fs.id.replace("fail:", "")[:60], confidence=0.55)
            self.graph.add_edge(
                "failure", fs.id.replace("fail:", "")[:60],
                "updates",
                "stance", output.stance,
                weight=0.4, confidence=0.55,
            )

            # immediate conditional repair application (same as composed deltas)
            if cause == "theory_prior":
                for t in output.matched_theories:
                    tid = t["id"]
                    self.weights[tid] = max(0.2, min(2.5, self.weights.get(tid, 1.0) - lr))
                    coord = t.get("coordinate") or ""
                    if coord:
                        self.weights[coord] = max(
                            0.2, min(2.5, self.weights.get(coord, 1.0) - lr * 0.5)
                        )
            elif cause == "profile":
                attr = (attrib.get("implicated_profile_attr") or "").strip()
                if attr:
                    hit = self._find_persona_attr(attr)
                    if hit:
                        self.layers.adjust_persona(hit, supported=False)
                        log["persona_downgraded"] = hit
            elif cause == "short_term_state":
                for k in list(self.layers.short_emotions):
                    self.layers.short_emotions[k] *= 0.5
                log["short_term_reset"] = True
            elif cause == "factor_extraction":
                for f in output.factors:
                    nid_ = _nid("factor", f.get("text", ""))
                    if nid_ in self.graph.nodes:
                        self.graph.nodes[nid_].confidence = max(
                            0.05, self.graph.nodes[nid_].confidence - lr
                        )
            # retrieval: edges already contested via reinforce(good=False)

        if gt_emotion_probs:
            self.layers.observe_emotions(gt_emotion_probs)
        if topic:
            self.layers.observe_topic(topic)
        # RTWI-style adaptive threshold tracking: record stage reliability for percentile calibration
        c_trace = output.c_trace or {}
        sr = c_trace.get("stage_reliability")
        if sr:
            self.adaptive_thresholds.record_from_trace(c_trace)
        gate = c_trace.get("gate") or {}
        if "surprise" in gate:
            self.adaptive_thresholds.record_surprise(float(gate["surprise"]))
        save_layers(self.state_dir, self.layers)
        self.save_weights()
        self.graph.save()
        self.failure_memory.save(self.state_dir)
        pruned = self.failure_memory.prune_to_budget(
            max_structures=int(self.fm_budget.get("max_structures", 80)),
            max_repairs=int(self.fm_budget.get("max_repairs", 120)),
        )
        if pruned.get("pruned_structures") or pruned.get("pruned_repairs"):
            self.failure_memory.save(self.state_dir)
            log["failure_prune"] = pruned
        # Skills distillation: periodically distill repairs into compact rules
        fm_stats = self.failure_memory.stats()
        if fm_stats.get("admitted_repairs", 0) >= 3 and helpful is not None:
            try:
                from .skills_distiller import distill_from_failure_memory, save_distilled
                skill_set = distill_from_failure_memory(self.failure_memory)
                dst_path = Path(self.state_dir) / f"{self.user_id}_distilled_skills.json"
                save_distilled(skill_set, dst_path)
                log["distilled_skills"] = {
                    "total_rules": len(skill_set.mining_rules) + len(skill_set.reasoning_rules) + len(skill_set.cross_cutting),
                    "by_stage": {
                        "mining": len(skill_set.mining_rules),
                        "reasoning": len(skill_set.reasoning_rules),
                        "cross": len(skill_set.cross_cutting),
                    },
                }
            except Exception:  # noqa: BLE001
                pass
        log["graph"] = self.graph.stats()
        log["failure_memory"] = fm_stats
        hist = Path(self.state_dir) / f"{self.user_id}_evolve.jsonl"
        with hist.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(log, ensure_ascii=False) + "\n")
        return log

    def _attribute_error(self, output: PathOutput, gt: str) -> dict[str, Any]:
        """RTWI-style error attribution using stage_reliability structural signals.

        Priority: rule engine (deterministic, fast, always produces a cause)
        → LLM fallback only if rule engine returns 'none' and we have budget.

        The rule engine uses stage_reliability data (mining/synthesis scores,
        leap verdict, coverage, confidence, etc.) which is already computed
        for 92%+ of predictions — no extra LLM call needed.
        """
        # 1) Rule engine: deterministic attribution from structural signals
        rule_result = self._rule_based_attribution(output)
        if rule_result.get("primary_cause") != "none":
            return rule_result

        # 2) LLM fallback: only when rule engine can't determine cause
        try:
            trace = output.c_trace or {}
            tool_trace_line = ""
            if trace.get("mode") == "agent_graph":
                tools_called = trace.get("tools_called") or {}
                tools_str = ",".join(f"{k}×{v}" for k, v in sorted(tools_called.items())) or "（无工具调用）"
                tool_trace_line = (
                    f"【工具轨迹】tools={tools_str}; rounds={trace.get('num_tool_rounds', 0)}"
                    f"; fallback={trace.get('fallback') or '无'}\n"
                )
            stage_hint = self._compute_stage_reliability_hint(output)
            raw = self.llm.chat(
                [{"role": "system", "content": _ATTRIB_SYSTEM},
                 {"role": "user", "content": (
                     f"【事件】{output.stimulus[:400]}\n"
                     f"【推理路径】\n{output.verbalization}\n"
                     f"{tool_trace_line}"
                     f"{stage_hint}"
                     f"【预测评论】{output.predicted_opinion}\n"
                     f"【真实评论】{gt[:400]}\n请归因。"
                 )}],
                temperature=0.0, max_tokens=450, disable_thinking=True,
            )
            obj = _parse_json(raw)
            cause = str(obj.get("primary_cause") or "none")
            if cause not in {"factor_extraction", "retrieval", "theory_prior",
                             "profile", "short_term_state", "context_shift", "none"}:
                cause = "none"
            error_stage = str(obj.get("error_stage") or "")
            if error_stage not in {"mining", "reasoning"}:
                error_stage = CAUSE_TO_STAGE.get(cause, "none")
            # If LLM also returns none, keep the rule engine's "none" result
            if cause != "none":
                return {
                    "primary_cause": cause,
                    "error_stage": error_stage,
                    "detail": str(obj.get("detail") or "")[:160],
                    "implicated_profile_attr": str(obj.get("implicated_profile_attr") or ""),
                    "transferable_strategy": str(obj.get("transferable_strategy") or "")[:160],
                }
        except Exception:  # noqa: BLE001
            pass
        return rule_result  # rule engine's "none" with detail

    def _compute_stage_reliability_hint(self, output: PathOutput) -> str:
        """RTWI-style: 从 trace 提取阶段可靠性指标，辅助归因 LLM 判断错误阶段。"""
        trace = output.c_trace or {}
        factors = output.factors or []
        pf = trace.get("per_factor") or {}
        # Mining stage metrics
        n_factors = len(factors)
        types = sorted({f.get("type", "other") for f in factors})
        has_other = "other" in types
        covered = sum(1 for f in factors if pf.get(f.get("id", ""), {}).get("events")
                      or pf.get(f.get("id", ""), {}).get("matched"))
        coverage = covered / max(1, n_factors)
        n_theories = trace.get("num_theories", 0)
        n_events = trace.get("num_events", 0)
        # Synthesis metrics
        used = output.used or []
        used_coverage = len(used) / max(1, n_factors)
        reason_len = len(output.verbalization or "")
        low_evidence = bool(output.low_evidence)
        confidence = float(output.confidence or 0)
        # Build hint string
        mining_quality = "强" if coverage >= 0.75 and n_theories >= 2 else ("中" if coverage >= 0.4 else "弱")
        synth_quality = "强" if confidence >= 0.7 and reason_len >= 60 else ("中" if confidence >= 0.4 else "弱")
        return (
            f"【阶段可靠性】信息采集={mining_quality}(覆盖{coverage:.0%}/{n_factors}因素,"
            f"{n_theories}理论/{n_events}证据,类型={types}); "
            f"推理合成={synth_quality}(置信{confidence:.2f},理由{reason_len}字,"
            f"引用覆盖{used_coverage:.0%},低证据={low_evidence})\n"
        )

    def _rule_based_attribution(self, output: PathOutput) -> dict[str, Any]:
        """Deterministic error attribution from stage_reliability structural signals.

        Uses mining/retrieval/synthesis scores + leap verdict + coverage/confidence
        to infer error_stage and primary_cause WITHOUT an extra LLM call.

        This is the primary attribution path. The LLM is only a fallback for
        genuinely ambiguous cases the rule engine can't classify.
        """
        trace = output.c_trace or {}
        sr = trace.get("stage_reliability") or {}
        mining = sr.get("mining") or {}
        synthesis = sr.get("synthesis") or {}

        n_factors = int(mining.get("n_factors") or 0)
        coverage = float(mining.get("coverage") or 0)
        has_other = bool(mining.get("has_other"))
        type_diversity = float(mining.get("type_diversity") or 0)
        types = mining.get("types") or []
        total_theories = int(mining.get("total_theories") or 0)
        total_events = int(mining.get("total_events") or 0)
        covered = int(mining.get("covered") or 0)

        confidence = float(synthesis.get("confidence") or 0)
        used_coverage = float(synthesis.get("used_coverage") or 0)
        low_evidence = bool(synthesis.get("low_evidence"))
        leap = float(sr.get("leap") or 0)
        leap_verdict = str(sr.get("leap_verdict") or "")

        # If no stage_reliability data at all, can't rule-attribute
        if n_factors == 0 and not trace:
            return {
                "primary_cause": "none", "error_stage": "none",
                "detail": "无 stage_reliability 数据，需 LLM 归因",
                "implicated_profile_attr": "", "transferable_strategy": "",
            }

        # ── MINING STAGE ERRORS ──
        # 1) factor_extraction: factors decomposed poorly (too many "other" or low diversity)
        if has_other and type_diversity < 0.35 and n_factors <= 3:
            return {
                "primary_cause": "factor_extraction", "error_stage": "mining",
                "detail": (
                    f"因素分解质量差：{n_factors}因素中'other'占比过高"
                    f"(类型多样性{type_diversity:.0%})，未捕捉到真正驱动该用户立场的关键维度"
                ),
                "implicated_profile_attr": "",
                "transferable_strategy": (
                    f"对该类事件（{','.join(types[:3])}），尝试从利益/身份/道德三个维度"
                    f"重新分解因素，避免全归为'other'"
                ),
            }

        # 2) retrieval: evidence coverage too thin
        if coverage < 0.35 or (total_events < 1 and total_theories < 1):
            return {
                "primary_cause": "retrieval", "error_stage": "mining",
                "detail": (
                    f"检索覆盖严重不足：仅{covered}/{n_factors}因素有材料"
                    f"(覆盖率{coverage:.0%}, {total_events}证据/{total_theories}理论)"
                ),
                "implicated_profile_attr": "",
                "transferable_strategy": (
                    "对该类情境加宽检索（增加top_k + LLM改写查询词），"
                    "不要只依赖一个维度的历史证据"
                ),
            }

        # 3) retrieval: moderate coverage but still mining-weak
        if coverage < 0.55 and total_events <= 2:
            return {
                "primary_cause": "retrieval", "error_stage": "mining",
                "detail": (
                    f"检索偏弱：覆盖率{coverage:.0%}({covered}/{n_factors}), "
                    f"仅{total_events}条历史证据"
                ),
                "implicated_profile_attr": "",
                "transferable_strategy": (
                    "证据不足时不要硬套理论，以用户画像和一贯立场为准做出保守预测"
                ),
            }

        # ── REASONING STAGE ERRORS ──
        # 4) theory_prior: suspicious overconfidence (RTWI leap → theory misled reasoning)
        if leap_verdict == "suspicious_overconfident" and coverage < 0.65:
            return {
                "primary_cause": "theory_prior", "error_stage": "reasoning",
                "detail": (
                    f"理论过度自信：低覆盖{coverage:.0%}但高置信{confidence:.2f}, "
                    f"RTWI跃升{leap:+.2f}(>0.25)，理论可能误导了推理"
                ),
                "implicated_profile_attr": "",
                "transferable_strategy": (
                    "降低理论坐标权重(-0.15)，优先信任该用户图中先验路径而非群体理论"
                ),
            }

        # 5) theory_prior: leap is high even if not flagged — theory dominated reasoning
        if leap > 0.20 and confidence > 0.55 and coverage < 0.6:
            return {
                "primary_cause": "theory_prior", "error_stage": "reasoning",
                "detail": (
                    f"推理跃升偏高({leap:+.2f})，理论权重可能压过了薄弱的个体证据"
                    f"(覆盖{coverage:.0%}, 置信{confidence:.2f})"
                ),
                "implicated_profile_attr": "",
                "transferable_strategy": (
                    "冲突时个体证据>群体理论。该用户历史原话权重应高于匹配理论坐标"
                ),
            }

        # 6) short_term_state: good coverage but low synthesis quality + low_evidence
        if low_evidence and used_coverage < 0.4:
            return {
                "primary_cause": "short_term_state", "error_stage": "reasoning",
                "detail": (
                    f"历史证据与当前情境不匹配(低证据标记)，用户短期情绪/话题状态可能已变化"
                ),
                "implicated_profile_attr": "",
                "transferable_strategy": (
                    "以最近30%帖子为重(recent_boost=0.35)，降低远期历史权重重新预测"
                ),
            }

        # 7) context_shift: decent info but model uncertain → user stance shifting?
        if coverage >= 0.5 and confidence < 0.35 and total_events >= 2:
            return {
                "primary_cause": "context_shift", "error_stage": "reasoning",
                "detail": (
                    f"证据充足(覆盖{coverage:.0%}, {total_events}证据)但模型很不确信"
                    f"(置信{confidence:.2f})，用户立场可能正在发生长期转变"
                ),
                "implicated_profile_attr": "",
                "transferable_strategy": (
                    "该用户对这类议题的立场可能已在近期转向，重新评估其最新帖子中的立场方向"
                ),
            }

        # ── CROSS-CUTTING ──
        # 8) profile: factor types include group_identity + moral but prediction wrong
        if {"group_identity", "moral", "interest"}.intersection(set(types)) and coverage < 0.5:
            return {
                "primary_cause": "profile", "error_stage": "reasoning",
                "detail": (
                    f"涉及身份/道德/利益因素但检索覆盖不足({coverage:.0%})，"
                    f"用户画像中相关属性可能不准确"
                ),
                "implicated_profile_attr": ",".join(types[:3]),
                "transferable_strategy": (
                    "重新评估该用户画像中与身份/道德/利益相关的属性，可能已过时"
                ),
            }

        # ── FALLBACK: ambiguous, can be either stage ──
        if coverage < 0.45:
            return {
                "primary_cause": "retrieval", "error_stage": "mining",
                "detail": f"最可能原因：检索不足(覆盖{coverage:.0%})，先补证据再判断推理质量",
                "implicated_profile_attr": "",
                "transferable_strategy": "优先补齐证据再评估，不要在没有足够材料的情况下强行推理",
            }

        # Genuinely ambiguous — let LLM handle
        return {
            "primary_cause": "none", "error_stage": "none",
            "detail": (
                f"规则引擎无法精确分类：覆盖{coverage:.0%}, 置信{confidence:.2f}, "
                f"leap={leap:+.2f}, 类型={types}"
            ),
            "implicated_profile_attr": "",
            "transferable_strategy": "该错误模式不匹配已知规则，优先信任个体证据路径",
        }

    def _find_persona_attr(self, text: str) -> str:
        from .genminds import _tokenize, _overlap
        q = _tokenize(text)
        best, best_s = "", 0.0
        for attr in self.layers.persona:
            s = _overlap(q, _tokenize(attr))
            if s > best_s:
                best, best_s = attr, s
        return best if best_s >= 0.3 else ""

    # ---------------- 维护 ----------------
    def compress_graph(self) -> int:
        n = self.graph.compress()
        self.graph.save()
        return n

    def evaluate_emotion(self, output: PathOutput, gt_probs: dict[str, float]) -> float:
        return emotion_alignment(output.emotion_probs, gt_probs)
