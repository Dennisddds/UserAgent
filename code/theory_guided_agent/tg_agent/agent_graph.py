"""Graph Agent：LangGraph 运行的自主 Agentic RAG 预测体（seq-CUV-Agent）。

与 PathAgent 的结构性区别：9 节点固定流水线里「路由/检索/修复」的硬编码
决策，现在由 LLM 通过工具调用自主完成（彻底 Agent 化）：

    resolve_context（确定性：u/v 快照、情境、错题本权重组合）
      → gate_router（fast/slow，novelty 门控与基线一致）
      → fast: _fast_predict（复用，单调用）
      → slow: agent_call ⇄ tool_exec（LLM 自主工具循环，≤max_tool_rounds）
      → ensure_final（耗尽兜底：forced finalize → fast_fallback）
      → calibrate → absorb_finalize

evolve_attributed 不变：输出形状（paths/matched_theories/factors/repairs_applied）
与图工作流一致，错误归因只需补一行工具轨迹（见 path_agent._attribute_error）。
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .agent_tools import (
    FINALIZE_TOOL,
    ToolCall,
    ToolExecutor,
    XML_PROTOCOL_HINT,
    build_tool_specs,
    finalize_tool_choice,
    parse_native_tool_calls,
    parse_xml_tool_calls,
)
from .llm import ToolCallingUnsupportedError, _rtwi_white_box
from .agent import _parse_json
from .path_agent import PathAgent, PathOutput
from .path_workflow import apply_repairs_to_weights

_AGENT_SYSTEM = """你是 Theory-Guided 图推理 Agent，扮演指定微博用户本人。
须遵守 Identity/人设表达特征（任意大V/KOL）：用其本人声口与惯用自称，勿写成旁观者点评。
你持有工具，检索什么、检索多少、何时收工都由你自主决定——不要不加选择地把所有工具都调一遍。
建议工作方式：
1. 先 decompose_event 把事件分解为因素（每次预测只需一次）；
2. 针对关键因素按需调用：retrieve_memory（个体证据，优先级最高）/ retrieve_theory（群体理论先验）/
   query_causal_graph（该用户过往学到的推理先验）/ read_failure_notes（错题本可迁移策略）/
   read_situational（发帖时刻情境环境）；
3. 个体证据优先：该用户历史原话 > 群体理论。冲突时以个体证据为准，理论只用于解释机制；
4. 证据足够后调用 finalize_prediction 收工：自由判断 stance，不强制结构化因果链；
   reason（<=150字）写出判断的实际依据——先点个体证据（如 证据标题），再点理论坐标，没用到的不许提；
   used 只列真正影响判断的材料；
5. predicted_opinion 用该用户本人声音写，<=120字；情绪分布结合其短期情绪状态与历史风格。
检索预算：工具调用不超过 {max_rounds} 轮，每轮可并行调用多个工具，请把钱花在刀刃上。
证据薄弱时不要硬套理论：以画像/身份/一贯立场为准，并在 low_evidence_factors 中标注。"""


class GraphAgentState(TypedDict, total=False):
    # inputs
    stimulus: str
    post_id: str
    bid: str
    date: str
    topic: str
    # resolve_context
    u_snapshot: dict[str, Any]
    v_snapshot: dict[str, Any]
    sit: dict[str, Any] | None
    sit_weights: dict[str, float]
    situational_missing: bool
    source_block: str
    context_block: str
    strategy_notes: list[str]
    repairs_applied: list[str]
    active_repairs: list[dict[str, Any]]
    _runtime_weights: dict[str, float]
    _recency_boost: float
    # gate
    gate: dict[str, Any]
    # agent loop
    messages: list[dict[str, Any]]
    tool_protocol: str
    tool_rounds: int
    parse_failures: int
    pending_calls: list[ToolCall]
    tool_history: list[dict[str, Any]]
    num_llm_calls: int
    factors: list[Any]
    per_factor: dict[str, dict[str, Any]]
    all_matched: dict[str, Any]
    all_events: dict[str, Any]
    theory_routes: list[dict[str, Any]]
    skeptic: dict[str, Any]
    caveats: list[str]
    # terminal
    final: dict[str, Any]
    fallback: str
    confidence: float
    low_evidence: bool
    output: Any
    route_history: list[str]


_AGENT_CFG_DEFAULTS: dict[str, Any] = {
    "max_tool_rounds": 8,
    "tool_result_max_chars": 1200,
    "context_char_budget": 5500,
    "temperature": 0.3,
    "on_exhaust": "finalize_forced",  # finalize_forced | fast_fallback
    "tool_protocol": "native",        # native | xml
    "skeptic_tool": False,
    "log_messages": False,
}


def _hist(state: dict[str, Any], node: str) -> list[str]:
    return list(state.get("route_history") or []) + [node]


def _compute_stage_reliability(
    *,
    factors: list[Any],
    per_factor: dict[str, dict[str, Any]],
    all_matched: dict[str, Any],
    all_events: dict[str, Any],
    final: dict[str, Any],
    confidence: float,
    low_evidence: bool,
) -> dict[str, Any]:
    """RTWI-style: decompose chain into mining (info gathering) vs reasoning (synthesis) reliability."""
    n_factors = len(factors)
    types = sorted({f.type for f in factors if hasattr(f, 'type') and f.type})
    # Mining: how well did we gather information?
    covered = 0
    total_theories = 0
    total_events = 0
    for f in factors:
        fid = f.id if hasattr(f, 'id') else str(f.get('id', ''))
        bucket = per_factor.get(fid, {})
        if bucket.get("events") or bucket.get("matched"):
            covered += 1
        total_theories += len(bucket.get("matched") or [])
        total_events += len(bucket.get("events") or [])
    coverage = covered / max(1, n_factors)
    type_diversity = len(types) / max(1, n_factors) if types else 0.0
    has_other = "other" in types
    # Normalize mining score to [0,1]
    mining_score = round(
        0.40 * coverage
        + 0.25 * min(1.0, total_theories / max(1, n_factors * 2))
        + 0.20 * min(1.0, total_events / max(1, n_factors * 3))
        + 0.10 * type_diversity
        - (0.12 if has_other else 0.0),
        4,
    )
    # Synthesis: how well did we use the info?
    used = final.get("used") or []
    used_coverage = len(used) / max(1, n_factors)
    reason_len = len(str(final.get("reason") or ""))
    synthesis_score = round(
        0.35 * min(1.0, confidence)
        + 0.25 * used_coverage
        + 0.20 * min(1.0, reason_len / 150)
        + 0.20 * (0.0 if low_evidence else 1.0),
        4,
    )
    # Reliability Leap: synthesis relative to mining baseline
    leap = round(synthesis_score - mining_score, 4)
    leap_verdict = (
        "normal" if abs(leap) < 0.25
        else ("suspicious_overconfident" if leap > 0 else "genuine_dilemma")
    )
    return {
        "mining": {
            "n_factors": n_factors,
            "type_diversity": round(type_diversity, 3),
            "types": types,
            "has_other": has_other,
            "coverage": round(coverage, 3),
            "covered": covered,
            "total_theories": total_theories,
            "total_events": total_events,
            "score": mining_score,
        },
        "synthesis": {
            "used_coverage": round(used_coverage, 3),
            "reason_length": reason_len,
            "confidence": round(confidence, 4),
            "low_evidence": low_evidence,
            "score": synthesis_score,
        },
        "leap": leap,
        "leap_verdict": leap_verdict,
    }


def _reliability_leap_check(
    *,
    factors: list[Any],
    per_factor: dict[str, dict[str, Any]],
    all_matched: dict[str, Any],
    all_events: dict[str, Any],
    final: dict[str, Any],
    confidence: float,
    low_evidence: bool,
    caveats: list[str],
    messages: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[str], list[dict[str, Any]]]:
    """RTWI-style Reliability Leap validator: 连续信号替代二值证据门控.

    - 证据弱但自信高 → suspicious_overconfident → 强制 retry
    - 证据强但自信低 → genuine_dilemma → 接受 uncertain（合理）
    - 正常跃升 → 通过
    """
    reli = _compute_stage_reliability(
        factors=factors,
        per_factor=per_factor,
        all_matched=all_matched,
        all_events=all_events,
        final=final,
        confidence=confidence,
        low_evidence=low_evidence,
    )
    mining = reli["mining"]
    leap = reli["leap"]
    verdict = reli["leap_verdict"]
    nf = mining["n_factors"]
    covered = mining["covered"]
    coverage = mining["coverage"]

    # Binary evidence gate (keep as hard floor)
    if nf > 0 and covered < max(1, nf * 0.5):
        caveats.append(f"evidence_gate: 覆盖率 {covered}/{nf}，强制加宽检索")
        messages.append({
            "role": "user",
            "content": (
                f"【证据门控】仅 {covered}/{nf} 因素有检索材料，"
                "证据薄弱。请对未覆盖因素调用 retrieve_memory / "
                "retrieve_theory，补齐后再 finalize。"
            ),
        })
        return None, caveats, messages

    # Reliability Leap: suspicious overconfidence → retry with skeptic hint
    if verdict == "suspicious_overconfident" and coverage < 0.6:
        caveats.append(
            f"reliability_leap: 证据{coverage:.0%}但置信{confidence:.2f}，"
            f"跃升{leap:+.2f}→可疑过度自信，要求skeptic复核"
        )
        messages.append({
            "role": "user",
            "content": (
                f"【可靠性跃升异常】检索覆盖率仅{coverage:.0%}({covered}/{nf}因素)，"
                f"但预测置信度达{confidence:.2f}。跃升幅度{leap:+.2f}异常。"
                "请自查：(1)是否硬套了理论？(2)低覆盖因素是否被忽略？"
                "如有必要请调用 skeptic_check，再重新 finalize。"
            ),
        })
        return None, caveats, messages

    # Genuine dilemma: evidence is there but still uncertain → this is valid
    if verdict == "genuine_dilemma" and not low_evidence:
        caveats.append(
            f"reliability_leap: 证据{coverage:.0%}但置信{confidence:.2f}，"
            f"跃升{leap:+.2f}→真诚两难，标记uncertain"
        )

    return reli, caveats, messages


def build_agent_graph(agent: "GraphAgent") -> Any:
    """Bind the GraphAgent's tools/prompts into a compiled LangGraph app."""

    cfg = agent.agent_cfg

    # ---------------- deterministic bookends ----------------
    def resolve_context(state: dict[str, Any]) -> dict[str, Any]:
        from .situational_env import (
            format_situational_block,
            resolve_situational,
            situational_env_weights,
        )
        from .user_actions import format_source_block

        stimulus = state.get("stimulus") or ""
        u = agent.memory.u_snapshot(max_motifs=agent.max_motifs)
        v = agent.memory.v_snapshot()
        sit = resolve_situational(
            agent.situational_store,
            post_id=state.get("post_id") or None,
            bid=state.get("bid") or None,
            date=state.get("date") or None,
            topic=state.get("topic") or None,
            text=stimulus,
        )
        sit_weights = situational_env_weights(sit, boost=1.85)
        situational_missing = bool(agent.use_situational and not sit_weights)
        source_block = format_source_block(getattr(agent, "source_profile", None) or {})

        # 错题本：权重类修复确定性组合（与基线一致；因素未分解，用图先验坐标做临时匹配）
        provisional_coords: list[str] = []
        for p in agent.graph.paths_for_factor(stimulus[:200], top_k=2):
            via = str(p.get("via") or "")
            if via.startswith("coordinate:"):
                provisional_coords.append(via.split(":", 1)[1])
        hits = agent.failure_memory.retrieve_repairs(
            factor_kinds=[], coordinates=provisional_coords, top_k=3
        )
        active = [
            {
                "id": r.id,
                "action": r.action,
                "payload": r.payload,
                "structure_id": fs.id,
                "overlap": ov,
                "cause": fs.primary_cause,
            }
            for r, fs, ov in hits
        ]
        weights, applied = apply_repairs_to_weights(agent.weights, active)
        strategy_notes: list[str] = []
        for r, fs, ov in hits:
            if r.action != "strategy_note":
                continue
            note = str((r.payload or {}).get("note") or "").strip()
            if not note:
                continue
            guard = fs.when_to_apply or " / ".join(fs.factor_kinds[:3])
            strategy_notes.append(f"当{guard}：{note}（源自{fs.freq}次类似失败）")
        recency_boost = 0.0
        for a in active:
            if a.get("cause") == "short_term_state" or a["action"] == "reset_short_term":
                recency_boost = max(
                    recency_boost, float((a.get("payload") or {}).get("recency_boost") or 0.35)
                )

        dom = agent.layers.dominant_emotions()
        strategy_block = ""
        if strategy_notes:
            strategy_block = (
                "## 错题本策略（该用户历史失败中学到的可迁移修正，优先遵守）\n"
                + "\n".join(f"- {n}" for n in strategy_notes[:4])
                + "\n\n"
            )
        # 蒸馏技能：定期从 failure_memory 蒸馏出的紧凑规则，注入 prompt
        distilled_block = ""
        try:
            from .skills_distiller import load_distilled
            dst_path = Path(agent.state_dir) / f"{agent.user_id}_distilled_skills.json"
            skill_set = load_distilled(dst_path)
            if skill_set:
                distilled_block = skill_set.to_prompt_block(stage="", max_rules=4)
                if distilled_block:
                    strategy_block = strategy_block + distilled_block
        except Exception:  # noqa: BLE001
            pass
        budget = int(cfg.get("context_char_budget", 5500))
        identity = (agent.memory.identity_block() or "")[:1200]
        persona = (agent.layers.persona_block() or "")[:1000]
        context_block = f"""## Identity
{identity}

## 用户画像（带置信度）
{persona}

## 短期情绪状态
{', '.join(f'{e}={v_}' for e, v_ in dom) or '(平静/无显著情绪)'}

## 信源画像
{source_block[:500]}

## 情境环境（摘要；完整版可调用 read_situational）
{format_situational_block(sit)[:400]}

{strategy_block}## 事件
{stimulus[:800]}

请以该用户身份预测其对事件的回应。先分解因素，再按需检索，最后 finalize_prediction。"""
        if len(context_block) > budget:
            context_block = context_block[:budget] + "\n...(context truncated for local Flash ctx)"

        return {
            "u_snapshot": u,
            "v_snapshot": v,
            "sit": sit,
            "sit_weights": sit_weights,
            "situational_missing": situational_missing,
            "source_block": source_block,
            "context_block": context_block,
            "strategy_notes": strategy_notes,
            "repairs_applied": applied,
            "active_repairs": active,
            "_runtime_weights": weights,
            "_recency_boost": recency_boost,
            "route_history": _hist(state, "resolve_context"),
        }

    def gate_node(state: dict[str, Any]) -> dict[str, Any]:
        fp = (agent.tuning or {}).get("fast_path") or {}
        if not fp.get("enabled"):
            return {"gate": {"route": "slow", "disabled": True}}
        from .novelty import compute_surprise

        gate = compute_surprise(
            agent,
            stimulus=state.get("stimulus") or "",
            topic=state.get("topic") or "",
            recent_window=int(fp.get("recent_window", 50)),
            w_topic=float(fp.get("w_topic", 0.35)),
            w_lexical=float(fp.get("w_lexical", 0.35)),
            w_prior=float(fp.get("w_prior", 0.30)),
        )
        threshold = float(fp.get("surprise_threshold", 0.35))
        gate["threshold"] = threshold
        gate["route"] = "fast" if gate["surprise"] < threshold else "slow"
        return {"gate": gate}

    def gate_router(state: dict[str, Any]) -> str:
        return str((state.get("gate") or {}).get("route") or "slow")

    def fast_predict_node(state: dict[str, Any]) -> dict[str, Any]:
        gate = state.get("gate") or {"surprise": 1.0, "threshold": 0.0, "route": "fast"}
        out = agent._fast_predict(
            state.get("stimulus") or "",
            post_id=state.get("post_id") or "",
            date=state.get("date") or "",
            topic=state.get("topic") or "",
            gate=gate,
        )
        return {"output": out, "gate": gate, "route_history": _hist(state, "fast_predict")}

    # ---------------- agent loop ----------------
    def _system_prompt(protocol: str) -> str:
        base = _AGENT_SYSTEM.replace("{max_rounds}", str(cfg.get("max_tool_rounds", 8)))
        if protocol == "xml":
            base += "\n\n" + XML_PROTOCOL_HINT
        return base

    def _context_guard(messages: list[dict[str, Any]], caveats: list[str]) -> None:
        budget = int(cfg.get("context_char_budget", 24000))
        total = sum(len(str(m.get("content") or "")) for m in messages)
        dropped = 0
        while total > budget:
            victim = next(
                (i for i, m in enumerate(messages) if m.get("role") == "tool"), None
            )
            if victim is None:
                break
            total -= len(str(messages[victim].get("content") or ""))
            messages.pop(victim)
            dropped += 1
        if dropped and not any("context_guard" in c for c in caveats):
            caveats.append(f"context_guard: dropped {dropped} oldest tool results")

    def agent_call(state: dict[str, Any]) -> dict[str, Any]:
        protocol = state.get("tool_protocol") or str(cfg.get("tool_protocol", "native"))
        caveats = list(state.get("caveats") or [])
        messages = list(state.get("messages") or [])
        if not messages:
            messages = [
                {"role": "system", "content": _system_prompt(protocol)},
                {"role": "user", "content": state.get("context_block") or state.get("stimulus")},
            ]
        _context_guard(messages, caveats)

        specs = build_tool_specs(skeptic_tool=bool(cfg.get("skeptic_tool", False)))
        try:
            msg = agent.llm.chat_completion(
                messages,
                tools=specs if protocol == "native" else None,
                tool_choice="auto" if protocol == "native" else None,
                temperature=float(cfg.get("temperature", 0.3)),
                max_tokens=2200,
            )
        except ToolCallingUnsupportedError:
            # 端点不支持 tool_calls → 降级 xml 协议，重建带协议提示的消息
            protocol = "xml"
            caveats.append("tool_protocol: endpoint 不支持 tool_calls，已降级 xml 协议")
            messages = [
                {"role": "system", "content": _system_prompt(protocol)},
                {"role": "user", "content": state.get("context_block") or state.get("stimulus")},
            ]
            msg = agent.llm.chat_completion(
                messages,
                temperature=float(cfg.get("temperature", 0.3)),
                max_tokens=2200,
            )
        num_llm_calls = int(state.get("num_llm_calls") or 0) + 1

        if protocol == "native":
            messages.append(msg)
            calls = parse_native_tool_calls(msg)
        else:
            content = str(msg.get("content") or "")
            messages.append({"role": "assistant", "content": content})
            calls = parse_xml_tool_calls(content)

        # 连续两轮解析失败且仍在 native → 自动降级 xml（prompt 协议问题，不是端点问题）
        parse_failures = 0 if calls else int(state.get("parse_failures") or 0) + 1
        if not calls and protocol == "native" and parse_failures >= 2:
            protocol = "xml"
            caveats.append("tool_protocol: 连续两轮未解析出工具调用，已降级 xml 协议")
            messages = [
                {"role": "system", "content": _system_prompt(protocol)},
                {"role": "user", "content": state.get("context_block") or state.get("stimulus")},
            ]
            parse_failures = 0

        # finalize 立即执行；其余工具留给 tool_exec
        final = state.get("final")
        pending: list[ToolCall] = []
        mutable = dict(state)
        mutable.update(
            {
                "factors": state.get("factors") or [],
                "per_factor": state.get("per_factor") or {},
                "all_matched": state.get("all_matched") or {},
                "all_events": state.get("all_events") or {},
                "theory_routes": state.get("theory_routes") or [],
                "tool_history": list(state.get("tool_history") or []),
                "tool_rounds": state.get("tool_rounds") or 0,
                "caveats": caveats,
            }
        )
        executor = ToolExecutor(agent, mutable)
        for call in calls:
            if call.name == FINALIZE_TOOL and final is None:
                executor.execute(call)
                final = mutable.get("final")
                # ── RTWI 可靠性门控 ──
                # 连续可靠性信号替代二值证据门控：
                #   - 证据弱+自信高 → suspicious → retry
                #   - 证据强+自信低 → genuine dilemma → accept uncertain
                #   - 证据弱(硬底线<50%) → binary gate retry
                if final is not None and not mutable.get("_evidence_gate_retried"):
                    reli_result, caveats, messages = _reliability_leap_check(
                        factors=mutable.get("factors") or [],
                        per_factor=mutable.get("per_factor") or {},
                        all_matched=mutable.get("all_matched") or {},
                        all_events=mutable.get("all_events") or {},
                        final=final,
                        confidence=float(final.get("confidence", 0.5)),
                        low_evidence=bool(final.get("low_evidence_factors")),
                        caveats=caveats,
                        messages=messages,
                    )
                    if reli_result is None:
                        # Gate triggered → retry
                        final = None
                        mutable["_evidence_gate_retried"] = True
                        mutable["final"] = None
                    else:
                        mutable["_stage_reliability"] = reli_result
                # ── end 可靠性门控 ──
            else:
                pending.append(call)
        if final is not None:
            pending = []

        return {
            "messages": messages,
            "tool_protocol": protocol,
            "parse_failures": parse_failures,
            "pending_calls": pending,
            "num_llm_calls": num_llm_calls,
            "final": final,
            "factors": mutable.get("factors"),
            "per_factor": mutable.get("per_factor"),
            "all_matched": mutable.get("all_matched"),
            "all_events": mutable.get("all_events"),
            "theory_routes": mutable.get("theory_routes"),
            "tool_history": mutable.get("tool_history"),
            "caveats": mutable.get("caveats"),
            "_evidence_gate_retried": mutable.get("_evidence_gate_retried"),
            "route_history": _hist(state, "agent_call"),
        }

    def after_agent_call(state: dict[str, Any]) -> str:
        if state.get("final"):
            return "done"
        # 证据门控触发 → 直接返回 agent_call 再来一轮（retry message 已在 messages 中）
        if state.get("_evidence_gate_retried") and not state.get("final"):
            return "retry"
        if not (state.get("pending_calls") or []):
            return "done"  # 无工具调用 → 走耗尽兜底
        if int(state.get("tool_rounds") or 0) >= int(cfg.get("max_tool_rounds", 8)):
            return "done"
        return "tools"

    def tool_exec(state: dict[str, Any]) -> dict[str, Any]:
        protocol = state.get("tool_protocol") or "native"
        rounds = int(state.get("tool_rounds") or 0) + 1
        mutable = dict(state)
        mutable["tool_rounds"] = rounds
        executor = ToolExecutor(agent, mutable)
        messages = list(state.get("messages") or [])
        xml_results: list[str] = []
        for call in state.get("pending_calls") or []:
            result = executor.execute(call)
            if protocol == "native":
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": result}
                )
            else:
                xml_results.append(f'<tool_result name="{call.name}">{result}</tool_result>')
        if xml_results:
            messages.append({"role": "user", "content": "\n".join(xml_results)})

        return {
            "messages": messages,
            "tool_rounds": rounds,
            "pending_calls": [],
            "final": mutable.get("final"),
            "factors": mutable.get("factors"),
            "per_factor": mutable.get("per_factor"),
            "all_matched": mutable.get("all_matched"),
            "all_events": mutable.get("all_events"),
            "theory_routes": mutable.get("theory_routes"),
            "tool_history": mutable.get("tool_history"),
            "skeptic": mutable.get("skeptic"),
            "caveats": mutable.get("caveats"),
            # 错题本飞轮：decompose 后的修复组合结果需透传到后续节点
            "repairs_applied": mutable.get("repairs_applied"),
            "active_repairs": mutable.get("active_repairs"),
            "_runtime_weights": mutable.get("_runtime_weights"),
            "strategy_notes": mutable.get("strategy_notes"),
            "_recency_boost": mutable.get("_recency_boost"),
            "route_history": _hist(state, "tool_exec"),
        }

    # ---------------- exhaustion & calibration ----------------
    def ensure_final(state: dict[str, Any]) -> dict[str, Any]:
        if state.get("final"):
            return {"route_history": _hist(state, "ensure_final")}
        caveats = list(state.get("caveats") or [])
        num_llm_calls = int(state.get("num_llm_calls") or 0)
        policy = str(cfg.get("on_exhaust", "finalize_forced"))
        final = None
        updates: dict[str, Any] = {}

        if policy == "finalize_forced":
            protocol = state.get("tool_protocol") or "native"
            messages = list(state.get("messages") or [])
            mutable = dict(state)
            mutable.setdefault("tool_history", list(state.get("tool_history") or []))
            executor = ToolExecutor(agent, mutable)
            try:
                if protocol == "native":
                    msg = agent.llm.chat_completion(
                        messages,
                        tools=build_tool_specs(skeptic_tool=False),
                        tool_choice=finalize_tool_choice(),
                        temperature=0.2,
                        max_tokens=1600,
                    )
                    calls = parse_native_tool_calls(msg)
                else:
                    messages.append(
                        {
                            "role": "user",
                            "content": "工具预算已用完。请立即调用 finalize_prediction 输出最终预测。",
                        }
                    )
                    msg = agent.llm.chat_completion(
                        messages, temperature=0.2, max_tokens=1600
                    )
                    calls = parse_xml_tool_calls(str(msg.get("content") or ""))
                num_llm_calls += 1
                for call in calls:
                    if call.name == FINALIZE_TOOL:
                        executor.execute(call)
                        break
                final = mutable.get("final")
                updates = {
                    "factors": mutable.get("factors"),
                    "per_factor": mutable.get("per_factor"),
                    "all_matched": mutable.get("all_matched"),
                    "all_events": mutable.get("all_events"),
                    "tool_history": mutable.get("tool_history"),
                }
            except Exception as e:  # noqa: BLE001
                caveats.append(f"forced_finalize_error: {e}")
            if final is None:
                caveats.append("forced_finalize_failed: 转 fast_fallback")
            else:
                caveats.append("finalize_forced: 工具预算耗尽后强制收工")

        if final is None:
            gate = dict(state.get("gate") or {})
            gate.setdefault("surprise", 1.0)
            gate["fallback"] = "agent_exhausted"
            out = agent._fast_predict(
                state.get("stimulus") or "",
                post_id=state.get("post_id") or "",
                date=state.get("date") or "",
                topic=state.get("topic") or "",
                gate=gate,
            )
            caveats.append("fast_fallback: agent 循环未收敛，退回单调用直觉预测")
            return {
                "output": out,
                "fallback": "fast_fallback",
                "caveats": caveats[:6],
                "num_llm_calls": num_llm_calls,
                "route_history": _hist(state, "ensure_final"),
            }

        return {
            "final": final,
            "caveats": caveats[:6],
            "num_llm_calls": num_llm_calls,
            "route_history": _hist(state, "ensure_final"),
            **updates,
        }

    def _per_factor_list(state: dict[str, Any]) -> list[dict[str, Any]]:
        per_factor = state.get("per_factor") or {}
        ordered: list[dict[str, Any]] = []
        seen: set[str] = set()
        for f in state.get("factors") or []:
            if f.id in per_factor:
                ordered.append(per_factor[f.id])
                seen.add(f.id)
        for fid, bucket in per_factor.items():
            if fid not in seen:
                ordered.append(bucket)
        return ordered

    def calibrate(state: dict[str, Any]) -> dict[str, Any]:
        if state.get("output") is not None:
            # fast_fallback 已产出完整 PathOutput，跳过校准
            return {"route_history": _hist(state, "calibrate")}
        final = dict(state.get("final") or {})
        sk = dict(state.get("skeptic") or {})
        confidence, low_ev = agent._calibrate(
            _per_factor_list(state),
            list(final.get("paths") or []),
            list(final.get("low_evidence_factors") or []),
            sk,
        )
        stance = str(final.get("stance") or "uncertain")
        opinion = str(final.get("opinion") or "")
        overturn_min_conf = float((agent.tuning or {}).get("skeptic_overturn_min_conf", 0.75))
        if sk.get("overturn") and sk.get("revised_opinion"):
            if (not low_ev) and confidence >= overturn_min_conf:
                stance = str(sk.get("revised_stance") or stance)
                if stance not in {"support", "oppose", "mixed", "uncertain"}:
                    stance = "uncertain"
                opinion = str(sk["revised_opinion"]).strip()
                sk["overturn_applied"] = True
            else:
                sk["overturn_applied"] = False
                sk["challenge"] = (sk.get("challenge") or "") + "（证据不足，维持原判）"
        final["stance"] = stance
        final["opinion"] = opinion
        return {
            "final": final,
            "confidence": confidence,
            "low_evidence": low_ev,
            "skeptic": sk,
            "route_history": _hist(state, "calibrate"),
        }

    def absorb_finalize(state: dict[str, Any]) -> dict[str, Any]:
        if state.get("output") is not None:
            # fast_fallback 已产出完整 PathOutput，跳过吸收
            return {"route_history": _hist(state, "absorb_finalize")}
        factors = list(state.get("factors") or [])
        final = state.get("final") or {}
        paths = list(final.get("paths") or [])
        stance = str(final.get("stance") or "uncertain")
        reason = str(final.get("reason") or "")
        per_factor_list = _per_factor_list(state)
        rendered = agent._render_paths(factors, paths, stance)
        verbalization = f"{reason}\n【图推理引用】\n{rendered}" if reason else rendered
        agent._graph_absorb(factors, per_factor_list, paths, stance)

        for a in state.get("active_repairs") or []:
            sid = a.get("structure_id") or ""
            if sid:
                agent.graph.touch_node("failure", sid.replace("fail:", "")[:60], confidence=0.4)
                agent.graph.add_edge(
                    "failure", sid.replace("fail:", "")[:60],
                    "derived_from",
                    "stance", stance,
                    weight=0.3, confidence=0.4,
                )
        agent.graph.save()

        all_matched = state.get("all_matched") or {}
        all_events = state.get("all_events") or {}
        sk = state.get("skeptic") or {}
        low_ev = bool(state.get("low_evidence"))
        caveats = list(state.get("caveats") or [])
        if state.get("situational_missing"):
            caveats.append("situational_missing: 无个人接收路径记录，仅理论先验+个体记忆参与")
        if sk.get("challenge"):
            caveats.append("skeptic: " + str(sk["challenge"])[:120])
        if low_ev:
            caveats.append("low_evidence: 证据覆盖不足，预测置信度低")
        if state.get("repairs_applied"):
            caveats.append("repairs_composed: " + ",".join(state["repairs_applied"][:3]))

        tool_history = list(state.get("tool_history") or [])
        tools_called: dict[str, int] = {}
        for h in tool_history:
            tools_called[h["name"]] = tools_called.get(h["name"], 0) + 1

        out = PathOutput(
            user_id=agent.user_id,
            stimulus=state.get("stimulus") or "",
            predicted_opinion=str(final.get("opinion") or ""),
            stance=stance,
            activated_coordinates=sorted({p.get("coordinate") for p in paths if p.get("coordinate")}),
            matched_theories=[
                {
                    "id": m.card.id,
                    "name": m.card.name,
                    "coordinate": m.card.coordinate,
                    "score": round(m.score, 4),
                    "why": m.why,
                    "mechanism": m.card.mechanism[:240],
                }
                for m in sorted(all_matched.values(), key=lambda m: -m.score)[:6]
            ],
            evidence_events=[
                {
                    "map_id": e.map_id,
                    "title": e.event_title,
                    "score": round(e.score, 4),
                    "opinion": (e.user_opinion or "")[:200],
                }
                for e in sorted(all_events.values(), key=lambda e: -e.score)[:8]
            ],
            verbalization=verbalization,
            c_trace={
                "mode": "agent_graph",
                "steps": list(state.get("route_history") or []),
                "reason": reason,
                "tool_history": tool_history,
                "tools_called": tools_called,
                "num_tool_rounds": int(state.get("tool_rounds") or 0),
                "num_llm_calls": int(state.get("num_llm_calls") or 0),
                "tool_protocol": state.get("tool_protocol") or "native",
                "fallback": state.get("fallback") or "",
                "repairs_applied": state.get("repairs_applied") or [],
                "strategy_notes": state.get("strategy_notes") or [],
                "active_repairs": [
                    {"id": a.get("id"), "action": a.get("action"), "overlap": a.get("overlap")}
                    for a in (state.get("active_repairs") or [])
                ],
                "failure_memory": agent.failure_memory.stats(),
                "repair_effectiveness": agent.failure_memory.repair_effectiveness(),
                "theory_routes": state.get("theory_routes") or [],
                "num_factors": len(factors),
                "num_paths": len(paths),
                "num_theories": len(all_matched),
                "num_events": len(all_events),
                "post_id": state.get("post_id"),
                "situational_missing": bool(state.get("situational_missing")),
                "graph": agent.graph.stats(),
                # RTWI-style stage reliability metrics
                "stage_reliability": _compute_stage_reliability(
                    factors=factors,
                    per_factor=state.get("per_factor") or {},
                    all_matched=all_matched,
                    all_events=all_events,
                    final=final,
                    confidence=float(state.get("confidence") or 0.0),
                    low_evidence=low_ev,
                ),
                # White-box CoT from last LLM turn (local Flash reasoning parser)
                "model_reasoning": getattr(agent.llm, "last_reasoning", "") or "",
                "white_box": _rtwi_white_box(
                    getattr(agent.llm, "last_reasoning", "") or "",
                    str(final.get("opinion") or ""),
                ),
            },
            u_snapshot=state.get("u_snapshot") or {},
            v_snapshot=state.get("v_snapshot") or {},
            caveats=caveats[:5],
            factors=[f.to_dict() for f in factors],
            paths=paths,
            emotion_probs=final.get("emotion_probs") or {},
            confidence=float(state.get("confidence") or 0.0),
            low_evidence=low_ev,
            skeptic=sk,
        )
        return {"output": out, "caveats": caveats[:5], "route_history": _hist(state, "absorb_finalize")}

    g: StateGraph = StateGraph(GraphAgentState)
    g.add_node("resolve_context", resolve_context)
    g.add_node("gate", gate_node)
    g.add_node("fast_predict", fast_predict_node)
    g.add_node("agent_call", agent_call)
    g.add_node("tool_exec", tool_exec)
    g.add_node("ensure_final", ensure_final)
    g.add_node("calibrate", calibrate)
    g.add_node("absorb_finalize", absorb_finalize)

    g.add_edge(START, "resolve_context")
    g.add_edge("resolve_context", "gate")
    g.add_conditional_edges("gate", gate_router, {"fast": "fast_predict", "slow": "agent_call"})
    g.add_edge("fast_predict", END)
    g.add_conditional_edges("agent_call", after_agent_call, {"tools": "tool_exec", "done": "ensure_final", "retry": "agent_call"})
    g.add_edge("tool_exec", "agent_call")
    g.add_edge("ensure_final", "calibrate")
    g.add_edge("calibrate", "absorb_finalize")
    g.add_edge("absorb_finalize", END)
    return g.compile()


def run_agent_predict(agent: "GraphAgent", **kwargs: Any) -> PathOutput:
    if agent._compiled_graph is None:
        agent._compiled_graph = build_agent_graph(agent)
    init: dict[str, Any] = {
        "stimulus": kwargs.get("stimulus") or "",
        "post_id": kwargs.get("post_id") or "",
        "bid": kwargs.get("bid") or "",
        "date": kwargs.get("date") or "",
        "topic": kwargs.get("topic") or "",
    }
    result = agent._compiled_graph.invoke(init, config={"recursion_limit": 64})
    out = result.get("output")
    if out is None:
        raise RuntimeError("agent graph finished without output")
    if isinstance(out.c_trace, dict):
        out.c_trace["route_history"] = list(result.get("route_history") or [])
    if agent.agent_cfg.get("log_messages"):
        try:
            dump = {
                "post_id": kwargs.get("post_id") or "",
                "messages": result.get("messages") or [],
            }
            path = Path(agent.state_dir) / f"agent_messages_{kwargs.get('post_id') or 'na'}.json"
            path.write_text(json.dumps(dump, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    return out


class GraphAgent(PathAgent):
    """彻底 Agent 化的预测体：LLM 持工具自主完成 Agentic RAG（fusion 哲学）。

    继承 PathAgent 的 memory/graph/failure_memory/_fast_predict/evolve_attributed；
    仅 predict 改为 LangGraph 工具循环。strict 模式不移植（旧 seq-CUV-Path 留作基线）。
    """

    def __init__(self, *args: Any, agent_cfg: dict[str, Any] | None = None, **kwargs: Any) -> None:
        kwargs["path_mode"] = "fusion"
        super().__init__(*args, **kwargs)
        self.agent_cfg: dict[str, Any] = dict(_AGENT_CFG_DEFAULTS) | dict(agent_cfg or {})
        self._compiled_graph: Any = None

    def predict(
        self,
        stimulus: str,
        *,
        post_id: str = "",
        bid: str = "",
        date: str = "",
        topic: str = "",
    ) -> PathOutput:
        return run_agent_predict(
            self, stimulus=stimulus, post_id=post_id, bid=bid, date=date, topic=topic
        )


# ── AgentX: paper-informed improvement layer (Mentis + Debate-on-Graph) ────
#
# GraphAgentX runs the unchanged GraphAgent first, then applies:
#   1. Candidate branch simulation (Mentis / Dreamer "imagine then choose"):
#      generate k candidate posts instead of a single sampling, decompose the
#      stance intent, and pick the branch that stays most consistent with the
#      user's mental state (persona + retrieved evidence + causal graph).
#   2. Multi-Agent Debate (Debate-on-Graph): a Proponent (the k candidates), a
#      Challenger that sees only *half* the evidence (information asymmetry),
#      and a Judge rubric with 1-5 -> [0,1] grades plus a strict relative rank.
#   3. Reliability-guided selection: candidates are scored on
#      mentally_consistent (0.5) / style_fidelity (0.3) / socially_appropriate
#      (0.2), with graph-support as a tie-breaker; the decision is deterministic
#      (rank then seeded hash), mirroring Mentis' tie handling.

_X_SYSTEM = """你是"个体对齐改进层"：给定同一个社交媒体用户的人设、证据与事件，做三件事：
1) 生成 K 个候选发帖（各有立场与理由，尽量不同措辞与侧重）；
2) 挑战者角色只能看到一半证据，挑出每个候选与用户一贯立场/证据最可能冲突的漏洞；
3) 裁决者给每个候选在「立场一致性(0.5)/文风保真(0.3)/平台得体(0.2)」三个维度打 1-5 档，
   并给出严格的 1..K 相对排名（不允许并列，每个整数用一次）。
只输出合法 JSON，不要 Markdown。个体历史原话优先于群体理论；不得编造用户没表达过的观点。"""


def _x_chat(agent: "GraphAgent", user: str, *, temperature: float, max_tokens: int) -> str:
    # The improvement layer is selection/scoring, not the audited reasoning path
    # (the base agent already captured its CoT). Thinking off keeps the whole
    # budget on the JSON answer instead of leaving content empty.
    msg = agent.llm.chat_with_trace(
        [{"role": "system", "content": _X_SYSTEM}, {"role": "user", "content": user}],
        temperature=temperature,
        max_tokens=max_tokens,
        disable_thinking=True,
    )
    return (msg.get("content") or "").strip()


def _x_evidence_block(base: PathOutput, limit: int | None = None) -> str:
    ev = (base.evidence_events or [])[:limit] if limit else (base.evidence_events or [])
    parts = []
    for e in ev[:6]:
        title = e.get("title") or ""
        opinion = e.get("opinion") or ""
        parts.append(f"- {title}: {opinion[:120]}")
    for t in (base.matched_theories or [])[:3]:
        parts.append(f"- 理论坐标 {t.get('coordinate')}: {(t.get('mechanism') or t.get('name') or '')[:80]}")
    return "\n".join(parts) or "- （无检索证据）"


def _x_context(agent: "GraphAgent", stimulus: str) -> str:
    return (
        f"用户人设（须沿用其声口）:\n{agent.memory.identity_block()}\n\n"
        f"当前事件:\n{stimulus}\n"
    )


def _x_candidates(agent: "GraphAgent", base: PathOutput, stimulus: str, k: int) -> list[dict]:
    user = (
        _x_context(agent, stimulus)
        + f"\n检索到的个体证据/理论:\n{_x_evidence_block(base)}\n"
        + f"\n生成 {k} 个候选发帖（每人一条，<=120字），输出 JSON: "
        + '{"candidates":[{"opinion":"...","stance":"support|oppose|mixed|uncertain","reason":"<=60字"}]}'
    )
    raw = _x_chat(agent, user, temperature=0.7, max_tokens=900)
    obj = _parse_json(raw)
    cands = obj.get("candidates") if isinstance(obj, dict) else None
    if not isinstance(cands, list) or not cands:
        # fall back to the base answer as a single candidate
        return [{"opinion": base.predicted_opinion, "stance": base.stance, "reason": base.verbalization[:60]}]
    out = []
    for c in cands:
        if isinstance(c, dict) and (c.get("opinion") or "").strip():
            out.append(c)
    return out[:k] or [{"opinion": base.predicted_opinion, "stance": base.stance, "reason": ""}]


def _x_challenge(agent: "GraphAgent", base: PathOutput, stimulus: str, candidates: list[dict]) -> list[dict]:
    cand_block = "\n".join(f"[{i}] ({c.get('stance')}) {c.get('opinion')}" for i, c in enumerate(candidates))
    half = max(1, len(base.evidence_events or []) // 2)
    user = (
        _x_context(agent, stimulus)
        + f"\n挑战者只看到以下一半证据（信息不对称）:\n{_x_evidence_block(base, limit=half)}\n\n"
        + f"候选:\n{cand_block}\n\n"
        + '对每个候选给出最可能违背用户一贯立场/证据的漏洞与修法，输出 JSON: '
        + '{"critiques":[{"candidate":0,"flaw":"<=60字","fix":"<=40字"}]}'
    )
    raw = _x_chat(agent, user, temperature=0.3, max_tokens=600)
    obj = _parse_json(raw)
    return obj.get("critiques") if isinstance(obj, dict) and isinstance(obj.get("critiques"), list) else []


def _x_rubric(agent: "GraphAgent", base: PathOutput, stimulus: str, candidates: list[dict], critiques: list[dict]) -> list[dict]:
    cand_block = "\n".join(f"[{i}] ({c.get('stance')}) {c.get('opinion')}" for i, c in enumerate(candidates))
    crit_block = "\n".join(
        f"- 候选{c.get('candidate')}: {c.get('flaw')} → {c.get('fix')}" for c in critiques
    ) or "- （无挑战意见）"
    user = (
        _x_context(agent, stimulus)
        + f"\n完整证据:\n{_x_evidence_block(base)}\n\n"
        + f"候选:\n{cand_block}\n\n挑战意见:\n{crit_block}\n\n"
        + '给每个候选打 1-5 档并输出 [0,1] 分数（档1=0,2=0.25,3=0.5,4=0.75,5=1）与唯一相对排名，输出 JSON: '
        + '{"scores":[{"candidate":0,"mentally_consistent":0.75,"style_fidelity":0.5,"socially_appropriate":0.75,"relative_rank":1,"reason":"<=50字"}]}'
    )
    raw = _x_chat(agent, user, temperature=0.2, max_tokens=800)
    obj = _parse_json(raw)
    return obj.get("scores") if isinstance(obj, dict) and isinstance(obj.get("scores"), list) else []


def _x_graph_support(agent: "GraphAgent", base: PathOutput, candidate: dict) -> float:
    """DoG-style reliability support: overlap of candidate content with the
    user's retrieved evidence / learned causal-graph labels."""
    import re as _re

    def toks(s: str) -> set[str]:
        s = _re.sub(r"\s+", "", s or "")
        return {s[i : i + 2] for i in range(max(len(s) - 1, 0))}

    c = toks(str(candidate.get("opinion") or ""))
    pool: set[str] = set()
    for e in (base.evidence_events or [])[:6]:
        pool |= toks(str(e.get("title") or "") + str(e.get("opinion") or ""))
    for p in (base.paths or [])[:5]:
        pool |= toks(str(p.get("evidence") or p.get("evidence_title") or p.get("history_title") or ""))
    if not pool:
        return 0.5
    return round(len(c & pool) / max(len(c), 1), 4)


def improve_output(agent: "GraphAgent", base: PathOutput, stimulus: str) -> PathOutput:
    cfg = dict(getattr(agent, "agent_cfg", {}) or {}).get("agentx") or {}
    k = int(cfg.get("num_candidates", 3))
    skip_confident = float(cfg.get("skip_confident", 0.85))
    override_margin = float(cfg.get("override_margin", 0.15))
    if (base.c_trace or {}).get("mode") == "fast_path":
        return base
    if getattr(base, "confidence", 0.0) >= skip_confident and not getattr(base, "low_evidence", False):
        return base

    # Keep the base answer as candidate 0 so the selector can *retain* it; the
    # improvement layer must be able to choose "no change" (adaptive override),
    # otherwise wholesale replacement drifts away from the GT-specific wording.
    base_candidate = {
        "opinion": base.predicted_opinion,
        "stance": base.stance,
        "reason": (base.verbalization or "")[:60],
        "__base__": True,
    }
    candidates = [base_candidate] + _x_candidates(agent, base, stimulus, k)
    if len(candidates) < 2:
        return base
    critiques = (
        _x_challenge(agent, base, stimulus, candidates)
        if getattr(base, "confidence", 0.0) < float(cfg.get("debate_below_conf", 0.6))
        or getattr(base, "low_evidence", False)
        else []
    )
    scores = _x_rubric(agent, base, stimulus, candidates, critiques)
    by_idx = {int(s.get("candidate", -1)): s for s in scores if isinstance(s.get("candidate"), (int, float))}

    best: dict | None = None
    best_key: tuple[float, int, float] | None = None
    base_key: tuple[float, int, float] | None = None
    for i, cand in enumerate(candidates):
        s = by_idx.get(i, {})
        mental = min(1.0, max(0.0, float(s.get("mentally_consistent", 0.5))))
        style = min(1.0, max(0.0, float(s.get("style_fidelity", 0.5))))
        social = min(1.0, max(0.0, float(s.get("socially_appropriate", 0.5))))
        weighted = round(0.5 * mental + 0.3 * style + 0.2 * social, 4)
        rank = int(s.get("relative_rank", len(candidates) + 1))
        support = _x_graph_support(agent, base, cand)
        key = (weighted, -rank, support)
        if best_key is None or key > best_key:
            best_key, best = key, cand
        if i == 0:
            base_key = key
    if best is None:
        return base
    if best.get("__base__") or (
        base_key is not None and best_key is not None and best_key[0] - base_key[0] < override_margin
    ):
        # Conservative adaptive override (DoG decision-agent pattern): keep the
        # baseline unless a candidate wins by a clear margin.
        base.c_trace = {
            **(base.c_trace or {}),
            "agentx": {
                "decision": "keep_base",
                "num_candidates": len(candidates),
                "rubric": scores,
                "base_kept": True,
                "base_score": base_key[0] if base_key else None,
                "best_score": best_key[0] if best_key else None,
                "override_margin": override_margin,
            },
        }
        return base

    improved = replace(
        base,
        predicted_opinion=str(best.get("opinion") or base.predicted_opinion),
        stance=str(best.get("stance") or base.stance),
        verbalization=str(best.get("reason") or base.verbalization)[:200],
        confidence=max(getattr(base, "confidence", 0.0), best_key[0]),
        caveats=(base.caveats or []) + ["agentx:候选分支推演+多智能体辩论选择"],
    )
    improved.c_trace = {
        **(base.c_trace or {}),
        "mode": "agentx",
        "agentx": {
            "num_candidates": len(candidates),
            "candidates": candidates,
            "critiques": critiques,
            "rubric": scores,
            "winner_opinion": best.get("opinion"),
            "graph_support": _x_graph_support(agent, base, best),
        },
    }
    return improved


class GraphAgentX(GraphAgent):
    """改进版 Graph Agent：基线 Agent 之上叠加论文机制（分支推演 + MAD 辩论 + 可靠性评分）。"""

    def predict(
        self,
        stimulus: str,
        *,
        post_id: str = "",
        bid: str = "",
        date: str = "",
        topic: str = "",
    ) -> PathOutput:
        base = super().predict(stimulus, post_id=post_id, bid=bid, date=date, topic=topic)
        return improve_output(self, base, stimulus)
