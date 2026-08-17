"""Graph Agent 工具层：LLM 自主 Agentic RAG 的工具定义与执行。

彻底 Agent 化：原来 retrieve_compose 节点里硬编码的「路由理论 + 检索记忆 +
查先验路径 + 组合错题修复」流程，现在全部变成 LLM 自主决定的工具调用。
权重类错题修复仍在 resolve_context 里确定性组合（与基线一致），这里只暴露
可读的错题策略笔记。

双协议：
- native：OpenAI 兼容 tools/tool_calls（默认，端点支持时用）
- xml：prompt 内 <tool_call>{"name":...,"args":{...}}</tool_call> 兜底，
  端点不支持 tool_calls 时自动降级，执行器与状态完全共用。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .factors import EventFactor

# ---------------------------------------------------------------- specs

_STANCE_ENUM = ["support", "oppose", "mixed", "uncertain"]

_EMOTION_PROPS = {
    k: {"type": "number"}
    for k in ["anger", "joy", "sadness", "fear", "disgust", "surprise", "neutral"]
}


def _spec(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


TOOL_SPECS: list[dict[str, Any]] = [
    _spec(
        "decompose_event",
        "把事件分解为 2-4 个事件因素（policy/interest/moral/source/group_identity/emotion/other），"
        "作为后续检索与推理的锚点。每次预测只需调用一次，重复调用返回缓存结果。",
        {},
        [],
    ),
    _spec(
        "retrieve_theory",
        "按查询检索社会科学理论详卡（群体先验）。经情境路由与置信门控，"
        "只返回高置信卡；可指定因素锚定检索。",
        {
            "query": {"type": "string", "description": "检索查询（因素文本+话题+机制关键词）"},
            "factor_id": {"type": "string", "description": "锚定的事件因素 id（如 f1），可选"},
            "factor_type": {"type": "string", "description": "因素类型（未分解因素时用），可选"},
            "top_k": {"type": "integer", "description": "返回卡数，默认 2"},
        },
        ["query"],
    ),
    _spec(
        "retrieve_memory",
        "检索该用户的历史发帖记忆（个体证据）。个体证据优先级高于群体理论。",
        {
            "query": {"type": "string", "description": "检索查询（因素文本或事件关键词）"},
            "factor_id": {"type": "string", "description": "锚定的事件因素 id，可选"},
            "top_k": {"type": "integer", "description": "返回条数，默认 4"},
        },
        ["query"],
    ),
    _spec(
        "query_causal_graph",
        "查询该用户个体认知图中的先验路径（因素→坐标→立场），"
        "即过去预测中学到的可复用推理链。",
        {
            "factor_text": {"type": "string", "description": "因素或事件文本"},
            "factor_id": {"type": "string", "description": "锚定的事件因素 id，可选"},
            "top_k": {"type": "integer", "description": "返回路径数，默认 3"},
        },
        ["factor_text"],
    ),
    _spec(
        "read_failure_notes",
        "读错题本：该用户历史预测失败中学到的可迁移策略（带适用条件与失败次数）。"
        "遇到同类情境时优先遵守。",
        {
            "factor_kinds": {"type": "array", "items": {"type": "string"}, "description": "因素类型过滤，可选"},
            "coordinates": {"type": "array", "items": {"type": "string"}, "description": "理论坐标过滤，可选"},
            "top_k": {"type": "integer", "description": "返回条数，默认 3"},
        },
        [],
    ),
    _spec(
        "read_situational",
        "读取该帖发布时刻的情境三维环境（传播/心理/社会气候，来自智搜检索）。",
        {},
        [],
    ),
    _spec(
        "skeptic_check",
        "反证质疑：对当前立场与评论草稿找茬，可能给出修正。"
        "证据不充分或立场不确定时建议调用。",
        {
            "stance": {"type": "string", "enum": _STANCE_ENUM},
            "opinion": {"type": "string", "description": "评论草稿"},
        },
        ["stance", "opinion"],
    ),
    _spec(
        "finalize_prediction",
        "证据足够后调用，输出最终预测并结束推理。"
        "reason 用自然语言写出判断依据（先点个体证据再点理论坐标，没用到的不许提）；"
        "used 只列真正影响判断的材料。",
        {
            "stance": {"type": "string", "enum": _STANCE_ENUM},
            "emotion_probs": {"type": "object", "properties": _EMOTION_PROPS},
            "predicted_opinion": {"type": "string", "description": "以该用户本人声口写的短评，<=120字"},
            "reason": {"type": "string", "description": "判断依据说明，<=150字"},
            "used": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "factor_id": {"type": "string"},
                        "coordinate": {"type": "string"},
                        "evidence_idx": {"type": "array", "items": {"type": "integer"}},
                    },
                },
            },
            "low_evidence_factors": {"type": "array", "items": {"type": "string"}},
        },
        ["stance", "predicted_opinion", "reason", "used"],
    ),
]

FINALIZE_TOOL = "finalize_prediction"


def build_tool_specs(*, skeptic_tool: bool = False) -> list[dict[str, Any]]:
    if skeptic_tool:
        return TOOL_SPECS
    return [s for s in TOOL_SPECS if s["function"]["name"] != "skeptic_check"]


def finalize_tool_choice() -> dict[str, Any]:
    return {"type": "function", "function": {"name": FINALIZE_TOOL}}


# ---------------------------------------------------------------- parsing


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict[str, Any] = field(default_factory=dict)


def parse_native_tool_calls(message: dict[str, Any]) -> list[ToolCall]:
    out: list[ToolCall] = []
    for i, tc in enumerate(message.get("tool_calls") or []):
        fn = (tc or {}).get("function") or {}
        name = str(fn.get("name") or "")
        if not name:
            continue
        raw_args = fn.get("arguments") or "{}"
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            if not isinstance(args, dict):
                args = {}
        except (json.JSONDecodeError, TypeError):
            args = {}
        out.append(ToolCall(id=str(tc.get("id") or f"call_{i}"), name=name, args=args))
    return out


_XML_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_xml_tool_calls(content: str) -> list[ToolCall]:
    out: list[ToolCall] = []
    for i, m in enumerate(_XML_CALL_RE.finditer(content or "")):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        name = str(obj.get("name") or "")
        if not name:
            continue
        args = obj.get("args")
        out.append(ToolCall(id=f"xml_{i}", name=name, args=args if isinstance(args, dict) else {}))
    return out


XML_PROTOCOL_HINT = """【工具调用协议】你不具备原生函数调用能力。调用工具时，在回复中输出一个或多个：
<tool_call>{"name": "工具名", "args": {"参数": "值"}}</tool_call>
工具结果会以 <tool_result name="工具名">...</tool_result> 的形式在下一轮发给你。
除 <tool_call> 块外不要输出其他内容；决定收工时调用 finalize_prediction。"""


# ---------------------------------------------------------------- executor


def _ser_theory(m: Any) -> dict[str, Any]:
    return {
        "id": m.card.id,
        "name": m.card.name,
        "coordinate": m.card.coordinate,
        "score": round(float(m.score), 4),
        "why": str(m.why or "")[:200],
        "mechanism": (m.card.mechanism or "")[:240],
    }


def _ser_event(e: Any) -> dict[str, Any]:
    return {
        "map_id": e.map_id,
        "title": e.event_title,
        "score": round(float(getattr(e, "score", 0.0) or 0.0), 4),
        "opinion": ((e.user_opinion or getattr(e, "text", "") or "")[:200]),
    }


class ToolExecutor:
    """Execute tool calls against the GraphAgent's memory/theory/graph state.

    `state` is the LangGraph agent state dict; retrieval results accumulate
    into per_factor / all_matched / all_events so calibrate/absorb can reuse
    the exact shapes the legacy workflow produced.
    """

    def __init__(self, agent: Any, state: dict[str, Any]) -> None:
        self.agent = agent
        self.state = state

    # -- helpers ----------------------------------------------------------
    def _factors(self) -> list[EventFactor]:
        return list(self.state.get("factors") or [])

    def _ensure_factor_bucket(self, fid: str) -> dict[str, Any]:
        per_factor = self.state.setdefault("per_factor", {})
        if fid not in per_factor:
            f = next((x for x in self._factors() if x.id == fid), None)
            if f is None:
                f = EventFactor(id=fid or "f?", type="other", text="", salience=0.0)
            per_factor[fid] = {"factor": f, "matched": [], "events": [], "priors": []}
        return per_factor[fid]

    def _record(self, call: ToolCall, ok: bool, result: str) -> None:
        hist = self.state.setdefault("tool_history", [])
        hist.append(
            {
                "round": int(self.state.get("tool_rounds") or 0),
                "name": call.name,
                "args_digest": json.dumps(call.args, ensure_ascii=False)[:120],
                "ok": ok,
                "result_chars": len(result),
            }
        )

    def _truncate(self, payload: Any) -> str:
        text = json.dumps(payload, ensure_ascii=False)
        limit = int(self.agent.agent_cfg.get("tool_result_max_chars", 1200))
        if len(text) > limit:
            text = text[:limit] + '..."_truncated":true}'
        return text

    # -- entry ------------------------------------------------------------
    def execute(self, call: ToolCall) -> str:
        handler = getattr(self, f"_tool_{call.name}", None)
        if handler is None:
            result = json.dumps({"error": f"unknown_tool:{call.name}"}, ensure_ascii=False)
            self._record(call, False, result)
            return result
        try:
            payload = handler(call.args)
            result = self._truncate(payload)
            self._record(call, True, result)
            return result
        except Exception as e:  # noqa: BLE001 — 工具失败不终止循环
            result = json.dumps({"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False)[:400]
            self._record(call, False, result)
            return result

    # -- tools ------------------------------------------------------------
    def _tool_decompose_event(self, args: dict) -> Any:
        cached = self._factors()
        if cached:
            return {"factors": [f.to_dict() for f in cached], "cached": True}
        from .factors import decompose_factors
        from .path_workflow import apply_repairs_to_weights

        factors = decompose_factors(
            self.agent.llm,
            str(self.state.get("stimulus") or ""),
            extra_context=str(self.state.get("topic") or ""),
        )
        self.state["factors"] = factors

        # ── 错题本飞轮修复 ──
        # resolve_context 阶段还没有因素类型，repairs 检索全空。
        # 这里 decompose 后有了真实 factor kinds，重新检索并组合错题修复。
        factor_kinds = sorted({f.type for f in factors if f.type})
        coords: list[str] = []
        for f in factors:
            priors = self.agent.graph.paths_for_factor(f.text[:200], top_k=2)
            for p in priors:
                via = str(p.get("via") or "")
                if via.startswith("coordinate:"):
                    coords.append(via.split(":", 1)[1])
        hits = self.agent.failure_memory.retrieve_repairs(
            factor_kinds=factor_kinds, coordinates=sorted(set(coords)), top_k=3,
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
        current_weights = dict(self.state.get("_runtime_weights") or self.agent.weights)
        weights, applied = apply_repairs_to_weights(current_weights, active)

        # strategy notes（异构修复知识）
        new_notes: list[str] = list(self.state.get("strategy_notes") or [])
        seen_notes: set[str] = set(new_notes)
        for r, fs, ov in hits:
            if r.action != "strategy_note":
                continue
            note = str((r.payload or {}).get("note") or "").strip()
            if not note or note in seen_notes:
                continue
            seen_notes.add(note)
            guard = fs.when_to_apply or " / ".join(fs.factor_kinds[:3])
            new_notes.append(f"当{guard}：{note}（源自{fs.freq}次类似失败）")

        # recency boost from short_term_state repairs
        recency_boost = float(self.state.get("_recency_boost") or 0.0)
        for a in active:
            if a.get("cause") == "short_term_state" or a["action"] == "reset_short_term":
                recency_boost = max(
                    recency_boost,
                    float((a.get("payload") or {}).get("recency_boost") or 0.35),
                )

        # merge with provisional repairs from resolve_context
        existing_active: list[dict] = list(self.state.get("active_repairs") or [])
        existing_applied: list[str] = list(self.state.get("repairs_applied") or [])
        seen_ids = {a.get("id") for a in existing_active}
        for a in active:
            if a["id"] not in seen_ids:
                existing_active.append(a)
                seen_ids.add(a["id"])
        for aid in applied:
            if aid not in existing_applied:
                existing_applied.append(aid)

        self.state["active_repairs"] = existing_active
        self.state["repairs_applied"] = existing_applied
        self.state["_runtime_weights"] = weights
        self.state["_recency_boost"] = recency_boost
        self.state["strategy_notes"] = new_notes

        return {"factors": [f.to_dict() for f in factors], "repairs_after_decompose": len(existing_applied)}

    def _tool_retrieve_theory(self, args: dict) -> Any:
        from .theory_router import route_theories

        query = str(args.get("query") or "")[:300]
        fid = str(args.get("factor_id") or "")
        f = next((x for x in self._factors() if x.id == fid), None)
        ftype = str(args.get("factor_type") or (f.type if f else "") or "other")
        top_k = max(1, min(5, int(args.get("top_k") or 2)))
        tune = getattr(self.agent, "tuning", {}) or {}
        decision = route_theories(
            self.agent.theories,
            factor_id=fid or ftype,
            factor_type=ftype,
            query=query or (f.text if f else str(self.state.get("stimulus") or "")[:200]),
            stimulus=str(self.state.get("stimulus") or ""),
            topic=str(self.state.get("topic") or ""),
            top_k=top_k,
            user_weights=self.state.get("_runtime_weights") or self.agent.weights,
            env_weights=self.state.get("sit_weights") or {},
            min_confidence=float(tune.get("min_confidence", 0.35)),
            min_richness=float(tune.get("min_richness", 0.35)),
        )
        if fid:
            bucket = self._ensure_factor_bucket(fid)
            bucket["matched"].extend(decision.matched)
        all_matched = self.state.setdefault("all_matched", {})
        for m in decision.matched:
            all_matched[m.card.id] = m
        routes = self.state.setdefault("theory_routes", [])
        routes.append(
            {
                "factor_id": fid,
                "type": ftype,
                "prefs": decision.preferred_coords[:4],
                "why": decision.why,
                "rejected": decision.rejected_low_conf[:4],
                "via": "agent_tool",
            }
        )
        return {
            "matched": [_ser_theory(m) for m in decision.matched],
            "rejected_low_conf": decision.rejected_low_conf[:4],
            "route": decision.why,
        }

    def _tool_retrieve_memory(self, args: dict) -> Any:
        query = str(args.get("query") or "")[:300]
        fid = str(args.get("factor_id") or "")
        top_k = max(1, min(8, int(args.get("top_k") or 4)))
        events = self.agent.memory.retrieve(
            query or str(self.state.get("stimulus") or "")[:200],
            top_k=top_k,
            recency_boost=float(self.state.get("_recency_boost") or 0.0),
        )
        if fid:
            bucket = self._ensure_factor_bucket(fid)
            bucket["events"].extend(events)
        all_events = self.state.setdefault("all_events", {})
        for e in events:
            all_events[e.map_id] = e
        return {"events": [_ser_event(e) for e in events]}

    def _tool_query_causal_graph(self, args: dict) -> Any:
        text = str(args.get("factor_text") or "")[:300]
        fid = str(args.get("factor_id") or "")
        top_k = max(1, min(8, int(args.get("top_k") or 3)))
        priors = self.agent.graph.paths_for_factor(
            text or str(self.state.get("stimulus") or "")[:200], top_k=top_k
        )
        if fid:
            bucket = self._ensure_factor_bucket(fid)
            bucket["priors"].extend(priors)
        return {"prior_paths": priors}

    def _tool_read_failure_notes(self, args: dict) -> Any:
        kinds = [str(k) for k in (args.get("factor_kinds") or []) if str(k).strip()]
        coords = [str(c) for c in (args.get("coordinates") or []) if str(c).strip()]
        if not kinds:
            kinds = [f.type for f in self._factors() if f.type]
        if not coords:
            coords = [
                m.card.coordinate
                for m in (self.state.get("all_matched") or {}).values()
                if m.card.coordinate
            ][:6]
        top_k = max(1, min(6, int(args.get("top_k") or 3)))
        hits = self.agent.failure_memory.retrieve_repairs(
            factor_kinds=kinds, coordinates=coords, top_k=top_k
        )
        notes = []
        for r, fs, ov in hits:
            notes.append(
                {
                    "repair_id": r.id,
                    "action": r.action,
                    "when_to_apply": fs.when_to_apply,
                    "note": str((r.payload or {}).get("note") or "")[:200],
                    "cause": fs.primary_cause,
                    "fail_count": fs.freq,
                    "overlap": round(float(ov), 3),
                }
            )
        return {"failure_notes": notes, "admitted_only": True}

    def _tool_read_situational(self, args: dict) -> Any:
        from .situational_env import format_situational_block

        return {"situational_env": format_situational_block(self.state.get("sit"))}

    def _tool_skeptic_check(self, args: dict) -> Any:
        sk = self.agent._skeptic_check(
            str(self.state.get("stimulus") or ""),
            self._factors(),
            list(self.state.get("_draft_paths") or []),
            str(args.get("stance") or "uncertain"),
            str(args.get("opinion") or ""),
        )
        self.state["skeptic"] = sk
        return sk

    def _tool_finalize_prediction(self, args: dict) -> Any:
        factors = self._factors()
        if not factors:
            # LLM 跳过 decompose：无 LLM 单因素兜底，记 caveat
            factors = [EventFactor(id="f1", type="other",
                                   text=str(self.state.get("stimulus") or "")[:200],
                                   salience=1.0)]
            self.state["factors"] = factors
            caveats = self.state.setdefault("caveats", [])
            caveats.append("no_decompose: agent 未调用 decompose_event，按单因素兜底")
        per_factor_list = [
            self._ensure_factor_bucket(f.id) for f in factors
        ]
        used = self.agent._validate_used(args.get("used"), factors, per_factor_list)
        stance = str(args.get("stance") or "uncertain")
        if stance not in _STANCE_ENUM:
            stance = "uncertain"
        opinion = str(args.get("predicted_opinion") or "").strip()
        if not opinion:
            raise ValueError("finalize_prediction: predicted_opinion empty")
        self.state["final"] = {
            "stance": stance,
            "emotion_probs": self.agent._validate_emotions(args.get("emotion_probs")),
            "opinion": opinion,
            "reason": str(args.get("reason") or "")[:400],
            "paths": used,
            "low_evidence_factors": [str(x) for x in (args.get("low_evidence_factors") or [])],
        }
        return {"ok": True, "stance": stance}
