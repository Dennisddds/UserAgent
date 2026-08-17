"""PathAgent workflow pathway (LangGraph-style).

Recipe (paper analogues):
  - Agentic RAG: retrieve → evidence_grade → (repair/re-retrieve | synthesize)
  - SQL repair loop: weak grade → conditional repair node with retry budget
  - Audit: route_history on typed AgentState

Update policy lives in failure_memory: structure + conditional repair + compose.
"""

from __future__ import annotations

from typing import Any

from .agent_state import AgentState, new_agent_state
from .failure_memory import FailureMemory
from .graph_engine import GraphRunResult, StateGraph


def _tuning(agent: Any) -> dict[str, Any]:
    return getattr(agent, "tuning", None) or {}


_REFORM_SYSTEM = """你是检索查询改写器。对某微博事件的上一轮理论检索证据不足，需要改写查询重试。
给定事件与话题，抽出背后的社会心理机制/构念，产出 2-3 条替代检索查询（中英关键词混合，
如 "identity threat 身份认同 威胁"），用于匹配社会科学理论卡。不要重复原词堆砌。
只输出 JSON：{"queries":["...","..."]}"""


def _reformulate_queries(agent: Any, stimulus: str, topic: str) -> list[str]:
    """Agentic RAG repair: LLM 改写检索查询（按 stimulus 缓存，每次预测最多一次调用）。"""
    cache = getattr(agent, "_reform_cache", None)
    if cache is None:
        cache = agent._reform_cache = {}
    key = (stimulus[:120], topic or "")
    if key in cache:
        return cache[key]
    from .agent import _parse_json

    queries: list[str] = []
    try:
        raw = agent.llm.chat(
            [
                {"role": "system", "content": _REFORM_SYSTEM},
                {"role": "user", "content": f"【事件】{stimulus[:400]}\n【话题】{topic}\n请改写检索查询。"},
            ],
            temperature=0.2,
            max_tokens=300,
            disable_thinking=True,
        )
        obj = _parse_json(raw)
        queries = [str(q)[:120] for q in (obj.get("queries") or []) if str(q).strip()][:3]
    except Exception:  # noqa: BLE001 — 改写失败则退回原查询加宽
        queries = []
    cache[key] = queries
    return queries


def apply_repairs_to_weights(
    weights: dict[str, float],
    repairs: list[dict[str, Any]],
) -> tuple[dict[str, float], list[str]]:
    """Compose conditional repair payloads onto theory/coord weights (no task replay)."""
    w = dict(weights)
    applied: list[str] = []
    for item in repairs:
        r = item.get("repair") or item
        action = r.get("action")
        payload = r.get("payload") or {}
        rid = r.get("id") or ""
        if action == "demote_theory_coords":
            delta = float(payload.get("delta", -0.15))
            for c in payload.get("coordinates") or []:
                w[c] = max(0.2, min(2.5, w.get(c, 1.0) + delta))
            applied.append(rid)
        elif action == "prefer_graph_priors":
            # soft marker only — retrieval node reads repairs_applied
            applied.append(rid)
        elif action in {"boost_retrieval_kinds", "demote_factor_kinds", "reset_short_term", "flag_profile_attr", "strategy_note"}:
            applied.append(rid)
    return w, [x for x in applied if x]


def build_path_workflow(agent: Any) -> StateGraph:
    """Bind PathAgent instance methods into an inspectable state graph."""

    g = StateGraph(name="path_agent")

    def resolve_context(state: dict[str, Any]) -> dict[str, Any]:
        from .situational_env import resolve_situational, situational_env_weights
        from .user_actions import format_source_block

        u = agent.memory.u_snapshot(max_motifs=agent.max_motifs)
        v = agent.memory.v_snapshot()
        sit = resolve_situational(
            agent.situational_store,
            post_id=state.get("post_id") or None,
            bid=state.get("bid") or None,
            date=state.get("date") or None,
            topic=state.get("topic") or None,
            text=state.get("stimulus"),
        )
        sit_weights = situational_env_weights(sit, boost=1.85)
        situational_missing = bool(agent.use_situational and not sit_weights)
        return {
            "u_snapshot": u,
            "v_snapshot": v,
            "sit": sit,
            "sit_weights": sit_weights,
            "situational_missing": situational_missing,
            "source_block": format_source_block(getattr(agent, "source_profile", None) or {}),
        }

    def decompose(state: dict[str, Any]) -> dict[str, Any]:
        from .factors import decompose_factors

        factors = decompose_factors(
            agent.llm, state["stimulus"], extra_context=state.get("topic") or ""
        )
        return {"factors": factors}

    def retrieve_compose(state: dict[str, Any]) -> dict[str, Any]:
        """Retrieve theories/evidence + compose matching conditional repairs."""
        factors = state.get("factors") or []
        topic = state.get("topic") or ""
        kinds = [
            getattr(f, "type", "") or (f.get("type") if isinstance(f, dict) else "")
            for f in factors
        ]
        # provisional coords from graph priors for repair retrieval
        provisional_coords: list[str] = []
        for f in factors:
            text = getattr(f, "text", None) or (f.get("text") if isinstance(f, dict) else "")
            for p in agent.graph.paths_for_factor(text or "", top_k=2):
                via = str(p.get("via") or "")
                if via.startswith("coordinate:"):
                    provisional_coords.append(via.split(":", 1)[1])

        fm: FailureMemory = agent.failure_memory
        hits = fm.retrieve_repairs(
            factor_kinds=[k for k in kinds if k],
            coordinates=provisional_coords,
            top_k=3,
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

        # ASPIRE strategy_note：准入的可迁移策略注入合成 prompt（异构修复知识）
        strategy_notes: list[str] = []
        for r, fs, ov in hits:
            if r.action != "strategy_note":
                continue
            note = str((r.payload or {}).get("note") or "").strip()
            if not note:
                continue
            guard = fs.when_to_apply or " / ".join(fs.factor_kinds[:3])
            strategy_notes.append(f"当{guard}：{note}（源自{fs.freq}次类似失败）")

        # retrieval boost from repairs + recency when opinion-drift repairs exist
        top_k_ev = 3
        recency_boost = 0.0
        for a in active:
            if a["action"] == "boost_retrieval_kinds":
                top_k_ev = max(top_k_ev, 3 + int((a.get("payload") or {}).get("top_k_boost", 2)))
            if a.get("cause") == "short_term_state" or a["action"] == "reset_short_term":
                rb = float((a.get("payload") or {}).get("recency_boost") or 0.35)
                recency_boost = max(recency_boost, rb)

        demote_kinds = set()
        for a in active:
            if a["action"] == "demote_factor_kinds":
                demote_kinds.update((a.get("payload") or {}).get("factor_kinds") or [])

        from .theory_router import route_theories

        tune = _tuning(agent)
        per_factor: list[dict[str, Any]] = []
        all_matched: dict[str, Any] = {}
        all_events: dict[str, Any] = {}
        route_log: list[dict[str, Any]] = []
        for f in factors:
            kind = getattr(f, "type", "")
            text = getattr(f, "text", "")
            fid = getattr(f, "id", "")
            if kind in demote_kinds:
                q = text
            else:
                q = f"{text} {topic}"
            decision = route_theories(
                agent.theories,
                factor_id=fid,
                factor_type=kind or "other",
                query=q,
                stimulus=state.get("stimulus") or "",
                topic=topic,
                top_k=2,
                user_weights=weights,
                env_weights=state.get("sit_weights") or {},
                min_confidence=float(tune.get("min_confidence", 0.35)),
                min_richness=float(tune.get("min_richness", 0.35)),
            )
            matched = decision.matched
            events = agent.memory.retrieve(text, top_k=top_k_ev, recency_boost=recency_boost)
            priors = agent.graph.paths_for_factor(text, top_k=3)
            per_factor.append({"factor": f, "matched": matched, "events": events, "priors": priors})
            route_log.append(
                {
                    "factor_id": fid,
                    "type": kind,
                    "prefs": decision.preferred_coords[:4],
                    "why": decision.why,
                    "rejected": decision.rejected_low_conf[:4],
                }
            )
            for m in matched:
                all_matched[m.card.id] = m
            for e in events:
                all_events[e.map_id] = e

        return {
            "per_factor": per_factor,
            "all_matched": all_matched,
            "all_events": all_events,
            "active_repairs": active,
            "repairs_applied": applied,
            "strategy_notes": strategy_notes,
            "theory_routes": route_log,
            "_runtime_weights": weights,
        }

    def grade_evidence(state: dict[str, Any]) -> dict[str, Any]:
        per_factor = state.get("per_factor") or []
        if not per_factor:
            return {"evidence_grade": "fail"}
        covered = 0
        for pf in per_factor:
            if (pf.get("events") or pf.get("matched") or pf.get("priors")):
                covered += 1
        ratio = covered / max(1, len(per_factor))
        avg_ev = 0.0
        n = 0
        for pf in per_factor:
            for e in pf.get("events") or []:
                avg_ev += float(getattr(e, "score", 0.0) or 0.0)
                n += 1
        avg_ev = avg_ev / max(1, n)
        if ratio >= 0.67 and avg_ev >= 0.05:
            grade = "strong"
        elif ratio >= 0.34:
            grade = "weak"
        else:
            grade = "fail"
        return {"evidence_grade": grade}

    def repair_retrieve(state: dict[str, Any]) -> dict[str, Any]:
        """Conditional repair path: widen retrieval + LLM 改写查询，never replay a past task."""
        attempts = int(state.get("repair_attempts") or 0) + 1
        factors = state.get("factors") or []
        topic = state.get("topic") or ""
        weights = state.get("_runtime_weights") or agent.weights
        tune = _tuning(agent)
        # Agentic RAG: 证据不足时改写检索查询（机制/构念层面的替代表述）
        reformulated: list[str] = []
        if tune.get("query_reformulation", True):
            reformulated = _reformulate_queries(agent, state.get("stimulus") or "", topic)
        extra_q = " ".join(reformulated)
        per_factor: list[dict[str, Any]] = []
        all_matched: dict[str, Any] = dict(state.get("all_matched") or {})
        all_events: dict[str, Any] = dict(state.get("all_events") or {})
        for f in factors:
            text = getattr(f, "text", "")
            matched = agent.theories.match(
                f"{text} {topic} {extra_q}",
                top_k=3,
                user_weights=weights,
                env_weights=state.get("sit_weights") or {},
            )
            events = agent.memory.retrieve(text, top_k=5)
            priors = agent.graph.paths_for_factor(text, top_k=5)
            per_factor.append({"factor": f, "matched": matched, "events": events, "priors": priors})
            for m in matched:
                all_matched[m.card.id] = m
            for e in events:
                all_events[e.map_id] = e
        caveats = list(state.get("caveats") or [])
        caveats.append("repair_retrieve: widened retrieval after weak evidence grade")
        if reformulated:
            caveats.append("query_reformulated: " + " | ".join(reformulated[:2]))
        return {
            "repair_attempts": attempts,
            "per_factor": per_factor,
            "all_matched": all_matched,
            "all_events": all_events,
            "reformulated_queries": reformulated,
            "caveats": caveats[:5],
        }

    def synthesize(state: dict[str, Any]) -> dict[str, Any]:
        factors = state["factors"]
        per_factor = state["per_factor"]
        fusion = getattr(agent, "path_mode", "strict") == "fusion"
        syn = agent._synthesize(
            state["stimulus"],
            state["u_snapshot"],
            state["v_snapshot"],
            state.get("sit"),
            factors,
            per_factor,
            fusion=fusion,
            strategy_notes=list(state.get("strategy_notes") or []),
            weak_evidence=(state.get("evidence_grade") == "fail"),
        )
        if fusion:
            # v1+v2 融合：不强制结构化因果链；used 引用宽松转图边，reason 即理由
            paths = agent._validate_used(syn.get("used"), factors, per_factor)
        else:
            paths = agent._validate_paths(syn.get("paths"), factors, per_factor)
        stance = str(syn.get("stance") or "uncertain")
        if stance not in {"support", "oppose", "mixed", "uncertain"}:
            stance = "uncertain"
        return {
            "paths": paths,
            "stance": stance,
            "emotion_probs": agent._validate_emotions(syn.get("emotion_probs")),
            "opinion": str(syn.get("predicted_opinion") or "").strip(),
            "reason": str(syn.get("reason") or "")[:400],
            "low_evidence_factors": list(syn.get("low_evidence_factors") or []),
        }

    def skeptic(state: dict[str, Any]) -> dict[str, Any]:
        sk = {"challenge": "", "overturn": False}
        if not (agent.use_skeptic and state.get("opinion")):
            return {"skeptic": sk}
        # instance-adaptive scaling：证据强 + 立场明确 + 无低证据因素 → 跳过质疑调用
        if _tuning(agent).get("skeptic_adaptive", True):
            if (
                state.get("evidence_grade") == "strong"
                and state.get("stance") in {"support", "oppose"}
                and not (state.get("low_evidence_factors") or [])
            ):
                sk["skipped"] = "adaptive:strong_decisive"
                return {"skeptic": sk}
        sk = agent._skeptic_check(
            state["stimulus"],
            state["factors"],
            state["paths"],
            state["stance"],
            state["opinion"],
        )
        return {"skeptic": sk}

    def calibrate(state: dict[str, Any]) -> dict[str, Any]:
        confidence, low_ev = agent._calibrate(
            state["per_factor"],
            state["paths"],
            state.get("low_evidence_factors") or [],
            state.get("skeptic") or {},
        )
        sk = dict(state.get("skeptic") or {})
        stance = state["stance"]
        opinion = state["opinion"]
        overturn_min_conf = float(_tuning(agent).get("skeptic_overturn_min_conf", 0.75))
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
        return {
            "confidence": confidence,
            "low_evidence": low_ev,
            "skeptic": sk,
            "stance": stance,
            "opinion": opinion,
        }

    def absorb_finalize(state: dict[str, Any]) -> dict[str, Any]:
        from .models import AgentOutput  # noqa: F401
        from .path_agent import PathOutput

        factors = state["factors"]
        paths = state["paths"]
        stance = state["stance"]
        reason = str(state.get("reason") or "")
        rendered = agent._render_paths(factors, paths, stance)
        # fusion：理由 = 图推理的自然语言说明 + 实际引用渲染；strict：路径渲染本身
        verbalization = f"{reason}\n【图推理引用】\n{rendered}" if reason else rendered
        agent._graph_absorb(factors, state["per_factor"], paths, stance)

        # link applied repairs into causal graph as derived_from edges (structure ids only)
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
        situational_missing = bool(state.get("situational_missing"))
        sk = state.get("skeptic") or {}
        low_ev = bool(state.get("low_evidence"))
        caveats = list(state.get("caveats") or [])
        if situational_missing:
            caveats.append("situational_missing: 无个人接收路径记录，仅理论先验+个体记忆参与")
        if sk.get("challenge"):
            caveats.append("skeptic: " + str(sk["challenge"])[:120])
        if low_ev:
            caveats.append("low_evidence: 证据覆盖不足，预测置信度低")
        if state.get("repairs_applied"):
            caveats.append(
                "repairs_composed: " + ",".join(state["repairs_applied"][:3])
            )

        out = PathOutput(
            user_id=agent.user_id,
            stimulus=state["stimulus"],
            predicted_opinion=state.get("opinion") or "",
            stance=stance,
            activated_coordinates=sorted(
                {p.get("coordinate") for p in paths if p.get("coordinate")}
            ),
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
                "steps": list(state.get("route_history") or []),
                "mode": (
                    "fusion_graph_workflow"
                    if getattr(agent, "path_mode", "strict") == "fusion"
                    else "path_graph_workflow"
                ),
                "reason": str(state.get("reason") or ""),
                "evidence_grade": state.get("evidence_grade"),
                "repair_attempts": state.get("repair_attempts"),
                "reformulated_queries": state.get("reformulated_queries") or [],
                "repairs_applied": state.get("repairs_applied") or [],
                "strategy_notes": state.get("strategy_notes") or [],
                "active_repairs": [
                    {"id": a.get("id"), "action": a.get("action"), "overlap": a.get("overlap")}
                    for a in (state.get("active_repairs") or [])
                ],
                "failure_memory": agent.failure_memory.stats(),
                "theory_routes": state.get("theory_routes") or [],
                "num_factors": len(factors),
                "num_paths": len(paths),
                "num_theories": len(all_matched),
                "num_events": len(all_events),
                "post_id": state.get("post_id"),
                "situational_missing": situational_missing,
                "graph": agent.graph.stats(),
            },
            u_snapshot=state.get("u_snapshot") or {},
            v_snapshot=state.get("v_snapshot") or {},
            caveats=caveats[:4],
            factors=[f.to_dict() for f in factors],
            paths=paths,
            emotion_probs=state.get("emotion_probs") or {},
            confidence=float(state.get("confidence") or 0.0),
            low_evidence=low_ev,
            skeptic=sk,
        )
        return {"verbalization": verbalization, "output": out, "caveats": caveats[:4], "status": "ok"}

    def route_after_grade(state: dict[str, Any]) -> str:
        grade = state.get("evidence_grade") or "weak"
        attempts = int(state.get("repair_attempts") or 0)
        budget = int(state.get("repair_budget") or 1)
        if grade in {"fail", "weak"} and attempts < budget:
            return "repair"
        return "synthesize"

    g.add_node("resolve_context", resolve_context)
    g.add_node("decompose", decompose)
    g.add_node("retrieve_compose", retrieve_compose)
    g.add_node("grade_evidence", grade_evidence)
    g.add_node("repair_retrieve", repair_retrieve)
    g.add_node("synthesize", synthesize)
    g.add_node("skeptic", skeptic)
    g.add_node("calibrate", calibrate)
    g.add_node("absorb_finalize", absorb_finalize)

    g.set_entry("resolve_context")
    g.add_edge("resolve_context", "decompose")
    g.add_edge("decompose", "retrieve_compose")
    g.add_edge("retrieve_compose", "grade_evidence")
    g.add_conditional_edges(
        "grade_evidence",
        route_after_grade,
        {"repair": "repair_retrieve", "synthesize": "synthesize"},
    )
    g.add_edge("repair_retrieve", "grade_evidence")
    g.add_edge("synthesize", "skeptic")
    g.add_edge("skeptic", "calibrate")
    g.add_edge("calibrate", "absorb_finalize")
    g.set_finish("absorb_finalize")
    return g


def run_path_predict(agent: Any, **kwargs: Any) -> Any:
    state = new_agent_state(**kwargs)
    graph = build_path_workflow(agent)
    result: GraphRunResult = graph.invoke(dict(state))
    out = result.state.get("output")
    if out is None:
        raise RuntimeError("workflow finished without output")
    # ensure route history landed in c_trace
    if hasattr(out, "c_trace") and isinstance(out.c_trace, dict):
        out.c_trace["route_history"] = result.route_history
    return out
