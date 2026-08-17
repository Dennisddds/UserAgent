from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .failure_memory import CAUSES, CAUSE_TO_STAGE, FailureMemory
from .genminds import GenMindsMemory
from .llm import DeepSeekClient
from .models import AgentOutput
from .situational_env import (
    format_situational_block,
    load_situational_store,
    resolve_situational,
    situational_env_weights,
)
from .theory_lib import TheoryLibrary
from .thinking_analyzer import (
    ThinkingError,
    attribute_thinking_error,
    format_thinking_error_for_memory,
    generate_targeted_repair,
    parse_thinking_trace,
)
from .user_env import load_env


class CUVAgent:
    """
    C: situational-env sparse theory match + GenMinds retrieve + verbalize loop
    U: GenMinds skills/memory
    V: persona values + situational coordinates (fallback: trait env)

    Weak theory match / missing situational env → GenMinds-style prediction (no theory dump).
    """

    def __init__(
        self,
        user_id: str,
        memory: GenMindsMemory,
        theories: TheoryLibrary,
        llm: DeepSeekClient,
        *,
        state_dir: str | Path,
        top_k_events: int = 6,
        top_k_theories: int = 3,
        max_motifs: int = 8,
        evolve_lr: float = 0.15,
        env_profile: dict[str, Any] | None = None,
        situational_store: dict[str, Any] | None = None,
        use_situational: bool = True,
        weak_match_threshold: float = 0.40,
    ) -> None:
        self.user_id = user_id
        self.memory = memory
        self.theories = theories
        self.llm = llm
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.top_k_events = top_k_events
        self.top_k_theories = top_k_theories
        self.max_motifs = max_motifs
        self.evolve_lr = evolve_lr
        self.weights = self._load_weights()
        self.env_profile = env_profile if env_profile is not None else load_env(self.state_dir, user_id)
        self.env_weights = dict((self.env_profile or {}).get("coordinate_weights") or {})
        self.use_situational = use_situational
        self.weak_match_threshold = float(weak_match_threshold)
        if situational_store is not None:
            self.situational_store = situational_store
        elif use_situational:
            sit_path = self.state_dir / f"{user_id}_situational_env.json"
            self.situational_store = load_situational_store(sit_path) if sit_path.exists() else None
        else:
            self.situational_store = None

    def _weights_path(self) -> Path:
        return self.state_dir / f"{self.user_id}_weights.json"

    def _load_weights(self) -> dict[str, float]:
        p = self._weights_path()
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        return {}

    def save_weights(self) -> None:
        self._weights_path().write_text(
            json.dumps(self.weights, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def predict(
        self,
        stimulus: str,
        *,
        post_id: str | None = None,
        bid: str | None = None,
        date: str | None = None,
        topic: str | None = None,
    ) -> AgentOutput:
        sit = None
        if self.use_situational and self.situational_store:
            sit = resolve_situational(
                self.situational_store,
                post_id=post_id,
                bid=bid,
                date=date,
                topic=topic,
                text=stimulus,
            )

        sit_weights = situational_env_weights(sit) if sit else {}
        # Prefer situational priors; only fall back to trait env when situational missing.
        active_env_weights = sit_weights if sit_weights else (self.env_weights or None)

        events = self.memory.retrieve(stimulus, top_k=self.top_k_events)
        u = self.memory.u_snapshot(max_motifs=self.max_motifs)
        v = self.memory.v_snapshot()

        # Gate: no situational coords → GenMinds (avoid trait-env theory dump)
        if self.use_situational and not sit_weights:
            return self._genminds_fallback(
                stimulus,
                events=events,
                u=u,
                v=v,
                reason="no_situational_env",
            )

        matched = self.theories.match(
            stimulus,
            top_k=self.top_k_theories,
            user_weights=self.weights,
            env_weights=active_env_weights,
            prefer_grounded=True,
        )
        top_score = float(matched[0].score) if matched else 0.0
        if not matched or top_score < self.weak_match_threshold:
            return self._genminds_fallback(
                stimulus,
                events=events,
                u=u,
                v=v,
                reason=f"weak_theory_match:{top_score:.3f}",
                matched=matched,
            )

        v["theory_coordinates"] = [m.card.coordinate for m in matched]
        v["personal_weights"] = {k: self.weights[k] for k in list(self.weights)[:12]}
        v["env_coordinates"] = list(sit_weights.keys())[:12]
        v["situational_summary"] = (sit or {}).get("summary") or ""

        theory_block = "\n".join(
            [
                (
                    f"- [{m.card.coordinate}|{m.card.source}|rich={m.card.richness:.2f}] "
                    f"{m.card.name} ({m.card.authors}, {m.card.year})\n"
                    f"  summary: {(m.card.summary or m.card.mechanism)[:280]}\n"
                    f"  mechanism: {m.card.mechanism[:180]}\n"
                    f"  prediction: {m.card.prediction[:160]}\n"
                    f"  boundary: {m.card.boundary[:120]}\n"
                    f"  constructs: {', '.join((m.card.constructs or [])[:6])}\n"
                    f"  (score={m.score:.3f}; {m.why})"
                )
                for m in matched
            ]
        ) or "- (no theory matched; rely on individual memory only)"

        evidence_block = "\n".join(
            [
                f"- ({e.score:.3f}) {e.event_title}: {e.user_opinion or e.text[:180]}"
                for e in events
            ]
        ) or "- (no similar past events)"

        sit_block = format_situational_block(sit)

        system = (
            "You are a Theory-Guided C–U–V cognitive agent.\n"
            "C=architecture: match sparse theories then ground in individual GenMinds memory.\n"
            "U=skills/memory from this user's history. V=values/identity preferences.\n"
            "Theories are calibrated by the EXTERNAL situational 3D environment at posting time "
            "(communication / psychological / social climate) — not by dumping trait labels.\n"
            "CRITICAL: Individual history dominates. Theories are sparse mechanism priors — "
            "do NOT override clear personal patterns.\n"
            "CRITICAL PERSONA: Obey the Identity block and persona voice hints for THIS account "
            "(any 大V / KOL). Write as the account holder, not as a bystander quoting them. "
            "If their persona uses third-person self-reference, treat it as first-person voice.\n"
            "Verbalization must cite theory coordinates + personal evidence. "
            "Do NOT claim activated words/CoT are direct proof of 'thinking'."
        )
        user_msg = f"""## Identity
{self.memory.identity_block()}

## Stimulus
{stimulus}

## V — values / interests (sample)
values: {u.get('beliefs', v.get('persona_values', []))[:8]}
interests: {u.get('interests', [])[:6]}
communication: {u.get('communication', [])[:4]}

## Situational 3D environment (at posting time)
{sit_block}

## U — GenMinds causal motifs (sample)
{u.get('motifs', [])[:8]}

## Matched theories (sparse priors from situational coords)
{theory_block}

## Retrieved personal events
{evidence_block}

Return STRICT JSON with keys:
predicted_opinion (string, user's likely voice, <=120 Chinese chars),
stance (support|oppose|mixed|uncertain),
activated_coordinates (array of short coordinate ids only),
verbalization (string, <=200 Chinese chars: theory coord + personal evidence),
caveats (array of short strings, <=3).
No markdown. No extra keys. Keep JSON compact.
"""
        # White-box CoT: keep model thinking when client.enable_thinking is on (RTWI audit).
        want_thinking = bool(getattr(self.llm, "enable_thinking", False))
        # Local Flash (max_model_len≈4096): budget for CoT + compact JSON answer.
        gen_max_tokens = 1200 if want_thinking else 800
        trace = self.llm.chat_with_trace(
            [{"role": "system", "content": system}, {"role": "user", "content": user_msg}],
            temperature=0.35,
            max_tokens=gen_max_tokens,
            disable_thinking=not want_thinking,
        )
        raw = trace.get("content") or ""
        parsed = _parse_json(raw)
        if not parsed.get("predicted_opinion"):
            # Retry compact JSON; disable thinking so budget goes to the answer.
            trace = self.llm.chat_with_trace(
                [
                    {
                        "role": "system",
                        "content": "Output one compact JSON object only. No thinking.",
                    },
                    {
                        "role": "user",
                        "content": user_msg
                        + "\n\nPrevious output was empty/invalid. Reply again with valid compact JSON.",
                    },
                ],
                temperature=0.2,
                max_tokens=800,
                disable_thinking=True,
            )
            raw = trace.get("content") or ""
            parsed = _parse_json(raw)
        opinion = str(
            parsed.get("predicted_opinion")
            or _regex_field(raw, "predicted_opinion")
            or raw.strip()[:400]
        )
        if opinion.strip().startswith("{") and "predicted_opinion" in opinion:
            inner = _parse_json(opinion)
            if inner.get("predicted_opinion"):
                parsed = {**parsed, **inner}
                opinion = str(inner["predicted_opinion"])
        out = AgentOutput(
            user_id=self.user_id,
            stimulus=stimulus,
            predicted_opinion=opinion,
            stance=str(parsed.get("stance") or _regex_field(raw, "stance") or "uncertain"),
            activated_coordinates=_as_str_list(
                parsed.get("activated_coordinates")
                or [m.card.coordinate for m in matched]
            ),
            matched_theories=[
                {
                    "id": m.card.id,
                    "name": m.card.name,
                    "coordinate": m.card.coordinate,
                    "score": m.score,
                    "why": m.why,
                    "mechanism": m.card.mechanism[:240],
                }
                for m in matched
            ],
            evidence_events=[
                {
                    "map_id": e.map_id,
                    "title": e.event_title,
                    "score": e.score,
                    "opinion": e.user_opinion[:200],
                }
                for e in events
            ],
            verbalization=str(
                parsed.get("verbalization") or _regex_field(raw, "verbalization") or ""
            ),
            c_trace={
                "steps": [
                    "resolve_situational_env",
                    "match_theories",
                    "retrieve_genminds",
                    "generate",
                    "verbalize",
                ],
                "num_theories": len(matched),
                "num_events": len(events),
                "top_theory_score": top_score,
                "post_id": post_id,
                "mode": "theory_guided",
                # RTWI white-box: raw model CoT + mining/reasoning excerpts
                "model_reasoning": getattr(self.llm, "last_reasoning", "") or "",
                "white_box": (trace.get("white_box") if isinstance(trace, dict) else None)
                or {
                    "has_reasoning": bool(getattr(self.llm, "last_reasoning", "")),
                    "reasoning_chars": len(getattr(self.llm, "last_reasoning", "") or ""),
                },
            },
            u_snapshot=u,
            v_snapshot=v,
            caveats=_as_str_list(
                parsed.get("caveats")
                or [
                    "Theory cards are group-level priors.",
                    "GenMinds history is the primary individual signal.",
                    "Situational env calibrates theory retrieval.",
                ]
            ),
        )
        return out

    def _genminds_fallback(
        self,
        stimulus: str,
        *,
        events: list[Any],
        u: dict[str, Any],
        v: dict[str, Any],
        reason: str,
        matched: list[Any] | None = None,
    ) -> AgentOutput:
        evidence = "\n".join(
            f"- {e.event_title}: {e.user_opinion or e.text[:160]}" for e in events
        ) or "- (no similar past events)"
        system = (
            "你正在扮演下方【身份/人设】所指定的微博账号本人发短评。"
            "严格贴合其人设表达特征、历史立场、信念与风格。"
            "你就是该账号本人，不是路人转述该大V；若人设含第三人称自称，视为本人惯用说法。"
            "只输出一条简短原创微博评论正文，不要解释，不要前缀标签。"
        )
        user = f"""【身份/人设】
{self.memory.identity_block()}

用户历史信念样例：{u.get('beliefs', [])[:6]}
表达风格：{u.get('communication', [])[:6]}
相关历史：
{evidence}

事件：
{stimulus}

请以该账号本人声口发表一条简短原创微博评论："""
        want_thinking = bool(getattr(self.llm, "enable_thinking", False))
        trace = self.llm.chat_with_trace(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.4,
            # Thinking models spend most budget on CoT; local Flash ctx is tight.
            max_tokens=1200 if want_thinking else 400,
            disable_thinking=not want_thinking,
        )
        pred = (trace.get("content") or "").strip()
        if pred.startswith("WEIBO:"):
            pred = pred[6:].strip()
        # Recover final comment from CoT when the answer channel is empty/truncated.
        if not pred or len(pred) < 4:
            import re as _re
            reasoning = (trace.get("reasoning_content") or getattr(self.llm, "last_reasoning", "") or "")
            quoted = _re.findall(r"[\"“]([^\"”]{8,120})[\"”]", reasoning)
            if quoted:
                pred = quoted[-1].strip()
            else:
                # last non-empty short line that looks like a weibo comment
                for line in reversed([ln.strip() for ln in reasoning.splitlines() if ln.strip()]):
                    if 8 <= len(line) <= 120 and not line.startswith(("-", "*", "#", "所以", "例如", "结构", "参考")):
                        pred = line
                        break
        # If the model leaked CoT into content, keep only the last short paragraph.
        if len(pred) > 180 and ("\n" in pred or "首先" in pred or "我需要" in pred):
            parts = [p.strip() for p in pred.split("\n") if p.strip()]
            cand = parts[-1] if parts else pred
            if len(cand) <= 160:
                pred = cand
            else:
                pred = pred[:160]
        matched = matched or []
        return AgentOutput(
            user_id=self.user_id,
            stimulus=stimulus,
            predicted_opinion=pred,
            stance="uncertain",
            activated_coordinates=[],
            matched_theories=[
                {
                    "id": m.card.id,
                    "name": m.card.name,
                    "coordinate": m.card.coordinate,
                    "score": m.score,
                    "why": m.why,
                    "mechanism": m.card.mechanism[:240],
                }
                for m in matched
            ],
            evidence_events=[
                {
                    "map_id": e.map_id,
                    "title": e.event_title,
                    "score": e.score,
                    "opinion": e.user_opinion[:200],
                }
                for e in events
            ],
            verbalization=f"fallback_genminds:{reason}",
            c_trace={
                "steps": ["resolve_situational_env", "weak_or_missing_gate", "genminds_fallback"],
                "num_theories": len(matched),
                "num_events": len(events),
                "mode": "genminds_fallback",
                "reason": reason,
                "model_reasoning": getattr(self.llm, "last_reasoning", "") or "",
                "white_box": {
                    **(trace.get("white_box") or {}),
                    "has_reasoning": bool(getattr(self.llm, "last_reasoning", "")),
                    "reasoning_chars": len(getattr(self.llm, "last_reasoning", "") or ""),
                    "answer_chars": len(pred or ""),
                    "answer_recovered_from_cot": not bool((trace.get("content") or "").strip()),
                },
            },
            u_snapshot=u,
            v_snapshot=v,
            caveats=[
                "Theory match gated; used GenMinds-only prediction.",
                reason,
            ],
        )

    def evolve(
        self,
        output: AgentOutput,
        *,
        feedback: str = "",
        helpful: bool | None = None,
        oa: float = 0.0,
        judge_scores: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Evolve from feedback — thinking-aware targeted repair.

        When model reasoning (CoT) is available and prediction failed:
          1. Parse the thinking trace to find the specific error
          2. Record thinking-level failure pattern in 错题本
          3. Apply targeted repair (not generic weight delta)
        Otherwise: fall back to simple weight adjustment.
        """
        if helpful is None:
            helpful = oa >= 0.5 if oa > 0 else any(
                k in feedback.lower()
                for k in ["对", "准", "正确", "好", "yes", "correct", "right", "accurate"]
            ) and not any(
                k in feedback.lower() for k in ["不对", "错", "不准", "wrong", "incorrect", "bad"]
            )

        # ── Extract thinking trace from last LLM call ──
        c_trace = getattr(output, "c_trace", None) or {}
        model_reasoning = c_trace.get("model_reasoning", "") or ""
        stage_reliability = c_trace.get("stage_reliability", {}) or {}

        thinking_trace = None
        thinking_error = None
        if model_reasoning:
            thinking_trace = parse_thinking_trace(model_reasoning)

        # ── If prediction failed and we have CoT: targeted repair ──
        if not helpful and thinking_trace and thinking_trace.has_reasoning:
            thinking_error = attribute_thinking_error(
                thinking_trace,
                oa=oa,
                judge_scores=judge_scores,
                stage_reliability=stage_reliability,
            )
        elif not helpful and not thinking_trace:
            # No CoT — use structural fallback
            thinking_error = attribute_thinking_error(
                parse_thinking_trace(""),  # empty trace triggers structural fallback
                oa=oa,
                judge_scores=judge_scores,
                stage_reliability=stage_reliability,
            )

        # ── Store in failure_memory ──
        fm_entry = None
        if hasattr(self, "failure_memory") and self.failure_memory is not None:
            try:
                factors = c_trace.get("factors", []) or []
                factor_kinds = [f.get("kind", "") if isinstance(f, dict) else str(f) for f in factors]
                coords = [
                    t.get("coordinate", "")
                    for t in (output.matched_theories or [])
                    if t.get("coordinate")
                ]
                cause = "none"
                error_stage = "none"
                thinking_info: dict[str, Any] = {}

                if thinking_error:
                    cause = _thinking_error_to_cause(thinking_error)
                    error_stage = thinking_error.error_phase
                    thinking_info = format_thinking_error_for_memory(thinking_error)

                fm_entry = self.failure_memory.observe_failure(
                    primary_cause=cause,
                    factor_kinds=factor_kinds,
                    coordinates=coords,
                    oa=oa,
                    detail=thinking_error.evidence[:200] if thinking_error else feedback[:200],
                    exemplar=thinking_error.correction[:120] if thinking_error else "",
                    error_stage=error_stage,
                )

                # Attach thinking-level info to the failure structure
                if thinking_info and fm_entry:
                    fs, repair = fm_entry
                    # Store thinking error metadata in the repair payload
                    if repair.action != "prefer_graph_priors":  # not the disabled sentinel
                        repair.payload["thinking_error"] = thinking_info
            except Exception:
                pass  # Failure memory is best-effort, don't block evolution

        # ── Apply targeted repair (thinking-aware) vs generic weight delta ──
        touched: dict[str, float] = {}
        repair_action = None
        if thinking_error and fm_entry:
            # Generate and apply targeted repair based on thinking error
            _, repair = fm_entry
            if repair.action != "prefer_graph_priors":
                repair_action = repair.action
            # Apply the repair: demote overconfident theories, boost retrieval, etc.
            touched = _apply_targeted_repair(
                self, output, thinking_error, repair, fm_entry
            )
        else:
            # Fallback: simple weight delta (original behavior)
            delta = self.evolve_lr if helpful else -self.evolve_lr
            for t in output.matched_theories:
                tid = t["id"]
                coord = t.get("coordinate") or ""
                self.weights[tid] = max(0.2, min(2.5, self.weights.get(tid, 1.0) + delta))
                if coord:
                    self.weights[coord] = max(
                        0.2, min(2.5, self.weights.get(coord, 1.0) + delta * 0.5)
                    )
                touched[tid] = self.weights[tid]

        self.save_weights()

        log = {
            "user_id": self.user_id,
            "helpful": helpful,
            "oa": oa,
            "touched": touched,
            "feedback": feedback[:300],
            "thinking_error": {
                "error_step": thinking_error.error_step,
                "error_phase": thinking_error.error_phase,
                "error_type": thinking_error.error_type,
                "severity": thinking_error.severity,
                "correction": thinking_error.correction[:150],
            } if thinking_error else None,
            "thinking_quality": thinking_trace.overall_quality if thinking_trace else None,
            "repair_action": repair_action,
        }
        hist = self.state_dir / f"{self.user_id}_evolve.jsonl"
        with hist.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log, ensure_ascii=False) + "\n")
        return log

# ── Thinking-aware evolve helpers ──────────────────────────────────────

def _thinking_error_to_cause(error: ThinkingError) -> str:
    """Map thinking error step to failure_memory primary_cause."""
    step_to_cause = {
        "factor_decomposition": "factor_extraction",
        "theory_retrieval": "retrieval",
        "evidence_query": "retrieval",
        "evidence_read": "retrieval",
        "theory_application": "theory_prior",
        "evidence_interpretation": "theory_prior",
        "profile_anchoring": "profile",
        "counter_argument": "context_shift",
        "confidence_calibration": "short_term_state",
    }
    return step_to_cause.get(error.error_step, "none")


def _apply_targeted_repair(
    agent: "Agent",
    output: AgentOutput,
    error: ThinkingError,
    repair: Any,
    fm_entry: Any,
) -> dict[str, float]:
    """Apply a targeted repair based on thinking error analysis."""
    touched: dict[str, float] = {}

    if error.error_type == "omission":
        if error.error_step == "factor_decomposition":
            # Boost retrieval diversity — don't demote anything
            pass  # Strategy note repair, no weight change
        elif error.error_step in ("evidence_query", "evidence_read"):
            # Boost evidence retrieval for the relevant factor kinds
            for t in (output.matched_theories or []):
                tid = t["id"]
                self_weight = agent.weights.get(tid, 1.0)
                agent.weights[tid] = min(2.5, self_weight + agent.evolve_lr * 0.5)
                touched[tid] = agent.weights[tid]
        elif error.error_step == "counter_argument":
            # Slightly boost skeptic-related weights
            pass  # Strategy note repair

    elif error.error_type == "overgeneralization":
        # Demote theories that were applied without evidence grounding
        for t in (output.matched_theories or []):
            tid = t["id"]
            coord = t.get("coordinate") or ""
            self_weight = agent.weights.get(tid, 1.0)
            agent.weights[tid] = max(0.2, self_weight - agent.evolve_lr * 1.2)
            if coord:
                agent.weights[coord] = max(
                    0.2, agent.weights.get(coord, 1.0) - agent.evolve_lr * 0.6
                )
            touched[tid] = agent.weights[tid]

    elif error.error_type == "contradiction":
        # Model was uncertain but predicted confidently → penalize confidence
        for t in (output.matched_theories or []):
            tid = t["id"]
            self_weight = agent.weights.get(tid, 1.0)
            agent.weights[tid] = max(0.2, self_weight - agent.evolve_lr * 0.8)
            touched[tid] = agent.weights[tid]

    elif error.error_type == "bias":
        # Profile bias → flag profile attributes, reduce profile weight
        for t in (output.matched_theories or []):
            tid = t["id"]
            self_weight = agent.weights.get(tid, 1.0)
            agent.weights[tid] = max(0.2, self_weight - agent.evolve_lr * 0.6)
            touched[tid] = agent.weights[tid]

    return touched


    def loop(
        self,
        stimulus: str,
        *,
        max_iterations: int = 3,
        auto_feedback: str | None = None,
    ) -> list[AgentOutput]:
        outputs: list[AgentOutput] = []
        current = stimulus
        for i in range(max_iterations):
            out = self.predict(current)
            outputs.append(out)
            if auto_feedback is None and i == 0:
                break
            fb = auto_feedback or "请根据上轮解释，收紧机制并更贴合个人历史。"
            self.evolve(out, feedback=fb, helpful=(i == 0))
            current = (
                f"{stimulus}\n\n[loop {i+1} refine]\nPrevious opinion: {out.predicted_opinion}\n"
                f"Feedback: {fb}"
            )
        return outputs


def _parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return {}
    blob = m.group(0)
    try:
        obj = json.loads(blob)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        repaired = blob
        if repaired.count('"') % 2 == 1:
            repaired += '"'
        repaired += "}" * max(0, repaired.count("{") - repaired.count("}"))
        try:
            obj = json.loads(repaired)
            return obj if isinstance(obj, dict) else {}
        except json.JSONDecodeError:
            return {}


def _regex_field(text: str, key: str) -> str:
    m = re.search(rf'"{re.escape(key)}"\s*:\s*"((?:\\.|[^"\\])*)"', text)
    if not m:
        return ""
    return m.group(1).replace('\\"', '"').replace("\\n", "\n")


def _as_str_list(val: Any) -> list[str]:
    if val is None:
        return []
    if isinstance(val, list):
        return [str(x) for x in val]
    return [str(val)]
