from __future__ import annotations

"""Leak-free online cognitive-map memory and V4-Pro self-improving agent.

The source bank may contain codings for the complete chronology, but those
codings live in a private lookup.  The visible graph starts empty and a post's
coding is revealed only by ``ingest_event`` after that post was predicted and
judged.
"""

import json
import math
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .failure_memory import FailureMemory
from .llm import DeepSeekClient
from .thinking_analyzer import (
    ThinkingError,
    attribute_thinking_error,
    format_thinking_error_for_memory,
    parse_thinking_trace,
)


def _tokens(text: str) -> set[str]:
    text = text.lower()
    out = set(re.findall(r"[a-z0-9_]{3,}", text))
    for chunk in re.findall(r"[\u4e00-\u9fff]{2,}", text):
        out.add(chunk)
        for n in (2, 3):
            out.update(chunk[i : i + n] for i in range(max(0, len(chunk) - n + 1)))
    return out


def _similarity(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / math.sqrt(len(a) * len(b))


def _event_blob(e: dict[str, Any]) -> str:
    return " ".join(
        [
            str(e.get("event_title") or ""),
            str(e.get("event_summary") or ""),
            str(e.get("user_opinion") or ""),
            str(e.get("feature_2d_text") or ""),
            str(e.get("feature_3d_text") or ""),
            " ".join(str(x) for x in e.get("topics") or []),
            " ".join(str(x) for x in e.get("entities") or []),
        ]
    )


class OnlinePaperCognitiveMemory:
    """Method-specific graph state that grows one observed post at a time."""

    def __init__(
        self,
        bank_path: str | Path,
        *,
        user_id: str,
        method_key: str,
        persona_path: str | Path | None = None,
    ) -> None:
        source = json.loads(Path(bank_path).read_text(encoding="utf-8"))
        self.user_id = user_id
        self.method_key = method_key
        self.method_name = str(source.get("method") or method_key)
        # Private oracle of per-post codings. Never included in prompts directly.
        self._coding_lookup = {
            str(e.get("post_id")): e
            for e in source.get("event_maps") or []
            if e.get("post_id")
        }
        self.persona: dict[str, Any] = {}
        if persona_path and Path(persona_path).exists():
            self.persona = json.loads(Path(persona_path).read_text(encoding="utf-8"))
        self.events: list[dict[str, Any]] = []
        self._event_tokens: list[set[str]] = []
        self.edge_stats: dict[tuple[str, str], dict[str, Any]] = {}
        self.node_stats: Counter[str] = Counter()
        self.motif_stats: Counter[str] = Counter()
        self.missing_codings: list[str] = []

    @property
    def values(self) -> list[str]:
        return list(self.persona.get("values") or [])[:8]

    @property
    def communication(self) -> list[str]:
        return list(self.persona.get("communication") or [])[:8]

    def identity_block(self) -> str:
        name = "胡锡进" if self.user_id == "1989660417" else ""
        base = (
            f"真实账号身份：{name}。你就是该账号本人，不是旁观者或转述者。"
            if name
            else "你就是该社交媒体账号本人。"
        )
        hints = "\n".join(f"- {x}" for x in self.communication[:5])
        return base + (f"\n表达习惯：\n{hints}" if hints else "")

    def ingest_event(self, event: dict[str, Any]) -> bool:
        """Reveal and merge this post's method-specific coding after prediction."""
        pid = str(event.get("post_id") or "")
        if pid and any(str(x.get("post_id") or "") == pid for x in self.events):
            return True
        coded = self._coding_lookup.get(pid)
        if coded is None:
            self.missing_codings.append(pid)
            coded = dict(event)
            coded["feature_3d_triples"] = []
            coded["feature_3d_text"] = ""
        self.events.append(coded)
        self._event_tokens.append(_tokens(_event_blob(coded)))
        if self.method_key in ("genminds", "genminds_v2"):
            self._ingest_genminds(coded)
        elif self.method_key == "cognitive_maps_1977":
            self._ingest_1977(coded)
        else:
            raise ValueError(f"unsupported method: {self.method_key}")
        return pid in self._coding_lookup

    def _merge_edge(
        self,
        src: str,
        dst: str,
        sign: int,
        confidence: float,
        post_id: str,
    ) -> None:
        if not src or not dst or src == dst:
            return
        rec = self.edge_stats.setdefault(
            (src, dst),
            {"pos": 0, "neg": 0, "confidence_sum": 0.0, "support": 0, "posts": []},
        )
        rec["pos" if sign > 0 else "neg"] += 1
        rec["confidence_sum"] += confidence
        rec["support"] += 1
        if len(rec["posts"]) < 5:
            rec["posts"].append(post_id)
        self.node_stats[src] += 1
        self.node_stats[dst] += 1

    def _ingest_genminds(self, coded: dict[str, Any]) -> None:
        pid = str(coded.get("post_id") or "")
        # V2 uses "causal_edges" with cause/effect; V1 uses "belief_edges" with src/dst
        edges = coded.get("causal_edges") or coded.get("belief_edges") or []
        for edge in edges:
            polarity = edge.get("polarity", 0)
            if isinstance(polarity, str):
                sign = -1 if polarity.startswith("-") else 1
            else:
                sign = -1 if float(polarity) < 0 else 1
            try:
                conf = float(edge.get("confidence") or 0.5)
            except (TypeError, ValueError):
                conf = 0.5
            src = str(edge.get("cause") or edge.get("src") or "")
            dst = str(edge.get("effect") or edge.get("dst") or "")
            self._merge_edge(src, dst, sign, conf, pid)
        # Per-event motifs (V1) or static_map motifs (V2)
        for motif in coded.get("cognitive_motifs") or []:
            name = str(motif.get("motif") or motif.get("name") or "")
            if name:
                self.motif_stats[name] += 1
        # V2: also ingest causal_concepts as nodes
        for concept in coded.get("causal_concepts") or []:
            if concept:
                self.node_stats[str(concept)] += 1

    def _ingest_1977(self, coded: dict[str, Any]) -> None:
        pid = str(coded.get("post_id") or "")
        labels = {
            str(v.get("code")): str(v.get("label") or v.get("code"))
            for v in coded.get("coded_variables") or []
        }
        for edge in coded.get("causal_assertions") or []:
            src_code = str(edge.get("cause") or "")
            dst_code = str(edge.get("effect") or "")
            src = labels.get(src_code, src_code)
            dst = labels.get(dst_code, dst_code)
            sign = -1 if str(edge.get("sign")).startswith("-") else 1
            self._merge_edge(src, dst, sign, 1.0, pid)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 6,
        recency_boost: float = 0.0,
    ) -> list[dict[str, Any]]:
        q = _tokens(query)
        n = len(self.events)
        scored: list[tuple[float, int]] = []
        for i, toks in enumerate(self._event_tokens):
            score = _similarity(q, toks)
            if recency_boost and n > 1:
                score *= 1.0 + recency_boost * i / (n - 1)
            if score > 0:
                scored.append((score, i))
        scored.sort(reverse=True)
        return [
            {"score": round(score, 4), **self.events[i]}
            for score, i in scored[:top_k]
        ]

    def relevant_edges(self, query: str, top_k: int = 12) -> list[dict[str, Any]]:
        q = _tokens(query)
        rows = []
        for (src, dst), rec in self.edge_stats.items():
            lexical = _similarity(q, _tokens(src + dst))
            support = math.log1p(rec["support"])
            score = lexical * 2.0 + 0.08 * support
            if lexical <= 0 and rec["support"] <= 1:
                continue
            sign = 1 if rec["pos"] >= rec["neg"] else -1
            rows.append(
                {
                    "src": src,
                    "dst": dst,
                    "sign": "+" if sign > 0 else "-",
                    "confidence": round(rec["confidence_sum"] / rec["support"], 3),
                    "support": rec["support"],
                    "score": score,
                }
            )
        rows.sort(key=lambda r: -r["score"])
        return rows[:top_k]

    def graph_snapshot(self, query: str) -> dict[str, Any]:
        edges = self.relevant_edges(query)
        return {
            "method": self.method_name,
            "observed_posts": len(self.events),
            "nodes": len(self.node_stats),
            "edges": len(self.edge_stats),
            "relevant_edges": edges,
            "top_motifs": self.motif_stats.most_common(8)
            if self.method_key in ("genminds", "genminds_v2")
            else [],
        }

    def coordinates(self, query: str) -> list[str]:
        return [
            f"{e['src']}->{e['dst']}"
            for e in self.relevant_edges(query, top_k=5)
        ]


def infer_factor_kinds(stimulus: str) -> list[str]:
    groups = {
        "identity": r"国家|民族|中国|美国|台湾|身份|爱国",
        "moral": r"正义|公平|道德|责任|应该|不该|可耻",
        "interest": r"利益|经济|增长|就业|贸易|成本",
        "emotion": r"愤怒|担忧|恐惧|高兴|悲剧|震惊|遗憾",
        "policy": r"政策|政府|法律|管理|制裁|改革|治理",
        "security": r"安全|战争|军事|冲突|威胁|风险",
    }
    found = [kind for kind, pattern in groups.items() if re.search(pattern, stimulus)]
    return found or ["context"]


def _clean_prediction(content: str, reasoning: str) -> str:
    text = (content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
        text = str(obj.get("comment") or obj.get("prediction") or "").strip()
    except (json.JSONDecodeError, AttributeError):
        pass
    if not text:
        quoted = re.findall(r"[\"“]([^\"”]{8,180})[\"”]", reasoning or "")
        if quoted:
            text = quoted[-1]
    if text.startswith("WEIBO:"):
        text = text[6:].strip()
    return text[:500]


class PaperCognitiveOnlineAgent:
    """V4-Pro agent whose only longitudinal cognition is one online paper map."""

    def __init__(
        self,
        *,
        user_id: str,
        memory: OnlinePaperCognitiveMemory,
        llm: DeepSeekClient,
        state_dir: str | Path,
        failure_threshold: float = 0.5,
    ) -> None:
        self.user_id = user_id
        self.memory = memory
        self.llm = llm
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.failure_threshold = failure_threshold
        self.failure_memory = FailureMemory.load(self.state_dir, user_id)
        self.failure_memory.enabled = True
        self.failure_memory.weight_repairs_enabled = True
        self.failure_memory.revoke_on_fail = True

    def predict(
        self,
        stimulus: str,
        *,
        immediate_guidance: str = "",
    ) -> dict[str, Any]:
        factor_kinds = infer_factor_kinds(stimulus)
        coordinates = self.memory.coordinates(stimulus)
        repair_hits = self.failure_memory.retrieve_repairs(
            factor_kinds=factor_kinds,
            coordinates=coordinates,
            top_k=4,
        )
        repair_ids = [r.id for r, _, _ in repair_hits]
        top_k = 6
        recency_boost = 0.0
        repair_notes = []
        for repair, structure, score in repair_hits:
            payload = repair.payload or {}
            if repair.action == "boost_retrieval_kinds":
                top_k += int(payload.get("top_k_boost") or 2)
            elif repair.action == "reset_short_term":
                recency_boost = max(
                    recency_boost, float(payload.get("recency_boost") or 0.35)
                )
            note = str(payload.get("note") or structure.thinking_correction or "")
            if note:
                repair_notes.append(note)
            else:
                repair_notes.append(f"{repair.action}: {json.dumps(payload, ensure_ascii=False)}")

        retrieved = self.memory.retrieve(
            stimulus,
            top_k=min(12, top_k),
            recency_boost=recency_boost,
        )
        history = "\n".join(
            f"- {r.get('event_title','')}｜{str(r.get('user_opinion') or '')[:180]}"
            f"｜graph={str(r.get('feature_3d_text') or '')[:220]}"
            for r in retrieved
        ) or "- 尚无可用历史"
        graph = self.memory.graph_snapshot(stimulus)
        edge_lines = "\n".join(
            f"- {e['src']} --({e['sign']}, conf={e['confidence']}, n={e['support']})--> {e['dst']}"
            for e in graph["relevant_edges"]
        ) or "- 当前图谱尚无相关边"
        motifs = "\n".join(
            f"- {name} (n={count})" for name, count in graph["top_motifs"]
        )
        repairs = "\n".join(f"- {x}" for x in repair_notes) or "- 无已准入修复"

        system = (
            "你是用户本人意见预测 Agent。必须只依据截至当前时刻已经观察到的历史与认知图谱，"
            "不能假装知道未来。请在 reasoning_content 中依次完成："
            "(1) mining：识别 identity/moral/interest/emotion/policy 等因素；"
            "(2) 查询并核对历史证据和图谱边；"
            "(3) reasoning：形成判断，执行反方检查和置信度校准。"
            "最终 content 只输出 JSON：{\"comment\":\"一条本人声口的原创微博评论\"}，"
            "不得在最终答案泄露分析过程。"
        )
        user = f"""【身份】
{self.memory.identity_block()}

【在线认知图谱】
方法={graph['method']}；已观察帖子={graph['observed_posts']}；
节点={graph['nodes']}；边={graph['edges']}
相关因果边：
{edge_lines}
{('相关认知母题：' + chr(10) + motifs) if motifs else ''}

【仅来自过去的相关发帖】
{history}

【错题本检索出的定向修复】
{repairs}

【仅限当前步骤的临时纠偏目标】
{immediate_guidance or '- 无'}

【当前事件】
{stimulus}

请预测该用户此刻会发表的评论。"""
        trace = self.llm.chat_with_trace(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.35,
            max_tokens=1800,
            disable_thinking=False,
        )
        reasoning = str(trace.get("reasoning_content") or "")
        prediction = _clean_prediction(str(trace.get("content") or ""), reasoning)
        parsed = parse_thinking_trace(reasoning)
        return {
            "prediction": prediction,
            "reasoning_content": reasoning,
            "white_box": trace.get("white_box") or {},
            "thinking_quality": parsed.overall_quality,
            "factor_kinds": factor_kinds,
            "coordinates": coordinates,
            "repairs_applied": repair_ids,
            "retrieved_post_ids": [str(r.get("post_id") or "") for r in retrieved],
            "graph_before": graph,
        }

    def evolve(
        self,
        *,
        trace: dict[str, Any],
        judge_scores: dict[str, Any],
        immediate_admit: bool = False,
        ground_truth: str = "",
        stimulus: str = "",
        repair_round: int = 0,
    ) -> dict[str, Any]:
        oa = float(judge_scores.get("opinion_alignment_score") or 0.0)
        helpful = oa >= self.failure_threshold
        applied = list(trace.get("repairs_applied") or [])
        if applied:
            self.failure_memory.feedback(applied, helpful=helpful)

        parsed = parse_thinking_trace(str(trace.get("reasoning_content") or ""))
        error = None
        diagnosis_reasoning = ""
        immediate_guidance = ""
        learned = None
        if not helpful:
            if ground_truth:
                error, diagnosis_reasoning, immediate_guidance = self._diagnose_from_feedback(
                    trace=trace,
                    judge_scores=judge_scores,
                    ground_truth=ground_truth,
                    stimulus=stimulus,
                    repair_round=repair_round,
                )
            if error is None:
                error = attribute_thinking_error(
                    parsed,
                    oa=oa,
                    judge_scores=judge_scores,
                    stage_reliability={},
                )
            if not immediate_guidance:
                # Diagnosis can occasionally return malformed JSON. Preserve
                # the strict no-advance contract with an ephemeral teacher
                # target for this event only; it is never stored in the
                # long-term failure memory.
                immediate_guidance = (
                    "仅限本步：请用账号本人声口准确改写以下真实反馈的核心判断，"
                    "不要增加反馈中没有的立场或事实："
                    + ground_truth[:500]
                )
            if error is not None:
                structure, repair = self.failure_memory.observe_failure(
                    primary_cause={
                        "factor_decomposition": "factor_extraction",
                        "theory_retrieval": "retrieval",
                        "evidence_query": "retrieval",
                        "evidence_read": "retrieval",
                        "theory_application": "theory_prior",
                        "evidence_interpretation": "theory_prior",
                        "profile_anchoring": "profile",
                        "counter_argument": "context_shift",
                        "confidence_calibration": "short_term_state",
                    }.get(error.error_step, "none"),
                    factor_kinds=list(trace.get("factor_kinds") or []),
                    coordinates=list(trace.get("coordinates") or []),
                    oa=oa,
                    detail=error.evidence,
                    strategy=error.correction,
                    exemplar=error.correction,
                    error_stage=error.error_phase,
                    thinking_error_step=error.error_step,
                    thinking_error_type=error.error_type,
                    thinking_correction=error.correction,
                )
                repair.payload["thinking_error"] = format_thinking_error_for_memory(error)
                if immediate_admit:
                    # Strict same-step repair: the current mistake must be
                    # actionable before another prediction of this same event.
                    # Admit every repair learned for this structure, including
                    # the natural-language strategy note.
                    for candidate in self.failure_memory.repairs.values():
                        if candidate.structure_id == structure.id and not candidate.revoked:
                            candidate.admitted = True
                learned = {
                    "structure_id": structure.id,
                    "structure_frequency": structure.freq,
                    "repair_id": repair.id,
                    "repair_action": repair.action,
                    "admitted": repair.admitted,
                }

        self.failure_memory.prune_to_budget(max_structures=80, max_repairs=120)
        self.failure_memory.save(self.state_dir)
        return {
            "helpful": helpful,
            "oa": oa,
            "thinking_error": asdict(error) if error else None,
            "diagnosis_reasoning": diagnosis_reasoning,
            # Ephemeral: used only to retry this same event; never persisted in
            # FailureMemory and never carried into the next chronological step.
            "immediate_guidance": immediate_guidance,
            "learned": learned,
            "failure_memory": self.failure_memory.stats(),
        }

    def _diagnose_from_feedback(
        self,
        *,
        trace: dict[str, Any],
        judge_scores: dict[str, Any],
        ground_truth: str,
        stimulus: str,
        repair_round: int,
    ) -> tuple[ThinkingError | None, str, str]:
        """Let V4-Pro critique its own failed CoT and derive a transferable fix.

        The real response is used as supervision for this step, but the stored
        correction must be an abstract rule and may not copy the response.
        """
        system = (
            "你是同一个预测Agent的自主纠错模块。比较本轮思维链、预测与真实反馈，"
            "精确定位导致意见不对齐的思维步骤，并给出可迁移到未来事件的抽象修复规则。"
            "不得把真实评论原句写入 correction，不得要求记住本题答案。"
            "只输出JSON："
            '{"error_step":"factor_decomposition|evidence_query|evidence_read|'
            'theory_application|evidence_interpretation|profile_anchoring|'
            'counter_argument|confidence_calibration",'
            '"error_phase":"mining|reasoning",'
            '"error_type":"omission|overgeneralization|contradiction|bias|misidentification",'
            '"evidence":"思维链中出错的具体片段",'
            '"correction":"面向以后同类问题的具体策略",'
            '"current_step_fix":"仅用于重做当前题的具体纠偏目标，可使用真实反馈中的关键信息",'
            '"severity":0.0}'
        )
        user = f"""修复轮次：{repair_round}
当前事件：
{stimulus}

本轮预测：
{trace.get('prediction') or ''}

真实反馈（仅用于归因，不得复制到修复规则）：
{ground_truth[:1200]}

评审分数：
{json.dumps(judge_scores, ensure_ascii=False)}

本轮V4-Pro思维链：
{str(trace.get('reasoning_content') or '')[:6000]}

请定位真正错误并生成新的、比上一轮更有针对性的抽象修复。"""
        try:
            result = self.llm.chat_with_trace(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.15,
                max_tokens=1200,
                disable_thinking=False,
            )
            content = str(result.get("content") or "").strip()
            match = re.search(r"\{.*\}", content, re.S)
            obj = json.loads(match.group(0) if match else content)
            step = str(obj.get("error_step") or "")
            phase = str(obj.get("error_phase") or "")
            error_type = str(obj.get("error_type") or "")
            if phase not in {"mining", "reasoning"}:
                phase = "mining" if step in {
                    "factor_decomposition",
                    "evidence_query",
                    "evidence_read",
                    "theory_retrieval",
                } else "reasoning"
            if error_type not in {
                "omission",
                "overgeneralization",
                "contradiction",
                "bias",
                "misidentification",
            }:
                error_type = "misidentification"
            return (
                ThinkingError(
                    error_step=step or "evidence_interpretation",
                    error_phase=phase,
                    error_type=error_type,
                    evidence=str(obj.get("evidence") or "")[:300],
                    correction=str(obj.get("correction") or "")[:300],
                    severity=max(0.0, min(1.0, float(obj.get("severity") or 0.7))),
                ),
                str(result.get("reasoning_content") or ""),
                str(obj.get("current_step_fix") or "")[:500],
            )
        except Exception:
            return None, "", ""
