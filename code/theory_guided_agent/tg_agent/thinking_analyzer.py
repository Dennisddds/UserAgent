"""Thinking Analyzer: parse model CoT into RTWI-structured trace for error attribution.

Takes raw `reasoning_content` from DeepSeek thinking mode and:
  1. Segments into Mining (information gathering) vs Reasoning (judgment) phases
  2. Extracts key claims, evidence references, and reasoning steps
  3. When prediction fails, identifies which thinking step was wrong
  4. Produces a ThinkingTrace suitable for 错题本 storage and targeted repair

Design principle: offline analysis that doesn't add latency to the prediction loop.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


# ── RTWI phase markers ──────────────────────────────────────────────
# These patterns indicate the transition from "gathering info" to "making judgment"
_REASONING_TRANSITIONS = [
    r"(?:现在|现在，|接下来|接下来，|然后|最后)(?:要|需要|我来|我要|开始)",
    r"(?:基于|根据|综合|结合)(?:以上|现有|这些|手头)",
    r"(?:所以|因此|综上|总的来说)",
    r"(?:in conclusion|therefore|based on the above|to summarize)",
    r"(?:现在写|现在输出|现在生成|现在预测)",
    r"(?:回复|评论|我的观点|我的立场|我?认为|我?觉得)",
    r"(?:output|prediction|final answer)",
]

# Steps within each phase that we can identify
_MINING_STEPS = {
    "factor_decomposition": [
        r"(?:分解|识别|提取|分析)(?:因素|因子|维度|方面)",
        r"(?:factor|decompose|identify).*(?:types?|kinds?|dimensions?)",
        r"(?:事件|话题|议题).*(?:涉及|包含|属于|关于)",
    ],
    "theory_retrieval": [
        r"(?:检索|匹配|查找|调用)(?:理论|坐标|机制|先验)",
        r"(?:theory|coordinate|retrieve|match|look.?up)",
        r"(?:理论库|TheoryLibrary|坐标匹配)",
    ],
    "evidence_query": [
        r"(?:查询|搜索|检索|查找)(?:证据|历史|微博|帖子|记录)",
        r"(?:query|search|retrieve).*(?:evidence|history|posts|records)",
        r"(?:用户|该用户|ta).*(?:历史|过去|以前|曾经|发过)",
    ],
    "evidence_read": [
        r"(?:读取|查看|阅读|检查)(?:证据|内容|结果|详情)",
        r"(?:read|examine|inspect|review).*(?:evidence|result|detail)",
        r"(?:证据|结果).*(?:显示|表明|包含|提到)",
    ],
}

_REASONING_STEPS = {
    "theory_application": [
        r"(?:理论|坐标|机制).*(?:适用|匹配|激活|触发|解释)",
        r"(?:根据|按照|依据)(?:理论|坐标|机制|先验)",
        r"(?:theory|coordinate|mechanism).*(?:applies|matches|activates|explains)",
    ],
    "evidence_interpretation": [
        r"(?:证据|历史|微博).*(?:说明|表明|暗示|指向|支持)",
        r"(?:evidence|history).*(?:suggests|indicates|points|supports)",
        r"(?:从|由).*(?:证据|历史|记录).*(?:可以|能).*(?:看出|推断|判断)",
    ],
    "profile_anchoring": [
        r"(?:用户|该用户|ta|画像).*(?:特征|特点|倾向|立场|人格)",
        r"(?:作为|身为).*(?:用户|该用户)",
        r"(?:profile|persona|identity|character).*(?:trait|tendency|stance)",
    ],
    "counter_argument": [
        r"(?:但是|不过|然而|另一方面|话说回来|质疑)",
        r"(?:however|but|on the other hand|skeptic|doubt)",
        r"(?:质疑|反驳|counter|alternative).*(?:观点|看法|角度|解释)",
    ],
    "confidence_calibration": [
        r"(?:置信度|把握|确定|不确定|可能|也许|大概)",
        r"(?:confidence|certainty|uncertain|probably|maybe|likely)",
        r"(?:证据|信息).*(?:不足|薄弱|有限|缺少|缺乏)",
    ],
}


@dataclass
class ThinkingStep:
    """A single identifiable step within the CoT."""
    step_type: str           # e.g., "factor_decomposition", "theory_application"
    phase: str               # "mining" or "reasoning"
    excerpt: str             # relevant text from CoT
    claims: list[str] = field(default_factory=list)  # specific claims made
    quality_flags: list[str] = field(default_factory=list)  # e.g. "incomplete", "contradictory"


@dataclass
class ThinkingTrace:
    """Parsed and structured thinking trace from model CoT."""
    has_reasoning: bool = False
    reasoning_chars: int = 0
    mining_phase: str = ""           # full mining excerpt
    reasoning_phase: str = ""        # full reasoning excerpt
    steps: list[ThinkingStep] = field(default_factory=list)
    factors_mentioned: list[str] = field(default_factory=list)
    theories_referenced: list[str] = field(default_factory=list)
    evidence_claimed: list[str] = field(default_factory=list)
    confidence_signals: list[str] = field(default_factory=list)  # high/low signals
    profile_references: list[str] = field(default_factory=list)
    split_marker: str = ""           # the marker that triggered the split
    overall_quality: dict[str, float] = field(default_factory=dict)


@dataclass
class ThinkingError:
    """An error found in the model's thinking process."""
    error_step: str          # which thinking step was wrong
    error_phase: str         # "mining" or "reasoning"
    error_type: str          # "misidentification" | "omission" | "overgeneralization" | "bias" | "contradiction"
    evidence: str            # quote from CoT showing the error
    correction: str          # what the thinking should have been
    severity: float = 0.5    # 0-1 how bad the error is


# ── Public API ──────────────────────────────────────────────────────

def parse_thinking_trace(reasoning_content: str) -> ThinkingTrace:
    """Parse raw reasoning_content into a structured ThinkingTrace.

    This is the main entry point — call after each prediction that has CoT.
    Runs entirely offline (no LLM calls), ~0.1ms latency.
    """
    if not reasoning_content or not reasoning_content.strip():
        return ThinkingTrace(has_reasoning=False)

    trace = ThinkingTrace(
        has_reasoning=True,
        reasoning_chars=len(reasoning_content),
    )

    # ── Phase 1: Split mining vs reasoning ──
    split_idx = _find_reasoning_split(reasoning_content)
    trace.mining_phase = reasoning_content[:split_idx] if split_idx > 0 else reasoning_content[:len(reasoning_content)//2]
    trace.reasoning_phase = reasoning_content[split_idx:] if split_idx > 0 else reasoning_content[len(reasoning_content)//2:]
    trace.split_marker = _identify_split_marker(reasoning_content, split_idx)

    # ── Phase 2: Extract steps from each phase ──
    _extract_steps(trace, trace.mining_phase, "mining", _MINING_STEPS)
    _extract_steps(trace, trace.reasoning_phase, "reasoning", _REASONING_STEPS)

    # ── Phase 3: Extract semantic signals ──
    trace.factors_mentioned = _extract_factors(reasoning_content)
    trace.theories_referenced = _extract_theory_refs(reasoning_content)
    trace.evidence_claimed = _extract_evidence_claims(reasoning_content)
    trace.confidence_signals = _extract_confidence_signals(reasoning_content)
    trace.profile_references = _extract_profile_refs(reasoning_content)

    # ── Phase 4: Quality assessment ──
    trace.overall_quality = _assess_thinking_quality(trace)

    return trace


def attribute_thinking_error(
    trace: ThinkingTrace,
    *,
    oa: float,
    judge_scores: dict[str, Any] | None = None,
    stage_reliability: dict[str, Any] | None = None,
) -> ThinkingError | None:
    """When prediction fails (OA < threshold), find the specific thinking error.

    Returns None if no clear error pattern can be identified.
    """
    if oa >= 0.5:  # Not a clear failure — skip detailed attribution
        return None

    if not trace.has_reasoning:
        # Fallback: use structural signals if no CoT available
        return _fallback_structural_error(stage_reliability or {}, oa)

    # ── Check each potential error type ──

    # 1. Factor misidentification: model claimed factors that don't match the event
    if not trace.factors_mentioned:
        return ThinkingError(
            error_step="factor_decomposition",
            error_phase="mining",
            error_type="omission",
            evidence="No factor decomposition found in thinking",
            correction="Explicitly decompose event factors before theory retrieval",
            severity=0.8,
        )

    if len(trace.factors_mentioned) <= 1 and oa < 0.4:
        return ThinkingError(
            error_step="factor_decomposition",
            error_phase="mining",
            error_type="omission",
            evidence=f"Only {len(trace.factors_mentioned)} factor(s) identified: {trace.factors_mentioned}",
            correction="Consider multiple factor dimensions (identity, moral, interest, emotion, policy)",
            severity=0.7,
        )

    # 2. Evidence retrieval failure: model didn't look for evidence
    has_evidence_query = any(
        s.step_type == "evidence_query" and s.phase == "mining"
        for s in trace.steps
    )
    has_evidence_read = any(
        s.step_type == "evidence_read" for s in trace.steps
    )
    if (not has_evidence_query and not has_evidence_read) and oa < 0.4:
        return ThinkingError(
            error_step="evidence_query",
            error_phase="mining",
            error_type="omission",
            evidence="Model didn't query or read user-specific evidence before judging",
            correction="Always retrieve individual evidence before forming judgment",
            severity=0.75,
        )

    # 3. Theory misapplication: model referenced theory but didn't ground in evidence
    has_theory = any(
        s.step_type == "theory_application" for s in trace.steps
    )
    has_evidence_interpretation = any(
        s.step_type == "evidence_interpretation" for s in trace.steps
    )
    if has_theory and not has_evidence_interpretation and trace.evidence_claimed:
        return ThinkingError(
            error_step="theory_application",
            error_phase="reasoning",
            error_type="overgeneralization",
            evidence=f"Theory applied but evidence {trace.evidence_claimed[:2]} not interpreted against it",
            correction="For each theory application, check against specific evidence items",
            severity=0.6,
        )

    # 4. Skipped skepticism: no counter-argument or confidence check
    has_counter = any(
        s.step_type == "counter_argument" for s in trace.steps
    )
    has_calibration = any(
        s.step_type == "confidence_calibration" for s in trace.steps
    )
    if not has_counter and not has_calibration:
        return ThinkingError(
            error_step="counter_argument",
            error_phase="reasoning",
            error_type="omission",
            evidence="Model skipped counter-argument and confidence calibration",
            correction="Add skeptic_check: consider alternative interpretations before finalizing",
            severity=0.5,
        )

    # 5. Low confidence recognized but not handled
    low_conf_signals = [s for s in trace.confidence_signals if "不足" in s or "薄弱" in s or "有限" in s or "uncertain" in s.lower()]
    if low_conf_signals and oa < 0.3:
        return ThinkingError(
            error_step="confidence_calibration",
            error_phase="reasoning",
            error_type="contradiction",
            evidence=f"Model noted uncertainty ({low_conf_signals[0][:80]}) but still made a confident wrong prediction",
            correction="When evidence is weak, lower confidence or output uncertain instead of guessing",
            severity=0.7,
        )

    # 6. Generic: if we can't pinpoint, use structural signals as fallback
    return _fallback_structural_error(stage_reliability or {}, oa)


def format_thinking_error_for_memory(error: ThinkingError) -> dict[str, Any]:
    """Convert ThinkingError to failure_memory-compatible format."""
    return {
        "thinking_error_step": error.error_step,
        "thinking_error_phase": error.error_phase,
        "thinking_error_type": error.error_type,
        "thinking_error_evidence": error.evidence[:200],
        "thinking_error_correction": error.correction[:200],
        "thinking_error_severity": error.severity,
    }


def generate_targeted_repair(
    error: ThinkingError,
    *,
    current_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a targeted repair action based on thinking error type.

    Returns a dict with 'action' and 'payload' matching ConditionalRepair format.
    """
    repairs: dict[str, dict[str, Any]] = {
        "omission": {
            "factor_decomposition": {
                "action": "strategy_note",
                "payload": {
                    "note": "分解事件因素时确保覆盖 identity/moral/interest/emotion/policy 五个维度，至少产出 3 个因素类型",
                    "stage": "mining",
                },
            },
            "evidence_query": {
                "action": "boost_retrieval_kinds",
                "payload": {
                    "factor_kinds": (current_state or {}).get("factor_kinds", []),
                    "top_k_boost": 3,
                    "stage": "mining",
                    "note": "必须在形成判断前查询个体证据",
                },
            },
            "counter_argument": {
                "action": "strategy_note",
                "payload": {
                    "note": "在 finalize 之前执行一次 skeptic_check：考虑对立观点和证据的局限性",
                    "stage": "reasoning",
                },
            },
        },
        "overgeneralization": {
            "theory_application": {
                "action": "demote_theory_coords",
                "payload": {
                    "note": "理论应用时必须逐条对照个体证据，不能仅凭理论先验判断",
                    "stage": "reasoning",
                },
            },
        },
        "contradiction": {
            "confidence_calibration": {
                "action": "strategy_note",
                "payload": {
                    "note": "证据薄弱或不确定时，输出 uncertain 立场并标注 low_evidence_factors",
                    "stage": "reasoning",
                },
            },
        },
        "bias": {
            "profile_anchoring": {
                "action": "flag_profile_attr",
                "payload": {
                    "note": "画像锚定时注意区分稳定特质和情境波动，使用 rolling window 而非全局画像",
                    "stage": "reasoning",
                },
            },
        },
    }

    # Try precise match first
    error_type_map = repairs.get(error.error_type, {})
    if error.error_step in error_type_map:
        return error_type_map[error.error_step]

    # Fallback: any repair for this error type
    if error_type_map:
        return next(iter(error_type_map.values()))

    # Ultimate fallback
    return {
        "action": "strategy_note",
        "payload": {
            "note": f"Thinking error: {error.error_type} at step {error.error_step}. Correction: {error.correction[:120]}",
            "stage": error.error_phase,
        },
    }


# ── Internal helpers ─────────────────────────────────────────────────

def _find_reasoning_split(text: str) -> int:
    """Find the transition point from mining to reasoning."""
    best_idx = -1
    text_lower = text.lower()
    for pattern in _REASONING_TRANSITIONS:
        for m in re.finditer(pattern, text):
            idx = m.start()
            # Must be at least 100 chars from start (need some mining)
            if idx >= 100 and (best_idx < 0 or idx < best_idx):
                best_idx = idx

    if best_idx < 0:
        # Heuristic: find where "现在" or "最后" appears in the second half
        half = len(text) // 2
        for marker in ["现在", "最后", "所以", "综上", "基于以上"]:
            idx = text.find(marker, half)
            if idx >= half:
                best_idx = idx
                break

    if best_idx < 0:
        best_idx = max(1, int(len(text) * 0.45))

    return best_idx


def _identify_split_marker(text: str, split_idx: int) -> str:
    """Identify which marker triggered the split."""
    if split_idx <= 0:
        return "heuristic_mid"
    window = text[max(0, split_idx - 10): split_idx + 30]
    for pattern in _REASONING_TRANSITIONS:
        m = re.search(pattern, window)
        if m:
            return m.group(0)[:40]
    return "heuristic_mid"


def _extract_steps(
    trace: ThinkingTrace,
    text: str,
    phase: str,
    step_patterns: dict[str, list[str]],
) -> None:
    """Extract identifiable thinking steps from a phase."""
    for step_type, patterns in step_patterns.items():
        for pattern in patterns:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                start = max(0, m.start() - 30)
                end = min(len(text), m.end() + 80)
                excerpt = text[start:end].strip()
                step = ThinkingStep(
                    step_type=step_type,
                    phase=phase,
                    excerpt=excerpt,
                    claims=_extract_claims_from_excerpt(excerpt),
                )
                trace.steps.append(step)
                break  # One match per step type per phase
            else:
                continue
            break


def _extract_claims_from_excerpt(excerpt: str) -> list[str]:
    """Extract specific claims from a thinking excerpt."""
    claims = []
    # Split on Chinese/English sentence boundaries
    sentences = re.split(r'[。；！？.!?;]\s*', excerpt)
    for s in sentences:
        s = s.strip()
        if len(s) > 10 and any(kw in s for kw in ["是", "有", "会", "要", "应该", "需要", "认为", "显示", "表明", "is", "has", "will", "should"]):
            claims.append(s[:120])
    return claims[:3]


def _extract_factors(text: str) -> list[str]:
    """Extract factor types mentioned in thinking."""
    factor_kws = [
        "identity", "身份", "认同", "moral", "道德", "emotion", "情绪", "情感",
        "interest", "利益", "policy", "政策", "source", "来源", "信息源",
        "framing", "框架", "agenda", "议程",
    ]
    found = []
    for kw in factor_kws:
        if kw.lower() in text.lower():
            found.append(kw)
    return list(dict.fromkeys(found))[:8]


def _extract_theory_refs(text: str) -> list[str]:
    """Extract theory/coordinate references from thinking."""
    theory_kws = [
        "identity_threat", "moral_superiority", "media_distrust", "framing_resistance",
        "agenda_setting", "national_identity", "group_bias", "confirmation_bias",
        "身份威胁", "道德优越", "媒体不信任", "框架抵抗", "议程设置",
        "国家认同", "群体偏见", "确认偏误",
    ]
    found = []
    for kw in theory_kws:
        if kw.lower() in text.lower():
            found.append(kw)
    return list(dict.fromkeys(found))[:8]


def _extract_evidence_claims(text: str) -> list[str]:
    """Extract evidence-related claims."""
    patterns = [
        r'(?:用户|该用户|ta).*?(?:历史|过去|以前|曾经|发过|说过)(.{5,60}?)(?:[。；！？]|$)',
        r'(?:证据|历史|微博|帖子).*?(?:显示|表明|说明|包含|提到)(.{5,60}?)(?:[。；！？]|$)',
        r'(?:evidence|history|posts).*?(?:shows?|suggests?|indicates?|contains?)(.{5,60}?)(?:[.!?]|$)',
    ]
    claims = []
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            claim = m.group(1).strip() if m.lastindex else m.group(0).strip()
            if len(claim) > 8:
                claims.append(claim[:120])
    return claims[:5]


def _extract_confidence_signals(text: str) -> list[str]:
    """Extract confidence-related signals."""
    high = ["很确定", "非常确定", "一定是", "显然是", "毫无疑问", "clearly", "definitely", "certainly", "obviously"]
    low = ["不确定", "可能", "也许", "大概", "证据不足", "信息有限", "uncertain", "probably", "maybe", "likely", "limited evidence", "insufficient"]
    signals = []
    for s in high:
        if s in text:
            signals.append(f"HIGH: {s}")
    for s in low:
        if s in text:
            signals.append(f"LOW: {s}")
    return signals


def _extract_profile_refs(text: str) -> list[str]:
    """Extract user profile references."""
    patterns = [
        r'(?:作为|身为).*?(?:用户|该用户).*?([。；！？])',
        r'(?:画像|人格|性格|立场|倾向).*?([。；！？])',
    ]
    refs = []
    for pat in patterns:
        for m in re.finditer(pat, text):
            refs.append(m.group(0)[:120])
    return refs[:5]


def _assess_thinking_quality(trace: ThinkingTrace) -> dict[str, float]:
    """Score the quality of the thinking process."""
    scores = {}

    # Completeness: did we cover all expected steps?
    expected_mining = {"factor_decomposition", "theory_retrieval", "evidence_query", "evidence_read"}
    expected_reasoning = {"theory_application", "evidence_interpretation", "profile_anchoring", "counter_argument", "confidence_calibration"}
    mining_done = {s.step_type for s in trace.steps if s.phase == "mining"}
    reasoning_done = {s.step_type for s in trace.steps if s.phase == "reasoning"}
    scores["mining_completeness"] = len(mining_done & expected_mining) / len(expected_mining) if expected_mining else 0.0
    scores["reasoning_completeness"] = len(reasoning_done & expected_reasoning) / len(expected_reasoning) if expected_reasoning else 0.0

    # Diversity: how many distinct factor types?
    scores["factor_diversity"] = min(1.0, len(trace.factors_mentioned) / 5.0)

    # Grounding: did we reference evidence?
    scores["evidence_grounding"] = min(1.0, len(trace.evidence_claimed) / 3.0) if trace.evidence_claimed else 0.0

    # Skepticism: did we do counter-argument?
    has_counter = any(s.step_type == "counter_argument" for s in trace.steps)
    has_calibration = any(s.step_type == "confidence_calibration" for s in trace.steps)
    scores["skepticism"] = 0.5 if has_counter else 0.0
    scores["calibration"] = 0.5 if has_calibration else 0.0

    # Overall
    scores["overall"] = sum(scores.values()) / max(1, len(scores))

    return {k: round(v, 3) for k, v in scores.items()}


def _fallback_structural_error(
    stage_reliability: dict[str, Any],
    oa: float,
) -> ThinkingError | None:
    """When no CoT is available, attribute error using structural signals."""
    mining = stage_reliability.get("mining", {})
    synthesis = stage_reliability.get("synthesis", {})
    leap_verdict = stage_reliability.get("leap_verdict", "normal")

    # Mining failures
    if mining.get("coverage", 0.0) < 0.35:
        return ThinkingError(
            error_step="evidence_query" if mining.get("total_events", 0) < 1 else "theory_retrieval",
            error_phase="mining",
            error_type="omission",
            evidence=f"Low coverage ({mining.get('coverage', 0):.2f}), {mining.get('total_events', 0)} evidence events found",
            correction="Widen retrieval scope: increase top_k and try LLM query rewriting",
            severity=0.7,
        )

    if mining.get("type_diversity", 1.0) < 0.35 and mining.get("n_factors", 0) <= 3:
        return ThinkingError(
            error_step="factor_decomposition",
            error_phase="mining",
            error_type="omission",
            evidence=f"Low type diversity ({mining.get('type_diversity', 0):.2f}), only {mining.get('n_factors', 0)} factor(s)",
            correction="Decompose event into more factor dimensions",
            severity=0.65,
        )

    # Reasoning failures
    if leap_verdict == "suspicious" and synthesis.get("used_coverage", 1.0) < 0.65:
        return ThinkingError(
            error_step="theory_application",
            error_phase="reasoning",
            error_type="overgeneralization",
            evidence=f"Suspicious leap: high confidence ({synthesis.get('confidence', 0):.2f}) with low evidence coverage",
            correction="Demote overconfident theory coords when evidence is thin",
            severity=0.6,
        )

    if synthesis.get("low_evidence") and oa < 0.3:
        return ThinkingError(
            error_step="confidence_calibration",
            error_phase="reasoning",
            error_type="contradiction",
            evidence="Model acknowledged low evidence but made confident wrong prediction",
            correction="Output 'uncertain' when evidence is insufficient",
            severity=0.55,
        )

    return None
