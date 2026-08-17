"""Theory router: 遇到什么情况 → 用什么理论 → 帮助预测.

Meeting contract:
  - Need routing (not dump-all theories)
  - Prefer high-confidence / rich / grounded cards (agentic RAG grade)
  - Explain why a theory was selected (condition match)

This is the "how to use theory" layer sitting on TheoryLibrary.match.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .models import MatchedTheory, TheoryCard


# factor type → preferred theory coordinates (C–U–V hat × elements)
FACTOR_COORD_PRIOR: dict[str, list[str]] = {
    "policy": ["fairness", "agenda_setting", "trust", "procedural_justice"],
    "interest": ["fairness", "prospect_theory", "risk_perception"],
    "moral": ["identity_threat", "cultural_cognition", "cognitive_dissonance", "fairness"],
    "source": ["source_credibility", "hostile_media", "trust", "two_step", "opinion_leadership"],
    "group_identity": ["identity_threat", "cultural_cognition", "spiral_of_silence", "selective_exposure"],
    "emotion": ["risk_perception", "prospect_theory", "uncertainty_reduction", "tech_threat"],
    "other": ["framing", "motivated_reasoning", "uncertainty_reduction"],
}

# lexical cues in stimulus → coordinates (routing hints)
CUE_COORD: list[tuple[tuple[str, ...], str]] = [
    (("崩溃", "崩盘", "危机", "风险", "灾难"), "risk_perception"),
    (("图表", "可视化", "数据", "辟谣", "澄清"), "uncertainty_reduction"),
    (("外媒", "西方", "辱华", "精日", "爱国"), "identity_threat"),
    (("双标", "不公", "特权", "程序"), "fairness"),
    (("造假", "撒谎", "信任", "官媒"), "trust"),
    (("框架", "叙事", "唱衰"), "framing"),
    (("损失", "亏了", "失去"), "prospect_theory"),
    (("谣言", "假新闻", "撤回"), "misinformation"),
    (("大V", "意见领袖", "转述"), "opinion_leadership"),
    (("AI", "算法", "监控"), "technology_threat"),
]


@dataclass
class RouteDecision:
    factor_id: str
    factor_type: str
    preferred_coords: list[str]
    cue_coords: list[str]
    matched: list[MatchedTheory] = field(default_factory=list)
    rejected_low_conf: list[str] = field(default_factory=list)
    why: str = ""


def cue_coordinates(text: str) -> list[str]:
    t = text or ""
    out: list[str] = []
    for cues, coord in CUE_COORD:
        if any(c in t for c in cues):
            out.append(coord)
    # dedupe keep order
    seen: set[str] = set()
    ordered: list[str] = []
    for c in out:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


def preferred_coordinates(factor_type: str, stimulus: str = "", topic: str = "") -> list[str]:
    base = list(FACTOR_COORD_PRIOR.get(factor_type, FACTOR_COORD_PRIOR["other"]))
    cues = cue_coordinates(f"{stimulus} {topic}")
    # cues first (situation-specific), then factor priors
    merged: list[str] = []
    seen: set[str] = set()
    for c in cues + base:
        if c not in seen:
            seen.add(c)
            merged.append(c)
    return merged


def theory_confidence(card: TheoryCard, score: float) -> float:
    """0–1 gate score for agentic-RAG style filtering."""
    rich = float(getattr(card, "richness", 0.0) or 0.0)
    grounded = 1.0 if getattr(card, "grounded", False) or card.source in {"canonical", "seed"} else 0.45
    src = 1.0 if card.source == "canonical" else (0.85 if card.source == "seed" else 0.7)
    # need conditions/propositions for "when to use"
    struct = 0.0
    if card.conditions:
        struct += 0.5
    if card.propositions:
        struct += 0.5
    return round(
        0.35 * min(1.0, score * 8)  # lexical/env match (scores are small)
        + 0.25 * rich
        + 0.20 * grounded
        + 0.10 * src
        + 0.10 * struct,
        4,
    )


def route_theories(
    theories: Any,
    *,
    factor_id: str,
    factor_type: str,
    query: str,
    stimulus: str = "",
    topic: str = "",
    top_k: int = 2,
    user_weights: dict[str, float] | None = None,
    env_weights: dict[str, float] | None = None,
    min_confidence: float = 0.35,
    min_richness: float = 0.35,
) -> RouteDecision:
    """Route: situation → preferred coords → match → confidence filter."""
    prefs = preferred_coordinates(factor_type, stimulus=stimulus, topic=topic)
    cues = cue_coordinates(f"{stimulus} {topic}")
    # boost preferred coords in env_weights for this call
    env = dict(env_weights or {})
    for i, c in enumerate(prefs[:6]):
        env[c] = max(float(env.get(c, 1.0)), 1.4 - 0.08 * i)

    raw = theories.match(
        query,
        top_k=max(top_k * 3, 6),
        user_weights=user_weights,
        env_weights=env,
        prefer_rich=True,
        prefer_grounded=True,
        min_richness=min_richness,
    )

    kept: list[MatchedTheory] = []
    rejected: list[str] = []
    for m in raw:
        conf = theory_confidence(m.card, m.score)
        # soft prefer preferred coords
        if m.card.coordinate in prefs:
            conf = min(1.0, conf + 0.08)
        if conf < min_confidence and m.card.source not in {"canonical", "seed"}:
            rejected.append(f"{m.card.id}:{conf:.2f}")
            continue
        # annotate why into MatchedTheory.why
        why_bits = [m.why]
        if m.card.coordinate in prefs:
            why_bits.append(f"route=coord:{m.card.coordinate}")
        if m.card.conditions:
            why_bits.append(f"cond={m.card.conditions[0][:40]}")
        why_bits.append(f"conf={conf:.2f}")
        m.why = "; ".join(x for x in why_bits if x)
        kept.append(m)
        if len(kept) >= top_k:
            break

    # if gate too strict, fall back to top canonical/raw
    if not kept and raw:
        kept = raw[:top_k]
        for m in kept:
            m.why = (m.why or "") + "; fallback_low_pool"

    why = (
        f"factor={factor_type}; prefs={prefs[:4]}; cues={cues[:3]}; "
        f"kept={len(kept)}; rejected_low_conf={len(rejected)}"
    )
    return RouteDecision(
        factor_id=factor_id,
        factor_type=factor_type,
        preferred_coords=prefs,
        cue_coords=cues,
        matched=kept,
        rejected_low_conf=rejected[:8],
        why=why,
    )
