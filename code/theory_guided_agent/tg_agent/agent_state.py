"""Typed workflow state for PathAgent (LangGraph paper: durable shared state).

Fields mirror product events that belong in route history / repair decisions,
not hidden prompt logic.
"""

from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    # --- inputs ---
    stimulus: str
    post_id: str
    bid: str
    date: str
    topic: str

    # --- context ---
    u_snapshot: dict[str, Any]
    v_snapshot: dict[str, Any]
    sit: dict[str, Any] | None
    sit_weights: dict[str, float]
    situational_missing: bool
    source_block: str

    # --- intermediate artifacts ---
    factors: list[Any]
    per_factor: list[dict[str, Any]]
    all_matched: dict[str, Any]
    all_events: dict[str, Any]
    paths: list[dict[str, Any]]
    stance: str
    emotion_probs: dict[str, float]
    opinion: str
    reason: str  # fusion 模式：图推理的自然语言说明（理由即图推理本身）
    low_evidence_factors: list[str]
    skeptic: dict[str, Any]
    confidence: float
    low_evidence: bool
    verbalization: str

    # --- evidence / repair gating (agentic RAG analogue) ---
    evidence_grade: str  # strong | weak | fail
    repair_attempts: int
    repair_budget: int
    active_repairs: list[dict[str, Any]]
    repairs_applied: list[str]
    reformulated_queries: list[str]  # repair 轮 LLM 改写出的替代检索查询
    strategy_notes: list[str]  # ASPIRE 式准入策略笔记（注入合成 prompt）

    # --- control ---
    route_history: list[str]
    current_node: str
    status: str
    caveats: list[str]
    output: Any


def new_agent_state(
    *,
    stimulus: str,
    post_id: str = "",
    bid: str = "",
    date: str = "",
    topic: str = "",
    repair_budget: int = 1,
) -> AgentState:
    return AgentState(
        stimulus=stimulus,
        post_id=post_id,
        bid=bid,
        date=date,
        topic=topic,
        sit=None,
        sit_weights={},
        situational_missing=False,
        factors=[],
        per_factor=[],
        all_matched={},
        all_events={},
        paths=[],
        stance="uncertain",
        emotion_probs={},
        opinion="",
        low_evidence_factors=[],
        skeptic={},
        confidence=0.0,
        low_evidence=False,
        verbalization="",
        evidence_grade="weak",
        repair_attempts=0,
        repair_budget=repair_budget,
        active_repairs=[],
        repairs_applied=[],
        route_history=[],
        status="ok",
        caveats=[],
    )
