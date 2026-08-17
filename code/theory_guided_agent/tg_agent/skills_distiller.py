"""Skills Distillation: compact reusable rules from failure memory repairs.

Motivation (会议: skills蒸馏 — 在较小训练代价得到全样本训练结果):
  - failure_memory grows unboundedly → costly to search on every step
  - Many repairs express the same underlying insight in slightly different forms
  - Distill high-frequency, high-success repairs into compact prompt rules
  - Rules are deduplicated, merged, and loaded on-demand (not all at once)

Conflict resolution:
  - Rules that contradict each other (e.g., "trust the theory" vs "trust the user")
    are flagged and the more successful one wins
  - Rules are scoped by error_stage (mining vs reasoning) to reduce interference
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DistilledRule:
    """A compact, reusable rule distilled from failure memory repairs."""

    id: str
    rule: str                       # compact natural-language rule (<=100 chars)
    source_repair_ids: list[str] = field(default_factory=list)
    error_stage: str = ""           # mining | reasoning
    primary_cause: str = ""
    factor_kinds: list[str] = field(default_factory=list)
    coordinates: list[str] = field(default_factory=list)
    # Quality
    support_count: int = 0          # number of repairs that contributed
    laplace_score: float = 0.5      # aggregate success rate
    last_updated: float = 0.0
    # Conflict detection
    conflicts_with: list[str] = field(default_factory=list)  # rule ids this contradicts
    superseded_by: str = ""         # if empty, this rule is active

    def to_prompt_line(self) -> str:
        """Render as a compact prompt injection line."""
        return f"- {self.rule} [来源:{self.support_count}次失败, 阶段:{self.error_stage}]"


@dataclass
class DistilledSkillSet:
    """A set of distilled rules, organized by error stage for prompt injection."""

    mining_rules: list[DistilledRule] = field(default_factory=list)      # factor/evidence fixes
    reasoning_rules: list[DistilledRule] = field(default_factory=list)   # theory/judgment fixes
    cross_cutting: list[DistilledRule] = field(default_factory=list)     # both stages
    metadata: dict[str, Any] = field(default_factory=dict)

    def top_rules(self, stage: str = "", max_rules: int = 5) -> list[DistilledRule]:
        """Get top rules for a stage (sorted by laplace_score × support_count)."""
        if stage == "mining":
            pool = self.mining_rules + self.cross_cutting
        elif stage == "reasoning":
            pool = self.reasoning_rules + self.cross_cutting
        else:
            pool = self.mining_rules + self.reasoning_rules + self.cross_cutting
        # Sort by score × log(support)
        import math
        ranked = sorted(pool, key=lambda r: -r.laplace_score * math.log1p(r.support_count))
        return ranked[:max_rules]

    def to_prompt_block(self, stage: str = "", max_rules: int = 5) -> str:
        rules = self.top_rules(stage=stage, max_rules=max_rules)
        if not rules:
            return ""
        lines = ["## 蒸馏策略（从历史失败中学到的可迁移规则，优先遵守）"]
        for r in rules:
            lines.append(r.to_prompt_line())
        return "\n".join(lines) + "\n"


def distill_from_failure_memory(
    fm: Any,  # FailureMemory
    *,
    min_support: int = 2,
    min_laplace: float = 0.55,
    max_rules: int = 20,
) -> DistilledSkillSet:
    """Distill FailureMemory repairs into a compact DistilledSkillSet.

    Rules are deduplicated by normalized text similarity (simple token overlap).
    Only admitted, non-revoked repairs with sufficient feedback are considered.
    """
    from .failure_memory import CAUSE_TO_STAGE

    # Collect candidate repairs
    candidates: list[tuple[Any, Any]] = []  # (repair, failure_structure)
    for r in fm.repairs.values():
        if not r.admitted or r.revoked:
            continue
        fs = fm.structures.get(r.structure_id)
        if not fs:
            continue
        score = r.score()
        n_feedback = r.success + r.fail
        if n_feedback < min_support or score < min_laplace:
            continue
        candidates.append((r, fs))

    # Generate rules from repairs
    raw_rules: list[DistilledRule] = []
    for r, fs in candidates:
        rule_text = _repair_to_rule(r, fs)
        if not rule_text:
            continue
        stage = fs.error_stage or CAUSE_TO_STAGE.get(fs.primary_cause, "none")
        raw_rules.append(DistilledRule(
            id=f"rule:{r.id.replace('repair:', '')}",
            rule=rule_text,
            source_repair_ids=[r.id],
            error_stage=stage,
            primary_cause=fs.primary_cause,
            factor_kinds=list(fs.factor_kinds),
            coordinates=list(fs.coordinates),
            support_count=fs.freq,
            laplace_score=r.score(),
            last_updated=r.last_seen or time.time(),
        ))

    # Deduplicate & merge similar rules
    merged = _deduplicate_rules(raw_rules)

    # Detect conflicts
    merged = _detect_conflicts(merged)

    # Sort and split by stage
    mining = [r for r in merged if r.error_stage == "mining"]
    reasoning = [r for r in merged if r.error_stage == "reasoning"]
    cross = [r for r in merged if r.error_stage not in {"mining", "reasoning"}]

    return DistilledSkillSet(
        mining_rules=sorted(mining, key=lambda r: -r.laplace_score)[:max_rules],
        reasoning_rules=sorted(reasoning, key=lambda r: -r.laplace_score)[:max_rules],
        cross_cutting=sorted(cross, key=lambda r: -r.laplace_score)[:max_rules],
        metadata={
            "total_candidates": len(candidates),
            "total_distilled": len(merged),
            "distilled_at": time.time(),
        },
    )


def _repair_to_rule(r: Any, fs: Any) -> str:
    """Convert a repair + failure structure into a compact natural language rule."""
    action = r.action
    payload = r.payload or {}
    if action == "strategy_note":
        note = str(payload.get("note") or "").strip()
        return note[:120] if note else ""
    if action == "demote_theory_coords":
        coords = payload.get("coordinates") or fs.coordinates or []
        return f"降低理论坐标 {', '.join(coords[:3])} 的权重，该用户对此类框架不敏感"
    if action == "boost_retrieval_kinds":
        kinds = payload.get("factor_kinds") or fs.factor_kinds or []
        return f"检索时加宽 {', '.join(kinds[:3])} 类因素的证据获取范围"
    if action == "demote_factor_kinds":
        kinds = payload.get("factor_kinds") or fs.factor_kinds or []
        return f"该用户对 {', '.join(kinds[:3])} 因素不敏感，不要以此为主推理依据"
    if action == "reset_short_term":
        boost = payload.get("recency_boost", 0.35)
        return f"该用户立场/情绪近期有变，以最近 {int(boost*100)}% 的帖子为准"
    if action == "flag_profile_attr":
        attr = str(payload.get("attr") or "")
        return f"画像属性 '{attr[:40]}' 可能不准确，需谨慎使用"
    if action == "prefer_graph_priors":
        return "优先信任该用户图中的先验路径，不要硬套群体理论"
    return ""


def _tokenize(text: str) -> set[str]:
    """Simple CJK + word tokenizer for dedup."""
    import re
    # Extract CJK chars and alphanumeric words
    tokens = set()
    for c in text:
        if '一' <= c <= '鿿' or '　' <= c <= '〿':
            tokens.add(c)
    for w in re.findall(r'[a-zA-Z0-9_]+', text):
        tokens.add(w.lower())
    return tokens


def _deduplicate_rules(rules: list[DistilledRule]) -> list[DistilledRule]:
    """Merge rules with high text overlap (>70% token Jaccard)."""
    if len(rules) <= 1:
        return rules
    merged: list[DistilledRule] = []
    used: set[int] = set()
    for i, r1 in enumerate(rules):
        if i in used:
            continue
        t1 = _tokenize(r1.rule)
        for j, r2 in enumerate(rules):
            if j <= i or j in used:
                continue
            t2 = _tokenize(r2.rule)
            if not t1 or not t2:
                continue
            jaccard = len(t1 & t2) / len(t1 | t2)
            if jaccard > 0.7:
                # Merge: keep the higher-scoring one, add support counts
                if r2.laplace_score > r1.laplace_score:
                    r1, r2 = r2, r1
                r1.support_count += r2.support_count
                r1.source_repair_ids.extend(r2.source_repair_ids)
                r1.laplace_score = max(r1.laplace_score, r2.laplace_score)
                used.add(j)
        merged.append(r1)
        used.add(i)
    return merged


def _detect_conflicts(rules: list[DistilledRule]) -> list[DistilledRule]:
    """Flag pairs of rules that could interfere."""
    conflict_keywords = [
        ({"信任", "理论"}, {"不信任", "不信", "忽略", "个体", "用户"}),
        ({"加宽", "加强", "boost"}, {"降低", "减弱", "demote", "忽略"}),
        ({"优先", "为主"}, {"不要", "避免", "谨慎"}),
    ]
    for i, r1 in enumerate(rules):
        t1 = _tokenize(r1.rule)
        for j, r2 in enumerate(rules):
            if j <= i:
                continue
            t2 = _tokenize(r2.rule)
            for set_a, set_b in conflict_keywords:
                if (t1 & set_a) and (t2 & set_b):
                    r1.conflicts_with.append(r2.id)
                    r2.conflicts_with.append(r1.id)
                    # Higher-score rule supersedes the other
                    if r1.laplace_score >= r2.laplace_score:
                        r2.superseded_by = r1.id
                    else:
                        r1.superseded_by = r2.id
                    break
    return rules


def save_distilled(skill_set: DistilledSkillSet, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "mining_rules": [
            {
                "id": r.id, "rule": r.rule, "error_stage": r.error_stage,
                "support_count": r.support_count, "laplace_score": r.laplace_score,
                "conflicts_with": r.conflicts_with, "superseded_by": r.superseded_by,
            }
            for r in skill_set.mining_rules
        ],
        "reasoning_rules": [
            {
                "id": r.id, "rule": r.rule, "error_stage": r.error_stage,
                "support_count": r.support_count, "laplace_score": r.laplace_score,
                "conflicts_with": r.conflicts_with, "superseded_by": r.superseded_by,
            }
            for r in skill_set.reasoning_rules
        ],
        "cross_cutting": [
            {
                "id": r.id, "rule": r.rule, "error_stage": r.error_stage,
                "support_count": r.support_count, "laplace_score": r.laplace_score,
                "conflicts_with": r.conflicts_with, "superseded_by": r.superseded_by,
            }
            for r in skill_set.cross_cutting
        ],
        "metadata": skill_set.metadata,
    }
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_distilled(path: str | Path) -> DistilledSkillSet | None:
    p = Path(path)
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    def _load_list(key: str) -> list[DistilledRule]:
        return [
            DistilledRule(
                id=d["id"], rule=d["rule"],
                error_stage=d.get("error_stage", ""),
                support_count=d.get("support_count", 0),
                laplace_score=d.get("laplace_score", 0.5),
                conflicts_with=d.get("conflicts_with", []),
                superseded_by=d.get("superseded_by", ""),
            )
            for d in data.get(key, [])
        ]
    return DistilledSkillSet(
        mining_rules=_load_list("mining_rules"),
        reasoning_rules=_load_list("reasoning_rules"),
        cross_cutting=_load_list("cross_cutting"),
        metadata=data.get("metadata", {}),
    )
