"""Failure-structure memory: learn conditional repairs, not full-task copies.

Update contract (user requirement):
  1) identify repeated failure structures
  2) learn conditional repairs
  3) retrieve + compose repairs for new tasks
  — never memorize and replay a complete past task episode.

A FailureStructure is an abstracted signature:
  (primary_cause, factor_kind_multiset, coordinate_set)
A ConditionalRepair is a small actionable delta applied when the signature
matches the current situation (retrieval boost, theory demote, etc.).
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


CAUSES = (
    "factor_extraction",
    "retrieval",
    "theory_prior",
    "profile",
    "short_term_state",
    "context_shift",
    "none",
)

# RTWI-style stage decomposition: mining (cue/evidence gathering) vs reasoning (using evidence)
ERROR_STAGES = ("mining", "reasoning", "none")

# Mapping: primary_cause → error_stage
CAUSE_TO_STAGE: dict[str, str] = {
    "factor_extraction": "mining",
    "retrieval": "mining",
    "theory_prior": "reasoning",
    "profile": "reasoning",
    "short_term_state": "reasoning",
    "context_shift": "reasoning",
    "none": "none",
}

REPAIR_ACTIONS = (
    "demote_theory_coords",
    "boost_retrieval_kinds",
    "demote_factor_kinds",
    "reset_short_term",
    "flag_profile_attr",
    "prefer_graph_priors",
    "strategy_note",
)


def _sig_hash(parts: list[str]) -> str:
    raw = "|".join(parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


@dataclass
class FailureStructure:
    """Abstract repeated failure pattern — not a full task transcript."""

    id: str
    primary_cause: str
    error_stage: str = ""  # RTWI-style: "mining" (factor/evidence) | "reasoning" (using info)
    factor_kinds: list[str] = field(default_factory=list)  # sorted multiset of kinds
    coordinates: list[str] = field(default_factory=list)  # sorted unique coords
    freq: int = 1
    last_seen: float = 0.0
    examples_oa: list[float] = field(default_factory=list)  # keep last few scores only
    # ASPIRE skill 格式：when-to-apply 条件描述 + 一行来源摘要（非完整任务回放）
    when_to_apply: str = ""
    exemplar: str = ""
    # ── Thinking-level fields (new) ──
    thinking_error_step: str = ""      # e.g. "theory_application", "evidence_query"
    thinking_error_type: str = ""      # "omission" | "overgeneralization" | "contradiction" | "bias"
    thinking_correction: str = ""      # what the model should have thought instead

    @classmethod
    def from_episode(
        cls,
        *,
        primary_cause: str,
        factor_kinds: list[str],
        coordinates: list[str],
        oa: float,
        when_to_apply: str = "",
        exemplar: str = "",
        error_stage: str = "",
        thinking_error_step: str = "",
        thinking_error_type: str = "",
        thinking_correction: str = "",
    ) -> "FailureStructure":
        kinds = sorted(k for k in factor_kinds if k)
        coords = sorted(set(c for c in coordinates if c))
        fid = "fail:" + _sig_hash([primary_cause, ",".join(kinds), ",".join(coords)])
        stage = error_stage if error_stage in ERROR_STAGES else CAUSE_TO_STAGE.get(primary_cause, "none")
        return cls(
            id=fid,
            primary_cause=primary_cause,
            error_stage=stage,
            factor_kinds=kinds,
            coordinates=coords,
            freq=1,
            last_seen=time.time(),
            examples_oa=[round(oa, 3)],
            when_to_apply=when_to_apply[:120],
            exemplar=exemplar[:120],
            thinking_error_step=thinking_error_step,
            thinking_error_type=thinking_error_type,
            thinking_correction=thinking_correction[:200],
        )

    def overlap(self, kinds: list[str], coords: list[str], cause: str | None = None, error_stage: str = "") -> float:
        """Structural similarity in [0,1] for retrieval (not episode identity)."""
        k1, k2 = set(self.factor_kinds), set(kinds)
        c1, c2 = set(self.coordinates), set(coords)
        kind_j = (len(k1 & k2) / len(k1 | k2)) if (k1 or k2) else 0.0
        coord_j = (len(c1 & c2) / len(c1 | c2)) if (c1 or c2) else 0.0
        cause_bonus = 0.25 if cause and cause == self.primary_cause else 0.0
        stage_bonus = 0.12 if error_stage and error_stage == self.error_stage else 0.0
        return round(0.40 * kind_j + 0.25 * coord_j + cause_bonus + stage_bonus, 4)


@dataclass
class ConditionalRepair:
    """If structure matches under conditions → apply composable deltas."""

    id: str
    structure_id: str
    action: str
    # action payload (small, composable — never a full prompt/task dump)
    payload: dict[str, Any] = field(default_factory=dict)
    # firing conditions
    min_overlap: float = 0.28
    require_cause: str = ""  # optional hard filter at apply time
    # ASPIRE 准入闸门：一次性失败是噪声；同 cause 失败复发 ≥N 次才 admitted 可组合
    admitted: bool = False
    # 负迁移防线：应用后反馈持续为差 → 永久撤销（不再被复发准入重新激活）
    revoked: bool = False
    success: int = 0
    fail: int = 0
    last_seen: float = 0.0

    def score(self) -> float:
        n = self.success + self.fail
        if n == 0:
            return 0.5
        return round((self.success + 1) / (n + 2), 4)  # Laplace


@dataclass
class FailureMemory:
    user_id: str
    structures: dict[str, FailureStructure] = field(default_factory=dict)
    repairs: dict[str, ConditionalRepair] = field(default_factory=dict)
    # ASPIRE 准入：同 cause 失败总频次 >= 此值，该 cause 的 repairs 才 admitted
    admission_min_cause_freq: int = 2
    # ablation / 负迁移防线（config failure_memory 段，run_sequential --fm-mode 覆盖）
    enabled: bool = True                 # False = 错题本整体关闭（V-off 变体）
    weight_repairs_enabled: bool = True  # False = 只学 strategy_note 软建议，不动权重（V-notes 变体）
    revoke_on_fail: bool = False         # True = 应用后反馈持续差则永久撤销准入
    revoke_min_feedback: int = 3         # 撤销所需最少反馈次数
    revoke_max_score: float = 0.4        # Laplace 分低于此值才撤销

    # ---------- persistence ----------
    @classmethod
    def load(cls, state_dir: str | Path, user_id: str) -> "FailureMemory":
        path = Path(state_dir) / f"{user_id}_failure_memory.json"
        mem = cls(user_id=user_id)
        if not path.exists():
            return mem
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return mem
        for s in d.get("structures") or []:
            fs = FailureStructure(**{k: v for k, v in s.items() if k in FailureStructure.__dataclass_fields__})
            mem.structures[fs.id] = fs
        for r in d.get("repairs") or []:
            legacy = "admitted" not in r  # 旧版存档无准入字段：保持旧的「总是可用」行为
            cr = ConditionalRepair(**{k: v for k, v in r.items() if k in ConditionalRepair.__dataclass_fields__})
            if legacy:
                cr.admitted = True
            mem.repairs[cr.id] = cr
        return mem

    def save(self, state_dir: str | Path) -> None:
        path = Path(state_dir) / f"{self.user_id}_failure_memory.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "user_id": self.user_id,
                    "structures": [asdict(s) for s in self.structures.values()],
                    "repairs": [asdict(r) for r in self.repairs.values()],
                    # explicit contract marker
                    "policy": "structure+conditional_repair+compose; never full-task replay",
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )

    # ---------- learn from failure ----------
    def observe_failure(
        self,
        *,
        primary_cause: str,
        factor_kinds: list[str],
        coordinates: list[str],
        oa: float,
        detail: str = "",
        implicated_profile_attr: str = "",
        strategy: str = "",
        exemplar: str = "",
        error_stage: str = "",
        thinking_error_step: str = "",
        thinking_error_type: str = "",
        thinking_correction: str = "",
    ) -> tuple[FailureStructure, ConditionalRepair]:
        cause = primary_cause if primary_cause in CAUSES else "none"
        fs = FailureStructure.from_episode(
            primary_cause=cause,
            factor_kinds=factor_kinds,
            coordinates=coordinates,
            oa=oa,
            when_to_apply=self._when_to_apply(cause, factor_kinds, coordinates),
            exemplar=exemplar,
            error_stage=error_stage,
            thinking_error_step=thinking_error_step,
            thinking_error_type=thinking_error_type,
            thinking_correction=thinking_correction,
        )
        if not self.enabled:
            # 错题本关闭（ablation）：照常返回结构供日志记录，但不落库、不学修复
            return fs, ConditionalRepair(
                id="repair:disabled", structure_id=fs.id, action="prefer_graph_priors"
            )
        if fs.id in self.structures:
            old = self.structures[fs.id]
            old.freq += 1
            old.last_seen = time.time()
            old.examples_oa = (old.examples_oa + [round(oa, 3)])[-8:]
            fs = old
        else:
            self.structures[fs.id] = fs

        repair = self._default_repair_for(fs, detail=detail, implicated_profile_attr=implicated_profile_attr)
        if self.weight_repairs_enabled:
            if repair.id in self.repairs:
                existing = self.repairs[repair.id]
                existing.last_seen = time.time()
                # merge payload keys lightly
                for k, v in repair.payload.items():
                    if k not in existing.payload:
                        existing.payload[k] = v
                repair = existing
            else:
                self.repairs[repair.id] = repair
        # 权重修复关闭（V-notes）：只学 strategy_note 软建议，不动检索/坐标权重

        # ASPIRE strategy_note：可迁移自然语言策略作为异构修复知识（权重 delta 之外）
        if strategy.strip():
            note_id = "repair:" + _sig_hash([fs.id, "strategy_note", strategy[:40]])
            if note_id not in self.repairs:
                self.repairs[note_id] = ConditionalRepair(
                    id=note_id,
                    structure_id=fs.id,
                    action="strategy_note",
                    payload={"note": strategy.strip()[:160]},
                    require_cause=cause if cause != "none" else "",
                    last_seen=time.time(),
                )

        # 准入闸门：同 cause 失败复发 >= 阈值 → 该 cause 全部 repairs 准入（一次性失败是噪声）
        # 已撤销（revoked）的修复不再重新准入——负迁移防线
        cause_freq = sum(s.freq for s in self.structures.values() if s.primary_cause == cause)
        if cause_freq >= max(1, int(self.admission_min_cause_freq)):
            for r in self.repairs.values():
                s = self.structures.get(r.structure_id)
                if s and s.primary_cause == cause and not r.revoked:
                    r.admitted = True
        return fs, repair

    @staticmethod
    def _when_to_apply(cause: str, kinds: list[str], coords: list[str], error_stage: str = "") -> str:
        """when-to-apply guard 的紧凑描述（ASPIRE skill 格式 + RTWI 阶段标签）。"""
        parts = []
        if error_stage and error_stage != "none":
            stage_label = "因素/证据层" if error_stage == "mining" else "推理/判断层"
            parts.append(f"错在{stage_label}")
        ks = sorted({k for k in kinds if k})
        if ks:
            parts.append("因素含 " + "/".join(ks[:4]))
        cs = sorted({c for c in coords if c})
        if cs:
            parts.append("召回坐标含 " + "/".join(cs[:3]))
        if cause and cause != "none":
            parts.append(f"历史错因={cause}")
        return "；".join(parts)[:120]

    def _default_repair_for(
        self,
        fs: FailureStructure,
        *,
        detail: str = "",
        implicated_profile_attr: str = "",
    ) -> ConditionalRepair:
        cause = fs.primary_cause
        stage = fs.error_stage
        # RTWI-style: mining-stage repairs fix retrieval/factor gaps; reasoning-stage repairs fix misuse
        if cause == "theory_prior":
            action, payload = "demote_theory_coords", {"coordinates": fs.coordinates, "delta": -0.15, "stage": "reasoning"}
        elif cause == "retrieval":
            action, payload = "boost_retrieval_kinds", {"factor_kinds": fs.factor_kinds, "top_k_boost": 2, "stage": "mining"}
        elif cause == "factor_extraction":
            action, payload = "demote_factor_kinds", {"factor_kinds": fs.factor_kinds, "delta": -0.1, "stage": "mining"}
        elif cause == "short_term_state":
            action, payload = "reset_short_term", {"scale": 0.5, "recency_boost": 0.35, "stage": "reasoning"}
        elif cause == "context_shift":
            # 用户立场/语境漂移：旧证据失真 → 更强地压旧提新
            action, payload = "reset_short_term", {"scale": 0.3, "recency_boost": 0.6, "stage": "reasoning"}
        elif cause == "profile":
            action, payload = "flag_profile_attr", {"attr": implicated_profile_attr or detail[:40], "stage": "reasoning"}
        else:
            action, payload = "prefer_graph_priors", {"top_k": 3}
        rid = "repair:" + _sig_hash([fs.id, action, json.dumps(payload, sort_keys=True, ensure_ascii=False)])
        return ConditionalRepair(
            id=rid,
            structure_id=fs.id,
            action=action,
            payload=payload,
            require_cause=cause if cause != "none" else "",
            last_seen=time.time(),
        )

    # ---------- retrieve + compose for new task ----------
    def retrieve_repairs(
        self,
        *,
        factor_kinds: list[str],
        coordinates: list[str],
        top_k: int = 3,
        min_overlap: float = 0.28,
    ) -> list[tuple[ConditionalRepair, FailureStructure, float]]:
        if not self.enabled:
            return []
        hits: list[tuple[ConditionalRepair, FailureStructure, float]] = []
        for r in self.repairs.values():
            if not r.admitted or r.revoked:
                continue  # ASPIRE 准入闸门 + 撤销防线
            fs = self.structures.get(r.structure_id)
            if not fs:
                continue
            ov = fs.overlap(factor_kinds, coordinates)
            thresh = max(min_overlap, float(r.min_overlap or 0.0))
            if ov < thresh:
                continue
            # prefer frequent structures + historically successful repairs
            score = ov * (0.5 + 0.5 * r.score()) * (1.0 + 0.1 * min(10, fs.freq))
            hits.append((r, fs, round(score, 4)))
        hits.sort(key=lambda x: -x[2])
        return hits[:top_k]

    def feedback(self, repair_ids: list[str], *, helpful: bool) -> None:
        for rid in repair_ids:
            r = self.repairs.get(rid)
            if not r:
                continue
            if helpful:
                r.success += 1
            else:
                r.fail += 1
            r.last_seen = time.time()
            # 负迁移防线：应用后反馈持续为差 → 永久撤销（ relapse 只能靠新失败重新学出新修复）
            if (
                self.revoke_on_fail
                and r.admitted
                and not r.revoked
                and (r.success + r.fail) >= max(1, int(self.revoke_min_feedback))
                and r.score() < float(self.revoke_max_score)
            ):
                r.admitted = False
                r.revoked = True

    def match_structure(
        self,
        *,
        factor_kinds: list[str],
        coordinates: list[str],
        min_overlap: float = 0.6,
    ) -> FailureStructure | None:
        """高重叠失败结构检索：命中则可复用其归因，省一次归因 LLM 调用。"""
        best: FailureStructure | None = None
        best_ov = 0.0
        for fs in self.structures.values():
            ov = fs.overlap(factor_kinds, coordinates)
            if ov >= min_overlap and ov > best_ov:
                best, best_ov = fs, ov
        return best

    def prune_to_budget(self, *, max_structures: int = 80, max_repairs: int = 120) -> dict[str, int]:
        """错题本容量约束：保留对未来仍有价值的失败结构（频次×修复成功率×新近性）。

        Never keep full-task transcripts — only compact structures + repairs.
        """
        before_s, before_r = len(self.structures), len(self.repairs)
        now = time.time()

        def struct_value(s: FailureStructure) -> float:
            age_days = max(0.0, (now - (s.last_seen or now)) / 86400.0)
            recency = math.exp(-age_days / 60.0)
            # linked repair success
            linked = [r for r in self.repairs.values() if r.structure_id == s.id]
            succ = sum(r.score() for r in linked) / max(1, len(linked))
            return s.freq * (0.4 + 0.6 * succ) * (0.3 + 0.7 * recency)

        if len(self.structures) > max_structures:
            ranked = sorted(self.structures.values(), key=struct_value, reverse=True)
            keep = {s.id for s in ranked[:max_structures]}
            self.structures = {k: v for k, v in self.structures.items() if k in keep}
            self.repairs = {
                k: v for k, v in self.repairs.items() if v.structure_id in keep
            }

        if len(self.repairs) > max_repairs:
            def repair_value(r: ConditionalRepair) -> float:
                fs = self.structures.get(r.structure_id)
                freq = float(fs.freq) if fs else 1.0
                age_days = max(0.0, (now - (r.last_seen or now)) / 86400.0)
                recency = math.exp(-age_days / 60.0)
                return r.score() * (1.0 + math.log1p(freq)) * (0.3 + 0.7 * recency)

            ranked_r = sorted(self.repairs.values(), key=repair_value, reverse=True)
            keep_r = {r.id for r in ranked_r[:max_repairs]}
            self.repairs = {k: v for k, v in self.repairs.items() if k in keep_r}

        return {
            "pruned_structures": before_s - len(self.structures),
            "pruned_repairs": before_r - len(self.repairs),
            "structures": len(self.structures),
            "repairs": len(self.repairs),
        }

    def stats(self) -> dict[str, int]:
        return {
            "structures": len(self.structures),
            "repairs": len(self.repairs),
            "admitted_repairs": sum(1 for r in self.repairs.values() if r.admitted),
            "revoked_repairs": sum(1 for r in self.repairs.values() if r.revoked),
            "repeated_structures": sum(1 for s in self.structures.values() if s.freq >= 2),
        }

    def repair_effectiveness(self) -> dict[str, Any]:
        """Aggregate repair effectiveness metrics for ablation reporting."""
        admitted = [r for r in self.repairs.values() if r.admitted and not r.revoked]
        if not admitted:
            return {"total_admitted": 0, "mean_score": 0.0, "by_action": {}, "by_stage": {}}
        scores = [r.score() for r in admitted]
        by_action: dict[str, dict[str, float]] = {}
        by_stage: dict[str, dict[str, float]] = {}
        for r in admitted:
            fs = self.structures.get(r.structure_id)
            stage = fs.error_stage if fs else "none"
            action = r.action
            if action not in by_action:
                by_action[action] = {"count": 0, "total_success": 0, "total_fail": 0}
            by_action[action]["count"] += 1
            by_action[action]["total_success"] += r.success
            by_action[action]["total_fail"] += r.fail
            if stage not in by_stage:
                by_stage[stage] = {"count": 0, "mean_score": 0.0, "scores": []}
            by_stage[stage]["count"] += 1
            by_stage[stage]["scores"].append(r.score())
        for stage, data in by_stage.items():
            data["mean_score"] = round(sum(data["scores"]) / max(1, len(data["scores"])), 4)
            del data["scores"]
        for action, data in by_action.items():
            n = data["total_success"] + data["total_fail"]
            data["laplace_score"] = round((data["total_success"] + 1) / (n + 2), 4) if n > 0 else 0.5
        return {
            "total_admitted": len(admitted),
            "mean_score": round(sum(scores) / len(scores), 4),
            "by_action": by_action,
            "by_stage": by_stage,
        }
