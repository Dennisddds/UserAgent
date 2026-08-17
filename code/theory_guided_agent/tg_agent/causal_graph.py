"""可进化的因果假设记忆网络（蓝图阶段三）。

每用户一张有向图：
- 节点：factor(事件因素模式) / coordinate(理论坐标) / evidence(历史事件) /
        value(用户价值) / emotion(情绪状态) / stance(态度结果) /
        failure(失败结构签名，非完整任务回放)
- 边（类型化）：supports / contradicts / triggers / moderates / updates / derived_from
- 边权重综合加权：语义相关度 × 置信度 × log(1+频次) × 时间新近性 × 信源可靠性 × 矛盾惩罚
- 时间衰减：半衰期指数衰减（lazy，读写时结算）
- 矛盾处理：contradicts 证据到来时降权并标记 contested
- 记忆压缩：相似 factor 节点按词面相似度聚簇合并，频次累加形成高置信记忆簇
持久化：data/users/{uid}_causal_graph.json
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

EDGE_TYPES = ["supports", "contradicts", "triggers", "moderates", "updates", "derived_from"]

# 边权重因素（蓝图：综合加权）
W_SEMANTIC = 1.0
W_CONFIDENCE = 1.0
W_FREQ = 0.4          # log(1+freq) 系数
W_RECENCY = 1.0       # 时间新近性（指数衰减后乘入）
W_RELIABILITY = 1.0
CONTRADICTION_PENALTY = 0.6   # contested 边的乘性惩罚

DEFAULT_HALF_LIFE_DAYS = 45.0  # 边权重时间半衰期


@dataclass
class GNode:
    id: str            # e.g. "factor:裁员补偿不足" / "coord:fairness" / "ev:map_id"
    kind: str          # factor | coordinate | evidence | value | emotion | stance | source
    label: str = ""
    freq: int = 1
    confidence: float = 0.5
    first_seen: float = 0.0
    last_seen: float = 0.0


@dataclass
class GEdge:
    src: str
    dst: str
    type: str          # EDGE_TYPES
    weight: float = 1.0        # 基础权重（语义相关度，外部给）
    confidence: float = 0.5
    freq: int = 1
    reliability: float = 1.0   # 信源可靠性
    contested: bool = False
    last_seen: float = 0.0     # epoch 秒

    def effective_weight(self, *, now: float | None = None, half_life_days: float = DEFAULT_HALF_LIFE_DAYS) -> float:
        now = now if now is not None else time.time()
        age_days = max(0.0, (now - (self.last_seen or now)) / 86400.0)
        recency = math.exp(-math.log(2) * age_days / max(1.0, half_life_days))
        w = (
            W_SEMANTIC * self.weight
            * (0.3 + 0.7 * W_CONFIDENCE * self.confidence)
            * (1.0 + W_FREQ * math.log1p(self.freq))
            * recency ** W_RECENCY
            * self.reliability
        )
        if self.contested:
            w *= CONTRADICTION_PENALTY
        return round(w, 4)


def _nid(kind: str, label: str) -> str:
    return f"{kind}:{label.strip()[:60]}"


class CausalGraph:
    def __init__(self, path: str | Path | None = None, *, half_life_days: float = DEFAULT_HALF_LIFE_DAYS):
        self.path = Path(path) if path else None
        self.half_life_days = half_life_days
        self.nodes: dict[str, GNode] = {}
        self.edges: dict[str, GEdge] = {}  # key = src|type|dst
        if self.path and self.path.exists():
            self._load()

    # ---------- persistence ----------
    def _load(self) -> None:
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return
        for n in d.get("nodes") or []:
            node = GNode(**{k: v for k, v in n.items() if k in GNode.__dataclass_fields__})
            self.nodes[node.id] = node
        for e in d.get("edges") or []:
            edge = GEdge(**{k: v for k, v in e.items() if k in GEdge.__dataclass_fields__})
            self.edges[self._ekey(edge.src, edge.type, edge.dst)] = edge

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        d = {
            "nodes": [asdict(n) for n in self.nodes.values()],
            "edges": [asdict(e) for e in self.edges.values()],
        }
        self.path.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")

    @staticmethod
    def _ekey(src: str, etype: str, dst: str) -> str:
        return f"{src}|{etype}|{dst}"

    # ---------- node / edge ops ----------
    def touch_node(self, kind: str, label: str, *, confidence: float = 0.5, now: float | None = None) -> GNode:
        now = now if now is not None else time.time()
        nid = _nid(kind, label)
        n = self.nodes.get(nid)
        if n is None:
            n = GNode(id=nid, kind=kind, label=label.strip()[:60], confidence=confidence,
                      first_seen=now, last_seen=now)
            self.nodes[nid] = n
        else:
            n.freq += 1
            n.last_seen = now
            n.confidence = min(1.0, 0.7 * n.confidence + 0.3 * confidence)
        return n

    def add_edge(
        self,
        src_kind: str, src_label: str,
        etype: str,
        dst_kind: str, dst_label: str,
        *,
        weight: float = 1.0,
        confidence: float = 0.5,
        reliability: float = 1.0,
        now: float | None = None,
    ) -> GEdge:
        assert etype in EDGE_TYPES, etype
        now = now if now is not None else time.time()
        src = self.touch_node(src_kind, src_label, confidence=confidence, now=now)
        dst = self.touch_node(dst_kind, dst_label, confidence=confidence, now=now)
        key = self._ekey(src.id, etype, dst.id)
        e = self.edges.get(key)
        if e is None:
            e = GEdge(src=src.id, dst=dst.id, type=etype, weight=weight,
                      confidence=confidence, reliability=reliability, last_seen=now)
            self.edges[key] = e
        else:
            e.freq += 1
            e.last_seen = now
            e.weight = min(2.0, 0.7 * e.weight + 0.3 * weight)
            e.confidence = min(1.0, 0.7 * e.confidence + 0.3 * confidence)
        return e

    def reinforce(self, src: str, etype: str, dst: str, *, good: bool, lr: float = 0.15) -> None:
        """按预测结果对错更新边（outcome feedback）。contradicts 到来时标记 contested。"""
        key = self._ekey(src, etype, dst)
        e = self.edges.get(key)
        if e is None:
            return
        if good:
            e.confidence = min(1.0, e.confidence + lr)
            if etype != "contradicts":
                e.contested = False
        else:
            e.confidence = max(0.05, e.confidence - lr)
            if etype in {"supports", "triggers"}:
                e.contested = True  # 支持边被结果证伪 → 矛盾惩罚
        e.last_seen = time.time()

    # ---------- query ----------
    def edges_from(self, nid: str, *, now: float | None = None) -> list[tuple[GEdge, float]]:
        out = [(e, e.effective_weight(now=now, half_life_days=self.half_life_days))
               for e in self.edges.values() if e.src == nid]
        out.sort(key=lambda x: -x[1])
        return out

    def paths_for_factor(self, factor_label: str, *, top_k: int = 5) -> list[dict[str, Any]]:
        """取该因素的最强先验路径：factor → coordinate（triggers）→ stance。"""
        fid = _nid("factor", factor_label)
        paths: list[dict[str, Any]] = []
        for e1, w1 in self.edges_from(fid):
            if e1.type not in {"triggers", "supports"}:
                continue
            for e2, w2 in self.edges_from(e1.dst):
                if e2.type not in {"supports", "contradicts", "moderates"}:
                    continue
                paths.append({
                    "factor": factor_label,
                    "via": e1.dst,
                    "edge1": e1.type, "w1": w1,
                    "to": e2.dst,
                    "edge2": e2.type, "w2": w2,
                    "score": round(w1 * w2, 4),
                })
        paths.sort(key=lambda p: -p["score"])
        return paths[:top_k]

    # ---------- compression ----------
    def compress(self, *, sim_threshold: float = 0.62) -> int:
        """记忆压缩：相似 factor 节点聚簇合并（频次累加、置信度取高），返回合并数。"""
        from .genminds import _tokenize, _overlap

        factors = [n for n in self.nodes.values() if n.kind == "factor"]
        merged = 0
        consumed: set[str] = set()
        toks = {n.id: _tokenize(n.label) for n in factors}
        for i, a in enumerate(factors):
            if a.id in consumed:
                continue
            for b in factors[i + 1:]:
                if b.id in consumed:
                    continue
                if _overlap(toks[a.id], toks[b.id]) >= sim_threshold:
                    # b 并入 a（保留频次高者为主）
                    if b.freq > a.freq:
                        a, b = b, a
                    a.freq += b.freq
                    a.confidence = max(a.confidence, b.confidence)
                    a.last_seen = max(a.last_seen, b.last_seen)
                    # 重挂边
                    for key, e in list(self.edges.items()):
                        if e.src == b.id:
                            e.src = a.id
                        if e.dst == b.id:
                            e.dst = a.id
                        newkey = self._ekey(e.src, e.type, e.dst)
                        if newkey != key:
                            self.edges.pop(key)
                            if newkey in self.edges:
                                self.edges[newkey].freq += e.freq
                            else:
                                self.edges[newkey] = e
                    self.nodes.pop(b.id, None)
                    consumed.add(b.id)
                    merged += 1
        return merged

    def stats(self) -> dict[str, Any]:
        kinds: dict[str, int] = {}
        for n in self.nodes.values():
            kinds[n.kind] = kinds.get(n.kind, 0) + 1
        return {"nodes": len(self.nodes), "edges": len(self.edges), "kinds": kinds}
