from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from .models import RetrievedEvent


def _tokenize(text: str) -> set[str]:
    text = text.lower()
    hans = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    words = re.findall(r"[a-z0-9_]{3,}", text)
    grams: set[str] = set(words)
    for h in hans:
        grams.add(h)
        if len(h) >= 4:
            for i in range(len(h) - 1):
                grams.add(h[i : i + 2])
    return grams


def _overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / math.sqrt(len(a) * len(b))


class GenMindsMemory:
    """Load best-method GenMinds bank; lexical retrieve without heavy deps."""

    def __init__(self, path: str | Path, persona_path: str | Path | None = None) -> None:
        self.path = Path(path)
        with self.path.open(encoding="utf-8") as f:
            self.bank: dict[str, Any] = json.load(f)
        self.static = self.bank.get("static_map", {})
        self.events: list[dict[str, Any]] = self.bank.get("event_maps", [])
        self.persona: dict[str, Any] = {}
        if persona_path and Path(persona_path).exists():
            with Path(persona_path).open(encoding="utf-8") as f:
                self.persona = json.load(f)
        self.identity = self._load_identity(persona_path)
        self._rebuild_event_index()

    def _load_identity(self, persona_path: str | Path | None) -> dict[str, Any]:
        """Ground-truth account identity (name), not LLM-inferred role."""
        candidates: list[Path] = []
        if persona_path:
            p = Path(persona_path)
            candidates.append(p.with_name("data_preview.json"))
            # outputs/weibo_user_{uid}/persona.json → sibling data_preview
            candidates.append(p.parent / "data_preview.json")
        # common layout
        uid = ""
        if persona_path:
            m = re.search(r"(\d{5,})", str(persona_path))
            if m:
                uid = m.group(1)
        if uid:
            candidates.append(Path(f"d:/UserAgent/outputs/weibo_user_{uid}/data_preview.json"))
        for c in candidates:
            if c.exists():
                try:
                    d = json.loads(c.read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001
                    continue
                name = str(d.get("user_name") or d.get("screen_name") or "").strip()
                if name:
                    return {
                        "user_id": str(d.get("user_id") or uid),
                        "user_name": name,
                        "source": str(c),
                    }
        # last-resort display name only (no persona-specific rules)
        known_names = {"1989660417": "胡锡进"}
        m = re.search(r"(\d{5,})", str(self.path))
        if m and m.group(1) in known_names:
            return {"user_id": m.group(1), "user_name": known_names[m.group(1)], "source": "builtin"}
        return {}

    def persona_voice_hints(self) -> list[str]:
        """Extract communication / self-reference cues from persona — not user-specific hardcodes."""
        hints: list[str] = []
        for src in (
            self.persona.get("communication") or [],
            (self.persona.get("analysis") or {}).get("communication")
            if isinstance(self.persona.get("analysis"), dict)
            else [],
            self.static.get("communication") or [],
        ):
            if isinstance(src, list):
                for x in src:
                    s = str(x).strip()
                    if s and s not in hints:
                        hints.append(s)
            elif isinstance(src, str) and src.strip():
                hints.append(src.strip())
        return hints[:8]

    def identity_block(self) -> str:
        """Describe who the agent is playing: account identity + persona voice (generic for any 大V)."""
        name = (self.identity or {}).get("user_name") or ""
        uid = (self.identity or {}).get("user_id") or ""
        lines: list[str] = []
        if name:
            lines.append(f"真实身份/昵称={name}（user_id={uid or 'unknown'}）。")
            lines.append(
                "硬约束：你就是该账号本人发帖，不是路人转述该大V；"
                "预测评论必须用其本人声口与惯用自称，勿写成旁观者点评。"
            )
        else:
            lines.append(f"user_id={uid or 'unknown'}（姓名未知）。请严格依据下方人设与历史发声。")

        voice = self.persona_voice_hints()
        if voice:
            lines.append("人设表达特征（来自 persona，须遵守）：")
            for h in voice:
                lines.append(f"- {h}")
            # Generic: third-person self-reference is first-person voice for that account
            blob = " ".join(voice)
            blob_l = blob.lower()
            if any(
                k in blob_l
                for k in (
                    "third-person",
                    "self-reference",
                    "third person",
                    "attribution",
                )
            ) or any(k in blob for k in ("自称", "第三人称", "以己名", "惯用自称")):
                lines.append(
                    "注意：若人设含第三人称自称（如「某某认为」），那是本人惯用第一人称写法，"
                    "不是在引用另一个人。"
                )
        return "\n".join(lines)

    def _event_token_blob(self, e: dict[str, Any]) -> str:
        return " ".join(
            [
                str(e.get("event_title") or ""),
                str(e.get("event_summary") or ""),
                str(e.get("user_opinion") or ""),
                " ".join(e.get("topics") or []),
                " ".join(e.get("entities") or []),
                " ".join(e.get("stance_keywords") or []),
                str(e.get("feature_2d_text") or ""),
            ]
        )

    def _rebuild_event_index(self) -> None:
        self._event_tokens = [_tokenize(self._event_token_blob(e)) for e in self.events]
        # Build causal concept index for graph-boosted retrieval
        self._causal_index: dict[str, list[int]] = {}
        for i, e in enumerate(self.events):
            for c in e.get("causal_concepts") or []:
                c = str(c).strip()
                if c and len(c) >= 2:
                    self._causal_index.setdefault(c, []).append(i)
            for edge in e.get("belief_edges") or []:
                for role in ("src", "dst"):
                    c = str(edge.get(role, "")).strip()
                    if c and len(c) >= 2:
                        self._causal_index.setdefault(c, []).append(i)

    def clone_for_sequential(self, *, keep_static: bool = True) -> "GenMindsMemory":
        """Fresh agent memory: persona/static priors, no past events yet."""
        import copy

        mem = GenMindsMemory.__new__(GenMindsMemory)
        mem.path = self.path
        mem.bank = copy.deepcopy(self.bank)
        if keep_static:
            mem.static = mem.bank.setdefault("static_map", {})
        else:
            mem.static = {}
            mem.bank["static_map"] = mem.static
        mem.events = []
        mem.bank["event_maps"] = mem.events
        mem.persona = copy.deepcopy(self.persona)
        mem.identity = copy.deepcopy(self.identity)
        mem._rebuild_event_index()
        return mem

    def ingest_event(self, event: dict[str, Any]) -> None:
        """Append one observed post/event (after prediction) to grow the agent."""
        e = dict(event)
        if not e.get("map_id"):
            e["map_id"] = str(e.get("post_id") or f"step_{len(self.events)}")
        # avoid duplicate post ingest
        pid = str(e.get("post_id") or "")
        if pid and any(str(x.get("post_id") or "") == pid for x in self.events):
            return
        self.events.append(e)
        self._event_tokens.append(_tokenize(self._event_token_blob(e)))

    @property
    def beliefs(self) -> list[str]:
        return list(self.static.get("beliefs") or [])

    @property
    def values(self) -> list[str]:
        return list(self.static.get("persona_values") or self.persona.get("values") or [])

    @property
    def interests(self) -> list[str]:
        return list(self.static.get("persona_interests") or self.persona.get("interests") or [])

    @property
    def communication(self) -> list[str]:
        return list(self.static.get("communication") or self.persona.get("communication") or [])

    @property
    def motifs(self) -> list[str]:
        return list(self.static.get("cognitive_motifs") or [])

    def retrieve(
        self,
        query: str,
        top_k: int = 6,
        *,
        recency_boost: float = 0.0,
    ) -> list[RetrievedEvent]:
        """Hybrid retrieve: lexical overlap + causal graph boost from belief_edges."""
        q = _tokenize(query)
        n = len(self.events)
        scored: dict[int, float] = {}
        # 1) Lexical base score
        for i, toks in enumerate(self._event_tokens):
            s = _overlap(q, toks)
            if s > 0:
                scored[i] = s
        # 2) Causal graph boost: query concepts → causally-linked events
        causal_index = getattr(self, "_causal_index", {})
        if causal_index:
            query_concepts = {c for c in causal_index if len(c) >= 3 and c in query}
            graph_hits: dict[int, float] = {}
            for c in query_concepts:
                for i in causal_index.get(c, []):
                    graph_hits[i] = graph_hits.get(i, 0.0) + 0.08
            for i, boost in graph_hits.items():
                scored[i] = scored.get(i, 0.01) + min(boost, 0.30)
        # 3) Recency boost
        if recency_boost > 0 and n > 1:
            for i in scored:
                scored[i] *= (1.0 + recency_boost * (i / (n - 1)))
        ranked = sorted(scored.items(), key=lambda x: -x[1])
        out: list[RetrievedEvent] = []
        for i, s in ranked[:top_k]:
            e = self.events[i]
            text = e.get("feature_2d_text") or e.get("event_summary") or e.get("user_opinion") or ""
            out.append(
                RetrievedEvent(
                    map_id=str(e.get("map_id") or e.get("post_id") or i),
                    text=str(text)[:600],
                    score=float(s),
                    event_title=str(e.get("event_title") or ""),
                    user_opinion=str(e.get("user_opinion") or ""),
                    topics=list(e.get("topics") or []),
                )
            )
        return out

    def u_snapshot(self, max_motifs: int = 8) -> dict[str, Any]:
        return {
            "method": self.bank.get("method", "GenMinds"),
            "num_events": len(self.events),
            "beliefs": self.beliefs[:12],
            "motifs": self.motifs[:max_motifs],
            "interests": self.interests[:10],
            "communication": self.communication[:6],
        }

    def v_snapshot(self) -> dict[str, Any]:
        return {
            "persona_values": self.values[:12],
            "entity_stance_sample": dict(list((self.static.get("entity_stance") or {}).items())[:8]),
        }
