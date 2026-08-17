"""记忆层次结构（蓝图阶段三右侧）：短期情绪 / 中期偏好 / 长期特质 + 画像置信度动态更新。

- 短期情绪状态：近窗（默认 10 步）真实行为检测到的情绪强度，随时间衰减；
- 中期话题敏感：近窗话题频次 → 话题敏感性；
- 长期特质/persona：每属性带置信度，证据支持时升、归因错误时降，低置信属性在 prompt 里降权展示。
持久化：data/users/{uid}_memory_layers.json
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

EMOTIONS = ["anger", "joy", "sadness", "fear", "disgust", "surprise", "neutral"]
EMOTION_ZH = {
    "anger": "愤怒", "joy": "喜悦", "sadness": "悲伤", "fear": "恐惧",
    "disgust": "厌恶", "surprise": "惊讶", "neutral": "中性",
}

SHORT_WINDOW = 10          # 短期记忆窗口（步）
EMOTION_DECAY = 0.85       # 每步乘性衰减
PERSONA_CONF_LR = 0.1


@dataclass
class PersonaAttr:
    text: str
    confidence: float = 0.6
    evidence_for: int = 0
    evidence_against: int = 0


@dataclass
class MemoryLayers:
    user_id: str
    short_emotions: dict[str, float] = field(default_factory=dict)   # emotion -> intensity 0..1
    recent_topics: dict[str, int] = field(default_factory=dict)      # 中期话题计数
    persona: dict[str, PersonaAttr] = field(default_factory=dict)    # 长期画像（带置信度）
    steps: int = 0

    # ---------- 短期情绪状态 ----------
    def observe_emotions(self, probs: dict[str, float]) -> None:
        """观察一步真实情绪分布（来自真实评论检测），滚动更新短期状态。"""
        for k in list(self.short_emotions):
            self.short_emotions[k] *= EMOTION_DECAY
        for emo, p in probs.items():
            if emo not in EMOTIONS:
                continue
            self.short_emotions[emo] = min(1.0, self.short_emotions.get(emo, 0.0) + float(p) * (1 - EMOTION_DECAY) * 2)
        self.steps += 1

    def dominant_emotions(self, top_k: int = 2) -> list[tuple[str, float]]:
        items = sorted(self.short_emotions.items(), key=lambda kv: -kv[1])
        return [(e, round(v, 3)) for e, v in items[:top_k] if v > 0.08]

    # ---------- 中期话题敏感 ----------
    def observe_topic(self, topic: str) -> None:
        topic = (topic or "").strip()
        if not topic:
            return
        self.recent_topics[topic] = self.recent_topics.get(topic, 0) + 1
        # 只保留窗口内话题
        if len(self.recent_topics) > 40:
            for t in sorted(self.recent_topics, key=lambda t: self.recent_topics[t])[:10]:
                self.recent_topics.pop(t, None)

    def sensitive_topics(self, top_k: int = 5) -> list[tuple[str, int]]:
        return sorted(self.recent_topics.items(), key=lambda kv: -kv[1])[:top_k]

    # ---------- 长期画像置信度 ----------
    def seed_persona(self, attrs: list[str], *, confidence: float = 0.6) -> None:
        for a in attrs:
            a = str(a).strip()
            if a and a not in self.persona:
                self.persona[a] = PersonaAttr(text=a, confidence=confidence)

    def adjust_persona(self, attr_text: str, *, supported: bool) -> None:
        pa = self.persona.get(attr_text)
        if pa is None:
            pa = PersonaAttr(text=attr_text, confidence=0.5)
            self.persona[attr_text] = pa
        if supported:
            pa.evidence_for += 1
            pa.confidence = min(1.0, pa.confidence + PERSONA_CONF_LR)
        else:
            pa.evidence_against += 1
            pa.confidence = max(0.05, pa.confidence - PERSONA_CONF_LR)

    def persona_block(self, *, max_attrs: int = 10) -> str:
        attrs = sorted(self.persona.values(), key=lambda a: -a.confidence)[:max_attrs]
        lines = []
        for a in attrs:
            tag = "高" if a.confidence >= 0.7 else ("中" if a.confidence >= 0.4 else "低")
            lines.append(f"- [{tag}置信 {a.confidence:.2f}] {a.text}")
        return "\n".join(lines) or "- (画像未初始化)"

    # ---------- persistence ----------
    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "short_emotions": self.short_emotions,
            "recent_topics": self.recent_topics,
            "persona": {k: asdict(v) for k, v in self.persona.items()},
            "steps": self.steps,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MemoryLayers":
        ml = cls(user_id=str(d.get("user_id") or ""))
        ml.short_emotions = {k: float(v) for k, v in (d.get("short_emotions") or {}).items()}
        ml.recent_topics = {k: int(v) for k, v in (d.get("recent_topics") or {}).items()}
        ml.persona = {
            k: PersonaAttr(**{f: vv for f, vv in v.items() if f in PersonaAttr.__dataclass_fields__})
            for k, v in (d.get("persona") or {}).items()
            if isinstance(v, dict)
        }
        ml.steps = int(d.get("steps") or 0)
        return ml


def layers_path(state_dir: str | Path, uid: str) -> Path:
    return Path(state_dir) / f"{uid}_memory_layers.json"


def load_layers(state_dir: str | Path, uid: str) -> MemoryLayers:
    p = layers_path(state_dir, uid)
    if p.exists():
        try:
            return MemoryLayers.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            pass
    return MemoryLayers(user_id=uid)


def save_layers(state_dir: str | Path, ml: MemoryLayers) -> None:
    p = layers_path(state_dir, ml.user_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(ml.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")


# ---------- 情绪检测（真实评论 → 7 情绪概率，供短期状态与情绪对齐评测共用） ----------

_EMOTION_DETECT_SYSTEM = """你是情绪分析器。判断下面这条微博评论表达的情绪分布。
七个标签：anger(愤怒) joy(喜悦) sadness(悲伤) fear(恐惧) disgust(厌恶) surprise(惊讶) neutral(中性)。
给出每个标签的概率（0-1，总和为 1）。讽刺/批评通常含 anger 或 disgust；陈述事实以 neutral 为主。
只输出 JSON：{"anger":0.0,"joy":0.0,"sadness":0.0,"fear":0.0,"disgust":0.0,"surprise":0.0,"neutral":1.0}"""


def detect_emotions(client: Any, text: str) -> dict[str, float]:
    """LLM 检测文本的 7 情绪概率分布；失败时返回 neutral=1。"""
    from .agent import _parse_json

    probs = {e: 0.0 for e in EMOTIONS}
    probs["neutral"] = 1.0
    if not text.strip():
        return probs
    try:
        raw = client.chat(
            [
                {"role": "system", "content": _EMOTION_DETECT_SYSTEM},
                {"role": "user", "content": f"评论：{text[:500]}"},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        obj = _parse_json(raw)
        out = {}
        for e in EMOTIONS:
            try:
                out[e] = max(0.0, float(obj.get(e, 0.0)))
            except (TypeError, ValueError):
                out[e] = 0.0
        total = sum(out.values())
        if total <= 0:
            return probs
        return {e: round(v / total, 4) for e, v in out.items()}
    except Exception:  # noqa: BLE001
        return probs


def emotion_alignment(pred_probs: dict[str, float], gt_probs: dict[str, float]) -> float:
    """预测情绪分布 vs 真实情绪分布的对齐度：cosine 相似度（0-1）。"""
    num = sum(pred_probs.get(e, 0.0) * gt_probs.get(e, 0.0) for e in EMOTIONS)
    d1 = math.sqrt(sum(pred_probs.get(e, 0.0) ** 2 for e in EMOTIONS))
    d2 = math.sqrt(sum(gt_probs.get(e, 0.0) ** 2 for e in EMOTIONS))
    if d1 <= 0 or d2 <= 0:
        return 0.0
    return round(num / (d1 * d2), 4)
