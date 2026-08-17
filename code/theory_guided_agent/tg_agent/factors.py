"""事件因素分解（蓝图阶段四·新事件因素分解）。

把一个新事件拆成 f1..fk 个「事件因素」，每个因素带类型：
policy(政策) / interest(利益) / moral(道德) / source(信源) / group_identity(群体身份) / emotion(情绪唤起) / other。
后续路径推理以因素为锚：因素 → 激活坐标/理论 → 历史证据 → 推出态度。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

FACTOR_TYPES = [
    "policy",          # 政策/制度因素
    "interest",        # 利益/资源分配因素
    "moral",           # 道德/价值判断因素
    "source",          # 信源/传播者因素
    "group_identity",  # 群体身份/立场阵营因素
    "emotion",         # 情绪唤起因素（事件本身的情绪钩子）
    "other",
]

_FACTOR_SYSTEM = """你是传播学事件分析器。把一个微博热点事件分解为 2-4 个「事件因素」，
每个因素是影响用户态度的一个独立维度。因素类型只能从以下选：
policy(政策/制度) | interest(利益/资源分配) | moral(道德/价值判断) | source(信源/传播者可信度) | group_identity(群体身份/阵营) | emotion(情绪唤起钩子) | other
要求：
- 每个因素是一句具体、可检验的描述（不是泛泛的"社会影响"）
- salience 为该因素对普通用户态度的影响权重，总和为 1
- 只输出 JSON，不要 markdown。
格式：{"factors":[{"id":"f1","type":"moral","text":"...","salience":0.4},...]}"""


@dataclass
class EventFactor:
    id: str
    type: str
    text: str
    salience: float = 0.25

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EventFactor":
        return cls(
            id=str(d.get("id") or "f?"),
            type=str(d.get("type") or "other"),
            text=str(d.get("text") or ""),
            salience=float(d.get("salience") or 0.25),
        )


def decompose_factors(
    llm: Any,
    stimulus: str,
    *,
    max_factors: int = 4,
    extra_context: str = "",
) -> list[EventFactor]:
    """LLM 分解事件因素；失败时退化为单因素（整个事件）。"""
    from .agent import _parse_json  # reuse robust JSON parsing

    user = f"【事件】\n{stimulus}\n"
    if extra_context:
        user += f"\n【补充语境】\n{extra_context}\n"
    user += "\n请分解。"
    try:
        raw = llm.chat(
            [
                {"role": "system", "content": _FACTOR_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            max_tokens=900,
            disable_thinking=True,
        )
        obj = _parse_json(raw)
        factors = [
            EventFactor.from_dict(d)
            for d in (obj.get("factors") or [])
            if isinstance(d, dict) and d.get("text")
        ]
        # normalize
        factors = [f for f in factors if f.type in FACTOR_TYPES or f.text]
        total = sum(f.salience for f in factors) or 1.0
        for f in factors:
            f.salience = round(f.salience / total, 3)
        factors = factors[:max_factors]
        # renumber ids deterministically
        for i, f in enumerate(factors, 1):
            f.id = f"f{i}"
        if factors:
            return factors
    except Exception:  # noqa: BLE001
        pass
    # fallback: whole event as one factor
    return [EventFactor(id="f1", type="other", text=stimulus[:200], salience=1.0)]


def ablate_stimulus(stimulus: str, factor: EventFactor) -> str:
    """反事实消融：从事件描述中剔除某因素（用于 counterfactual probe）。"""
    return (
        f"{stimulus}\n\n"
        f"[反事实设定：请假设该事件【不涉及】「{factor.text}」这一因素，"
        f"其余事实不变。]"
    )
