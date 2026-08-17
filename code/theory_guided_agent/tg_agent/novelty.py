"""Surprise-gated fast/slow routing — D-MEM / EM-LLM / Titans 本地化。

论文依据：
- D-MEM (arXiv 2603.14597): RPE 门控——routine 输入绕过昂贵的记忆演化管线（>80% token 削减）
- EM-LLM (2024): Bayesian surprise 做事件分割，training-free 新奇检测
- Titans / Nested Learning: prediction-error gating，只有「打破模式的时刻」进长时记忆

本地化为 training-free 三路信号（全程无 LLM 调用，O(近窗) 词面计算）：
  topic_novelty    话题在近期记忆中出现的频次（越少越惊讶）
  lexical_surprise 刺激与近窗历史证据的最大词面差异
  prior_strength   因果图中是否已有可复用先验路径（有则不惊讶）
"""

from __future__ import annotations

import math
from typing import Any


def _topic_novelty(agent: Any, topic: str) -> float:
    if not topic:
        return 0.5  # 无话题信息时保持中性，不单独决定路由
    try:
        count = int(agent.layers.recent_topics.get(topic, 0))
    except Exception:  # noqa: BLE001
        return 0.5
    if count <= 0:
        return 1.0
    return round(1.0 / (1.0 + math.log1p(count)), 4)


def _lexical_surprise(agent: Any, stimulus: str, recent_window: int) -> float:
    from .genminds import _overlap, _tokenize

    q = _tokenize(stimulus)
    if not q:
        return 0.5
    try:
        toks_list = list(agent.memory._event_tokens[-recent_window:])
    except Exception:  # noqa: BLE001
        return 0.5
    if not toks_list:
        return 1.0  # 没有任何历史 → 一切皆新
    best = max(_overlap(q, t) for t in toks_list)
    return round(1.0 - best, 4)


def _prior_strength(agent: Any, stimulus: str) -> float:
    try:
        priors = agent.graph.paths_for_factor(stimulus[:200], top_k=2)
    except Exception:  # noqa: BLE001
        return 0.0
    if not priors:
        return 0.0
    best = max(float(p.get("score") or 0.0) for p in priors)
    return round(max(0.0, min(1.0, best)), 4)


def compute_surprise(
    agent: Any,
    *,
    stimulus: str,
    topic: str = "",
    recent_window: int = 50,
    w_topic: float = 0.35,
    w_lexical: float = 0.35,
    w_prior: float = 0.30,
) -> dict[str, Any]:
    """返回 surprise ∈ [0,1] 与分信号；route 由调用方按阈值决定。"""
    tn = _topic_novelty(agent, topic)
    ls = _lexical_surprise(agent, stimulus, recent_window)
    ps = _prior_strength(agent, stimulus)
    surprise = w_topic * tn + w_lexical * ls + w_prior * (1.0 - ps)
    return {
        "surprise": round(surprise, 4),
        "topic_novelty": tn,
        "lexical_surprise": ls,
        "prior_strength": ps,
        "weights": {"topic": w_topic, "lexical": w_lexical, "prior": w_prior},
    }
