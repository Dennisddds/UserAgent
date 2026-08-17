# -*- coding: utf-8 -*-
"""Independent builders for personality-detection graph papers."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any

from ..common import base_event_fields, build_static_from_persona, encode_text, write_memory_bank

# Compact LIWC-like Chinese psycholinguistic categories (TrigNet / Semi-PerGCN)
LIWC_CATS: dict[str, list[str]] = {
    "affect": ["爱", "恨", "喜", "怒", "悲", "恐", "惊", "感动", "愤怒", "失望", "欣慰", "自豪"],
    "social": ["我们", "他们", "朋友", "家人", "社会", "公众", "网友", "人民", "大家"],
    "cogmech": ["因为", "所以", "但是", "如果", "认为", "觉得", "知道", "理解", "思考", "分析"],
    "percept": ["看", "听", "感觉", "观察", "注意", "发现"],
    "drives": ["成就", "权力", "风险", "安全", "利益", "目标", "成功", "失败"],
    "relativ": ["今天", "现在", "过去", "未来", "这里", "那里", "之前", "之后"],
    "bio": ["身体", "健康", "生命", "死亡", "疾病"],
    "informal": ["哈哈", "嗯", "吧", "啊", "嘛", "啦"],
    "work": ["工作", "经济", "增长", "就业", "市场", "企业", "政策"],
    "money": ["钱", "价格", "成本", "财富", "贫困", "收入"],
    "relig": ["信仰", "宗教", "神"],
    "death": ["死", "牺牲", "遇难"],
    "swear": ["滚", "傻", "蠢"],
    "assent": ["是的", "对", "同意", "支持"],
    "negate": ["不", "没", "非", "无", "别"],
}


def _psych_words(text: str) -> list[tuple[str, str]]:
    hits = []
    for cat, words in LIWC_CATS.items():
        for w in words:
            if w in text:
                hits.append((w, cat))
    return hits


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def build_ddgcn(user_id: str, bundle: dict[str, Any]):
    """D-DGCN: user hub + posts; L2C-style similarity edges (τ=0.5)."""
    events = bundle["events"]
    static = build_static_from_persona(bundle["persona"], events)
    # Encode each post for dynamic connectivity
    vecs = []
    for ev in events:
        text = " ".join(
            [
                str(ev.get("raw_text") or ""),
                str(ev.get("event_summary") or ""),
                str(ev.get("user_opinion") or ""),
            ]
        )
        vecs.append(encode_text(text))

    maps = []
    for i, ev in enumerate(events):
        base = base_event_fields(ev)
        pid = base["post_id"]
        triples = [f"(user, aggregates, post:{pid})"]
        # L2C: connect to other posts with cosine > 0.5 (cap neighbors)
        sims = []
        for j, v in enumerate(vecs):
            if i == j:
                continue
            sims.append((j, _cos(vecs[i], v)))
        sims.sort(key=lambda x: -x[1])
        neighbors = []
        for j, s in sims[:8]:
            if s > 0.5:
                other = str(events[j].get("post_id"))
                triples.append(f"(post:{pid}, l2c_edge, post:{other})")
                neighbors.append({"post_id": other, "score": round(s, 4)})
        for ent in (ev.get("entities") or [])[:4]:
            triples.append(f"(post:{pid}, mentions, {ent})")
        base["feature_3d_triples"] = triples
        base["graph_neighbors"] = neighbors
        maps.append(base)
    return write_memory_bank(
        user_id=user_id,
        method_key="ddgcn",
        method_name="D-DGCN",
        paper_ref="D-DGCN: Dynamic Deep Graph Convolutional Network (AAAI 2023)",
        analogy={"l2c": "similarity-threshold dynamic edges among posts + user hub"},
        static_map=static,
        event_maps=maps,
        method_extras={"retriever": "default", "tau": 0.5},
    )


def build_semipergcn(user_id: str, bundle: dict[str, Any]):
    """Semi-PerGCN: heterogeneous user–word–LIWC graph with sliding co-occurrence."""
    events = bundle["events"]
    static = build_static_from_persona(bundle["persona"], events)
    maps = []
    for ev in events:
        base = base_event_fields(ev)
        text = str(ev.get("user_opinion") or ev.get("event_summary") or ev.get("raw_text") or "")
        # word nodes: Chinese bigrams from opinion (psych-focused)
        words = re.findall(r"[\u4e00-\u9fff]{2}", text)[:20]
        psych = _psych_words(text)
        triples = []
        for w in words:
            triples.append(f"(word:{w}, of_user, user)")
        for w, cat in psych[:12]:
            triples.append(f"(word:{w}, in_liwc, {cat})")
            triples.append(f"(liwc:{cat}, of_user, user)")
        # word-word sliding window co-occurrence
        for i in range(len(words) - 1):
            triples.append(f"(word:{words[i]}, cooccur, word:{words[i+1]})")
        if not triples:
            triples = ["(user, has_post, empty_psych)"]
        base["feature_3d_triples"] = triples[:40]
        maps.append(base)
    return write_memory_bank(
        user_id=user_id,
        method_key="semipergcn",
        method_name="Semi-PerGCN",
        paper_ref="Semi-PerGCN: Data Augmented GNN for Personality (AAAI 2024)",
        analogy={"graph": "user-word-psycholinguistic category heterogeneous graph"},
        static_map=static,
        event_maps=maps,
        method_extras={"retriever": "default"},
    )


def build_trignet(user_id: str, bundle: dict[str, Any]):
    """TrigNet: tripartite post–psych-word–LIWC category with pwp / pwcwp flows."""
    events = bundle["events"]
    static = build_static_from_persona(bundle["persona"], events)
    # Index posts sharing psych words / categories for flow edges
    word_posts: dict[str, list[str]] = defaultdict(list)
    cat_posts: dict[str, list[str]] = defaultdict(list)
    post_psych: dict[str, list[tuple[str, str]]] = {}
    for ev in events:
        pid = str(ev.get("post_id"))
        text = str(ev.get("raw_text") or "") + str(ev.get("user_opinion") or "")
        psych = _psych_words(text)
        post_psych[pid] = psych
        for w, cat in psych:
            word_posts[w].append(pid)
            cat_posts[cat].append(pid)

    maps = []
    for ev in events:
        base = base_event_fields(ev)
        pid = base["post_id"]
        psych = post_psych.get(pid, [])
        triples = []
        if psych:
            triples.append("(post, flow_pwp, psych)")
            # pwp: connect via shared psych word to another post
            for w, cat in psych[:5]:
                triples.append(f"(post, has_word, {w})")
                triples.append(f"(word:{w}, in_category, {cat})")
                others = [p for p in word_posts.get(w, []) if p != pid][:2]
                for op in others:
                    triples.append(f"(post:{pid}, pwp_via:{w}, post:{op})")
                # pwcwp: via same LIWC category
                cat_others = [p for p in cat_posts.get(cat, []) if p != pid][:2]
                for op in cat_others:
                    triples.append(f"(post:{pid}, pwcwp_via:{cat}, post:{op})")
            triples.append("(post, flow_pwcwp, category)")
        for ent in (ev.get("entities") or [])[:3]:
            triples.append(f"(post, discusses, {ent})")
        if not triples:
            triples = ["(post, flow_pwp, none)"]
        base["feature_3d_triples"] = triples[:40]
        base["psych_words"] = [w for w, _ in psych[:10]]
        maps.append(base)
    return write_memory_bank(
        user_id=user_id,
        method_key="trignet",
        method_name="TrigNet",
        paper_ref="TrigNet: Psycholinguistic Tripartite Graph Network (ACL 2021)",
        analogy={"flows": "post-word-category tripartite"},
        static_map=static,
        event_maps=maps,
        method_extras={"retriever": "default"},
    )


def build_kgrat(user_id: str, bundle: dict[str, Any]):
    """KGrAt-Net: prune entity co-occurrence KG; attach essay/post nodes with attention."""
    events = bundle["events"]
    static = build_static_from_persona(bundle["persona"], events)
    # Aggregate entity-entity edges only when both appear in same text (paper pruning)
    ent_edges: Counter = Counter()
    for ev in events:
        ents = [str(e) for e in (ev.get("entities") or []) if e]
        for i, a in enumerate(ents):
            for b in ents[i + 1 :]:
                if a != b:
                    ent_edges[tuple(sorted((a, b)))] += 1
    # Keep frequent pruned edges as background KG
    bg = [f"({a}, related, {b})" for (a, b), c in ent_edges.most_common(200) if c >= 2]

    maps = []
    for ev in events:
        base = base_event_fields(ev)
        pid = base["post_id"]
        ents = [str(e) for e in (ev.get("entities") or []) if e]
        triples = []
        for e in ents[:6]:
            triples.append(f"(post:{pid}, mentions, {e})")
            triples.append(f"(attention, focus, {e})")
        for kw in (ev.get("stance_keywords") or [])[:2]:
            triples.append(f"(post:{pid}, stance, {kw})")
        # local pruned KG snippet among entities in this post
        for i, a in enumerate(ents):
            for b in ents[i + 1 :]:
                triples.append(f"({a}, related, {b})")
        base["feature_3d_triples"] = triples[:30]
        maps.append(base)
    # store aggregated KG size in static extras
    static["aggregated_kg_edges"] = bg[:50]
    return write_memory_bank(
        user_id=user_id,
        method_key="kgrat",
        method_name="KGrAt-Net",
        paper_ref="KGrAt-Net: Knowledge Graph Attention Network (Scientific Reports 2022)",
        analogy={"kg": "entity aggregation + post-entity attention edges"},
        static_map=static,
        event_maps=maps,
        method_extras={"retriever": "multihop"},
    )
