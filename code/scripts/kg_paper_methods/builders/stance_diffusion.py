# -*- coding: utf-8 -*-
"""Independent builders for stance / signed-social / diffusion / recommendation papers."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

from ..common import base_event_fields, build_static_from_persona, write_memory_bank


def _edge_sign(text: str, default: int = 1) -> int:
    neg = len(re.findall(r"批评|质疑|反对|谴责|不满|威胁|损害|错误|可耻|双标", text))
    pos = len(re.findall(r"支持|赞|期望|希望|相信|爱国|公正|进步|欣慰", text))
    if neg > pos:
        return -1
    if pos > neg:
        return 1
    return default


def build_cttn(user_id: str, bundle: dict[str, Any]):
    """CT-TN: target-conditioned text component (target + context) for stance."""
    events = bundle["events"]
    static = build_static_from_persona(bundle["persona"], events)
    maps = []
    for ev in events:
        base = base_event_fields(ev)
        ents = [str(e) for e in (ev.get("entities") or []) if e]
        target = ents[0] if ents else (base["topics"][0] if base["topics"] else "event")
        context = (ev.get("event_summary") or ev.get("raw_text") or "")[:60]
        text = " ".join([str(ev.get("user_opinion") or ""), " ".join(ev.get("stance_keywords") or [])])
        # favor / against / none (paper binary favor/against; keep none when flat)
        if re.search(r"支持|赞|期望|相信", text):
            label = "favor"
        elif re.search(r"批评|质疑|反对|谴责", text):
            label = "against"
        else:
            label = "none"
        triples = [
            f"(target, is, {target})",
            f"(context, text, {context})",
            f"(stance, label, {label})",
            "(text_encoder, conditions_on, target+context)",
        ]
        for e in ents[1:3]:
            triples.append(f"(related_target, is, {e})")
        # graph-component proxy without external follower graph: co-target links
        for t in (ev.get("topics") or [])[:2]:
            triples.append(f"(user, network_proxy_likes, topic:{t})")
        base["feature_3d_triples"] = triples
        base["stance_label"] = label
        base["target"] = target
        maps.append(base)
    return write_memory_bank(
        user_id=user_id,
        method_key="cttn",
        method_name="CT-TN",
        paper_ref="CT-TN: Few-shot Cross-Target Stance Detection (IEEE TCSS 2023)",
        analogy={"components": "target-conditioned text + stance graph"},
        static_map=static,
        event_maps=maps,
        method_extras={"retriever": "signed"},
    )


def build_enm_senm(user_id: str, bundle: dict[str, Any]):
    """ENM/SENM: ego–alter circles by interaction frequency + signed valence."""
    events = bundle["events"]
    static = build_static_from_persona(bundle["persona"], events)
    # alter = entity; frequency = mention count; valence from co-text
    freq: Counter = Counter()
    valence_sum: dict[str, float] = defaultdict(float)
    valence_n: dict[str, int] = defaultdict(int)
    for ev in events:
        text = " ".join(
            [str(ev.get("user_opinion") or ""), " ".join(ev.get("stance_keywords") or [])]
        )
        s = _edge_sign(text, default=0)
        for ent in (ev.get("entities") or []):
            ent = str(ent)
            freq[ent] += 1
            if s:
                valence_sum[ent] += s
                valence_n[ent] += 1
    # MeanShift-like circle assignment by frequency ranks (inner=1-2, outer=3+)
    ranked = [e for e, _ in freq.most_common()]
    n = max(len(ranked), 1)
    circles = {}
    for i, e in enumerate(ranked):
        # 4 circles max as typical ENM
        if i < max(1, n // 10):
            circles[e] = "inner_1"
        elif i < max(2, n // 5):
            circles[e] = "inner_2"
        elif i < max(3, n // 2):
            circles[e] = "outer_3"
        else:
            circles[e] = "outer_4"
    # Gottman 17% negative threshold for signed tie
    ego_alters = []
    for e, c in freq.most_common(100):
        neg_ratio = 0.0
        if valence_n[e]:
            # fraction of negative interactions approx
            avg = valence_sum[e] / valence_n[e]
            neg_ratio = max(0.0, (1 - avg) / 2)
        sign = "-" if neg_ratio >= 0.17 and valence_sum[e] < 0 else "+"
        ego_alters.append(
            {
                "alter": e,
                "freq": c,
                "circle": circles.get(e, "outer_4"),
                "sign": sign,
                "neg_ratio": round(neg_ratio, 3),
            }
        )
    static["ego_alters"] = ego_alters

    maps = []
    for ev in events:
        base = base_event_fields(ev)
        triples = []
        for ent in (ev.get("entities") or [])[:5]:
            ent = str(ent)
            info = next((x for x in ego_alters if x["alter"] == ent), None)
            sign = (info or {}).get("sign", "+")
            circle = (info or {}).get("circle", "outer_4")
            ring = "inner" if circle.startswith("inner") else "outer"
            triples.append(f"(ego, {sign}_alter, {ent})")
            triples.append(f"({ent}, circle, {ring})")
        if not triples:
            triples = ["(ego, has_alter, none)"]
        base["feature_3d_triples"] = triples
        maps.append(base)
    return write_memory_bank(
        user_id=user_id,
        method_key="enm_senm",
        method_name="ENM-SENM",
        paper_ref="ENM/SENM: Ego Network Model for Cross-Target Stance (2024)",
        analogy={"ego": "signed ego-alter entity circles"},
        static_map=static,
        event_maps=maps,
        method_extras={"retriever": "signed"},
    )


def build_sem(user_id: str, bundle: dict[str, Any]):
    """SEM: signed topic/entity edges + Heider balance on triples."""
    events = bundle["events"]
    static = build_static_from_persona(bundle["persona"], events)
    maps = []
    for ev in events:
        base = base_event_fields(ev)
        text = " ".join(
            [str(ev.get("user_opinion") or ""), " ".join(ev.get("stance_keywords") or [])]
        )
        s = _edge_sign(text, default=1)
        sign = f"{s:+d}"
        triples = [f"(walk, biased_by, p=1.5_q=0.5)"]
        for t in (ev.get("topics") or [])[:3]:
            triples.append(f"(user, {sign}, topic:{t})")
        ents = [str(e) for e in (ev.get("entities") or []) if e]
        for e in ents[:4]:
            triples.append(f"(user, {sign}, entity:{e})")
        # Heider balance between entities under same topic polarity product
        if len(ents) >= 2:
            # friend of friend / enemy of enemy
            triples.append(f"(entity:{ents[0]}, balance_with, entity:{ents[1]})")
            triples.append(f"(balance_sign, is, {sign})")
        base["feature_3d_triples"] = triples
        base["edge_sign"] = s
        base["topic_key"] = (ev.get("topics") or ["untopic"])[0]
        maps.append(base)
    return write_memory_bank(
        user_id=user_id,
        method_key="sem",
        method_name="SEM",
        paper_ref="SEM: Stance Embeddings from Signed Social Graphs (WSDM 2023)",
        analogy={"signed": "topic/entity signed edges + balance"},
        static_map=static,
        event_maps=maps,
        method_extras={"retriever": "signed"},
    )


def build_rotdiff(user_id: str, bundle: dict[str, Any]):
    """RotDiff: time-ordered diffusion cascade + social rotation around entities."""
    events = sorted(bundle["events"], key=lambda e: float(e.get("timestamp") or 0.0))
    static = build_static_from_persona(bundle["persona"], events)
    maps = []
    for i, ev in enumerate(events):
        base = base_event_fields(ev)
        triples = [f"(cascade, position, {i})"]
        if i + 1 < len(events):
            nxt = str(events[i + 1].get("post_id"))
            triples.append(f"(post:{base['post_id']}, diffuses_to, post:{nxt})")
        if i > 0:
            prev = str(events[i - 1].get("post_id"))
            triples.append(f"(post:{prev}, diffuses_to, post:{base['post_id']})")
        for ent in (ev.get("entities") or [])[:3]:
            triples.append(f"(social_rot, around, {ent})")
            triples.append(f"(diffusion_rot, around, {ent})")
        # Lorentz rotation proxy markers (social vs diffusion views)
        triples.append("(view, social_rotation, active)")
        triples.append("(view, diffusion_rotation, active)")
        base["feature_3d_triples"] = triples
        base["cascade_index"] = i
        maps.append(base)
    # restore original train order by post_id map for retrieval alignment with train
    by_id = {m["post_id"]: m for m in maps}
    ordered = [by_id[str(e.get("post_id"))] for e in bundle["events"] if str(e.get("post_id")) in by_id]
    return write_memory_bank(
        user_id=user_id,
        method_key="rotdiff",
        method_name="RotDiff",
        paper_ref="RotDiff: Hyperbolic Rotation for Information Diffusion (CIKM 2023)",
        analogy={"diffusion": "time-ordered cascade + entity social rotation proxy"},
        static_map=static,
        event_maps=ordered,
        method_extras={"retriever": "temporal"},
    )


def build_gorec(user_id: str, bundle: dict[str, Any]):
    """GoRec: KOL–opinion–item translation triples + opinion diffusion marker."""
    events = bundle["events"]
    static = build_static_from_persona(bundle["persona"], events)
    maps = []
    for ev in events:
        base = base_event_fields(ev)
        opinions = [str(k) for k in (ev.get("stance_keywords") or []) if k] or ["opinion"]
        items = [str(x) for x in (ev.get("topics") or []) + (ev.get("entities") or []) if x]
        op = opinions[0]
        triples = []
        for item in items[:5]:
            triples.append(f"(KOL:user, opinion:{op}, item:{item})")
            triples.append(f"({item}, translated_by, {op})")
        triples.append("(diffusion, from_kol, followers_proxy)")
        if not items:
            triples = [f"(KOL:user, opinion:{op}, item:event)"]
        base["feature_3d_triples"] = triples
        base["opinion_relation"] = op
        maps.append(base)
    return write_memory_bank(
        user_id=user_id,
        method_key="gorec",
        method_name="GoRec",
        paper_ref="GoRec: Key Opinion Leaders in Recommendation (WSDM 2020)",
        analogy={"translation": "KOL-opinion-item triples + diffusion"},
        static_map=static,
        event_maps=maps,
        method_extras={"retriever": "default"},
    )


def build_cogigraph(user_id: str, bundle: dict[str, Any]):
    """CogiGraph: OpenIE-style triples + entity alignment into a local KG."""
    events = bundle["events"]
    static = build_static_from_persona(bundle["persona"], events)
    # Local KG mentions = frequent entities/topics
    mention_freq: Counter = Counter()
    for ev in events:
        for x in list(ev.get("entities") or []) + list(ev.get("topics") or []):
            mention_freq[str(x)] += 1
    kg_mentions = {m for m, c in mention_freq.items() if c >= 2}

    maps = []
    for ev in events:
        base = base_event_fields(ev)
        title = base["event_title"] or "event"
        ents = [str(e) for e in (ev.get("entities") or []) if e]
        kws = [str(k) for k in (ev.get("stance_keywords") or []) if k]
        openie = []
        # OpenIE-like SVO from entities + stance + event
        if ents and kws:
            openie.append(f"OpenIE:({ents[0]}, {kws[0]}, {title})")
        elif ents:
            openie.append(f"OpenIE:({ents[0]}, about, {title})")
        for a, b in zip(ents, ents[1:]):
            openie.append(f"OpenIE:({a}, co_mentioned_with, {b})")
        triples = list(openie)
        for e in ents[:4]:
            # alignment score proxy: 1 if in aggregated KG mentions
            if e in kg_mentions:
                triples.append(f"(context_embed, aligns_to, kg_mention:{e})")
                triples.append(f"({e}, aligned_entity, {title})")
            else:
                triples.append(f"({e}, unaligned_mention, {title})")
        if not triples:
            triples = ["OpenIE:(user, comments, event)"]
        base["feature_3d_triples"] = triples
        base["openie"] = openie
        maps.append(base)
    return write_memory_bank(
        user_id=user_id,
        method_key="cogigraph",
        method_name="CogiGraph",
        paper_ref="CogiGraph: Semantic KG Fusion for Fake News Detection (PLOS ONE 2025)",
        analogy={"fusion": "OpenIE triples + entity alignment"},
        static_map=static,
        event_maps=maps,
        method_extras={"retriever": "default"},
    )
