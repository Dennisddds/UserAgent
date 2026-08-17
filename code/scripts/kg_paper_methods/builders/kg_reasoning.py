# -*- coding: utf-8 -*-
"""Independent builders for KG-reasoning-style papers."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

from ..common import (
    base_event_fields,
    build_static_from_persona,
    extract_signed_causals,
    write_memory_bank,
)


def build_cogkr(user_id: str, bundle: dict[str, Any]):
    """CogKR: System-1 frontier expansion + System-2 multi-hop stance reasoning.

    Paper: cognitive graph for multi-hop KG reasoning — start from seed entities,
    expand hop edges, then reason along attended paths to a conclusion.
    """
    events = bundle["events"]
    static = build_static_from_persona(bundle["persona"], events)
    # Background co-occurrence KG (entity/topic) for multi-hop expansion
    co: Counter = Counter()
    for ev in events:
        nodes = [str(x) for x in (ev.get("entities") or []) + (ev.get("topics") or []) if x]
        for i, a in enumerate(nodes):
            for b in nodes[i + 1 :]:
                if a != b:
                    co[tuple(sorted((a, b)))] += 1
    nbrs: dict[str, list[str]] = defaultdict(list)
    for (a, b), c in co.most_common(5000):
        nbrs[a].append(b)
        nbrs[b].append(a)

    maps = []
    for ev in events:
        base = base_event_fields(ev)
        seeds = [str(x) for x in (ev.get("entities") or [])[:4] if x]
        if not seeds:
            seeds = [str(x) for x in (ev.get("topics") or [])[:2] if x]
        triples = []
        frontier = list(seeds)
        visited = set(seeds)
        # System-1: expand up to 2 hops with top neighbors
        for hop in range(1, 3):
            nxt = []
            for e in frontier:
                for nb in nbrs.get(e, [])[:3]:
                    if nb in visited:
                        continue
                    visited.add(nb)
                    nxt.append(nb)
                    triples.append(f"({e}, hop{hop}_expand, {nb})")
            frontier = nxt
            if not frontier:
                break
        # System-2: attend seed → stance keyword → event conclusion
        title = base["event_title"] or "event"
        for e in seeds[:3]:
            triples.append(f"(user, attend, {e})")
            triples.append(f"({e}, hop1_related, {title})")
        for kw in (ev.get("stance_keywords") or [])[:3]:
            triples.append(f"(user, reason_via, {kw})")
            triples.append(f"({kw}, concludes, {title})")
        # co-occur among seed entities (local cognitive graph edges)
        for i, a in enumerate(seeds):
            for b in seeds[i + 1 :]:
                triples.append(f"({a}, co_occur, {b})")
        base["feature_3d_triples"] = triples
        base["cognitive_hops"] = min(2, max(1, len(visited) // max(len(seeds), 1)))
        maps.append(base)

    return write_memory_bank(
        user_id=user_id,
        method_key="cogkr",
        method_name="CogKR",
        paper_ref="CogKR: Cognitive Graph for Multi-Hop Knowledge Reasoning (IEEE TKDE 2023)",
        analogy={"system1": "entity frontier expansion", "system2": "stance path reasoning"},
        static_map=static,
        event_maps=maps,
        method_extras={"retriever": "multihop", "max_hops": 2},
    )


def build_cognet3(user_id: str, bundle: dict[str, Any]):
    """CogNet3: semantic event + homophilous personality group + Plutchik emotions."""
    events = bundle["events"]
    persona = bundle["persona"]
    static = build_static_from_persona(persona, events)

    # Homophilous group attributes (paper Frame 2) — heuristic from persona text
    blob = " ".join(
        [
            str(persona.get("analysis") or ""),
            " ".join(persona.get("values") or []),
            " ".join(persona.get("communication") or []),
        ]
    )
    if re.search(r"爱国|主流|体制|领导", blob):
        stance_t = "centrist"
    elif re.search(r"左|公平|劳工", blob):
        stance_t = "left"
    else:
        stance_t = "centrist"
    firmness = "stable" if "rhetorical" not in blob.lower() else "depends"
    aggress = "medium" if re.search(r"批评|质疑|嘲讽", blob) else "low"
    logical = "high" if re.search(r"summar|客观|论证|分析", blob, re.I) else "medium"
    personality_group = {
        "stance_tendency": stance_t,
        "stance_firmness": firmness,
        "expression_aggressiveness": aggress,
        "expression_logicality": logical,
        "big_five": {
            "openness": "medium",
            "conscientiousness": "high",
            "extraversion": "medium",
            "agreeableness": "medium",
            "neuroticism": "low",
        },
    }
    static["personality_group"] = personality_group

    # Plutchik 8 emotions × intensity histogram aggregated
    emo_keys = [
        "Anger", "Fear", "Sadness", "Joy", "Disgust", "Surprise", "Trust", "Anticipation"
    ]
    hist = {k: Counter() for k in emo_keys}
    maps = []
    for ev in events:
        base = base_event_fields(ev)
        text = " ".join(
            [
                str(ev.get("raw_text") or ""),
                str(ev.get("user_opinion") or ""),
                " ".join(ev.get("stance_keywords") or []),
            ]
        )
        dist = {k: "none" for k in emo_keys}
        rules = [
            ("Anger", r"愤怒|气愤|怒|谴责|痛批", "high"),
            ("Fear", r"担心|担忧|恐慌|危险|威胁", "medium"),
            ("Sadness", r"遗憾|痛心|悲伤|失望|寒心", "medium"),
            ("Joy", r"欣慰|高兴|喜悦|振奋|自豪", "medium"),
            ("Disgust", r"恶心|可耻|丑陋|虚伪|反感", "high"),
            ("Surprise", r"竟然|震惊|意外|没想到", "medium"),
            ("Trust", r"相信|信任|靠谱|支持|期望", "medium"),
            ("Anticipation", r"希望|期待|展望|将会|未来", "medium"),
        ]
        for emo, pat, lvl in rules:
            if re.search(pat, text):
                dist[emo] = lvl
                hist[emo][lvl] += 1
            else:
                hist[emo]["none"] += 1
        triples = [f"(semantic_event, is, {base['event_title'] or 'event'})"]
        for ent in (ev.get("entities") or [])[:4]:
            triples.append(f"(group, reacts_to, {ent})")
        for emo, lvl in dist.items():
            if lvl != "none":
                triples.append(f"(group_emotion:{emo}, level, {lvl})")
        triples.append(
            f"(personality_group, stance_tendency, {personality_group['stance_tendency']})"
        )
        base["feature_3d_triples"] = triples
        base["emotion_dist"] = dist
        maps.append(base)

    # percentage emotion histogram for static map
    static["emotion_histogram"] = {
        emo: {
            lvl: round(cnt / max(sum(hist[emo].values()), 1), 4)
            for lvl, cnt in hist[emo].items()
        }
        for emo in emo_keys
    }
    return write_memory_bank(
        user_id=user_id,
        method_key="cognet3",
        method_name="CogNet3",
        paper_ref="CogNet3: Dynamic Emotional Knowledge Fusion (ISWC 2025 Companion)",
        analogy={"frames": "semantic event + homophilous group + group emotion"},
        static_map=static,
        event_maps=maps,
        method_extras={"retriever": "default"},
    )


def build_cimplekg(user_id: str, bundle: dict[str, Any]):
    """CimpleKG: ClaimReview pipeline — claim, normalized rating, entities, factors."""
    events = bundle["events"]
    static = build_static_from_persona(bundle["persona"], events)
    maps = []
    for ev in events:
        base = base_event_fields(ev)
        claim = (ev.get("user_opinion") or ev.get("event_summary") or "")[:120]
        text = " ".join(
            [str(ev.get("raw_text") or ""), str(ev.get("user_opinion") or "")]
        )
        # 5-level rating mapping (paper Step 4)
        if re.search(r"假的|谣言|不实|造谣|虚假", text):
            rating = "not_credible"
        elif re.search(r"存疑|难辨|不确定|传闻|未证实", text):
            rating = "uncertain"
        elif re.search(r"属实|证实|确认|官方|证据", text):
            rating = "mostly_credible"
        elif re.search(r"真的|确实|无疑", text):
            rating = "credible"
        else:
            rating = "not_verifiable"
        triples = [
            f"(ClaimReview, reviews, {claim[:40]})",
            f"(ClaimReview, rating, {rating})",
            f"(Claim, appearance, {base['event_title'] or 'post'})",
        ]
        for ent in (ev.get("entities") or [])[:4]:
            triples.append(f"(Claim, mentions, {ent})")
            triples.append(f"(factor:stance, about, {ent})")
        # factor extraction proxies (emotion / political / propaganda-ish cues)
        if re.search(r"愤怒|谴责|痛批", text):
            triples.append("(factor:emotion, is, anger)")
        if re.search(r"美国|西方|民主|自由|体制", text):
            triples.append("(factor:political_leaning, is, present)")
        if re.search(r"阴谋|暗箱|操弄|带节奏", text):
            triples.append("(factor:conspiracy, is, present)")
        base["feature_3d_triples"] = triples
        base["claim_rating"] = rating
        base["claim_text"] = claim
        maps.append(base)
    return write_memory_bank(
        user_id=user_id,
        method_key="cimplekg",
        method_name="CimpleKG",
        paper_ref="CimpleKG: Continuously Updated Misinformation KG (2024)",
        analogy={"pipeline": "claim extraction + rating + entity factors"},
        static_map=static,
        event_maps=maps,
        method_extras={"retriever": "default"},
    )


def build_claimskg(user_id: str, bundle: dict[str, Any]):
    """ClaimsKG: Schema.org Claim / ClaimReview graph of fact-checked claims."""
    events = bundle["events"]
    static = build_static_from_persona(bundle["persona"], events)
    maps = []
    for ev in events:
        base = base_event_fields(ev)
        claim_text = (ev.get("user_opinion") or "")[:80]
        triples = [
            f"(schema:Claim, about, {base['event_title'] or 'event'})",
            "(schema:ClaimReview, author, user)",
            f"(schema:ClaimReview, text, {claim_text})",
        ]
        for t in (ev.get("topics") or [])[:4]:
            triples.append(f"(schema:Claim, keywords, {t})")
        for ent in (ev.get("entities") or [])[:4]:
            triples.append(f"(schema:Claim, entity, {ent})")
        if claim_text:
            triples.append("(schema:ClaimReview, itemReviewed, schema:Claim)")
        base["feature_3d_triples"] = triples
        maps.append(base)
    return write_memory_bank(
        user_id=user_id,
        method_key="claimskg",
        method_name="ClaimsKG",
        paper_ref="ClaimsKG: Knowledge Graph of Fact-Checked Claims (ISWC 2019)",
        analogy={"schema": "ClaimReview/Claim simplification"},
        static_map=static,
        event_maps=maps,
        method_extras={"retriever": "default"},
    )
