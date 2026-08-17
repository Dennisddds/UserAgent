# -*- coding: utf-8 -*-
"""Independent builders for cognitive-modeling papers."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from ..common import (
    base_event_fields,
    build_static_from_persona,
    extract_signed_causals,
    write_memory_bank,
)


def build_genminds(user_id: str, bundle: dict[str, Any]):
    """GenMinds: belief DAG from causal explanations + cognitive motifs.

    Paper: explanations parsed into directed causal graphs; nodes=concepts,
    edges=directed causality with polarity; motifs are minimal causal units.
    """
    events = bundle["events"]
    static = build_static_from_persona(bundle["persona"], events)
    motif_counter: Counter = Counter()
    maps = []
    for ev in events:
        base = base_event_fields(ev)
        text = " ".join(
            [
                str(ev.get("raw_text") or ""),
                str(ev.get("event_summary") or ""),
                str(ev.get("user_opinion") or ""),
            ]
        )
        causals = extract_signed_causals(text)
        # Fallback: entity/topic chain ONLY when no causal language — marked as weak motifs
        concepts = []
        triples = []
        if causals:
            for a, b, s in causals[:6]:
                pol = "+" if s > 0 else "-"
                triples.append(f"({a}, causes, {b})")
                triples.append(f"(belief_edge, polarity, {pol})")
                motif = f"{a}→{b}"
                motif_counter[motif] += 1
                concepts.extend([a, b])
                triples.append(f"(user, holds_motif, {motif})")
        else:
            # Still build a belief stub from opinion keywords / entities without fake causes
            ents = [str(x) for x in (ev.get("entities") or [])[:3] if x]
            topics = [str(x) for x in (ev.get("topics") or [])[:2] if x]
            for e in ents:
                triples.append(f"(belief_node, is, {e})")
                concepts.append(e)
            for t in topics:
                triples.append(f"(belief_node, is, {t})")
            if ents and topics:
                triples.append(f"({ents[0]}, associated_with, {topics[0]})")
                triples.append("(belief_edge, polarity, 0)")
            title = base["event_title"] or "event"
            triples.append(f"(user, holds_motif, {title})")
            motif_counter[title] += 1
        # confidence proxy from stance keyword strength
        conf = min(1.0, 0.4 + 0.15 * len(ev.get("stance_keywords") or []))
        triples.append(f"(motif_confidence, is, {conf:.2f})")
        base["feature_3d_triples"] = triples
        base["causal_concepts"] = list(dict.fromkeys(concepts))[:12]
        maps.append(base)

    static["cognitive_motifs"] = [
        {"motif": m, "count": c} for m, c in motif_counter.most_common(40)
    ]
    return write_memory_bank(
        user_id=user_id,
        method_key="genminds",
        method_name="GenMinds",
        paper_ref="GenMinds: Simulating Society Requires Simulating Thought (NeurIPS 2025)",
        analogy={"dag": "belief causal motifs"},
        static_map=static,
        event_maps=maps,
        method_extras={"retriever": "multihop"},
    )


def build_lcg(user_id: str, bundle: dict[str, Any]):
    """LCG: dynamic cognition graph with multi-scale temporal memory.

    Paper: G_t=(V_t,E_t) with learner/concept/question; event-driven updates;
    short/mid/long memory via different time-decay scales.
    """
    events = sorted(bundle["events"], key=lambda e: float(e.get("timestamp") or 0.0))
    static = build_static_from_persona(bundle["persona"], events)
    if not events:
        ordered = []
    else:
        t0 = float(events[0].get("timestamp") or 0.0)
        t1 = float(events[-1].get("timestamp") or t0) or t0 + 1.0
        span = max(t1 - t0, 1.0)
        maps_chrono = []
        for i, ev in enumerate(events):
            base = base_event_fields(ev)
            ts = float(ev.get("timestamp") or t0)
            # relative age → memory scale (paper multi-scale λ)
            age = (t1 - ts) / span
            if age < 0.2:
                scale = "short"
            elif age < 0.55:
                scale = "mid"
            else:
                scale = "long"
            title = base["event_title"] or "event"
            triples = [
                f"(learner:user, interacts, concept:{title})",
                f"(memory_scale, is, {scale})",
                f"(event, time_index, {i})",
            ]
            for ent in (ev.get("entities") or [])[:4]:
                triples.append(f"(concept:{ent}, updated_by, event)")
            for t in (ev.get("topics") or [])[:3]:
                triples.append(f"(question_proxy, probes, concept:{t})")
            # event-driven edge from previous concept (temporal message)
            if i > 0:
                prev_title = str(events[i - 1].get("event_title") or "prev")
                triples.append(f"(concept:{prev_title}, temporal_msg, concept:{title})")
            base["feature_3d_triples"] = triples
            base["memory_scale"] = scale
            base["time_index"] = i
            maps_chrono.append(base)
        by_id = {m["post_id"]: m for m in maps_chrono}
        ordered = [
            by_id[str(e.get("post_id"))]
            for e in bundle["events"]
            if str(e.get("post_id")) in by_id
        ]
    return write_memory_bank(
        user_id=user_id,
        method_key="lcg",
        method_name="LCG",
        paper_ref="LCG: Dynamic Cognition Graph for Adaptive Learning (Applied Sciences 2026)",
        analogy={"dynamic": "event-driven updates + multi-scale temporal memory"},
        static_map=static,
        event_maps=ordered,
        method_extras={"retriever": "temporal"},
    )


def build_cognimap(user_id: str, bundle: dict[str, Any]):
    """CogniMap3D-inspired local baseline: related_to / stance_on memory map."""
    events = bundle["events"]
    static = build_static_from_persona(bundle["persona"], events)
    maps = []
    for ev in events:
        base = base_event_fields(ev)
        triples = []
        ents = [str(e) for e in (ev.get("entities") or []) if e]
        topics = [str(t) for t in (ev.get("topics") or []) if t]
        for e in ents:
            triples.append(f"(user, related_to, {e})")
        for t in topics:
            triples.append(f"(user, related_to, {t})")
        for e in ents[:3]:
            for kw in (ev.get("stance_keywords") or [])[:2]:
                triples.append(f"(user, stance_on, {e})")
                triples.append(f"({e}, stance, {kw})")
        if not triples:
            triples = ["(user, related_to, event)"]
        base["feature_3d_triples"] = triples
        maps.append(base)
    return write_memory_bank(
        user_id=user_id,
        method_key="cognimap",
        method_name="CogniMap3D-inspired",
        paper_ref="CogniMap3D (ICLR 2026) / local baseline",
        analogy={"map": "related_to / stance_on entity memory"},
        static_map=static,
        event_maps=maps,
        method_extras={"retriever": "default"},
    )


def build_cognitive_maps_1977(user_id: str, bundle: dict[str, Any]):
    """Hart 1977 cognitive maps: signed causal assertions A--(+/-)-->B + Utility."""
    events = bundle["events"]
    static = build_static_from_persona(bundle["persona"], events)
    UTILITY = "Utility"
    POS = re.compile(r"支持|赞|期望|希望|相信|爱国|正能量|欣慰|进步|公正|应该")
    NEG = re.compile(r"批评|质疑|反对|谴责|愤怒|失望|担忧|可耻|错误|歪风|双标|不满")
    agg: Counter = Counter()
    maps = []
    for ev in events:
        base = base_event_fields(ev)
        text = " ".join(
            [
                str(ev.get("raw_text") or ""),
                str(ev.get("event_summary") or ""),
                str(ev.get("user_opinion") or ""),
            ]
        )
        causals = extract_signed_causals(text)
        anchors = [str(x) for x in (ev.get("entities") or []) + (ev.get("topics") or []) if x]

        def snap(p: str) -> str:
            p = re.sub(r"\s+", "", p)[:24]
            for a in anchors:
                if a in p or p in a:
                    return a
            return p

        edges = []
        for a, b, s in causals:
            a, b = snap(a), snap(b)
            if len(a) >= 2 and len(b) >= 2 and a != b:
                edges.append((a, b, s))
        # Utility linkage from evaluations
        opinion = str(ev.get("user_opinion") or "")
        vars_ = list(dict.fromkeys([a for a, _, _ in edges] + [b for _, b, _ in edges] + anchors))
        for v in vars_:
            if v in opinion:
                if POS.search(opinion):
                    edges.append((v, UTILITY, 1))
                elif NEG.search(opinion):
                    edges.append((v, UTILITY, -1))
        if not any(b == UTILITY for _, b, _ in edges) and vars_ and base["polarity"]:
            edges.append((vars_[0], UTILITY, 1 if base["polarity"] > 0 else -1))

        triples = [f"({a}, --({'+' if s > 0 else '-'})-->, {b})" for a, b, s in edges]
        for a, b, s in edges:
            agg[(a, b, "+" if s > 0 else "-")] += 1
        u_signs = [s for a, b, s in edges if b == UTILITY]
        pol = round(sum(u_signs) / len(u_signs), 3) if u_signs else 0.0
        base["feature_3d_triples"] = triples
        base["variables"] = vars_
        base["map_sign"] = "+" if pol > 0 else ("-" if pol < 0 else "0")
        base["polarity"] = pol
        base["causal_assertions"] = [
            {"cause": a, "effect": b, "sign": "+" if s > 0 else "-"} for a, b, s in edges
        ]
        maps.append(base)

    static["causal_beliefs"] = [
        {"cause": a, "effect": b, "sign": s, "count": c}
        for (a, b, s), c in agg.most_common(200)
    ]
    return write_memory_bank(
        user_id=user_id,
        method_key="cognitive_maps_1977",
        method_name="CognitiveMaps1977",
        paper_ref="Cognitive Maps of Three Latin American Policy Makers (World Politics 1977)",
        analogy={"manual_coding": "signed causal concept maps (Hart 1977 documentary coding)"},
        static_map=static,
        event_maps=maps,
        method_extras={"retriever": "signed"},
    )
