# -*- coding: utf-8 -*-
"""LCG — Dynamic Cognition Graph for Adaptive Learning (Applied Sciences 2026).

Faithful structure:
  * Dynamic graph G_t = (V_t, E_t, R, X_t) with V = learner(user) + knowledge
    concepts (topics/entities) + items (posts)  (Eq. 5).
  * Events e_i = (u_i, v_i, r_i, t_i, a_i) (Eq. 7); the reasoning-evidence
    attribute a_i = [correctness-analog: stance polarity, response-time
    analog: gap to previous post, reasoning embedding of the user's opinion]
    (Eq. 4 analog).
  * Event-driven message passing with exponential time decay
    m_bar_v(t) = sum exp(-lambda (t - t_i)) m_i  (Eq. 11), with the paper's
    three decay scales lambda in {0.01, 0.05, 0.1} (per-day units).
  * Multi-scale memory: h_v = h_short || h_mid || h_long (Eq. 13); GRU update
    replaced by exponential moving updates (deterministic).
  * The RL intervention module is not applicable (no tutoring actions);
    documented.
"""

import math
from collections import defaultdict

import numpy as np

from cogmap_common import (assemble_bank, base_map, encode_text, load_events,
                    sentiment_score, write_bank)

LAMBDAS = {"short": 0.1, "mid": 0.05, "long": 0.01}   # per-day decay
DAY = 86400.0


def build_user(uid):
    events = load_events(uid)
    events = sorted(events, key=lambda e: e.get("timestamp") or 0)

    # multi-scale concept memories: concept -> {scale: (mass, vec)}
    mem_mass = {s: defaultdict(float) for s in LAMBDAS}
    mem_vec = {s: defaultdict(lambda: np.zeros(64)) for s in LAMBDAS}
    last_t = {}
    prev_ts = None

    maps = []
    for ti, ev in enumerate(events):
        m = base_map(ev)
        t = ev.get("timestamp") or 0
        concepts = list(dict.fromkeys((ev.get("topics") or []) + (ev.get("entities") or [])))
        pol = sentiment_score((ev.get("user_opinion") or "")
                              + " ".join(ev.get("stance_keywords") or []))
        gap_days = (t - prev_ts) / DAY if prev_ts else 0.0
        prev_ts = t
        z = np.array(encode_text(ev.get("user_opinion") or m["event_summary"]))[:64]
        a_i = {"correctness_analog": round(pol, 3),
               "response_gap_days": round(gap_days, 2)}

        # event-driven decay update for touched concepts
        for c in concepts:
            dt_days = (t - last_t.get(c, t)) / DAY
            for s, lam in LAMBDAS.items():
                decay = math.exp(-lam * dt_days)
                mem_mass[s][c] = mem_mass[s][c] * decay + 1.0
                mem_vec[s][c] = mem_vec[s][c] * decay + z
            last_t[c] = t

        # dominant memory scale for this event's concepts
        scale_mass = {s: sum(mem_mass[s][c] for c in concepts) for s in LAMBDAS}
        dom = max(scale_mass, key=scale_mass.get) if concepts else "short"

        triples = []
        for c in concepts[:4]:
            triples.append(f"(learner:user, interacts[r=comment], concept:{c})")
            triples.append(f"(item:post:{ev['post_id']}, involves, concept:{c})")
            for s in LAMBDAS:
                triples.append(f"(concept:{c}, memory_{s}, {round(mem_mass[s][c], 2)})")
        triples.append(f"(event, time_index, {ti})")
        triples.append(f"(event, evidence, corr={a_i['correctness_analog']}"
                       f"_gap={a_i['response_gap_days']}d)")
        m.update({
            "memory_scale": dom,
            "time_index": ti,
            "event_attributes": a_i,
            "feature_3d_triples": triples,
            "feature_3d_text": " ; ".join(triples),
        })
        maps.append(m)

    top_long = sorted(mem_mass["long"].items(), key=lambda kv: -kv[1])[:20]
    static_extra = {
        "multi_scale_top_concepts": {
            s: sorted(mem_mass[s].items(), key=lambda kv: -kv[1])[:10]
            for s in LAMBDAS},
        "long_term_knowledge_state": [{"concept": c, "mass": round(v, 2)}
                                      for c, v in top_long],
    }
    bank = assemble_bank(
        uid, "LCG",
        "LCG: Dynamic Cognition Graph for Adaptive Learning (Applied Sciences 2026)",
        {"dynamic": "event-driven updates e=(u,v,r,t,a) + exp time decay + "
                    "multi-scale (0.01/0.05/0.1) temporal memory"},
        maps, static_extra,
        {"retriever": "temporal", "lambdas": LAMBDAS,
         "approximation": "GRU state update replaced by exponential moving "
                          "update; RL intervention module not applicable "
                          "(no tutoring actions); reverse-Turing prompts "
                          "replaced by the user's own opinions as evidence"},
        events)
    return write_bank(uid, "lcg", bank,
                      "re-built per LCG: dynamic tripartite graph "
                      "(learner/concept/item), event tuples with evidence "
                      "attributes, Eq.11 time-decay message passing, Eq.13 "
                      "multi-scale memory")


if __name__ == "__main__":
    for u in ["1989660417", "7463374646"]:
        print(build_user(u))
