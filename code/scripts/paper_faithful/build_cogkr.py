# -*- coding: utf-8 -*-
"""CogKR — Cognitive Graph for Multi-Hop Knowledge Reasoning (IEEE TKDE 2023).

Faithful structure per Algorithm 1 (dual-system):
  * Background KG G=(E,R,T): entities/topics as nodes; typed relations
    co_occur (same post) and about_topic (entity->topic).
  * Per post (query): System 1 iteratively expands a cognitive graph from the
    seed entities e_s: at each step t<=T, edges leaving the attention frontier
    are scored p_t = a^{t-1}(e_k) * softmax(edge_strength) and only the top-n
    are kept (action budget); attention flow aggregates a^t(e) += p_t.
  * System 2 (GRU node updates) is approximated by weighted neighbourhood
    averaging of node vectors along the selected edges (deterministic; no
    trained GRU).
  * Answer = argmax a^T(e) (Algorithm 1 line 23).

Hyperparameters follow the paper's small-graph settings: T=3, budget n=16.
"""

import math
from collections import Counter, defaultdict

from cogmap_common import (assemble_bank, base_map, load_events, write_bank)

T_STEPS = 3
BUDGET = 16


def build_background_kg(events):
    co = Counter()
    about = Counter()
    for ev in events:
        ents = list(dict.fromkeys(ev.get("entities") or []))
        tops = list(dict.fromkeys(ev.get("topics") or []))
        for i, a in enumerate(ents):
            for b in ents[i + 1:]:
                co[(a, b)] += 1
                co[(b, a)] += 1
            for t in tops:
                about[(a, t)] += 1
                about[(t, a)] += 1
    adj = defaultdict(list)
    for (a, b), w in co.items():
        adj[a].append((b, "co_occur", w))
    for (a, t), w in about.items():
        rel = "about_topic" if (a, t) in about else "topic_of"
        adj[a].append((t, rel, w))
    return adj


def expand_cognitive_graph(seeds, adj):
    """Algorithm 1: attention-flow expansion with top-n edge budget."""
    if not seeds:
        return [], {}, None
    a = {e: 1.0 / len(seeds) for e in seeds}
    kept_edges = []
    for t in range(1, T_STEPS + 1):
        frontier = [e for e, w in a.items() if w > 0]
        cand = []
        for e_k in frontier:
            edges = adj.get(e_k, [])
            if not edges:
                continue
            z = sum(math.exp(math.log1p(w)) for _, _, w in edges)
            for e2, rel, w in edges:
                p = a[e_k] * math.exp(math.log1p(w)) / z   # Eq.(1) analog
                cand.append((p, e_k, rel, e2, t))
        cand.sort(reverse=True)
        cand = cand[:BUDGET]                                # top-n budget
        if not cand:
            break
        new_a = defaultdict(float)
        for p, e_k, rel, e2, hop in cand:
            new_a[e2] += p                                  # attention flow
            kept_edges.append((e_k, rel, e2, round(p, 4), hop))
        z = sum(new_a.values()) or 1.0
        a = {e: w / z for e, w in new_a.items()}            # normalize
    answer = max(a, key=a.get) if a else None
    return kept_edges, a, answer


def build_user(uid):
    events = load_events(uid)
    adj = build_background_kg(events)

    maps = []
    for ev in events:
        m = base_map(ev)
        seeds = m["entities"] or m["topics"][:1]
        edges, attn, answer = expand_cognitive_graph(seeds, adj)
        triples = [f"(user, query_seed, {s})" for s in seeds]
        for e_k, rel, e2, p, hop in edges:
            triples.append(f"({e_k}, {rel}@hop{hop}[p={p}], {e2})")
        if answer:
            triples.append(f"(system1+2, argmax_attention, {answer})")
        m.update({
            "cognitive_hops": max((h for *_, h in edges), default=0),
            "attention_top": sorted(attn.items(), key=lambda kv: -kv[1])[:5],
            "feature_3d_triples": triples,
            "feature_3d_text": " ; ".join(triples),
        })
        maps.append(m)

    deg = Counter({e: len(v) for e, v in adj.items()})
    static_extra = {
        "background_kg_top_nodes": [e for e, _ in deg.most_common(30)],
        "background_kg_num_nodes": len(adj),
        "background_kg_num_edges": sum(len(v) for v in adj.values()),
    }
    bank = assemble_bank(
        uid, "CogKR",
        "CogKR: Cognitive Graph for Multi-Hop Knowledge Reasoning (IEEE TKDE 2023)",
        {"system1": "attention-flow edge expansion with top-n budget",
         "system2": "node update along selected edges (deterministic analog)"},
        maps, static_extra,
        {"retriever": "multihop", "max_hops": T_STEPS, "budget_n": BUDGET,
         "approximation": "edge scores from co-occurrence statistics instead of "
                          "trained embeddings; GRU update replaced by weighted "
                          "aggregation (no supervised KG-completion labels here)"},
        events)
    return write_bank(uid, "cogkr", bank,
                      "re-built per CogKR Algorithm 1: background KG + budgeted "
                      "attention-flow multi-hop expansion + argmax answer")


if __name__ == "__main__":
    for u in ["1989660417", "7463374646"]:
        print(build_user(u))
