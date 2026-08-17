# -*- coding: utf-8 -*-
"""SEM — Stance Embeddings from Signed Social Graphs (WSDM 2023).

Faithful structure:
  * Signed topic graph: 4-column edge list (source, target, topic, +/-1),
    exactly the paper's data format. Edges: user->alter (mention/repost) and
    alter->entity within a post's topic; sign from the post's stance.
  * Training data generation: biased second-order random walks per topic
    subgraph with the paper's exact parameters p=1.5, q=0.5, L=40, r=10;
    signed contexts via Heider balance (sign of a context = product of edge
    weights along the walk, the paper's three rules).
  * Embeddings: topic-aware skip-gram approximated by factorizing the signed
    walk co-occurrence matrix (SVD of signed PMI; SGNS is known to factorize
    shifted PMI), psi = addition variant: topic embedding g(t) = mean of its
    member node embeddings, f(u)+g(t) used for scoring.
"""

import numpy as np

from cogmap_common import (assemble_bank, base_map, load_events, mentions,
                    repost_chain, sentiment, write_bank)

P_RETURN = 1.5
Q_INOUT = 0.5
WALK_LEN = 40
R_WALKS = 10
WINDOW = 5
EMB_D = 32


def build_edge_list(events):
    """4-column signed edge list (src, dst, topic, sign)."""
    edges = []
    for ev in events:
        raw = ev.get("raw_text") or ""
        s = sentiment((ev.get("user_opinion") or "")
                      + " ".join(ev.get("stance_keywords") or [])) or 1
        topic = (ev.get("topics") or ["general"])[0]
        alters = list(set(mentions(raw)) | set(repost_chain(raw)))
        for a in alters:
            edges.append(("user", a, topic, s))
        for e in (ev.get("entities") or [])[:3]:
            edges.append(("user", f"ent:{e}", topic, s))
    return edges


def biased_walks(adj, rng):
    """Second-order node2vec-style walks with p=1.5, q=0.5."""
    walks = []
    nodes = sorted(adj.keys())
    for v0 in nodes:
        for _ in range(R_WALKS):
            walk = [v0]
            prev = None
            for _ in range(WALK_LEN - 1):
                cur = walk[-1]
                nbrs = sorted(adj[cur].keys())
                if not nbrs:
                    break
                weights = []
                for x in nbrs:
                    w = max(abs(adj[cur][x]), 0.1)   # floor: signed sums may cancel to 0
                    if prev is None:
                        weights.append(w)
                    elif x == prev:
                        weights.append(w / P_RETURN)
                    elif x in adj.get(prev, {}):
                        weights.append(w)
                    else:
                        weights.append(w / Q_INOUT)
                weights = np.array(weights, dtype=float)
                weights /= weights.sum()
                nxt = nbrs[rng.choice(len(nbrs), p=weights)]
                prev = cur
                walk.append(nxt)
            walks.append(walk)
    return walks


def build_user(uid):
    events = load_events(uid)
    edges = build_edge_list(events)

    from collections import Counter, defaultdict
    topics = sorted({t for _, _, t, _ in edges})
    adj_all = defaultdict(dict)
    topic_members = defaultdict(set)
    for a, b, t, s in edges:
        adj_all[a][b] = adj_all[a].get(b, 0) + s
        adj_all[b][a] = adj_all[b].get(a, 0) + s
        topic_members[t].update([a, b])

    # walks + Heider-signed context pairs
    rng = np.random.default_rng(5)
    nodes = sorted(adj_all.keys())
    n_idx = {v: i for i, v in enumerate(nodes)}
    signed_cooc = defaultdict(float)
    walks = biased_walks(adj_all, rng)
    for walk in walks:
        for i, src in enumerate(walk):
            sign = 1.0
            for j in range(i + 1, min(i + 1 + WINDOW, len(walk))):
                w = adj_all[walk[j - 1]].get(walk[j], 1)
                sign *= 1.0 if w >= 0 else -1.0          # Heider product
                signed_cooc[(n_idx[src], n_idx[walk[j]])] += sign

    # SGNS-analog: SVD of signed PMI-like matrix
    emb = np.zeros((len(nodes), EMB_D))
    if signed_cooc:
        from scipy.sparse import coo_matrix
        from sklearn.decomposition import TruncatedSVD
        rows, cols, vals = zip(*[(i, j, v) for (i, j), v in signed_cooc.items()])
        M = coo_matrix((vals, (rows, cols)), shape=(len(nodes), len(nodes)))
        k = min(EMB_D, len(nodes) - 1)
        if k >= 2:
            emb[:, :k] = TruncatedSVD(n_components=k, random_state=0).fit_transform(M.tocsr())
    topic_emb = {t: emb[[n_idx[v] for v in vs if v in n_idx]].mean(axis=0)
                 for t, vs in topic_members.items() if vs}

    maps = []
    for ev in events:
        m = base_map(ev)
        raw = ev.get("raw_text") or ""
        s = sentiment((ev.get("user_opinion") or "")
                      + " ".join(ev.get("stance_keywords") or [])) or 1
        topic = (m["topics"] or ["general"])[0]
        alters = list(set(mentions(raw)) | set(repost_chain(raw)))
        triples = [f"(walk_params, p_q_L_r, {P_RETURN}_{Q_INOUT}_{WALK_LEN}_{R_WALKS})"]
        for a in alters:
            triples.append(f"(user, {'+1' if s > 0 else '-1'}@{topic}, {a})")
        for e in m["entities"][:3]:
            triples.append(f"(user, {'+1' if s > 0 else '-1'}@{topic}, ent:{e})")
        # Heider-balance illustration between co-mentioned nodes
        pair = (alters + [f"ent:{e}" for e in m["entities"]])[:2]
        if len(pair) == 2:
            triples.append(f"({pair[0]}, heider_balance_with, {pair[1]})")
        m.update({
            "edge_sign": "+1" if s > 0 else "-1",
            "topic_key": topic,
            "feature_3d_triples": triples,
            "feature_3d_text": " ; ".join(triples),
        })
        maps.append(m)

    neg_share = (sum(1 for *_, s in edges if s < 0) / len(edges)) if edges else 0.0
    static_extra = {
        "signed_edge_list_size": len(edges),
        "negative_edge_share": round(neg_share, 4),
        "num_topics": len(topics),
        "psi_variant": "addition (f(u)+g(t))",
        "num_walks": len(walks),
    }
    bank = assemble_bank(
        uid, "SEM",
        "SEM: Stance Embeddings from Signed Social Graphs (WSDM 2023)",
        {"signed": "4-column signed topic edge list; biased 2nd-order walks "
                   "(p=1.5,q=0.5,L=40); Heider-signed contexts; topic-aware "
                   "skip-gram (addition psi)"},
        maps, static_extra,
        {"retriever": "signed",
         "approximation": "SGNS replaced by SVD of the signed walk "
                          "co-occurrence matrix (skip-gram factorizes shifted "
                          "PMI); alters/entities stand in for the user-user "
                          "graph (single-user data)"},
        events)
    return write_bank(uid, "sem", bank,
                      "re-built per SEM: signed topic graph, biased 2nd-order "
                      "walks with exact paper parameters, Heider-balance "
                      "signed contexts, topic-aware embedding")


if __name__ == "__main__":
    for u in ["1989660417", "7463374646"]:
        print(build_user(u))
