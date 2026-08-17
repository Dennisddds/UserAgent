# -*- coding: utf-8 -*-
"""KGrAt-Net — Knowledge Graph Attention Network (Scientific Reports 2022).

Faithful to Algorithm 1:
  Phase 1 preprocessing per essay(post): tokenization, noise/stopword removal,
    normalization, NER (extraction-layer entities as the NER analog),
    dedup/final preparation.
  Phase 2 knowledge representation: per-post knowledge graph from a
    DBpedia-analog background graph (entity-entity edges from corpus-level
    co-occurrence, since DBpedia has no coverage of these Weibo entities),
    then *pruning*: keep only edges whose subject AND object occur in the
    post (the paper's pruning rule); edge labels removed, multigraph->simple.
  Aggregation: all pruned per-post KGs merged; each post node appended and
    connected to its entities (Algorithm 1 lines 10-14).
  Embedding: RDF2Vec analog — truncated random walks (depth 5, 5 walks/node,
    the paper's walk parameters) turned into co-occurrence PPMI + SVD vectors.
  Attention: per post node, GAT attention (LeakyReLU) over 1-hop entities.
"""

import numpy as np

from cogmap_common import assemble_bank, base_map, load_events, write_bank

WALK_DEPTH = 5
WALKS_PER_NODE = 5
EMB_D = 64


def build_user(uid):
    events = load_events(uid)
    from collections import Counter, defaultdict

    # ---- Phase 2: background KG (DBpedia analog) ----------------------------
    co = Counter()
    for ev in events:
        ents = list(dict.fromkeys(ev.get("entities") or []))
        for i, a in enumerate(ents):
            for b in ents[i + 1:]:
                co[tuple(sorted((a, b)))] += 1
    bg_edges = {e for e, c in co.items() if c >= 2}   # simple graph, unlabeled

    # ---- per-post pruned KGs + aggregation ----------------------------------
    agg_adj = defaultdict(set)
    post_edges = []
    for ev in events:
        text = (ev.get("raw_text") or "") + (ev.get("event_summary") or "")
        ents = [e for e in (ev.get("entities") or []) if e]
        pruned = [(a, b) for (a, b) in bg_edges
                  if a in ents and b in ents and a in text and b in text]
        post_edges.append(pruned)
        for a, b in pruned:
            agg_adj[a].add(b)
            agg_adj[b].add(a)
    # append essay nodes (Algorithm 1 lines 11-14)
    for ev in events:
        p = f"post:{ev['post_id']}"
        for e in ev.get("entities") or []:
            agg_adj[p].add(e)
            agg_adj[e].add(p)

    # ---- RDF2Vec-analog embeddings: random walks -> PPMI -> SVD -------------
    nodes = sorted(agg_adj.keys())
    n_idx = {v: i for i, v in enumerate(nodes)}
    rng = np.random.default_rng(3)
    cooc = Counter()
    for v in nodes:
        for _ in range(WALKS_PER_NODE):
            cur = v
            walk = [cur]
            for _ in range(WALK_DEPTH):
                nbrs = sorted(agg_adj[cur])
                if not nbrs:
                    break
                cur = nbrs[rng.integers(len(nbrs))]
                walk.append(cur)
            for i, a in enumerate(walk):
                for b in walk[i + 1:i + 3]:
                    if a != b:
                        cooc[(n_idx[a], n_idx[b])] += 1
                        cooc[(n_idx[b], n_idx[a])] += 1
    emb = np.zeros((len(nodes), EMB_D))
    if cooc:
        from scipy.sparse import coo_matrix
        from sklearn.decomposition import TruncatedSVD
        rows, cols, vals = zip(*[(i, j, c) for (i, j), c in cooc.items()])
        M = coo_matrix((vals, (rows, cols)), shape=(len(nodes), len(nodes))).tocsr()
        total = M.sum()
        row_sum = np.asarray(M.sum(axis=1)).ravel() + 1e-9
        col_sum = np.asarray(M.sum(axis=0)).ravel() + 1e-9
        M = M.tocoo()
        ppmi_vals = np.maximum(np.log((M.data * total) / (row_sum[M.row] * col_sum[M.col])), 0)
        P = coo_matrix((ppmi_vals, (M.row, M.col)), shape=(len(nodes), len(nodes)))
        k = min(EMB_D, len(nodes) - 1)
        if k >= 2:
            emb[:, :k] = TruncatedSVD(n_components=k, random_state=0).fit_transform(P.tocsr())

    # ---- per-post GAT attention over 1-hop entities -------------------------
    maps = []
    for pi, ev in enumerate(events):
        m = base_map(ev)
        p = f"post:{ev['post_id']}"
        ents = [e for e in m["entities"] if e in n_idx]
        triples = []
        att = []
        if ents and p in n_idx:
            h_p = emb[n_idx[p]]
            scores = np.array([float(emb[n_idx[e]] @ h_p) for e in ents])
            scores = np.where(scores > 0, scores, 0.2 * scores)   # LeakyReLU
            a = np.exp(scores - scores.max())
            a = a / a.sum()
            att = sorted(zip(ents, a), key=lambda kv: -kv[1])
        for e in ents:
            triples.append(f"({p}, connected_to, {e})")
        for a, b in post_edges[pi][:6]:
            triples.append(f"({a}, kg_edge, {b})")
        for e, w in att[:4]:
            triples.append(f"(attention, focus[{round(float(w), 3)}], {e})")
        m.update({
            "feature_3d_triples": triples,
            "feature_3d_text": " ; ".join(triples),
        })
        maps.append(m)

    static_extra = {
        "aggregated_kg": {"nodes": len(nodes),
                          "entity_entity_edges": len(bg_edges),
                          "post_entity_edges": sum(len(ev.get("entities") or [])
                                                   for ev in events)},
    }
    bank = assemble_bank(
        uid, "KGrAt-Net",
        "KGrAt-Net: Knowledge Graph Attention Network (Scientific Reports 2022)",
        {"kg": "per-post pruned KGs aggregated + essay nodes; RDF2Vec-analog "
               "walk embeddings; GAT attention over 1-hop entities"},
        maps, static_extra,
        {"retriever": "multihop", "walk_depth": WALK_DEPTH,
         "walks_per_node": WALKS_PER_NODE,
         "approximation": "DBpedia replaced by corpus co-occurrence background "
                          "KG (no DBpedia coverage for these entities); "
                          "RDF2Vec word2vec replaced by walk-PPMI+SVD"},
        events)
    return write_bank(uid, "kgrat", bank,
                      "re-built per KGrAt-Net Algorithm 1: preprocessing, "
                      "per-essay KG + pruning rule, aggregation with essay "
                      "nodes, walk embeddings, graph attention")


if __name__ == "__main__":
    for u in ["1989660417", "7463374646"]:
        print(build_user(u))
