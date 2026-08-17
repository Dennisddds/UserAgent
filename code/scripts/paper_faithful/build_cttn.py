# -*- coding: utf-8 -*-
"""CT-TN — Few-shot Cross-Target Stance Detection (IEEE TCSS 2023).

Faithful structure: three parallel components + majority voting:
  Component 1 (text): target-conditioned encoding "[CLS] target [SEP] context"
    (RoBERTa replaced by lexicon stance scoring of the target-conditioned
    text) -> favor/against/none.
  Component 2 (graphs x3): followers / friends / likes network encodings.
    Twitter network data is unavailable for Weibo, so the paper's three
    network types are mapped to their closest observable analogs:
      followers-analog: accounts this user reposts from (//@source),
      friends-analog:   accounts mentioned directly (@name),
      likes-analog:     entity graph of the user's highly-liked posts.
    Each graph gets Node2Vec-analog embeddings (adjacency SVD) and votes by
    the aggregated stance of the target's graph neighbourhood.
  Component 3 (aggregation): majority vote over the 4 votes, exactly the
    paper's output aggregation.
"""

import numpy as np

from cogmap_common import (assemble_bank, base_map, load_events, mentions,
                    repost_chain, sentiment, write_bank)


def stance_label(s):
    return "favor" if s > 0 else ("against" if s < 0 else "none")


def build_graphs(events):
    """followers/friends/likes analog graphs: node -> Counter(neighbor)."""
    from collections import Counter, defaultdict
    g = {"followers": defaultdict(Counter), "friends": defaultdict(Counter),
         "likes": defaultdict(Counter)}
    likes_median = sorted(ev.get("likes") or 0 for ev in events)[len(events) // 2]
    for ev in events:
        raw = ev.get("raw_text") or ""
        rc = repost_chain(raw)
        for src in rc:
            g["followers"]["user"][src] += 1
        for mnt in mentions(raw):
            if mnt not in rc:
                g["friends"]["user"][mnt] += 1
        if (ev.get("likes") or 0) >= likes_median:
            ents = ev.get("entities") or []
            for i, a in enumerate(ents):
                g["likes"]["user"][a] += 1
                for b in ents[i + 1:]:
                    g["likes"][a][b] += 1
                    g["likes"][b][a] += 1
    return g


def graph_embeddings(graph):
    """Node2Vec analog: normalized adjacency SVD (128-d in paper; d
    bounded by graph size here)."""
    nodes = sorted(set(graph.keys()) | {n for c in graph.values() for n in c})
    if len(nodes) < 3:
        return {}, nodes
    idx = {v: i for i, v in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)))
    for a, cnt in graph.items():
        for b, w in cnt.items():
            A[idx[a], idx[b]] = np.log1p(w)
            A[idx[b], idx[a]] = np.log1p(w)
    from sklearn.decomposition import TruncatedSVD
    k = min(32, len(nodes) - 1)
    emb = TruncatedSVD(n_components=k, random_state=0).fit_transform(A)
    return {v: emb[idx[v]] for v in nodes}, nodes


def build_user(uid):
    events = load_events(uid)
    graphs = build_graphs(events)
    embs = {name: graph_embeddings(g)[0] for name, g in graphs.items()}

    # per-target neighbourhood stance from each graph
    from collections import Counter, defaultdict
    target_stance = defaultdict(lambda: defaultdict(list))
    for ev in events:
        s = sentiment((ev.get("user_opinion") or "")
                      + " ".join(ev.get("stance_keywords") or []))
        for name, g in graphs.items():
            for tgt in (ev.get("entities") or [])[:2]:
                if tgt in g or tgt in g.get("user", {}):
                    target_stance[name][tgt].append(s)

    maps = []
    for ev in events:
        m = base_map(ev)
        target = (m["entities"] or m["topics"] or ["无明确对象"])[0]
        ctx = m["event_summary"] or (ev.get("raw_text") or "")[:80]

        # component 1: text (target-conditioned)
        s_text = sentiment(f"{target} {ctx} {m['user_opinion']} "
                           + " ".join(m["stance_keywords"]))
        votes = {"text": stance_label(s_text)}
        # components 2-4: graph votes
        for name in ("followers", "friends", "likes"):
            hist = target_stance[name].get(target)
            votes[name] = stance_label(int(np.sign(sum(hist))) if hist else 0)
        tally = Counter(votes.values())
        final = tally.most_common(1)[0][0]

        triples = [f"(target, is, {target})",
                   f"(text_encoder, conditions_on, [CLS] {target} [SEP] {ctx[:40]})"]
        for comp, v in votes.items():
            triples.append(f"(component:{comp}, votes, {v})")
        triples.append(f"(majority_vote, stance, {final})")
        for e in m["entities"][1:3]:
            triples.append(f"(related_target, is, {e})")
        m.update({
            "target": target,
            "stance_label": final,
            "component_votes": votes,
            "feature_3d_triples": triples,
            "feature_3d_text": " ; ".join(triples),
        })
        maps.append(m)

    static_extra = {
        "network_sizes": {k: len(v) for k, v in embs.items()},
        "followers_analog_top": Counter(graphs["followers"]["user"]).most_common(10),
        "friends_analog_top": Counter(graphs["friends"]["user"]).most_common(10),
    }
    bank = assemble_bank(
        uid, "CT-TN",
        "CT-TN: Few-shot Cross-Target Stance Detection (IEEE TCSS 2023)",
        {"components": "target-conditioned text + followers/friends/likes "
                       "graph encodings + majority voting"},
        maps, static_extra,
        {"retriever": "signed",
         "approximation": "RoBERTa replaced by lexicon stance scoring; Twitter "
                          "follower/friend/like networks replaced by repost/"
                          "mention/high-like analogs observable in Weibo data; "
                          "Node2Vec replaced by adjacency SVD"},
        events)
    return write_bank(uid, "cttn", bank,
                      "re-built per CT-TN: 4 parallel components (1 text + 3 "
                      "networks) with majority-vote aggregation per target")


if __name__ == "__main__":
    for u in ["1989660417", "7463374646"]:
        print(build_user(u))
