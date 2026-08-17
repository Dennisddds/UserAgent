# -*- coding: utf-8 -*-
"""TrigNet — Psycholinguistic Tripartite Graph Network (ACL 2021).

Faithful structure:
  * Tripartite heterogeneous graph G=(V,E): post nodes V_p, psycholinguistic
    word nodes V_w (words appearing in posts AND matching the LIWC-analog
    dictionary), category nodes V_c (15 = 9 main + 6 personal-concern,
    exactly the paper's node scheme).
  * Two interaction flows only (the paper's key design): "pwp" (posts interact
    through shared psycholinguistic words) and "pwcwp" (posts interact through
    words sharing a LIWC category). No direct post-post edges.
  * Flow GAT: attention message passing along each flow (similarity-based
    attention with LeakyReLU slope 0.2, tanh combination, residual), L=1
    layer as selected in the paper.
  * User representation u = mean of final post nodes.
"""

import numpy as np

from cogmap_common import assemble_bank, base_map, encode_text, load_events, tokens, write_bank
from zh_liwc import CATEGORIES, word_categories

MAX_WORDS = 1200


def build_user(uid):
    events = load_events(uid)
    cats = list(CATEGORIES.keys())

    # ---- graph construction -------------------------------------------------
    from collections import Counter, defaultdict
    tf = Counter()
    post_words = []
    for ev in events:
        toks = [w for w in dict.fromkeys(tokens((ev.get("raw_text") or "")
                                                + " " + (ev.get("user_opinion") or "")))
                if word_categories(w)]
        post_words.append(toks)
        tf.update(toks)
    words = [w for w, _ in tf.most_common(MAX_WORDS)]
    wset = set(words)
    word_cat = {w: word_categories(w) for w in words}

    w2p = defaultdict(list)   # word -> posts
    for pi, toks in enumerate(post_words):
        for w in toks:
            if w in wset:
                w2p[w].append(pi)
    c2w = defaultdict(list)   # category -> words
    for w in words:
        for c in word_cat[w]:
            c2w[c].append(w)

    # ---- node init (BERT replaced by the hashing sentence encoder) ---------
    P = np.array([encode_text((ev.get("event_summary") or "") + " "
                              + (ev.get("raw_text") or "")) for ev in events])
    W = np.array([encode_text(w * 3) for w in words]) if words else np.zeros((0, P.shape[1]))
    C = np.array([encode_text(CATEGORIES[c][:60]) for c in cats])

    def gat(dst, srcs):
        """similarity attention + tanh + residual (single head analog)."""
        if len(srcs) == 0:
            return dst
        sims = srcs @ dst
        sims = np.where(sims > 0, sims, 0.2 * sims)             # LeakyReLU
        a = np.exp(sims - sims.max())
        a = a / a.sum()
        return dst + np.tanh((a[:, None] * srcs).sum(axis=0))    # residual

    # ---- Flow GAT (1 layer): word<-post, category<-word, word<-category,
    #      post<-word ; flows pwp and pwcwp ----------------------------------
    W1 = np.array([gat(W[i], P[w2p[w]]) for i, w in enumerate(words)]) if words else W
    C1 = np.array([gat(C[i], W1[[words.index(w) for w in c2w[c]]] if c2w[c] else np.zeros((0, P.shape[1])))
                   for i, c in enumerate(cats)])
    Wc = np.array([gat(W1[i], C1[[cats.index(c) for c in word_cat[w]]])
                   for i, w in enumerate(words)]) if words else W1
    P_pwp = np.array([gat(P[pi], W1[[words.index(w) for w in post_words[pi] if w in wset]]
                          if any(w in wset for w in post_words[pi]) else np.zeros((0, P.shape[1])))
                      for pi in range(len(events))])
    P_pwcwp = np.array([gat(P[pi], Wc[[words.index(w) for w in post_words[pi] if w in wset]]
                            if any(w in wset for w in post_words[pi]) else np.zeros((0, P.shape[1])))
                        for pi in range(len(events))])
    P_final = (P_pwp + P_pwcwp) / 2.0                            # two-flow mean
    user_vec = P_final.mean(axis=0)

    # ---- per-post maps ------------------------------------------------------
    maps = []
    for pi, ev in enumerate(events):
        m = base_map(ev)
        pw = [w for w in post_words[pi] if w in wset][:8]
        triples = []
        for w in pw:
            triples.append(f"(post, contains_psych_word, {w})")
            for c in word_cat[w]:
                triples.append(f"(word:{w}, in_category, {c})")
        # pwp / pwcwp neighbours: strongest co-word posts
        nbr = Counter()
        for w in pw:
            for pj in w2p[w]:
                if pj != pi:
                    nbr[pj] += 1
        for pj, _ in nbr.most_common(2):
            triples.append(f"(post, flow_pwp, post:{events[pj]['post_id']})")
        cat_nbr = Counter()
        for w in pw:
            for c in word_cat[w]:
                cat_nbr[c] += 1
        for c, _ in cat_nbr.most_common(2):
            triples.append(f"(post, flow_pwcwp_via, {c})")
        m.update({
            "psych_words": pw,
            "feature_3d_triples": triples,
            "feature_3d_text": " ; ".join(triples),
        })
        maps.append(m)

    static_extra = {
        "tripartite_size": {"posts": len(events), "psych_words": len(words),
                            "categories": len(cats)},
        "user_vector_norm": round(float(np.linalg.norm(user_vec)), 4),
        "top_categories": Counter(c for w in words for c in word_cat[w]).most_common(8),
    }
    bank = assemble_bank(
        uid, "TrigNet",
        "TrigNet: Psycholinguistic Tripartite Graph Network (ACL 2021)",
        {"flows": "tripartite post-word-category graph; pwp and pwcwp flows "
                  "with Flow-GAT attention; user = mean of post nodes"},
        maps, static_extra,
        {"retriever": "default",
         "approximation": "BERT init replaced by hashing sentence encoder; "
                          "single-head similarity attention; LIWC 2015 "
                          "replaced by Chinese analog"},
        events)
    return write_bank(uid, "trignet", bank,
                      "re-built per TrigNet: true tripartite graph with only "
                      "pwp/pwcwp flows, Flow-GAT with LeakyReLU(0.2)+tanh+"
                      "residual, 15 category nodes")


if __name__ == "__main__":
    for u in ["1989660417", "7463374646"]:
        print(build_user(u))
