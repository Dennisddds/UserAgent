# -*- coding: utf-8 -*-
"""GoRec — Key Opinion Leaders in Recommendation (WSDM 2020).

Faithful structure:
  Part 1 Translation-based opinion elicitation:
    * KOLs = accounts the user reposts/mentions most; items = topics;
      opinion types = stance keywords (paper: each unique opinion word is an
      opinion type).
    * Opinion graph G_o of (KOL, opinion, item) triples from posts where a
      KOL and topic co-occur.
    * TransD-style translation embeddings trained with the paper's margin
      loss [gamma + s(k,o,i) - s(k,o,i')]+ (gamma=1), s = -||k + o - i||^2,
      via a small seeded SGD (numpy).
  Part 2 Neural graph-based opinion diffusion:
    * Fusing layer: user representation attends over followed KOLs' opinion
      embeddings (alpha_up softmax).
    * GNN propagation over the user-item bipartite graph with symmetric
      1/sqrt(|Nu||Ni|) normalization (3 layers, the paper's depth).
    * Recommendation scores y_u = sigma(V x_u).
"""

import numpy as np

from cogmap_common import (assemble_bank, base_map, load_events, mentions,
                    repost_chain, write_bank)

GAMMA = 1.0
EMB_D = 32
EPOCHS = 30
LR = 0.05


def build_user(uid):
    events = load_events(uid)
    from collections import Counter, defaultdict

    kol_freq = Counter()
    for ev in events:
        raw = ev.get("raw_text") or ""
        for a in repost_chain(raw):
            kol_freq[a] += 2
        for a in mentions(raw):
            kol_freq[a] += 1
    kols = [k for k, _ in kol_freq.most_common(30)]
    kset = set(kols)

    triples_go = []      # (kol, opinion, item)
    user_items = Counter()
    for ev in events:
        raw = ev.get("raw_text") or ""
        post_kols = [a for a in set(mentions(raw)) | set(repost_chain(raw)) if a in kset]
        ops = ev.get("stance_keywords") or []
        for t in ev.get("topics") or []:
            user_items[t] += 1
            for k in post_kols:
                for o in ops[:2]:
                    triples_go.append((k, o, t))
    items = sorted(user_items)
    opinions = sorted({o for _, o, _ in triples_go})

    # ---- Part 1: TransD-style translation embeddings (margin SGD) ----------
    rng = np.random.default_rng(17)
    K = {k: rng.standard_normal(EMB_D) / np.sqrt(EMB_D) for k in kols}
    O = {o: rng.standard_normal(EMB_D) / np.sqrt(EMB_D) for o in opinions}
    I = {i: rng.standard_normal(EMB_D) / np.sqrt(EMB_D) for i in items}

    def score(k, o, i):
        d = K[k] + O[o] - I[i]
        return -float(d @ d)

    if triples_go and len(items) > 1:
        for _ in range(EPOCHS):
            for (k, o, i) in triples_go:
                i_neg = items[rng.integers(len(items))]
                if i_neg == i:
                    continue
                margin = GAMMA + score(k, o, i_neg) - score(k, o, i)
                if margin > 0:   # hinge active: push apart
                    d_pos = K[k] + O[o] - I[i]
                    d_neg = K[k] + O[o] - I[i_neg]
                    K[k] -= LR * 2 * (d_pos - d_neg)
                    O[o] -= LR * 2 * (d_pos - d_neg)
                    I[i] += LR * 2 * d_pos
                    I[i_neg] -= LR * 2 * d_neg

    # ---- Part 2: fusing layer + bipartite GNN -------------------------------
    e_u = np.mean([I[i] * user_items[i] for i in items], axis=0) if items else np.zeros(EMB_D)
    if kols:
        att = np.array([float(K[k] @ e_u) for k in kols])
        att = np.exp(att - att.max())
        att /= att.sum()
        n_u = (att[:, None] * np.stack([K[k] for k in kols])).sum(axis=0)
    else:
        att = np.array([])
        n_u = np.zeros(EMB_D)
    x_u = np.maximum(np.concatenate([n_u, e_u]), 0)

    x_items = {i: np.maximum(np.concatenate([I[i], I[i]]), 0) for i in items}
    Nu = sum(user_items.values()) or 1
    for _ in range(3):   # 3-layer propagation
        agg = np.zeros_like(x_u)
        for i in items:
            agg += x_items[i] * user_items[i] / np.sqrt(Nu * user_items[i])
        x_u_new = np.maximum(agg, 0)
        for i in items:
            x_items[i] = np.maximum(x_u * user_items[i] / np.sqrt(Nu * user_items[i]), 0)
        x_u = x_u_new
    rec_scores = {i: float(x_u @ x_items[i]) for i in items}
    top_rec = sorted(rec_scores, key=lambda i: -rec_scores[i])[:10]
    kol_att = dict(zip(kols, [round(float(a), 4) for a in att])) if kols else {}

    maps = []
    for ev in events:
        m = base_map(ev)
        raw = ev.get("raw_text") or ""
        post_kols = [a for a in set(mentions(raw)) | set(repost_chain(raw)) if a in kset]
        ops = m["stance_keywords"][:2]
        triples = []
        for k in post_kols:
            for o in ops:
                for t in m["topics"][:2]:
                    sc = round(score(k, o, t), 3) if (o in O and t in I) else 0.0
                    triples.append(f"(KOL:{k}, opinion:{o}[s={sc}], item:{t})")
            triples.append(f"(user, fuses_kol[alpha={kol_att.get(k, 0)}], {k})")
        if not post_kols:
            for o in ops:
                for t in m["topics"][:2]:
                    triples.append(f"(user_as_kol, opinion:{o}, item:{t})")
        for t in m["topics"]:
            if t in top_rec:
                triples.append(f"(gnn_diffusion, recommends, item:{t})")
        m.update({
            "opinion_relation": ops[0] if ops else None,
            "feature_3d_triples": triples,
            "feature_3d_text": " ; ".join(triples),
        })
        maps.append(m)

    static_extra = {
        "kols": kols,
        "kol_attention": kol_att,
        "num_opinion_triples": len(triples_go),
        "top_recommended_items": top_rec,
    }
    bank = assemble_bank(
        uid, "GoRec",
        "GoRec: Key Opinion Leaders in Recommendation (WSDM 2020)",
        {"translation": "(KOL, opinion, item) triples with TransD margin "
                        "training; fusing-layer KOL attention; 3-layer "
                        "bipartite GNN diffusion"},
        maps, static_extra,
        {"retriever": "default", "gamma": GAMMA, "emb_dim": EMB_D,
         "approximation": "KOLs proxied by reposted/mentioned accounts; items "
                          "= topics; small seeded SGD instead of full-scale "
                          "training; single-user bipartite graph"},
        events)
    return write_bank(uid, "gorec", bank,
                      "re-built per GoRec: translation-based opinion "
                      "elicitation (margin loss, TransD scoring) + KOL "
                      "attention fusing + normalized bipartite GNN diffusion")


if __name__ == "__main__":
    for u in ["1989660417", "7463374646"]:
        print(build_user(u))
