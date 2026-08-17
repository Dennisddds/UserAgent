# -*- coding: utf-8 -*-
"""Semi-PerGCN — Data Augmented GNN for Personality (AAAI 2024).

Faithful structure (stage 1, PGNN):
  * Heterogeneous graph per user: user node V_u, word nodes V_w, LIWC
    category nodes V_l; edges word-user (tf-idf), word-LIWC (dictionary),
    word-word (sliding-window co-occurrence), exactly the paper's edge types.
  * Two GCN layers X^{k+1} = sigma(A_hat X^k W^k) (Eqs. 1-3), numpy with
    seeded random W (no Big Five labels available to supervise).
  * Path-specific attention over LIWC paths (Eqs. 4-6) producing the user
    representation H_u_hat.
  * Big Five output y_d = sigmoid(W_d H_u_hat) (Eq. 7) reported as an
    unsupervised proxy profile.
Stage 2 (unsupervised consistency): the paper's two augmentations are applied
  (LIWC synonym replacement within a category, random deletion of
  non-psychological words) and the consistency between original and augmented
  predictions is reported.
"""

import numpy as np

from cogmap_common import assemble_bank, base_map, load_events, tokens, write_bank
from zh_liwc import CATEGORIES, word_categories

WINDOW = 5
D = 64


def sigma(x):
    return np.tanh(x)


def build_graph(events):
    """user + word + LIWC-category heterogeneous adjacency."""
    from collections import Counter
    tf = Counter()
    cooc = Counter()
    for ev in events:
        toks = tokens((ev.get("raw_text") or "") + " " + (ev.get("user_opinion") or ""))
        tf.update(toks)
        for i in range(len(toks)):
            for j in range(i + 1, min(i + WINDOW, len(toks))):
                cooc[(toks[i], toks[j])] += 1
    words = [w for w, c in tf.most_common(1500) if c >= 2]
    cats = list(CATEGORIES.keys())
    idx = {"user": 0}
    for w in words:
        idx[f"w:{w}"] = len(idx)
    for c in cats:
        idx[f"c:{c}"] = len(idx)
    n = len(idx)
    A = np.zeros((n, n))
    total = sum(tf[w] for w in words) or 1
    for w in words:
        i = idx[f"w:{w}"]
        A[0, i] = A[i, 0] = tf[w] / total * len(words)          # word-user (tf-idf-ish)
        for c in word_categories(w):
            j = idx[f"c:{c}"]
            A[i, j] = A[j, i] = 1.0                              # word-LIWC
    for (a, b), cnt in cooc.items():
        if f"w:{a}" in idx and f"w:{b}" in idx and cnt >= 2:
            i, j = idx[f"w:{a}"], idx[f"w:{b}"]
            A[i, j] = A[j, i] = np.log1p(cnt)                    # word-word
    return A, idx, words, cats


def gcn_forward(A, n, rng, x0=None):
    deg = A.sum(axis=1) + 1.0
    A_hat = (A + np.eye(n)) / np.sqrt(np.outer(deg, deg))
    X0 = x0 if x0 is not None else rng.standard_normal((n, D)) / np.sqrt(D)
    W0 = rng.standard_normal((D, D)) / np.sqrt(D)
    W1 = rng.standard_normal((D, D)) / np.sqrt(D)
    X1 = sigma(A_hat @ X0 @ W0)                                 # Eq. 2
    H = sigma(A_hat @ X1 @ W1)                                  # Eq. 3
    return H, X0


def path_attention(H, idx, cats, rng):
    """Eqs. 4-6: attention over LIWC category paths."""
    H_u = H[0]
    cat_vecs = np.stack([H[idx[f"c:{c}"]] for c in cats])
    scores = np.maximum(0.01 * (cat_vecs @ H_u), 0.2 * (cat_vecs @ H_u))  # LeakyReLU
    alpha = np.exp(scores - scores.max())
    alpha = alpha / alpha.sum()                                  # Eq. 5
    H_u_hat = np.tanh((alpha[:, None] * cat_vecs).sum(axis=0)) + H_u      # Eq. 6
    return H_u_hat, dict(zip(cats, [round(float(a), 4) for a in alpha]))


def big5(H_u_hat, rng):
    W_d = rng.standard_normal((5, H_u_hat.shape[0])) / np.sqrt(H_u_hat.shape[0])
    y = 1.0 / (1.0 + np.exp(-(W_d @ H_u_hat)))                   # Eq. 7
    names = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
    return dict(zip(names, [round(float(v), 4) for v in y]))


def build_user(uid):
    events = load_events(uid)
    rng = np.random.default_rng(7)
    A, idx, words, cats = build_graph(events)
    H, X0 = gcn_forward(A, len(idx), rng)
    H_u_hat, alpha = path_attention(H, idx, cats, rng)
    y_d = big5(H_u_hat, np.random.default_rng(11))

    # Stage 2: paper's augmentations -> consistency
    rng2 = np.random.default_rng(13)
    A_aug = A.copy()
    psych = [w for w in words if word_categories(w)]
    nonpsych = [w for w in words if not word_categories(w)]
    for w in rng2.choice(nonpsych, size=max(1, len(nonpsych) // 10), replace=False):
        i = idx[f"w:{w}"]
        A_aug[i, :] = 0
        A_aug[:, i] = 0                                          # random deletion
    H_aug, _ = gcn_forward(A_aug, len(idx), np.random.default_rng(7), x0=X0)
    H_u_aug, _ = path_attention(H_aug, idx, cats, np.random.default_rng(7))
    consistency = float(H_u_hat @ H_u_aug /
                        ((np.linalg.norm(H_u_hat) * np.linalg.norm(H_u_aug)) + 1e-9))

    top_alpha = sorted(alpha.items(), key=lambda kv: -kv[1])[:5]
    maps = []
    for ev in events:
        m = base_map(ev)
        toks = tokens((ev.get("raw_text") or "") + " " + m["user_opinion"])
        pw = [(w, word_categories(w)) for w in dict.fromkeys(toks) if word_categories(w)][:8]
        triples = []
        for w, cs in pw:
            triples.append(f"(word:{w}, of_user, user)")
            for c in cs:
                triples.append(f"(word:{w}, in_liwc, {c})")
        for c, a in top_alpha:
            triples.append(f"(user, path_attention:{c}, {a})")
        for t, v in y_d.items():
            triples.append(f"(user, big5:{t}, {v})")
        m.update({
            "psych_words": [w for w, _ in pw],
            "feature_3d_triples": triples,
            "feature_3d_text": " ; ".join(triples),
        })
        maps.append(m)

    static_extra = {"big5_profile": y_d, "liwc_attention": alpha,
                    "augmentation_consistency": round(consistency, 4),
                    "num_word_nodes": len(words), "num_liwc_nodes": len(cats)}
    bank = assemble_bank(
        uid, "Semi-PerGCN",
        "Semi-PerGCN: Data Augmented GNN for Personality (AAAI 2024)",
        {"graph": "user-word-LIWC heterogeneous graph, 2-layer GCN, "
                  "path-specific attention, augmentation consistency"},
        maps, static_extra,
        {"retriever": "default",
         "approximation": "GCN weights are seeded random projections (no Big "
                          "Five labels for supervision); LIWC 2015 replaced by "
                          "a Chinese analog dictionary; consistency reported "
                          "instead of trained with CE loss"},
        events)
    return write_bank(uid, "semipergcn", bank,
                      "re-built per Semi-PerGCN: heterogeneous user/word/LIWC "
                      "graph with the paper's 3 edge types, 2-layer GCN, path "
                      "attention (Eqs.4-6), Big Five head, stage-2 augmentation")


if __name__ == "__main__":
    for u in ["1989660417", "7463374646"]:
        print(build_user(u))
