# -*- coding: utf-8 -*-
"""D-DGCN — Dynamic Deep Graph Convolutional Network (AAAI 2023).

Faithful structure:
  * Post encoder: each post independently encoded (BERT replaced by TF-IDF +
    SVD sentence vectors, documented).
  * A special user node u initialized as the mean of all post vectors and
    connected through the learned graph (paper Sec. 3.1).
  * L2C (learn-to-connect): r_ij = sigmoid(ReLU(h_i W_Q) . (h_j W_K)^T);
    a_ij = r_ij normalized, kept only when r_ij > tau (tau = 0.5, the paper's
    threshold), giving a *dynamic directed* adjacency per layer.
  * DGCN: decoupled propagation H^{k+1} = A_hat^k H^k (no weight matrix),
    L = 3 layers, adjacency re-learned from the previous layer's output.
  * Layer attention fusion over [H^0..H^L] (paper Eq. S = sigma(H c)).

W_Q / W_K are deterministic random projections (seeded); no supervised
personality labels exist here, so the classifier head is not trained; the
deliverable is the learned dynamic graph itself.
"""

import numpy as np

from cogmap_common import assemble_bank, base_map, load_events, tokens, write_bank

TAU = 0.5
L_LAYERS = 3
D = 64


def post_vectors(events):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    docs = [" ".join(tokens((ev.get("raw_text") or "") + " " + (ev.get("event_summary") or "")))
            for ev in events]
    tfidf = TfidfVectorizer(max_features=20000).fit_transform(docs)
    d = min(D, tfidf.shape[1] - 1, tfidf.shape[0] - 1)
    H = TruncatedSVD(n_components=max(d, 2), random_state=0).fit_transform(tfidf)
    H = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-9)
    return H


def l2c(H, rng):
    """Learn-to-connect: dynamic directed adjacency from current states."""
    d = H.shape[1]
    W_Q = rng.standard_normal((d, d)) / np.sqrt(d)
    W_K = rng.standard_normal((d, d)) / np.sqrt(d)
    Q = np.maximum(H @ W_Q, 0.0)
    K = H @ W_K
    R = 1.0 / (1.0 + np.exp(-(Q @ K.T)))            # r_ij
    A = np.where(R > TAU, R, 0.0)                   # threshold tau
    A = A / (A.sum(axis=1, keepdims=True) + 1e-9)   # a_hat normalization
    return A


def build_user(uid):
    events = load_events(uid)
    H0 = post_vectors(events)
    n = len(events)
    u0 = H0.mean(axis=0, keepdims=True)             # user node init = mean
    H = np.vstack([H0, u0])                         # user node index = n

    rng = np.random.default_rng(42)
    layers = [H]
    adjs = []
    for k in range(L_LAYERS):
        A = l2c(layers[-1], rng)
        A[:, n] = np.maximum(A[:, n], 1e-3)         # keep user node reachable
        adjs.append(A)
        layers.append(A @ layers[-1])               # DGCN: propagation only

    # layer attention fusion: S = sigma(H . c)
    stack = np.stack(layers, axis=1)                # (n+1, L+1, d)
    c = stack.mean(axis=(0, 1))
    c /= (np.linalg.norm(c) + 1e-9)
    S = 1.0 / (1.0 + np.exp(-(stack @ c)))          # (n+1, L+1)
    S = S / S.sum(axis=1, keepdims=True)
    H_out = np.einsum("nl,nld->nd", S, stack)

    A_last = adjs[-1]
    maps = []
    for i, ev in enumerate(events):
        m = base_map(ev)
        nbr_idx = np.argsort(-A_last[i, :n])[:5]
        nbrs = [(events[j]["post_id"], round(float(A_last[i, j]), 3))
                for j in nbr_idx if A_last[i, j] > 0]
        triples = [f"(user, aggregates, post:{ev['post_id']})"]
        for pid, w in nbrs:
            triples.append(f"(post:{ev['post_id']}, l2c_edge[a={w}], post:{pid})")
        lw = [round(float(x), 3) for x in S[i]]
        triples.append(f"(layer_attention, weights, {lw})")
        for e in m["entities"]:
            triples.append(f"(post:{ev['post_id']}, mentions, {e})")
        m.update({
            "graph_neighbors": [pid for pid, _ in nbrs],
            "feature_3d_triples": triples,
            "feature_3d_text": " ; ".join(triples),
        })
        maps.append(m)

    static_extra = {
        "user_node_norm": round(float(np.linalg.norm(H_out[n])), 4),
        "dynamic_graph_density": [round(float((A > 0).mean()), 4) for A in adjs],
    }
    bank = assemble_bank(
        uid, "D-DGCN",
        "D-DGCN: Dynamic Deep Graph Convolutional Network (AAAI 2023)",
        {"l2c": "sigmoid(ReLU(hWQ).(hWK)^T) thresholded at tau=0.5, per-layer "
                "dynamic directed adjacency",
         "dgcn": "decoupled propagation H<-A_hat H, layer-attention fusion, "
                 "special user node"},
        maps, static_extra,
        {"retriever": "default", "tau": TAU, "layers": L_LAYERS,
         "approximation": "BERT post encoder replaced by TF-IDF+SVD; W_Q/W_K "
                          "are seeded random projections (no labels to train "
                          "the L2C end-to-end)"},
        events)
    return write_bank(uid, "ddgcn", bank,
                      "re-built per D-DGCN: post encoder + user node + L2C "
                      "dynamic adjacency (tau=0.5) + decoupled DGCN propagation "
                      "+ layer attention")


if __name__ == "__main__":
    for u in ["1989660417", "7463374646"]:
        print(build_user(u))
