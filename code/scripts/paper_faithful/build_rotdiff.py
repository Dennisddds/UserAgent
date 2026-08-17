# -*- coding: utf-8 -*-
"""RotDiff — Hyperbolic Rotation for Information Diffusion (CIKM 2023).

Faithful structure:
  * Two relational graphs: social graph (mention edges) and diffusion graph
    (repost cascades parsed from '//@A: ... //@B: ...' chains, giving
    time-ordered diffusion paths B -> A -> user).
  * Lorentz model L^d with curvature c=1: nodes embedded as
    x = (x0, z), x0 = sqrt(1/c + ||z||^2) (the paper's phi map); the spatial
    part z comes from adjacency SVD scaled so that structurally central nodes
    sit near the origin (hierarchy-aware radial placement).
  * Distinct block-diagonal rotations Rot_S / Rot_D applied to the spatial
    part for the social and diffusion spaces (angles seeded), matching the
    paper's use of rotations to separate the two relation types.
  * Scoring f(u,v) = -d_L^2(x_u, x_v) with the squared Lorentz distance
    d_L^2 = -2/c - 2<x,y>_L; per cascade, next-user candidates ranked by
    diffusion-space score.
"""

import numpy as np

from cogmap_common import (assemble_bank, base_map, load_events, mentions,
                    repost_chain, write_bank)

C_CURV = 1.0
EMB_D = 16  # spatial dim (even, for 2x2 rotation blocks)


def lorentz_lift(z):
    x0 = np.sqrt(1.0 / C_CURV + (z ** 2).sum(axis=-1, keepdims=True))
    return np.concatenate([x0, z], axis=-1)


def lorentz_sqdist(x, y):
    inner = -x[..., 0] * y[..., 0] + (x[..., 1:] * y[..., 1:]).sum(axis=-1)
    return -2.0 / C_CURV - 2.0 * inner


def block_rotation(angles):
    """diag(1, R(theta_1), ..., R(theta_{d/2})) applied to spatial part."""
    def rot(z):
        out = z.copy()
        for k, th in enumerate(angles):
            i, j = 2 * k, 2 * k + 1
            a, b = z[..., i].copy(), z[..., j].copy()
            out[..., i] = np.cos(th) * a + np.sin(th) * b
            out[..., j] = -np.sin(th) * a + np.cos(th) * b
        return out
    return rot


def build_user(uid):
    events = load_events(uid)
    from collections import Counter, defaultdict

    social = Counter()
    cascades = []           # (post_idx, [path nodes source->...->user], timestamp)
    for i, ev in enumerate(events):
        raw = ev.get("raw_text") or ""
        for mnt in mentions(raw):
            social[("user", mnt)] += 1
        chain = repost_chain(raw)
        if chain:
            path = list(reversed(chain)) + ["user"]   # earliest source first
            cascades.append((i, path, ev.get("timestamp")))

    diff = Counter()
    for _, path, _ in cascades:
        for a, b in zip(path, path[1:]):
            diff[(a, b)] += 1

    nodes = sorted({n for e in list(social) + list(diff) for n in e} | {"user"})
    idx = {v: i for i, v in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)))
    for (a, b), w in list(social.items()) + list(diff.items()):
        A[idx[a], idx[b]] += np.log1p(w)
        A[idx[b], idx[a]] += np.log1p(w)

    # spatial embedding: SVD + hierarchy-aware radius (high degree -> small r)
    z = np.zeros((len(nodes), EMB_D))
    if len(nodes) >= 3:
        from sklearn.decomposition import TruncatedSVD
        k = min(EMB_D, len(nodes) - 1)
        z[:, :k] = TruncatedSVD(n_components=k, random_state=0).fit_transform(A)
        norms = np.linalg.norm(z, axis=1, keepdims=True) + 1e-9
        deg = A.sum(axis=1)
        radius = 1.0 / (1.0 + np.log1p(deg))[:, None]     # central nodes near origin
        z = z / norms * radius

    rng = np.random.default_rng(9)
    rot_s = block_rotation(rng.uniform(0, np.pi, EMB_D // 2))
    rot_d = block_rotation(rng.uniform(0, np.pi, EMB_D // 2))
    X_s = lorentz_lift(rot_s(z))
    X_d = lorentz_lift(rot_d(z))

    cascade_of_post = {i: (ci, path) for ci, (i, path, _) in enumerate(cascades)}

    maps = []
    for i, ev in enumerate(events):
        m = base_map(ev)
        triples = []
        if i in cascade_of_post:
            ci, path = cascade_of_post[i]
            m["cascade_index"] = ci
            for pos, (a, b) in enumerate(zip(path, path[1:])):
                d2 = float(lorentz_sqdist(X_d[idx[a]], X_d[idx[b]]))
                triples.append(f"({a}, diffuses_to[dL2={round(d2, 3)}], {b})")
                triples.append(f"(cascade:{ci}, position_{pos}, {a})")
            # next-infected prediction: nearest nodes in diffusion space
            last = idx[path[-1]]
            d2_all = lorentz_sqdist(X_d[last][None, :], X_d)
            cand = [nodes[j] for j in np.argsort(d2_all)
                    if nodes[j] not in path][:3]
            for cd in cand:
                triples.append(f"(prediction, next_infected, {cd})")
        else:
            m["cascade_index"] = None
            for mnt in mentions(ev.get("raw_text") or "")[:3]:
                if mnt in idx:
                    d2 = float(lorentz_sqdist(X_s[idx['user']], X_s[idx[mnt]]))
                    triples.append(f"(user, social_edge[dL2={round(d2, 3)}], {mnt})")
        triples.append("(embedding, space, Lorentz_c=1_rotated)")
        m.update({
            "feature_3d_triples": triples,
            "feature_3d_text": " ; ".join(triples),
        })
        maps.append(m)

    static_extra = {
        "num_cascades": len(cascades),
        "num_social_edges": len(social),
        "num_diffusion_edges": len(diff),
        "top_diffusion_sources": Counter(p[0] for _, p, _ in cascades).most_common(10),
    }
    bank = assemble_bank(
        uid, "RotDiff",
        "RotDiff: Hyperbolic Rotation for Information Diffusion (CIKM 2023)",
        {"diffusion": "repost-cascade diffusion graph + mention social graph, "
                      "Lorentz embeddings with distinct social/diffusion "
                      "rotations, Lorentz-distance scoring"},
        maps, static_extra,
        {"retriever": "temporal", "curvature": C_CURV, "emb_dim": EMB_D,
         "approximation": "Riemannian-Adam training replaced by deterministic "
                          "hierarchy-aware Lorentz placement (degree->radius) "
                          "+ seeded rotations; only the user's own cascades "
                          "are observable"},
        events)
    return write_bank(uid, "rotdiff", bank,
                      "re-built per RotDiff: true Lorentz-model embeddings "
                      "(c=1) with block rotations for social vs diffusion "
                      "spaces, cascade paths, Lorentz-distance ranking")


if __name__ == "__main__":
    for u in ["1989660417", "7463374646"]:
        print(build_user(u))
