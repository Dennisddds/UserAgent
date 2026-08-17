# -*- coding: utf-8 -*-
"""ENM/SENM — Ego Network Model for Cross-Target Stance Detection (2024).

Faithful structure:
  ENM: ego = the user; alters = accounts interacted with (@mentions and //@
    repost sources); per-alter interaction frequency computed; circle
    structure via MeanShift clustering of contact frequencies (the paper's
    method for determining circle numbers automatically).
  SENM: per ego-alter pair, interactions grouped and signed with a sentiment
    model (VADER replaced by the Chinese lexicon); a relationship is negative
    when the share of negative interactions exceeds the Gottman 17% threshold
    (the paper's exact rule).
  Features are separated into inner circles (1-2) and outer circles (3+).
"""

import numpy as np

from cogmap_common import (assemble_bank, base_map, load_events, mentions,
                    repost_chain, sentiment, write_bank)

GOTTMAN = 0.17


def build_user(uid):
    events = load_events(uid)
    from collections import Counter, defaultdict

    inter = defaultdict(list)     # alter -> [sentiment per interaction]
    for ev in events:
        raw = ev.get("raw_text") or ""
        s = sentiment(raw + " " + (ev.get("user_opinion") or ""))
        for alt in set(mentions(raw)) | set(repost_chain(raw)):
            inter[alt].append(s)

    alters = {a: len(ss) for a, ss in inter.items()}
    # MeanShift clustering of contact frequencies -> circles
    circles = {}
    n_circles = 0
    if len(alters) >= 3:
        from sklearn.cluster import MeanShift
        freqs = np.log1p(np.array(list(alters.values()), dtype=float)).reshape(-1, 1)
        ms = MeanShift().fit(freqs)
        centers = ms.cluster_centers_.ravel()
        order = np.argsort(-centers)          # circle 1 = highest frequency
        rank = {int(c): i + 1 for i, c in enumerate(order)}
        for a, lab in zip(alters.keys(), ms.labels_):
            circles[a] = rank[int(lab)]
        n_circles = len(centers)
    else:
        circles = {a: 1 for a in alters}
        n_circles = 1 if alters else 0

    signs = {}
    for a, ss in inter.items():
        neg_share = sum(1 for x in ss if x < 0) / len(ss)
        signs[a] = "-" if neg_share > GOTTMAN else "+"

    ego_alters = [{"alter": a, "freq": alters[a], "circle": circles[a],
                   "sign": signs[a],
                   "ring": "inner" if circles[a] <= 2 else "outer"}
                  for a in sorted(alters, key=lambda x: -alters[x])]

    maps = []
    for ev in events:
        m = base_map(ev)
        raw = ev.get("raw_text") or ""
        alts = list(set(mentions(raw)) | set(repost_chain(raw)))
        triples = []
        for a in alts:
            triples.append(f"(ego, {signs.get(a, '+')}_alter, {a})")
            c = circles.get(a, 1)
            triples.append(f"({a}, circle_{c}, {'inner' if c <= 2 else 'outer'})")
        if not alts:
            for e in m["entities"][:2]:
                triples.append(f"(ego, discusses_without_interaction, {e})")
        m.update({
            "feature_3d_triples": triples,
            "feature_3d_text": " ; ".join(triples),
        })
        maps.append(m)

    neg_alters = [a for a, s in signs.items() if s == "-"]
    static_extra = {
        "ego_alters": ego_alters[:80],
        "num_circles": n_circles,
        "num_alters": len(alters),
        "negative_relationships": neg_alters[:30],
        "inner_circle_alters": [a["alter"] for a in ego_alters if a["ring"] == "inner"][:30],
    }
    bank = assemble_bank(
        uid, "ENM-SENM",
        "ENM/SENM: Ego Network Model for Cross-Target Stance (2024)",
        {"ego": "MeanShift circles over interaction frequencies; SENM signs "
                "via Gottman 17% negative-interaction threshold; inner/outer "
                "circle separation"},
        maps, static_extra,
        {"retriever": "signed", "gottman_threshold": GOTTMAN,
         "approximation": "VADER replaced by Chinese sentiment lexicon; "
                          "interactions limited to mentions/reposts observable "
                          "in the user's own posts"},
        events)
    return write_bank(uid, "enm_senm", bank,
                      "re-built per ENM/SENM: interaction-frequency ego "
                      "network, MeanShift circle detection, Gottman-threshold "
                      "signed relationships, inner/outer features")


if __name__ == "__main__":
    for u in ["1989660417", "7463374646"]:
        print(build_user(u))
