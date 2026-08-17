# -*- coding: utf-8 -*-
"""CogiGraph — Semantic KG Fusion for Fake News Detection (PLOS ONE 2025).

Faithful to Algorithm 1:
  3: preprocess text (URLs/mentions removed — tweet-preprocessor analog)
  4: E_context = DistilBERT(X)  -> hashing sentence encoder (512-d)
  5-6: build domain KG from the corpus (COVID ontology + CORD-19 replaced by
       the corpus's own entity/topic ontology, since the domain here is
       Chinese social commentary, not COVID)
  7: Triples = OpenIE(X) -> jieba-posseg SVO clause extraction
  8-9: alignment score(m) = similarity(m, kg_mention)/max_length (Eq. 4),
       aligned entity = argmax score (Eq. 6)
  10: E_graph = alignment-score-weighted average of aligned entity/relation
      embeddings (SimplE embeddings replaced by seeded random projections)
      (Eq. 7)
  11: E(X) = E_graph + E_context (Eq. 8)
  12: MLP prediction head replaced by a credibility flag from rumor markers
      (no fake/real labels exist for these posts).
"""

import re

import numpy as np

from cogmap_common import (assemble_bank, base_map, encode_text, load_events,
                    strip_urls, write_bank)

RUMOR = re.compile(r"谣言|传闻|假的|不实|未经证实|真假难辨|辟谣|捏造|虚假|据称")
_CLAUSE = re.compile(r"[，,。！？!?；;：:\n]+")


def openie_svo(text):
    """OpenIE analog: SVO triple extraction with jieba POS tagging."""
    import jieba.posseg as pseg
    triples = []
    for clause in _CLAUSE.split(strip_urls(text or "")):
        if not (4 <= len(clause) <= 60):
            continue
        words = [(w, f) for w, f in pseg.cut(clause) if w.strip()]
        subj = verb = None
        for i, (w, f) in enumerate(words):
            if f.startswith("n") and len(w) >= 2 and subj is None:
                subj = w
            elif f.startswith("v") and subj is not None and verb is None and len(w) >= 1:
                verb = w
            elif f.startswith(("n", "t")) and len(w) >= 2 and subj and verb and w != subj:
                triples.append((subj, verb, w))
                break
    return triples[:6]


def char_sim(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(len(sa | sb), 1)


def align(mention, kg_mentions):
    """Eqs. 4-6: score = similarity / max_length; argmax alignment."""
    best, best_s = None, 0.0
    for kg_m in kg_mentions:
        s = char_sim(mention, kg_m) / max(len(mention), len(kg_m))
        if s > best_s:
            best, best_s = kg_m, s
    return best, round(best_s, 4)


def build_user(uid):
    events = load_events(uid)
    # domain KG: corpus ontology of entities + topics
    kg_entities = sorted({e for ev in events for e in (ev.get("entities") or [])})
    kg_relations = sorted({t for ev in events for t in (ev.get("topics") or [])})
    rng = np.random.default_rng(23)
    ent_emb = {e: rng.standard_normal(512) / 22.6 for e in kg_entities}
    rel_emb = {r: rng.standard_normal(512) / 22.6 for r in kg_relations}

    maps = []
    n_fake_flags = 0
    for ev in events:
        m = base_map(ev)
        raw = ev.get("raw_text") or ""
        text = raw + " " + m["event_summary"]
        e_context = np.array(encode_text(m["event_summary"] or raw))

        svo = openie_svo(text)
        aligned = []
        vecs = []
        scores = []
        for s, v, o in svo:
            for mention in (s, o):
                ent, sc = align(mention, kg_entities)
                if ent and sc > 0:
                    aligned.append((mention, ent, sc))
                    vecs.append(ent_emb[ent])
                    scores.append(sc)
            rel, sc = align(v, kg_relations)
            if rel and sc > 0:
                vecs.append(rel_emb[rel])
                scores.append(sc)
        if vecs:
            w = np.array(scores)[:, None]
            e_graph = (w * np.stack(vecs)).sum(axis=0) / len(vecs)   # Eq. 7
        else:
            e_graph = np.zeros(512)
        e_fused = e_graph + e_context                                # Eq. 8
        cred_flag = bool(RUMOR.search(text))
        n_fake_flags += cred_flag

        triples = [f"OpenIE:({s}, {v}, {o})" for s, v, o in svo]
        for mention, ent, sc in aligned[:5]:
            triples.append(f"({mention}, aligned_to[score={sc}], kg:{ent})")
        triples.append(f"(fusion, E_graph_plus_E_context, norm={round(float(np.linalg.norm(e_fused)), 3)})")
        triples.append(f"(credibility_flag, rumor_markers, {'present' if cred_flag else 'absent'})")
        m.update({
            "openie": [list(t) for t in svo],
            "feature_3d_triples": triples,
            "feature_3d_text": " ; ".join(triples),
        })
        maps.append(m)

    static_extra = {
        "domain_kg": {"entities": len(kg_entities), "relations": len(kg_relations)},
        "rumor_flagged_posts": n_fake_flags,
    }
    bank = assemble_bank(
        uid, "CogiGraph",
        "CogiGraph: Semantic KG Fusion for Fake News Detection (PLOS ONE 2025)",
        {"fusion": "OpenIE SVO triples + Eq.4 alignment scoring + Eq.7 "
                   "score-weighted graph embedding + Eq.8 additive fusion"},
        maps, static_extra,
        {"retriever": "default",
         "approximation": "DistilBERT -> hashing encoder; SimplE embeddings -> "
                          "seeded projections; COVID ontology/CORD-19 -> corpus "
                          "ontology; no fake/real labels so MLP head replaced "
                          "by rumor-marker flag"},
        events)
    return write_bank(uid, "cogigraph", bank,
                      "re-built per CogiGraph Algorithm 1: OpenIE extraction, "
                      "Eq.4/6 alignment, Eq.7 weighted graph embedding, Eq.8 "
                      "fusion")


if __name__ == "__main__":
    for u in ["1989660417", "7463374646"]:
        print(build_user(u))
