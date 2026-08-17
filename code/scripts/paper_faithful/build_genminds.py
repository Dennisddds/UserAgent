# -*- coding: utf-8 -*-
"""GenMinds — Simulating Society Requires Simulating Thought (NeurIPS 2025).

Faithful structure (position paper's framework):
  * Structured thought capture: causal *explanations* are elicited from each
    post (the LLM interview is replaced by parsing the user's own explanatory
    statements) and parsed into a directed acyclic graph:
      node  = concept
      edge  = directed causal relation with a confidence score AND a polarity
              score (the paper's exact edge attributes).
    Confidence comes from linguistic certainty markers (一定/肯定 > 会/使 >
    可能/或许); polarity from promote/suppress verbs. Acyclicity is enforced
    per belief graph (later edges closing a cycle are dropped).
  * Cognitive motifs: minimal causal reasoning units A -> B -> C extracted
    from the DAG (the paper's "Surveillance -> Crime Rate -> Public Safety").
  * RECAP evaluation: traceability (share of edges with a recorded source
    span), motif compositionality (motifs recurring across posts), belief
    coherence (sign-consistency of repeated edges) are computed and stored.
"""

import re
from collections import Counter, defaultdict

from cogmap_common import assemble_bank, base_map, load_events, strip_urls, write_bank

_SENT = re.compile(r"[。！？!?；;\n]+")
C = r"[^，,。！？!?；;：:]"

# explanation frames with (pattern, cause_group, effect_group, polarity)
FRAMES = [
    (re.compile(r"(?:因为|由于)(" + C + r"{2,25})[，,](?:所以|因此|才|就)?(" + C + r"{2,30})"), 1, 2, None),
    (re.compile(r"(" + C + r"{2,25})[，,](?:所以|因此|因而|从而|于是)(" + C + r"{2,30})"), 1, 2, None),
    (re.compile(r"(" + C + r"{2,25}?)(?:促进|推动|增强|提升|有利于|带动|激发)(" + C + r"{2,28})"), 1, 2, +1),
    (re.compile(r"(" + C + r"{2,25}?)(?:损害|破坏|削弱|阻碍|抑制|打击|威胁|不利于)(" + C + r"{2,28})"), 1, 2, -1),
    (re.compile(r"(" + C + r"{2,25}?)(?:导致|造成|引发|使得|促使)(" + C + r"{2,28})"), 1, 2, None),
    (re.compile(r"(?:如果|一旦)(" + C + r"{2,22})[，,]?(?:就|将|则)(" + C + r"{2,30})"), 1, 2, None),
]
CERTAIN_HI = re.compile(r"一定|肯定|必然|毫无疑问|绝对|无疑")
CERTAIN_LO = re.compile(r"可能|或许|恐怕|大概|也许|未必|似乎")
NEG_STATE = re.compile(r"受损|下降|变差|更糟|失败|倒退|恶化|受害|吃亏|危机")
_LEAD = re.compile(r"^(?:虽然|尽管|但是|但|而且|而|其实|就是|这|那|我们|我|他们|他)+")
_TAIL = re.compile(r"(?:的话|而已|的|了|吧|啊|呢)+$")


def concept(p, anchors):
    p = re.sub(r"[\s“”\"@#【】()（）]+", "", p)
    p = _LEAD.sub("", p)
    p = _TAIL.sub("", p)[:20]
    for a in anchors:
        if a and len(a) >= 2 and (a in p or p in a):
            return a
    return p if len(p) >= 2 else None


def elicit(text, anchors):
    """Interview analog: parse causal explanations into attributed edges."""
    edges = []
    for sent in _SENT.split(strip_urls(text or "")):
        if len(sent) < 6:
            continue
        conf = 0.7
        if CERTAIN_HI.search(sent):
            conf = 0.9
        elif CERTAIN_LO.search(sent):
            conf = 0.5
        for rx, gi, gj, pol in FRAMES:
            mt = rx.search(sent)
            if not mt:
                continue
            c1, c2 = concept(mt.group(gi), anchors), concept(mt.group(gj), anchors)
            if not c1 or not c2 or c1 == c2:
                continue
            polarity = pol if pol is not None else (-1 if NEG_STATE.search(mt.group(gj)) else 1)
            edges.append((c1, c2, polarity, conf, sent[:40]))
    return edges


def enforce_dag(edges):
    """Drop edges that would close a cycle (insertion order priority)."""
    succ = defaultdict(set)

    def reaches(a, b, seen=None):
        seen = seen or set()
        if a == b:
            return True
        for x in succ[a]:
            if x not in seen:
                seen.add(x)
                if reaches(x, b, seen):
                    return True
        return False

    kept = []
    for c1, c2, pol, conf, src in edges:
        if reaches(c2, c1):
            continue
        succ[c1].add(c2)
        kept.append((c1, c2, pol, conf, src))
    return kept


def motifs_of(edges):
    succ = defaultdict(list)
    for c1, c2, pol, *_ in edges:
        succ[c1].append((c2, pol))
    out = []
    for a in succ:
        for b, p1 in succ[a]:
            for c, p2 in succ.get(b, []):
                if c != a:
                    out.append((a, b, c, p1 * p2))
    return out


def build_user(uid):
    events = load_events(uid)
    belief = Counter()       # (c1,c2) -> weighted polarity
    belief_n = Counter()
    motif_count = Counter()
    traced = total_edges = 0

    per_post = []
    for ev in events:
        anchors = (ev.get("entities") or []) + (ev.get("topics") or [])
        edges = enforce_dag(elicit((ev.get("raw_text") or "") + "。"
                                   + (ev.get("user_opinion") or ""), anchors))
        per_post.append(edges)
        for c1, c2, pol, conf, src in edges:
            belief[(c1, c2)] += pol * conf
            belief_n[(c1, c2)] += 1
            total_edges += 1
            traced += bool(src)
        for a, b, c, sgn in motifs_of(edges):
            motif_count[(a, b, c)] += 1

    # RECAP metrics
    repeated = [k for k, n in belief_n.items() if n >= 2]
    coherence = (sum(abs(belief[k]) / belief_n[k] for k in repeated) / len(repeated)
                 if repeated else 1.0)
    shared_motifs = [m for m, n in motif_count.items() if n >= 2]
    recap = {
        "traceability": round(traced / total_edges, 4) if total_edges else 0.0,
        "belief_coherence": round(coherence, 4),
        "motif_compositionality": len(shared_motifs),
    }

    maps = []
    for ev, edges in zip(events, per_post):
        m = base_map(ev)
        triples = []
        concepts = []
        for c1, c2, pol, conf, src in edges:
            sign = "+" if pol > 0 else "-"
            triples.append(f"({c1}, causes[{sign}{conf}], {c2})")
            concepts += [c1, c2]
        for a, b, c, sgn in motifs_of(edges)[:2]:
            triples.append(f"(user, holds_motif, {a}->{b}->{c})")
        if not edges:
            for e in (m["entities"] or m["topics"])[:2]:
                triples.append(f"(concept, observed_without_causal_claim, {e})")
        belief_edges = [{"src": c1, "dst": c2, "polarity": pol,
                         "confidence": conf, "source_span": src}
                        for c1, c2, pol, conf, src in edges]
        m.update({
            "causal_concepts": list(dict.fromkeys(concepts)),
            "feature_3d_triples": triples,
            "feature_3d_text": " ; ".join(triples),
            "belief_edges": belief_edges,
        })
        maps.append(m)

    top_motifs = [f"{a}->{b}->{c}" for (a, b, c), _ in motif_count.most_common(20)]
    top_beliefs = [{"cause": a, "effect": b,
                    "polarity": round(belief[(a, b)] / belief_n[(a, b)], 3),
                    "count": belief_n[(a, b)]}
                   for (a, b), _ in belief_n.most_common(50)]
    static_extra = {"cognitive_motifs": top_motifs, "belief_graph_top": top_beliefs,
                    "recap_metrics": recap}
    bank = assemble_bank(
        uid, "GenMinds",
        "GenMinds: Simulating Society Requires Simulating Thought (NeurIPS 2025)",
        {"dag": "belief DAGs with confidence+polarity edges, cognitive motifs, "
                "RECAP metrics (traceability/coherence/compositionality)"},
        maps, static_extra,
        {"retriever": "multihop",
         "approximation": "LLM semi-structured interview replaced by parsing "
                          "the user's own explanatory statements; do-calculus "
                          "simulation not run (no interventions available)"},
        events)
    return write_bank(uid, "genminds", bank,
                      "re-built per GenMinds framework: per-post causal DAGs "
                      "with confidence+polarity, enforced acyclicity, motif "
                      "extraction, RECAP metrics")


if __name__ == "__main__":
    for u in ["1989660417", "7463374646"]:
        print(build_user(u))
