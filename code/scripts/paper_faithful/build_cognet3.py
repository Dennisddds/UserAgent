# -*- coding: utf-8 -*-
"""CogNet3 — Dynamic Emotional Knowledge of Personality Groups (ISWC 2025).

Three data frames per the paper:
  Frame 1 Semantic Event: event instances linked to parent event labels.
    Parent labels obtained by clustering topic labels (co-occurrence label
    propagation; cluster centre = most frequent topic), mirroring the paper's
    K-Means-over-label-embeddings + hierarchy cleaning.
  Frame 2 Homophilous Group: the user's personality profile on the paper's
    exact attribute scheme: stance tendency (5 levels), stance firmness (2),
    expression aggressiveness (3), expression logicality (3), Big Five (3
    levels each). LLM annotation is replaced by deterministic lexical scoring
    over the user's history (documented approximation).
  Frame 3 Group Emotion: Plutchik 8-emotion x 4-level distribution per event,
    aggregated over the (single-member) homophilous group.
"""

import re
from collections import Counter, defaultdict

from cogmap_common import (assemble_bank, base_map, emotion_counts, emotion_level,
                    load_events, sentiment_score, write_bank)

AGGR = re.compile(r"！{2,}|无耻|滚|蠢|可耻|荒唐|放屁|闭嘴|叫嚣|嘴脸")
LOGIC = re.compile(r"因为|所以|因此|首先|其次|然而|但是|如果|一方面|另一方面|综上|事实上")
BIG5 = {
    "openness": re.compile(r"新|创新|科技|探索|未来|想象|变革|开放"),
    "conscientiousness": re.compile(r"应该|必须|规则|落实|责任|严格|纪律|规范"),
    "extraversion": re.compile(r"大家|我们|朋友们|一起|！|来吧|快来"),
    "agreeableness": re.compile(r"感谢|尊重|理解|包容|善意|温暖|支持"),
    "neuroticism": re.compile(r"担忧|焦虑|愤怒|恐慌|不安|失望|痛心"),
}


def level3(x, lo, hi):
    return "low" if x < lo else ("high" if x >= hi else "medium")


def cluster_topics(events):
    """Frame 1: parent event labels via topic co-occurrence label propagation."""
    co = Counter()
    freq = Counter()
    for ev in events:
        tops = list(dict.fromkeys(ev.get("topics") or []))
        freq.update(tops)
        for i, a in enumerate(tops):
            for b in tops[i + 1:]:
                co[(a, b)] += 1
                co[(b, a)] += 1
    parent = {t: t for t in freq}
    for _ in range(3):  # label propagation rounds
        for t in freq:
            nbrs = Counter()
            for (a, b), w in co.items():
                if a == t:
                    nbrs[parent[b]] += w
            if nbrs:
                best = max(nbrs, key=lambda l: (nbrs[l], freq[l]))
                if freq[best] >= freq[parent[t]]:
                    parent[t] = best
    return parent


def personality_group(events):
    """Frame 2: the paper's exact attribute scheme, lexically scored."""
    text = " ".join((ev.get("raw_text") or "") + " " + (ev.get("user_opinion") or "")
                    for ev in events)
    n = max(len(text), 1)
    pols = [sentiment_score((ev.get("user_opinion") or "")
                            + " ".join(ev.get("stance_keywords") or [])) for ev in events]
    mean_pol = sum(pols) / max(len(pols), 1)
    var_pol = sum((p - mean_pol) ** 2 for p in pols) / max(len(pols), 1)
    tendency_idx = max(0, min(4, int(round((mean_pol + 1) * 2))))
    tendency = ["far_left", "left", "centrist", "right", "far_right"][tendency_idx]
    group = {
        "stance_tendency": tendency,
        "stance_firmness": "stable" if var_pol < 0.35 else "depends",
        "expression_aggressiveness": level3(len(AGGR.findall(text)) / n * 1e4, 1.0, 4.0),
        "expression_logicality": level3(len(LOGIC.findall(text)) / n * 1e4, 5.0, 15.0),
    }
    for trait, rx in BIG5.items():
        group[f"big5_{trait}"] = level3(len(rx.findall(text)) / n * 1e4, 2.0, 8.0)
    return group


def build_user(uid):
    events = load_events(uid)
    parent = cluster_topics(events)
    group = personality_group(events)
    group_label = "+".join(f"{k}={v}" for k, v in list(group.items())[:2])

    emo_agg = defaultdict(Counter)
    maps = []
    for ev in events:
        m = base_map(ev)
        text = (ev.get("raw_text") or "") + " " + m["user_opinion"]
        ec = emotion_counts(text)
        dist = {k: emotion_level(v, len(text)) for k, v in ec.items()}
        for k, lv in dist.items():
            emo_agg[k][lv] += 1

        tops = m["topics"]
        parents = list(dict.fromkeys(parent.get(t, t) for t in tops))
        triples = [f"(semantic_event, is, {m['event_title']})"]
        for t in tops:
            if parent.get(t, t) != t:
                triples.append(f"(event:{t}, has_parent_event, {parent[t]})")
        for p in parents:
            triples.append(f"(semantic_event, typed_as, {p})")
        triples.append(f"(homophilous_group, profile, {group_label})")
        for emo, lv in dist.items():
            if lv != "none":
                triples.append(f"(group, emotion:{emo}={lv}, {m['event_title']})")
        for e in m["entities"]:
            triples.append(f"(group, reacts_to, {e})")
        m.update({
            "emotion_dist": dist,
            "feature_3d_triples": triples,
            "feature_3d_text": " ; ".join(triples),
        })
        maps.append(m)

    hist = {emo: {lv: cnt for lv, cnt in cnts.items()} for emo, cnts in emo_agg.items()}
    static_extra = {"personality_group": group, "emotion_histogram": hist,
                    "parent_event_labels": sorted({p for p in parent.values()})[:50]}
    bank = assemble_bank(
        uid, "CogNet3",
        "CogNet3: Dynamic Emotional Knowledge Fusion (ISWC 2025 Companion)",
        {"frames": "semantic event hierarchy + homophilous group profile + "
                   "Plutchik 8x4 group emotion distributions"},
        maps, static_extra,
        {"retriever": "default",
         "approximation": "LLM annotation replaced by deterministic lexical "
                          "scoring; single-user group (no cross-user homophily "
                          "data available)"},
        events)
    return write_bank(uid, "cognet3", bank,
                      "re-built per CogNet3 three-frame scheme: event hierarchy, "
                      "personality attributes (paper's exact levels), Plutchik "
                      "emotion distributions")


if __name__ == "__main__":
    for u in ["1989660417", "7463374646"]:
        print(build_user(u))
