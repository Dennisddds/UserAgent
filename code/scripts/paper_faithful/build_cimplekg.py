# -*- coding: utf-8 -*-
"""CimpleKG — Continuously Updated Misinformation KG (2024).

Per the paper's two pipelines, adapted to one user's posts:
  Collection: each post is treated as a claim occurrence; ratings are
    normalized onto the paper's exact 5-level scheme (credible,
    mostly_credible, uncertain, not_credible, not_verifiable) using
    assertion/hedge/rumor markers (replaces IFCN fact-checker verdicts).
  KG integration: entity extraction (DBpedia-Spotlight analog: the extraction
    layer's entities), factor extraction with the paper's exact factor set
    (emotion, sentiment, political leaning, conspiracy, propaganda technique),
    SHA224 URIs, and Schema.org-style properties (sc:ClaimReview, sc:Claim,
    sc:itemReviewed, co:normalizedReviewRating, sc:appearance, sc:mentions).
"""

import hashlib
import re
from collections import Counter

from cogmap_common import (assemble_bank, base_map, emotion_counts, load_events,
                    sentiment, write_bank)

RUMOR = re.compile(r"谣言|传闻|假的|不实|未经证实|真假难辨|辟谣|捏造|虚假")
CONFIRM = re.compile(r"证实|官方|通报|公布|确认|判决|数据显示|白皮书")
HEDGE = re.compile(r"可能|或许|据称|据说|疑似|待证|难说|不确定")
QUESTION = re.compile(r"[？?]\s*$|该不该|是不是|真的吗")
CONSPIRACY = re.compile(r"阴谋|暗中|操纵|幕后|勾结|渗透|别有用心|递刀")
PROPAGANDA = {
    "name_calling": re.compile(r"爱国贼|叼飞盘|恨国党|公知|水军|汉奸"),
    "appeal_to_fear": re.compile(r"威胁|危险|后果不堪设想|警惕|亡国"),
    "exaggeration": re.compile(r"史上最|空前|绝无仅有|彻底|全都|永远"),
    "flag_waving": re.compile(r"祖国|民族大义|爱国|国家利益|人民"),
}
POLITICAL = {"pro_establishment": re.compile(r"党的领导|正能量|国家|政府|主旋律|体制优势"),
             "critical": re.compile(r"批评|质疑|追责|问责|反思|漏洞|官僚")}


def normalized_rating(text):
    """The paper's 5-level normalized review rating."""
    if RUMOR.search(text):
        return "not_credible"
    if CONFIRM.search(text):
        return "credible"
    if HEDGE.search(text):
        return "uncertain"
    if QUESTION.search(text):
        return "not_verifiable"
    return "mostly_credible"


def factors(text):
    """The paper's factor set (Step 3 of KG integration)."""
    s = sentiment(text)
    emos = emotion_counts(text)
    top_emo = max(emos, key=emos.get) if any(emos.values()) else "none"
    pol = "neutral"
    if POLITICAL["pro_establishment"].search(text):
        pol = "pro_establishment"
    if POLITICAL["critical"].search(text):
        pol = "critical" if pol == "neutral" else "mixed"
    props = [k for k, rx in PROPAGANDA.items() if rx.search(text)]
    return {
        "sentiment": {1: "positive", 0: "neutral", -1: "negative"}[s],
        "emotion": top_emo,
        "political_leaning": pol,
        "conspiracy_related": bool(CONSPIRACY.search(text)),
        "propaganda_techniques": props,
    }


def build_user(uid):
    events = load_events(uid)
    maps = []
    rating_hist = Counter()
    factor_hist = Counter()
    for ev in events:
        m = base_map(ev)
        text = (ev.get("raw_text") or "") + " " + m["event_summary"]
        claim_text = m["event_summary"] or (ev.get("raw_text") or "")[:80]
        uri = hashlib.sha224(ev["post_id"].encode()).hexdigest()[:16]
        rating = normalized_rating(text)
        fac = factors(text)
        rating_hist[rating] += 1
        factor_hist[fac["sentiment"]] += 1

        triples = [
            f"(sc:ClaimReview:{uri}, sc:itemReviewed, sc:Claim:{uri})",
            f"(sc:ClaimReview:{uri}, co:normalizedReviewRating, {rating})",
            f"(sc:Claim:{uri}, sc:text, {claim_text[:60]})",
            f"(sc:Claim:{uri}, sc:appearance, post:{ev['post_id']})",
            f"(sc:Claim:{uri}, sc:author, user)",
        ]
        for e in m["entities"]:
            triples.append(f"(sc:Claim:{uri}, sc:mentions, {e})")
        triples.append(f"(sc:Claim:{uri}, co:factor:sentiment, {fac['sentiment']})")
        if fac["emotion"] != "none":
            triples.append(f"(sc:Claim:{uri}, co:factor:emotion, {fac['emotion']})")
        triples.append(f"(sc:Claim:{uri}, co:factor:political_leaning, {fac['political_leaning']})")
        if fac["conspiracy_related"]:
            triples.append(f"(sc:Claim:{uri}, co:factor:conspiracy, true)")
        for p in fac["propaganda_techniques"]:
            triples.append(f"(sc:Claim:{uri}, co:factor:propaganda, {p})")

        m.update({
            "claim_rating": rating,
            "claim_text": claim_text,
            "claim_factors": fac,
            "feature_3d_triples": triples,
            "feature_3d_text": " ; ".join(triples),
        })
        maps.append(m)

    static_extra = {"rating_distribution": dict(rating_hist),
                    "factor_sentiment_distribution": dict(factor_hist)}
    bank = assemble_bank(
        uid, "CimpleKG",
        "CimpleKG: Continuously Updated Misinformation KG (2024)",
        {"pipeline": "claim collection + 5-level normalized rating + entity/"
                     "factor extraction + Schema.org RDF triples (SHA224 URIs)"},
        maps, static_extra,
        {"retriever": "default",
         "approximation": "fact-checker verdicts replaced by assertion/hedge/"
                          "rumor lexical markers; BERT factor models replaced "
                          "by lexicons (same factor taxonomy)"},
        events)
    return write_bank(uid, "cimplekg", bank,
                      "re-built per CimpleKG pipelines: 5-level ratings, factor "
                      "set (emotion/sentiment/political/conspiracy/propaganda), "
                      "Schema.org triples with SHA224 URIs")


if __name__ == "__main__":
    for u in ["1989660417", "7463374646"]:
        print(build_user(u))
