# -*- coding: utf-8 -*-
"""ClaimsKG — A Knowledge Graph of Fact-Checked Claims (ISWC 2019).

ClaimsKG models Claim / ClaimReview pairs with: truth ratings normalized to
the 4 classes TRUE / FALSE / MIXTURE / OTHER, claim keywords, entities
annotated in claim text (DBpedia-linked in the paper), claim author and date.
Distinct from CimpleKG (which adds factors and 5-level ratings): here we keep
strictly to the ClaimsKG data model.
"""

import re
import time
from collections import Counter

from cogmap_common import assemble_bank, base_map, load_events, write_bank

TRUE_RX = re.compile(r"证实|属实|官方确认|判决|数据显示|确凿")
FALSE_RX = re.compile(r"谣言|假的|不实|捏造|辟谣|虚假|骗局")
MIX_RX = re.compile(r"部分属实|夸大|断章取义|真假难辨|有出入|片面")


def truth_rating(text):
    t = FALSE_RX.search(text)
    if MIX_RX.search(text):
        return "MIXTURE"
    if t:
        return "FALSE"
    if TRUE_RX.search(text):
        return "TRUE"
    return "OTHER"


def build_user(uid):
    events = load_events(uid)
    maps = []
    hist = Counter()
    for ev in events:
        m = base_map(ev)
        text = (ev.get("raw_text") or "") + " " + m["event_summary"]
        rating = truth_rating(text)
        hist[rating] += 1
        date = time.strftime("%Y-%m-%d", time.localtime(m["timestamp"] or 0))
        claim = m["event_summary"][:60] or (ev.get("raw_text") or "")[:60]
        triples = [
            f"(schema:Claim, text, {claim})",
            f"(schema:ClaimReview, reviews, schema:Claim)",
            f"(schema:ClaimReview, truthRating, {rating})",
            f"(schema:Claim, author, user)",
            f"(schema:Claim, datePublished, {date})",
        ]
        for t in m["topics"]:
            triples.append(f"(schema:Claim, keywords, {t})")
        for e in m["entities"]:
            triples.append(f"(schema:Claim, entityAnnotation, dbpedia-analog:{e})")
        if m["user_opinion"]:
            triples.append(f"(schema:ClaimReview, reviewBody, {m['user_opinion'][:60]})")
        m.update({
            "truth_rating": rating,
            "feature_3d_triples": triples,
            "feature_3d_text": " ; ".join(triples),
        })
        maps.append(m)

    bank = assemble_bank(
        uid, "ClaimsKG",
        "ClaimsKG: Knowledge Graph of Fact-Checked Claims (ISWC 2019)",
        {"schema": "Claim/ClaimReview pairs, TRUE/FALSE/MIXTURE/OTHER ratings, "
                   "keywords + entity annotations"},
        maps, {"truth_rating_distribution": dict(hist)},
        {"retriever": "default",
         "approximation": "fact-checker verdicts inferred from lexical markers; "
                          "DBpedia entity linking replaced by canonical entity "
                          "names from the extraction layer"},
        events)
    return write_bank(uid, "claimskg", bank,
                      "re-built per ClaimsKG data model: Claim/ClaimReview with "
                      "normalized 4-class truth ratings, keywords, entity "
                      "annotations, author/date")


if __name__ == "__main__":
    for u in ["1989660417", "7463374646"]:
        print(build_user(u))
