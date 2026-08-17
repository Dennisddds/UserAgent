# -*- coding: utf-8 -*-
"""Sanity-check rebuilt banks: each method must have distinct paper-aligned structure."""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "outputs"

CHECKS = {
    "genminds": {
        "need_any_triple_substr": ["causes", "holds_motif", "belief_node"],
        "static_keys": ["cognitive_motifs"],
        "forbid_old": ["(胡老师, causes, 我是一只小蘑菇)"],  # old naive chain pattern ok if rare
    },
    "lcg": {"need_any_triple_substr": ["memory_scale", "learner:user", "temporal_msg"], "static_keys": []},
    "cogkr": {"need_any_triple_substr": ["hop1_expand", "attend", "reason_via"], "static_keys": []},
    "cognet3": {"need_any_triple_substr": ["semantic_event", "group_emotion", "personality_group"], "static_keys": ["personality_group", "emotion_histogram"]},
    "cimplekg": {"need_any_triple_substr": ["ClaimReview", "rating"], "static_keys": []},
    "claimskg": {"need_any_triple_substr": ["schema:Claim", "schema:ClaimReview"], "static_keys": []},
    "ddgcn": {"need_any_triple_substr": ["aggregates", "l2c_edge", "mentions"], "static_keys": []},
    "semipergcn": {"need_any_triple_substr": ["of_user", "in_liwc"], "static_keys": []},
    "trignet": {"need_any_triple_substr": ["flow_pwp", "pwp_via", "pwcwp_via", "in_category"], "static_keys": []},
    "kgrat": {"need_any_triple_substr": ["attention", "mentions"], "static_keys": []},
    "cttn": {"need_any_triple_substr": ["target", "stance, label"], "static_keys": []},
    "enm_senm": {"need_any_triple_substr": ["_alter", "circle"], "static_keys": ["ego_alters"]},
    "sem": {"need_any_triple_substr": ["+1", "-1", "balance"], "static_keys": []},
    "rotdiff": {"need_any_triple_substr": ["diffuses_to", "social_rot", "cascade"], "static_keys": []},
    "gorec": {"need_any_triple_substr": ["KOL:user", "translated_by"], "static_keys": []},
    "cogigraph": {"need_any_triple_substr": ["OpenIE:", "aligns_to", "aligned_entity"], "static_keys": []},
    "cognimap": {"need_any_triple_substr": ["related_to", "stance_on"], "static_keys": []},
    "cognitive_maps_1977": {"need_any_triple_substr": ["--(+)", "--(-)", "Utility"], "static_keys": ["causal_beliefs"]},
}


def main() -> None:
    uid = "1989660417"
    report = []
    for key, spec in CHECKS.items():
        mb = json.loads((OUT / f"weibo_kg_{key}_{uid}" / "memory_bank.json").read_text(encoding="utf-8"))
        triples = " || ".join(
            " ; ".join(m.get("feature_3d_triples") or []) for m in mb["event_maps"][:50]
        )
        ok_sub = any(s in triples for s in spec["need_any_triple_substr"])
        ok_static = all(k in mb["static_map"] for k in spec["static_keys"])
        sample = (mb["event_maps"][0].get("feature_3d_triples") or [])[:6]
        row = {
            "method": key,
            "paper_method": mb.get("method"),
            "triple_check": ok_sub,
            "static_check": ok_static,
            "n_maps": len(mb["event_maps"]),
            "sample_triples": sample,
            "analogy": mb.get("analogy"),
        }
        report.append(row)
        status = "OK" if ok_sub and ok_static else "FAIL"
        print(f"[{status}] {key}: triples={ok_sub} static={ok_static} sample={sample[:3]}")
    out = OUT / "paper_kg_structure_verify_1989660417.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
