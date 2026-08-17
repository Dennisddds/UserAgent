# -*- coding: utf-8 -*-
"""Sanity validation of rebuilt memory banks (schema + samples)."""

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "outputs"
DIRS = ["cogkr", "cognet3", "cimplekg", "claimskg", "ddgcn", "semipergcn",
        "trignet", "kgrat", "cttn", "enm_senm", "sem", "rotdiff", "gorec",
        "cogigraph", "genminds", "lcg", "cognitive_maps_1977"]
USERS = ["1989660417", "7463374646"]

REQ_MAP = ["map_id", "post_id", "event_title", "feature_2d_text",
           "feature_3d_text", "feature_3d_triples"]

lines = []
for d in DIRS:
    for uid in USERS:
        p = OUT / f"weibo_kg_{d}_{uid}" / "memory_bank.json"
        if not p.exists():
            lines.append(f"[MISSING] {d}:{uid}")
            continue
        b = json.loads(p.read_text(encoding="utf-8"))
        maps = b["event_maps"]
        n = len(maps)
        missing = [k for k in REQ_MAP if k not in maps[0]]
        empty3d = sum(1 for m in maps if not m.get("feature_3d_triples"))
        vecs = b["retrieval_index"]["vectors"]
        ok_vec = len(vecs) == n and len(vecs[0]) == 512
        avg_triples = sum(len(m.get("feature_3d_triples", [])) for m in maps) / n
        lines.append(f"=== {b['method']} ({uid}) maps={n} miss={missing or '-'} "
                     f"empty3d={empty3d} ({empty3d/n:.0%}) vec512={ok_vec} "
                     f"avg_triples={avg_triples:.1f} retriever={b.get('method_extras', {}).get('retriever')}")
        if uid == USERS[0]:
            # first non-empty sample
            for m in maps:
                if m.get("feature_3d_triples"):
                    lines.append("  sample: " + m["feature_3d_text"][:300])
                    break
            extras = {k: v for k, v in b["static_map"].items()
                      if k not in ("beliefs", "persona_values", "persona_interests",
                                   "communication", "entity_stance")}
            lines.append("  static_extra_keys: " + ", ".join(extras.keys()))

out = Path(__file__).parent / "validate_banks_report.txt"
out.write_text("\n".join(lines), encoding="utf-8")
print(out)
print("\n".join(l for l in lines if l.startswith(("[MISSING", "==="))))
