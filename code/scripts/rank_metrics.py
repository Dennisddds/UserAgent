# -*- coding: utf-8 -*-
import json
from pathlib import Path

OUT = Path("outputs")
rows = []
for p in sorted(OUT.glob("weibo_kg_*_7463374646/metrics.json")):
    m = json.loads(p.read_text(encoding="utf-8"))
    b = m.get("benchmark") or {}
    rows.append((float(b.get("opinion_alignment_score") or 0), m.get("method"), b, p.parent.name))
rows.sort(reverse=True)
lines = [
    "# Small user ranking (rebuilt paper-faithful KGs)",
    "",
    "| rank | method | opinion | stance | core | belief | value |",
    "|---:|---|---:|---:|---:|---:|---:|",
]
print("=== SMALL USER RANKING ===")
for i, (oa, name, b, d) in enumerate(rows, 1):
    print(f"{i:2d}. {name:22s} OA={oa:.4f}")
    lines.append(
        f"| {i} | {name} | {oa:.4f} | {b.get('stance',0):.3f} | {b.get('core_judgment',0):.3f} | {b.get('belief',0):.3f} | {b.get('value',0):.3f} |"
    )
Path("outputs/paper_kg_rerank_small_7463374646.md").write_text("\n".join(lines), encoding="utf-8")
