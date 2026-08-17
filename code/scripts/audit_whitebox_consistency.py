"""White-box verbalized-trace consistency audit (read-only).

For every scored prediction that carries `agent_trace.c_trace.model_reasoning`,
check whether the theory coordinates / evidence titles the model *claims* in its
self-reported CoT actually appear in what was retrieved for that step. This is
the honest, verifiable version of "CoT is real": it does not prove internal
thinking, but it does prove the verbalized trace is grounded in the step's
actual tool/memory returns instead of being post-hoc confabulation.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOTS = [
    Path(r"D:\UserSimuAgent\项目最新版\exp_outputs"),
    Path(r"D:\UserSimuAgent\项目最新版\exp_x_outputs"),
]

def _ngrams(text: str, n: int = 2) -> set[str]:
    text = re.sub(r"\s+", "", text or "")
    return {text[i : i + n] for i in range(max(len(text) - n + 1, 0))}


def _appears(needle: str, haystack: str) -> bool:
    needle = (needle or "").strip()
    haystack = haystack or ""
    if len(needle) < 2:
        return False
    return needle in haystack or haystack in needle


def grounded_items(row: dict) -> dict[str, list[str]]:
    at = row.get("agent_trace") or {}
    items: dict[str, list[str]] = {"theory": [], "evidence": [], "coordinate": []}
    for m in at.get("matched_theories") or []:
        if m.get("coordinate"):
            items["coordinate"].append(str(m["coordinate"]))
        if m.get("name"):
            items["theory"].append(str(m["name"]))
    for e in at.get("c_trace", {}).get("evidence_events") or at.get("evidence_events") or []:
        if isinstance(e, dict) and e.get("title"):
            items["evidence"].append(str(e["title"]))
    for p in at.get("paths") or []:
        if isinstance(p, dict):
            for k in ("evidence", "evidence_title", "history_title"):
                v = p.get(k)
                if isinstance(v, str) and v:
                    items["evidence"].append(v)
    for c in at.get("activated_coordinates") or []:
        if c:
            items["coordinate"].append(str(c))
    return items


def audit(path: Path) -> dict:
    n_rows = n_reasoning = 0
    theory_hits = theory_total = 0
    coord_hits = coord_total = 0
    ev_hits = ev_total = 0
    cot_verb_overlap = 0.0
    hallucinated_examples: list[str] = []
    reasoning_chars = 0
    rows_by_step: dict[int, dict] = {}
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                step = int(r.get("step", -1))
            except (TypeError, ValueError):
                step = -1
            if step >= 0:
                rows_by_step[step] = r
    for r in rows_by_step.values():
        at = r.get("agent_trace") or {}
        ct = at.get("c_trace") or {}
        cot = (ct.get("model_reasoning") or "").strip()
        if r.get("warmup") or not (r.get("prediction") or "").strip():
            continue
        n_rows += 1
        if not cot:
            continue
        n_reasoning += 1
        reasoning_chars += len(cot)
        items = grounded_items(r)
        for t in items["theory"]:
            theory_total += 1
            theory_hits += int(_appears(t, cot))
        for c in items["coordinate"]:
            coord_total += 1
            coord_hits += int(_appears(c, cot))
        for e in items["evidence"]:
            ev_total += 1
            ev_hits += int(_appears(e, cot))
        verb = (at.get("verbalization") or (r.get("prediction") or ""))[:400]
        cot_ng = _ngrams(cot[:4000])
        verb_ng = _ngrams(verb)
        if cot_ng and verb_ng:
            cot_verb_overlap += len(cot_ng & verb_ng) / len(verb_ng)
        for c in items["coordinate"][:8]:
            if not _appears(c, cot) and len(hallucinated_examples) < 8:
                hallucinated_examples.append(c)
    return {
        "rows": n_rows,
        "with_reasoning": n_reasoning,
        "reasoning_rate": round(n_reasoning / n_rows, 4) if n_rows else 0.0,
        "avg_reasoning_chars": round(reasoning_chars / n_reasoning, 1) if n_reasoning else 0.0,
        "theory_grounding_rate": round(theory_hits / theory_total, 4) if theory_total else None,
        "coordinate_grounding_rate": round(coord_hits / coord_total, 4) if coord_total else None,
        "evidence_grounding_rate": round(ev_hits / ev_total, 4) if ev_total else None,
        "cot_verbalization_overlap": round(cot_verb_overlap / n_reasoning, 4) if n_reasoning else 0.0,
        "hallucinated_examples": hallucinated_examples,
    }


def main() -> None:
    out_rows = []
    for root in ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("sequential_predictions.jsonl")):
            if any("CORRUPT" in p or p.lower().startswith("smoke") for p in path.parts):
                continue
            rel = path.relative_to(root)
            cell = rel.parts[0] if rel.parts else ""
            method_user = rel.parts[1] if len(rel.parts) > 1 else path.stem
            stats = audit(path)
            if stats["rows"]:
                out_rows.append({"cell": cell, "method_user": method_user, **stats})
    out = Path(r"D:\UserSimuAgent\项目最新版\whitebox_consistency.json")
    out.write_text(json.dumps(out_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out} with {len(out_rows)} groups")
    for row in out_rows:
        print(
            f"{row['cell']:20s} {row['method_user']:34s} rows={row['rows']:5d} "
            f"cot_rate={row['reasoning_rate']:.2f} coord_ground={row['coordinate_grounding_rate']} "
            f"ev_ground={row['evidence_grounding_rate']} cot_verb_overlap={row['cot_verbalization_overlap']:.2f}"
        )


if __name__ == "__main__":
    main()
