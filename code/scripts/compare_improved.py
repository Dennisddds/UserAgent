"""Improved vs baseline comparison summary (read-only).

Pairs: exp_outputs_v2/small_all_methods/seq-CUV-Agent_<uid>
   vs  exp_outputs_improved/small_agentx/seq-CUV-AgentX_<uid>
and the X pairs under exp_outputs_improved (x_<uid>_agent vs x_<uid>_agentx).
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path


def cell_oa(path: Path) -> dict:
    last: dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(r.get("step"), int):
            last[r["step"]] = r
    oas = []
    fails = 0
    for r in last.values():
        js = r.get("judge_scores") or {}
        if r.get("warmup") or not (r.get("prediction") or "").strip():
            continue
        if js.get("error") or r.get("error"):
            fails += 1
            continue
        v = js.get("opinion_alignment_score")
        if isinstance(v, (int, float)):
            oas.append(v)
    def mean(xs):
        return round(statistics.mean(xs), 4) if xs else None
    return {
        "n": len(oas),
        "fails": fails,
        "oa": mean(oas),
        "first5": mean(oas[:5]),
        "last5": mean(oas[-5:]),
    }


PAIRS = [
    (
        "weibo-small",
        Path(r"D:\UserSimuAgent\项目最新版\exp_outputs_v2\small_all_methods\seq-CUV-Agent_7463374646\sequential_predictions.jsonl"),
        Path(r"D:\UserSimuAgent\项目最新版\exp_outputs_improved\small_agentx\seq-CUV-AgentX_7463374646\sequential_predictions.jsonl"),
    ),
]
for uid in ("geoffreyhinton", "johnschulman2", "lilianweng"):
    PAIRS.append(
        (
            f"x-{uid}",
            Path(rf"D:\UserSimuAgent\项目最新版\exp_outputs_improved\x_{uid}_agent\seq-CUV-Agent_{uid}\sequential_predictions.jsonl"),
            Path(rf"D:\UserSimuAgent\项目最新版\exp_outputs_improved\x_{uid}_agentx\seq-CUV-AgentX_{uid}\sequential_predictions.jsonl"),
        )
    )


def main() -> None:
    for label, base_path, imp_path in PAIRS:
        if not base_path.exists() and not imp_path.exists():
            continue
        b = cell_oa(base_path) if base_path.exists() else {}
        i = cell_oa(imp_path) if imp_path.exists() else {}
        if not b or not i:
            print(f"{label:14s} baseline={'pending' if not b else b} improved={'pending' if not i else i}")
            continue
        d = round(i["oa"] - b["oa"], 4) if i["oa"] is not None and b["oa"] is not None else None
        print(
            f"{label:14s} base OA={b['oa']} (n={b['n']}, first5={b['first5']}, last5={b['last5']}) | "
            f"AgentX OA={i['oa']} (n={i['n']}, first5={i['first5']}, last5={i['last5']}) | Δ={d}"
        )


if __name__ == "__main__":
    main()
