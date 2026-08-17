#!/usr/bin/env python3
"""Export local-Flash scores + CoT + failure-memory notebook for a run dir."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path, help=".../seq-CUV-Agent_<uid>")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    run = args.run_dir
    uid = run.name.split("_")[-1]
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(l)
        for l in (run / "sequential_predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    fm_path = run / "agent_state" / f"{uid}_failure_memory.json"
    fm = json.loads(fm_path.read_text(encoding="utf-8")) if fm_path.exists() else {}

    lines: list[str] = []
    b = metrics.get("benchmark") or {}
    lines += [
        f"# White-box report — user {uid}",
        "",
        f"- model: `{metrics.get('predict_model')}`",
        f"- scored: {metrics.get('num_scored')}/{metrics.get('num_steps')}",
        f"- OA: **{b.get('opinion_alignment_score')}**",
        f"- stance/core/belief/value: {b.get('stance')} / {b.get('core_judgment')} / {b.get('belief')} / {b.get('value')}",
        f"- late5: {metrics.get('late_alignment', {}).get('last_5')}",
        f"- failure structures: {len(fm.get('structures') or [])}, repairs: {len(fm.get('repairs') or [])}",
        "",
        "## Per-step (scored)",
        "",
    ]
    for r in rows:
        if r.get("warmup"):
            continue
        at = r.get("agent_trace") or {}
        ct = at.get("c_trace") or {}
        wb = ct.get("white_box") or {}
        oa = (r.get("judge_scores") or {}).get("opinion_alignment_score")
        lines += [
            f"### step {r.get('step')} | OA={oa}",
            f"- topic: {r.get('topic')}",
            f"- pred: {(r.get('prediction') or '')[:240]}",
            f"- gt: {(r.get('ground_truth') or '')[:240]}",
            f"- stage_reliability: `{json.dumps(ct.get('stage_reliability') or {}, ensure_ascii=False)[:400]}`",
            f"- CoT chars: {wb.get('reasoning_chars') or len(ct.get('model_reasoning') or '')}",
            "",
            "**mining**",
            "```",
            (wb.get("mining_excerpt") or (ct.get("model_reasoning") or "")[:600]) or "(empty)",
            "```",
            "",
            "**reasoning**",
            "```",
            (wb.get("reasoning_excerpt") or (ct.get("model_reasoning") or "")[-600:]) or "(empty)",
            "```",
            "",
        ]

    lines += ["## 错题本 (failure_memory)", ""]
    for s in (fm.get("structures") or [])[:20]:
        lines += [
            f"- **{s.get('primary_cause')}** freq={s.get('freq')} oa={s.get('examples_oa')}",
            f"  - when: {s.get('when_to_apply')}",
            f"  - exemplar: {(s.get('exemplar') or '')[:160]}",
        ]
    for rp in (fm.get("repairs") or [])[:20]:
        lines += [
            f"- repair `{rp.get('id') or rp.get('action')}`: "
            f"{json.dumps(rp, ensure_ascii=False)[:220]}"
        ]

    out = args.out or (run / "WHITEBOX_REPORT.md")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
