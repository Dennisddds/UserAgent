#!/usr/bin/env python3
"""Aggregate token/cost/FLOPs estimates from existing prediction JSONL files.

The current prediction JSONL files record `num_llm_calls` and tool traces but
not exact provider token usage. This script therefore estimates token counts
from character lengths. It must be reported as an ESTIMATE, not measured usage.

Usage:
    python scripts/run_cost_report.py --outputs-root ../outputs
    python scripts/run_cost_report.py --outputs-root ../outputs --model-params 685000000000
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _pick_trace(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("predict_trace"):
        return row["predict_trace"]
    agent_trace = row.get("agent_trace")
    if isinstance(agent_trace, dict) and agent_trace.get("c_trace"):
        return agent_trace["c_trace"]
    return {}


def estimate_tokens(text: str, chars_per_token: float) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / max(0.1, chars_per_token)))


def row_summary(row: dict[str, Any], chars_per_token: float) -> dict[str, Any]:
    trace = _pick_trace(row)
    agent_trace = row.get("agent_trace") or {}
    judge = row.get("judge_scores") or {}

    tool_history = trace.get("tool_history") or []
    tools_called = trace.get("tools_called") or {}
    tool_result_chars = sum(int(x.get("result_chars") or 0) for x in tool_history)

    prediction = str(row.get("prediction") or "")
    ground_truth = str(row.get("ground_truth") or "")
    context = str(row.get("context") or "")
    reasoning = (
        str(trace.get("reason") or "")
        or str(agent_trace.get("verbalization") or "")
    )

    prompt_chars = len(context) + len(ground_truth) + len(reasoning)
    completion_chars = len(prediction) + len(reasoning)

    return {
        "step": _int(row.get("step"), 0),
        "post_id": str(row.get("post_id") or ""),
        "method": str(row.get("method") or ""),
        "user_id": str(row.get("user_id") or ""),
        "num_llm_calls": _int(trace.get("num_llm_calls"), 1),
        "tool_calls": _int(sum(tools_called.values()), len(tool_history)),
        "tool_result_chars": tool_result_chars,
        "prediction_chars": len(prediction),
        "reasoning_chars": len(reasoning),
        "prompt_est_tokens": estimate_tokens(
            context + "\n" + ground_truth + "\n" + reasoning, chars_per_token
        ),
        "completion_est_tokens": estimate_tokens(
            prediction + "\n" + reasoning, chars_per_token
        ),
        "judge_oa": _num(judge.get("opinion_alignment_score")),
        "judge_stance": _num(judge.get("stance")),
        "judge_core": _num(judge.get("core_judgment")),
        "judge_belief": _num(judge.get("belief")),
        "judge_value": _num(judge.get("value")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--chars-per-token", type=float, default=2.0,
                        help="Rough token estimator; lower is more conservative.")
    parser.add_argument("--model-params", type=float, default=0.0,
                        help="Optional model parameters for dense FLOPs estimate.")
    parser.add_argument("--out", type=Path, default=Path("cost_report.csv"))
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    pred_files = sorted(args.outputs_root.rglob("sequential_predictions.jsonl"))
    pred_files += sorted(args.outputs_root.rglob("predictions.jsonl"))
    seen = set()
    for path in pred_files:
        if str(path) in seen:
            continue
        seen.add(str(path))
        for row in _read_jsonl(path):
            if row.get("judge_scores") is None:
                continue
            s = row_summary(row, args.chars_per_token)
            s["source"] = str(path)
            rows.append(s)

    if not rows:
        raise SystemExit("No judged prediction rows found under " + str(args.outputs_root))

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total_calls = sum(r["num_llm_calls"] for r in rows)
    total_prompt_tokens = sum(r["prompt_est_tokens"] * r["num_llm_calls"] for r in rows)
    total_completion_tokens = sum(r["completion_est_tokens"] * r["num_llm_calls"] for r in rows)
    total_tokens = total_prompt_tokens + total_completion_tokens
    scored = [r for r in rows if r["judge_oa"] > 0]
    mean_oa = sum(r["judge_oa"] for r in scored) / len(scored) if scored else 0.0

    summary: dict[str, Any] = {
        "sources": len(seen),
        "scored_rows": len(scored),
        "total_llm_calls": total_calls,
        "estimated_prompt_tokens": total_prompt_tokens,
        "estimated_completion_tokens": total_completion_tokens,
        "estimated_total_tokens": total_tokens,
        "mean_judge_oa": round(mean_oa, 4),
        "oa_per_1m_tokens": round(mean_oa / (total_tokens / 1_000_000), 6)
        if total_tokens else 0.0,
        "tokens_per_scored_row": round(total_tokens / len(scored), 2) if scored else 0.0,
    }
    if args.model_params > 0:
        summary["flops_estimate_2N_tokens"] = 2.0 * args.model_params * total_tokens
        summary["flops_estimate_note"] = (
            "approximate dense inference: 2 * params * estimated_tokens; no KV-cache/batch correction"
        )

    summary_path = out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Wrote", out)
    print("Wrote", summary_path)


if __name__ == "__main__":
    main()
