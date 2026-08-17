"""Judge robustness audit: self-consistency + failed-row recovery (read-only).

1. Self-consistency: sample already-judged rows and re-judge the identical
   (context, gt, pred) twice with the same Qwen judge; report exact agreement,
   mean absolute delta, and correlation.
2. Failed-row recovery: rows whose Qwen judge returned `error` (content
   moderation 400 etc.) are re-judged with DeepSeek; report recovery rate.

No experiment outputs are modified; results go to a sidecar JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "theory_guided_agent"))

from dotenv import load_dotenv

from tg_agent.benchmark_core import (
    JUDGE_SYSTEM,
    LLMConfig,
    OpenAICompatClient,
    build_judge_user,
    parse_judge,
)


ROOTS = [
    Path(r"D:\UserSimuAgent\项目最新版\exp_outputs"),
    Path(r"D:\UserSimuAgent\项目最新版\exp_outputs_v2"),
    Path(r"D:\UserSimuAgent\项目最新版\exp_x_outputs"),
]


def collect(root: Path) -> tuple[list[dict], list[dict]]:
    scored: list[dict] = []
    failed: list[dict] = []
    for path in sorted(root.rglob("sequential_predictions.jsonl")):
        if any("CORRUPT" in p or p.lower().startswith("smoke") for p in path.parts):
            continue
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
            js = r.get("judge_scores") or {}
            has_error = bool(js.get("error") or r.get("error"))
            if r.get("warmup") or (not (r.get("prediction") or "").strip() and not has_error):
                continue
            row = {
                "path": str(path),
                "step": r.get("step"),
                "post_id": r.get("post_id"),
                "context": r.get("context") or "",
                "gt": r.get("ground_truth") or "",
                "pred": r.get("prediction") or "",
                "oa": js.get("opinion_alignment_score"),
                "error": js.get("error") or r.get("error") or "",
            }
            if has_error:
                failed.append(row)
            elif row["oa"] is not None:
                scored.append(row)
    return scored, failed


def judge_call(client: OpenAICompatClient, row: dict) -> dict:
    raw = client.chat(
        [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": build_judge_user(row["context"], row["gt"], row["pred"])},
        ],
        temperature=0.0,
    )
    return parse_judge(raw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-consistency-n", type=int, default=40)
    ap.add_argument("--failed-limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path(r"D:\UserSimuAgent\项目最新版\judge_robustness.json"))
    args = ap.parse_args()

    load_dotenv(r"D:\UserSimuAgent\UserAgent\agentic-harness-engineering\.env")
    qwen_key = os.environ.get("QWEN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY", "")
    ds_key = os.environ.get("LLM_API_KEY", "")

    qwen = OpenAICompatClient(
        LLMConfig(
            api_key=qwen_key,
            base_url=os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            model=os.environ.get("QWEN_MODEL", "qwen3.7-plus"),
            disable_thinking=True,
        )
    )
    deepseek = OpenAICompatClient(
        LLMConfig(
            api_key=ds_key,
            base_url=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com"),
            model=os.environ.get("LLM_MODEL", "deepseek-v4-pro"),
            disable_thinking=False,
        )
    )

    scored: list[dict] = []
    failed: list[dict] = []
    for root in ROOTS:
        if root.exists():
            s, f = collect(root)
            scored.extend(s)
            failed.extend(f)
    print(f"scored={len(scored)} failed={len(failed)}", flush=True)

    rng = random.Random(args.seed)
    sample = rng.sample(scored, min(args.self_consistency_n, len(scored)))
    pairs = []
    exact = 0
    deltas = []
    for i, row in enumerate(sample, 1):
        try:
            a = judge_call(qwen, row)
            b = judge_call(qwen, row)
        except Exception as e:
            pairs.append({"post_id": row["post_id"], "error": str(e)[:200]})
            continue
        oa_a = float(a.get("opinion_alignment_score") or 0.0)
        oa_b = float(b.get("opinion_alignment_score") or 0.0)
        if oa_a == oa_b:
            exact += 1
        deltas.append(abs(oa_a - oa_b))
        pairs.append(
            {
                "post_id": row["post_id"],
                "original_oa": row["oa"],
                "rejudge_a": oa_a,
                "rejudge_b": oa_b,
            }
        )
        print(f"self-consistency {i}/{len(sample)}", flush=True)

    recovered = []
    failed_to_test = failed if not args.failed_limit else failed[: args.failed_limit]
    for row in failed_to_test:
        try:
            ds = judge_call(deepseek, row)
            oa = float(ds.get("opinion_alignment_score") or 0.0)
            recovered.append({"post_id": row["post_id"], "deepseek_oa": oa, "error": row["error"][:120]})
        except Exception as e:
            recovered.append({"post_id": row["post_id"], "deepseek_error": str(e)[:160]})
        print(f"recovery {len(recovered)}/{len(failed_to_test)}", flush=True)

    n_ok = len(deltas)
    report = {
        "scored_rows": len(scored),
        "failed_rows": len(failed),
        "self_consistency": {
            "n": n_ok,
            "exact_match_rate": round(exact / n_ok, 4) if n_ok else 0.0,
            "mean_abs_delta": round(sum(deltas) / n_ok, 4) if n_ok else 0.0,
            "max_abs_delta": round(max(deltas), 4) if deltas else 0.0,
            "pairs": pairs,
        },
        "failed_recovery": recovered,
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "scored_rows": report["scored_rows"],
        "failed_rows": report["failed_rows"],
        "self_consistency": {k: v for k, v in report["self_consistency"].items() if k != "pairs"},
        "recovered": len(report["failed_recovery"]),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
