"""Gap-fill judge errors in a sequential_predictions.jsonl in place.

Re-judges ONLY rows whose judge_scores has an error (prediction kept as-is),
then recomputes the comparison_summary metrics.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tg_agent.benchmark_core import (  # noqa: E402
    JUDGE_SYSTEM,
    LLMConfig,
    OpenAICompatClient,
    build_judge_user,
    parse_judge,
)
from tg_agent.llm import load_env  # noqa: E402


def main() -> None:
    pred_path = Path(sys.argv[1])
    summary_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    load_env(cfg["paths"]["env_file"])
    key = os.environ.get("QWEN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY", "")
    judge = OpenAICompatClient(
        LLMConfig(
            api_key=key,
            base_url=os.environ.get(
                "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
            ),
            model=os.environ.get("QWEN_MODEL", "qwen3.7-plus"),
            disable_thinking=True,
        )
    )

    rows = [json.loads(l) for l in pred_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    todo = [
        r for r in rows
        if not r.get("skipped_reason")
        and (r.get("judge_scores") or {}).get("error")
        and (r.get("prediction") or "").strip()
    ]
    print(f"rows={len(rows)} to_rejudge={len(todo)}")
    fixed = 0
    for r in todo:
        try:
            raw = judge.chat(
                [
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": build_judge_user(r["context"], r["ground_truth"], r["prediction"])},
                ],
                temperature=0.0,
                max_tokens=300,
            )
            scores = parse_judge(raw)
            if scores.get("error"):
                print(f"  step {r.get('step')}: still error: {scores['error'][:80]}")
                continue
            keep = {k: v for k, v in (r.get("judge_scores") or {}).items()
                    if k in {"emotion_alignment"}}
            keep.update(scores)
            r["judge_scores"] = keep
            fixed += 1
            print(f"  step {r.get('step')}: oa={scores.get('opinion_alignment_score')}")
        except Exception as ex:  # noqa: BLE001
            print(f"  step {r.get('step')}: exception {ex}")

    pred_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    print(f"fixed={fixed}")

    scored = [
        r for r in rows
        if not r.get("skipped_reason") and not r.get("warmup")
        and r.get("judge_scores") and not (r.get("judge_scores") or {}).get("error")
        and (r.get("prediction") or "").strip()
    ]
    scored.sort(key=lambda r: int(r.get("step", 0)))
    oa = [float(r["judge_scores"].get("opinion_alignment_score") or 0.0) for r in scored]
    if not oa:
        print("no scored rows")
        return
    dims = ["stance", "core_judgment", "belief", "value"]
    bench = {d: round(sum(float(r["judge_scores"].get(d) or 0.0) for r in scored) / len(scored), 4) for d in dims}
    bench["opinion_alignment_score"] = round(sum(oa) / len(oa), 4)
    bench["n"] = len(scored)
    late = {
        "last_5": round(sum(oa[-5:]) / min(5, len(oa)), 4),
        "last_10": round(sum(oa[-10:]) / min(10, len(oa)), 4),
        "first_5": round(sum(oa[:5]) / min(5, len(oa)), 4),
    }
    print(json.dumps({"benchmark": bench, "late_alignment": late}, ensure_ascii=False, indent=2))

    if summary_path and summary_path.exists():
        summ = json.loads(summary_path.read_text(encoding="utf-8"))
        entry = summ[0] if isinstance(summ, list) and summ else summ
        entry["benchmark"] = bench
        entry["late_alignment"] = late
        entry["num_scored"] = len(scored)
        entry["oa_series"] = oa
        summary_path.write_text(json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"updated {summary_path}")


if __name__ == "__main__":
    main()
