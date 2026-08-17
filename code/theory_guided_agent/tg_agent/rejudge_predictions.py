from __future__ import annotations

"""Re-judge existing GenMinds predictions with current Qwen judge for fair comparison."""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tg_agent.benchmark_core import (  # noqa: E402
    LLMConfig,
    OpenAICompatClient,
    aggregate_metrics,
    JUDGE_SYSTEM,
    build_judge_user,
    parse_judge,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source predictions.jsonl")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--method", default="GenMinds")
    ap.add_argument("--user-id", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    load_dotenv("d:/UserAgent/agentic-harness-engineering/.env")
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

    src = Path(args.src)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "predictions.jsonl"

    done = set()
    if args.resume and out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(str(json.loads(line).get("post_id")))
    elif out_path.exists() and not args.resume:
        out_path.unlink()

    rows = []
    for i, line in enumerate(src.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        d = json.loads(line)
        if args.limit and len(rows) >= args.limit:
            break
        pid = str(d.get("post_id") or i)
        if pid in done:
            continue
        rows.append(d)

    print(f"rejudge pending={len(rows)} done={len(done)}", flush=True)

    def work(d):
        pred = d.get("prediction") or ""
        ctx = d.get("context") or ""
        gt = d.get("ground_truth") or ""
        try:
            # inline to avoid import cycle confusion
            raw = judge.chat(
                [
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": build_judge_user(ctx, gt, pred)},
                ],
                temperature=0.0,
                max_tokens=300,
            )
            scores = parse_judge(raw)
        except Exception as e:  # noqa: BLE001
            scores = {
                "stance": 0.0,
                "core_judgment": 0.0,
                "belief": 0.0,
                "value": 0.0,
                "opinion_alignment_score": 0.0,
                "error": str(e),
            }
        out = dict(d)
        out["method"] = args.method
        out["user_id"] = args.user_id
        out["judge_scores"] = scores
        out["judge_model"] = "dashscope/qwen3.7-plus"
        return out

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, r) for r in rows]
        n = 0
        for fut in as_completed(futs):
            row = fut.result()
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            if n % 50 == 0 or n == len(rows):
                print(f"rejudge {n}/{len(rows)}", flush=True)

    all_rows = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    agg = aggregate_metrics(all_rows)
    metrics = {
        "user_id": args.user_id,
        "method": args.method,
        "predict_model": "deepseek/deepseek-v4-pro (reused predictions)",
        "judge_model": "dashscope/qwen3.7-plus",
        "benchmark_type": "opinion_alignment",
        "num_samples": len(all_rows),
        "benchmark": {
            "stance": agg["stance"],
            "core_judgment": agg["core_judgment"],
            "belief": agg["belief"],
            "value": agg["value"],
            "opinion_alignment_score": agg["opinion_alignment_score"],
        },
        "coverage": f"{agg['n']}/{len(all_rows)}",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": f"rejudged from {src}",
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
