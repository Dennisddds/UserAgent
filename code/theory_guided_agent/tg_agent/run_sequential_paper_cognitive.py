from __future__ import annotations

"""Compare GenMinds and Hart-1977 as online cognitive maps.

Strict step protocol:
  predict from posts < t -> judge against post t -> learn from V4-Pro CoT
  failure -> append the real post's method-specific coding to the graph.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tg_agent.benchmark_core import (  # noqa: E402
    JUDGE_SYSTEM,
    LLMConfig,
    OpenAICompatClient,
    aggregate_metrics,
    build_judge_user,
    parse_judge,
)
from tg_agent.llm import DeepSeekClient  # noqa: E402
from tg_agent.paper_cognitive_online import (  # noqa: E402
    OnlinePaperCognitiveMemory,
    PaperCognitiveOnlineAgent,
)
from tg_agent.run_sequential import (  # noqa: E402
    ground_truth,
    load_chrono_events,
    stimulus_from_event,
)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def make_v4_pro() -> DeepSeekClient:
    key = os.environ.get("ADB_LLM_API_KEY") or os.environ.get("LLM_API_KEY")
    base = os.environ.get("ADB_LLM_BASE_URL") or os.environ.get("LLM_BASE_URL")
    return DeepSeekClient(
        api_key=key,
        base_url=base,
        model="deepseek-v4-pro",
        enable_thinking=True,
        reasoning_effort="high",
    )


def make_judge() -> OpenAICompatClient:
    return OpenAICompatClient(
        LLMConfig(
            api_key=os.environ.get("QWEN_API_KEY")
            or os.environ.get("DASHSCOPE_API_KEY", ""),
            base_url=os.environ.get(
                "QWEN_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
            model=os.environ.get("QWEN_MODEL") or "qwen3.7-plus",
            disable_thinking=True,
        )
    )


def judge_one(
    judge: OpenAICompatClient,
    context: str,
    gt: str,
    prediction: str,
    *,
    fallback: DeepSeekClient | None = None,
) -> dict[str, float]:
    if not prediction.strip():
        return {
            "stance": 0.0,
            "core_judgment": 0.0,
            "belief": 0.0,
            "value": 0.0,
            "opinion_alignment_score": 0.0,
        }
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {
            "role": "user",
            "content": build_judge_user(context, gt, prediction),
        },
    ]
    try:
        raw = judge.chat(messages, temperature=0.0, max_tokens=300)
    except RuntimeError as exc:
        if fallback is None or "DataInspectionFailed" not in str(exc):
            raise
        # A judge-side moderation rejection is not an Agent reasoning error.
        # Use the configured V4-Pro endpoint as a deterministic fallback so
        # the strict temporal loop does not advance without a verdict.
        raw = fallback.chat(
            messages,
            temperature=0.0,
            max_tokens=300,
            disable_thinking=True,
        )
    return parse_judge(raw)


def write_metrics(
    out_dir: Path,
    *,
    user_id: str,
    method: str,
    rows: list[dict[str, Any]],
    num_events: int,
    warmup: int,
) -> dict[str, Any]:
    scored = [
        r
        for r in rows
        if not r.get("warmup")
        and r.get("judge_scores")
        and not r.get("error")
    ]
    oa = [
        float((r.get("judge_scores") or {}).get("opinion_alignment_score") or 0.0)
        for r in scored
    ]
    failures = [r for r in scored if not (r.get("evolution") or {}).get("helpful")]
    first = oa[: min(50, len(oa))]
    last = oa[-min(50, len(oa)) :] if oa else []
    metrics = {
        "user_id": user_id,
        "method": method,
        "protocol": "strict_online_predict_judge_evolve_then_ingest",
        "predict_model": "deepseek-v4-pro (thinking=enabled, reasoning_effort=high)",
        "judge_model": os.environ.get("QWEN_MODEL") or "qwen3.7-plus",
        "num_events": num_events,
        "warmup": warmup,
        "num_scored": len(scored),
        "benchmark": aggregate_metrics(scored) if scored else {},
        "online_improvement": {
            "first_50_oa": round(sum(first) / len(first), 4) if first else 0.0,
            "last_50_oa": round(sum(last) / len(last), 4) if last else 0.0,
            "delta": round(
                (sum(last) / len(last) if last else 0.0)
                - (sum(first) / len(first) if first else 0.0),
                4,
            ),
        },
        "failure_learning": {
            "num_failures": len(failures),
            "num_with_thinking_error": sum(
                1 for r in failures if (r.get("evolution") or {}).get("thinking_error")
            ),
            "num_repairs_learned": sum(
                1 for r in failures if (r.get("evolution") or {}).get("learned")
            ),
            "num_steps_using_repairs": sum(
                1 for r in scored if (r.get("agent_trace") or {}).get("repairs_applied")
            ),
            "num_same_step_repair_rounds": sum(
                int(r.get("repair_rounds") or 0) for r in scored
            ),
            "num_repaired_before_advance": sum(
                1 for r in scored if int(r.get("repair_rounds") or 0) > 0
            ),
            "num_unresolved": sum(
                1 for r in rows if r.get("repair_status") == "unresolved_stop"
            ),
        },
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metrics


def run_method(
    *,
    user_id: str,
    method_key: str,
    events: list[dict[str, Any]],
    warmup: int,
    out_root: Path,
    resume: bool,
    pass_threshold: float,
    max_repair_rounds: int,
) -> dict[str, Any]:
    out_dir = out_root / f"seq-{method_key}_{user_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "sequential_predictions.jsonl"
    existing = load_jsonl(pred_path) if resume else []
    if not resume and pred_path.exists():
        pred_path.unlink()
        existing = []
    # Resume after the latest successfully ingested row. Strict-loop failures
    # are appended for audit but do not cross the reveal boundary, so retry
    # that same step.
    rows_by_step = {int(r["step"]): r for r in existing if "step" in r}
    latest = max(rows_by_step, default=-1)
    if latest >= 0 and not rows_by_step[latest].get("coding_found_at_ingest", True):
        # Strict-loop unresolved/technical stop: current event was not revealed,
        # so resume must retry the same step.
        start = latest
    else:
        start = latest + 1

    bank_path = (
        ROOT.parent
        / "outputs"
        / f"weibo_kg_{method_key}_{user_id}"
        / "memory_bank.json"
    )
    persona_path = (
        ROOT.parent / "outputs" / f"weibo_user_{user_id}" / "persona.json"
    )
    memory = OnlinePaperCognitiveMemory(
        bank_path,
        user_id=user_id,
        method_key=method_key,
        persona_path=persona_path,
    )
    llm = make_v4_pro()
    judge = make_judge()
    agent = PaperCognitiveOnlineAgent(
        user_id=user_id,
        memory=memory,
        llm=llm,
        state_dir=out_dir / "agent_state",
        failure_threshold=pass_threshold,
    )

    # Deterministically reconstruct the visible graph through the checkpoint.
    for prior_step in range(start):
        memory.ingest_event(events[prior_step])
    print(
        f"[{method_key}] strict online events={len(events)} warmup={warmup} "
        f"resume_from={start} V4-Pro-CoT=on",
        flush=True,
    )

    for step in range(start, len(events)):
        event = events[step]
        context = stimulus_from_event(event)
        gt = ground_truth(event)
        row: dict[str, Any] = {
            "step": step,
            "post_id": str(event.get("post_id") or step),
            "timestamp": event.get("timestamp"),
            "method": method_key,
            "num_graph_posts_before": len(memory.events),
            "warmup": step < warmup,
            "context": context,
            "ground_truth": gt,
        }
        should_stop = False
        try:
            if step < warmup:
                row["prediction"] = ""
                row["judge_scores"] = None
                row["skipped_reason"] = "warmup_ingest_only"
            else:
                attempts: list[dict[str, Any]] = []
                final_trace: dict[str, Any] = {}
                final_scores: dict[str, float] = {}
                final_evolution: dict[str, Any] = {}
                immediate_guidance = ""
                for repair_round in range(max_repair_rounds + 1):
                    trace = agent.predict(
                        context,
                        immediate_guidance=immediate_guidance,
                    )
                    scores = judge_one(
                        judge,
                        context,
                        gt,
                        trace["prediction"],
                        fallback=llm,
                    )
                    passed = (
                        float(scores.get("opinion_alignment_score") or 0.0)
                        >= pass_threshold
                    )
                    evolution = agent.evolve(
                        trace=trace,
                        judge_scores=scores,
                        immediate_admit=not passed,
                        ground_truth=gt,
                        stimulus=context,
                        repair_round=repair_round,
                    )
                    attempts.append(
                        {
                            "repair_round": repair_round,
                            "prediction": trace["prediction"],
                            "judge_scores": scores,
                            # Keep every V4-Pro CoT, not only the final attempt.
                            "reasoning_content": trace.get("reasoning_content") or "",
                            "thinking_quality": trace.get("thinking_quality") or {},
                            "repairs_applied": trace.get("repairs_applied") or [],
                            "evolution": evolution,
                        }
                    )
                    final_trace, final_scores, final_evolution = (
                        trace,
                        scores,
                        evolution,
                    )
                    immediate_guidance = str(
                        evolution.get("immediate_guidance") or ""
                    )
                    if passed:
                        break

                row["attempts"] = attempts
                row["repair_rounds"] = max(0, len(attempts) - 1)
                row["prediction"] = final_trace.get("prediction") or ""
                row["judge_scores"] = final_scores
                row["agent_trace"] = final_trace
                row["evolution"] = final_evolution
                if (
                    float(final_scores.get("opinion_alignment_score") or 0.0)
                    < pass_threshold
                ):
                    row["repair_status"] = "unresolved_stop"
                    row["error"] = (
                        f"same-step autonomous repair did not reach OA "
                        f"{pass_threshold:.3f} after {max_repair_rounds} rounds"
                    )
                    should_stop = True
                elif row["repair_rounds"]:
                    row["repair_status"] = "repaired_before_advance"
                else:
                    row["repair_status"] = "passed_first_attempt"
        except Exception as exc:  # noqa: BLE001
            row["error"] = str(exc)
            row["prediction"] = ""
            row["judge_scores"] = None
            row["repair_status"] = "technical_error_stop"
            should_stop = True
            print(f"[{method_key}] step {step} STOPPED: {exc}", flush=True)

        # Critical anti-leakage boundary: reveal current post only after the
        # current prediction has passed. An unresolved error halts chronology.
        if not should_stop:
            row["coding_found_at_ingest"] = memory.ingest_event(event)
        else:
            row["coding_found_at_ingest"] = False
        row["num_graph_posts_after"] = len(memory.events)
        append_jsonl(pred_path, row)
        rows_by_step[step] = row

        if should_stop:
            write_metrics(
                out_dir,
                user_id=user_id,
                method=method_key,
                rows=[rows_by_step[k] for k in sorted(rows_by_step)],
                num_events=len(events),
                warmup=warmup,
            )
            raise RuntimeError(
                f"{method_key} halted at step {step}: {row['error']}"
            )

        if (step + 1) % 10 == 0 or step + 1 == len(events):
            metrics = write_metrics(
                out_dir,
                user_id=user_id,
                method=method_key,
                rows=[rows_by_step[k] for k in sorted(rows_by_step)],
                num_events=len(events),
                warmup=warmup,
            )
            oa = (metrics.get("benchmark") or {}).get(
                "opinion_alignment_score", 0.0
            )
            fm = agent.failure_memory.stats()
            print(
                f"[{method_key}] step {step+1}/{len(events)} "
                f"graph={len(memory.events)} OA={oa:.3f} "
                f"fail_structures={fm['structures']} "
                f"admitted_repairs={fm['admitted_repairs']}",
                flush=True,
            )

    return write_metrics(
        out_dir,
        user_id=user_id,
        method=method_key,
        rows=[rows_by_step[k] for k in sorted(rows_by_step)],
        num_events=len(events),
        warmup=warmup,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="1989660417")
    ap.add_argument(
        "--methods",
        default="genminds,cognitive_maps_1977",
        help="comma-separated: genminds,cognitive_maps_1977",
    )
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="chronological event cap; 0 means all",
    )
    ap.add_argument(
        "--out-root",
        default=str(
            ROOT.parent / "outputs" / "benchmark_sequential_paper_cognitive"
        ),
    )
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--pass-threshold",
        type=float,
        default=0.5,
        help="OA required before the current event may be ingested.",
    )
    ap.add_argument(
        "--max-repair-rounds",
        type=int,
        default=5,
        help="Maximum autonomous same-step repair attempts; unresolved stops the run.",
    )
    args = ap.parse_args()

    load_dotenv(ROOT.parent / "agentic-harness-engineering" / ".env")
    events = load_chrono_events(args.user, root=ROOT.parent)
    if args.limit:
        events = events[: args.limit]
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for method in methods:
        summaries.append(
            run_method(
                user_id=args.user,
                method_key=method,
                events=events,
                warmup=args.warmup,
                out_root=out_root,
                resume=args.resume,
                pass_threshold=args.pass_threshold,
                max_repair_rounds=args.max_repair_rounds,
            )
        )

    summary_path = out_root / f"comparison_summary_{args.user}.json"
    summary_path.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Online Cognitive-Map Comparison",
        "",
        "- protocol: predict → judge → CoT failure repair → ingest current post",
        "- predictor: DeepSeek-V4-Pro, thinking enabled",
        "",
        "| method | OA | first50 | last50 | delta | failures | repairs used |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for m in summaries:
        b = m.get("benchmark") or {}
        imp = m["online_improvement"]
        fl = m["failure_learning"]
        lines.append(
            f"| {m['method']} | {b.get('opinion_alignment_score',0):.4f} | "
            f"{imp['first_50_oa']:.4f} | {imp['last_50_oa']:.4f} | "
            f"{imp['delta']:+.4f} | {fl['num_failures']} | "
            f"{fl['num_steps_using_repairs']} |"
        )
    (out_root / f"comparison_report_{args.user}.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print(f"Wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
