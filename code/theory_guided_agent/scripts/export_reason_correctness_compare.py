#!/usr/bin/env python3
"""Export reason-path correctness compare: local Flash vs existing API (aligned post_ids)."""
from __future__ import annotations

import csv
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

from tg_agent.benchmark_core import LLMConfig, OpenAICompatClient
from tg_agent.llm import load_env
from tg_agent.rejudge_reason import (
    REASON_JUDGE_SYSTEM,
    REASON_KEYS,
    build_reason_user,
    extract_reasoning,
    parse_reason,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = Path("/root/autodl-tmp/UserAgent/outputs/flash_full_vs_existing/export_reason_correctness")

PAIRS = [
    {
        "uid": "7463374646",
        "local": Path(
            "/root/autodl-tmp/UserAgent/outputs/flash_full_vs_existing/"
            "seq-CUV-Agent_7463374646/sequential_predictions.jsonl"
        ),
        "api": Path(
            "/root/autodl-tmp/UserAgent/outputs/benchmark_agent_small/"
            "seq-CUV-Agent_7463374646/sequential_predictions.jsonl"
        ),
    },
    {
        "uid": "1989660417",
        "local": Path(
            "/root/autodl-tmp/UserAgent/outputs/flash_full_vs_existing/"
            "seq-CUV-Agent_1989660417/sequential_predictions.jsonl"
        ),
        "api": Path(
            "/root/autodl-tmp/UserAgent/outputs/benchmark_agent_big/"
            "seq-CUV-Agent_1989660417/sequential_predictions.jsonl"
        ),
    },
]


def load_scored(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("warmup"):
            continue
        if not (r.get("prediction") or "").strip():
            continue
        rows.append(r)
    return rows


def index_by_post(rows: list[dict]) -> dict[str, dict]:
    out = {}
    for r in rows:
        pid = str(r.get("post_id") or r.get("bid") or "")
        if pid:
            out[pid] = r
    return out


def mean_scores(rows: list[dict]) -> dict[str, float]:
    n = max(1, len(rows))
    return {
        k: round(sum(float((r.get("reason_scores") or {}).get(k, 0.0)) for r in rows) / n, 4)
        for k in REASON_KEYS
    }


def judge_rows(judge: OpenAICompatClient, rows: list[dict], *, verb_only: bool, workers: int) -> list[dict]:
    def work(r: dict) -> dict:
        context = r.get("context") or ""
        gt = r.get("ground_truth") or ""
        pred = r.get("prediction") or ""
        reasoning = extract_reasoning(r, verb_only=verb_only)
        try:
            raw = judge.chat(
                [
                    {"role": "system", "content": REASON_JUDGE_SYSTEM},
                    {"role": "user", "content": build_reason_user(context, gt, pred, reasoning)},
                ],
                temperature=0.0,
                max_tokens=300,
            )
            scores = parse_reason(raw)
        except Exception as e:  # noqa: BLE001
            scores = {k: 0.0 for k in REASON_KEYS} | {"error": str(e)[:200]}
        out = {
            "post_id": r.get("post_id"),
            "step": r.get("step"),
            "topic": r.get("topic"),
            "warmup": r.get("warmup"),
            "oa": (r.get("judge_scores") or {}).get("opinion_alignment_score"),
            "prediction": (r.get("prediction") or "")[:300],
            "ground_truth": (r.get("ground_truth") or "")[:300],
            "verbalization": ((r.get("agent_trace") or {}).get("verbalization") or "")[:500],
            "reasoning_for_judge": reasoning[:1200],
            "reason_scores": scores,
            "stage_reliability": ((r.get("agent_trace") or {}).get("c_trace") or {}).get(
                "stage_reliability"
            ),
            "white_box": ((r.get("agent_trace") or {}).get("c_trace") or {}).get("white_box"),
            "num_paths": len((r.get("agent_trace") or {}).get("paths") or []),
            "num_factors": len((r.get("agent_trace") or {}).get("factors") or []),
            "num_theories": len((r.get("agent_trace") or {}).get("matched_theories") or []),
        }
        return out

    out_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, r) for r in rows]
        for i, fut in enumerate(as_completed(futs), 1):
            out_rows.append(fut.result())
            if i % 10 == 0 or i == len(futs):
                print(f"  judged {i}/{len(futs)}", flush=True)
    out_rows.sort(key=lambda r: (r.get("step") is None, r.get("step") or 0))
    return out_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "side",
        "user_id",
        "post_id",
        "step",
        "topic",
        "oa",
        "reason_correctness",
        "reason_consistency",
        "reason_grounding",
        "num_factors",
        "num_paths",
        "num_theories",
        "mining_score",
        "synthesis_score",
        "has_reasoning",
        "prediction",
        "ground_truth",
        "verbalization",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            sr = r.get("stage_reliability") or {}
            wb = r.get("white_box") or {}
            rs = r.get("reason_scores") or {}
            w.writerow(
                {
                    "side": r.get("side"),
                    "user_id": r.get("user_id"),
                    "post_id": r.get("post_id"),
                    "step": r.get("step"),
                    "topic": r.get("topic"),
                    "oa": r.get("oa"),
                    "reason_correctness": rs.get("reason_correctness"),
                    "reason_consistency": rs.get("reason_consistency"),
                    "reason_grounding": rs.get("reason_grounding"),
                    "num_factors": r.get("num_factors"),
                    "num_paths": r.get("num_paths"),
                    "num_theories": r.get("num_theories"),
                    "mining_score": (sr.get("mining") or {}).get("score"),
                    "synthesis_score": (sr.get("synthesis") or {}).get("score"),
                    "has_reasoning": (wb.get("has_reasoning") if wb else None),
                    "prediction": r.get("prediction"),
                    "ground_truth": r.get("ground_truth"),
                    "verbalization": r.get("verbalization"),
                }
            )


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    load_env(cfg["paths"]["env_file"])
    qwen_key = os.environ.get("QWEN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY", "")
    if not qwen_key:
        raise SystemExit("QWEN_API_KEY missing")
    judge = OpenAICompatClient(
        LLMConfig(
            api_key=qwen_key,
            base_url=os.environ.get(
                "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
            ),
            model=os.environ.get("QWEN_MODEL", "qwen3.7-plus"),
            disable_thinking=True,
        )
    )

    OUT.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    all_csv_rows: list[dict] = []

    for pair in PAIRS:
        uid = pair["uid"]
        local_rows = load_scored(pair["local"])
        api_idx = index_by_post(load_scored(pair["api"]))
        # Align by local post_id order
        aligned_local = []
        aligned_api = []
        missing = []
        for r in local_rows:
            pid = str(r.get("post_id") or "")
            if pid in api_idx:
                aligned_local.append(r)
                aligned_api.append(api_idx[pid])
            else:
                missing.append(pid)
        print(
            f"[{uid}] local={len(local_rows)} aligned={len(aligned_local)} missing_in_api={len(missing)}",
            flush=True,
        )

        print(f"[{uid}] judging LOCAL (verb-only path)...", flush=True)
        local_judged = judge_rows(judge, aligned_local, verb_only=True, workers=6)
        print(f"[{uid}] judging API aligned (verb-only path)...", flush=True)
        api_judged = judge_rows(judge, aligned_api, verb_only=True, workers=6)

        for r in local_judged:
            r["side"] = "local_Flash"
            r["user_id"] = uid
        for r in api_judged:
            r["side"] = "existing_API"
            r["user_id"] = uid

        udir = OUT / uid
        udir.mkdir(parents=True, exist_ok=True)
        (udir / "local_reason_judge_verbonly.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in local_judged) + "\n",
            encoding="utf-8",
        )
        (udir / "api_reason_judge_verbonly.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in api_judged) + "\n",
            encoding="utf-8",
        )

        lm, am = mean_scores(local_judged), mean_scores(api_judged)
        local_oa = round(
            sum(float(r.get("oa") or 0.0) for r in local_judged) / max(1, len(local_judged)), 4
        )
        api_oa = round(
            sum(float(r.get("oa") or 0.0) for r in api_judged) / max(1, len(api_judged)), 4
        )
        summary_rows.append(
            {
                "user_id": uid,
                "n_aligned": len(aligned_local),
                "local_OA": local_oa,
                "api_OA": api_oa,
                "delta_OA": round(local_oa - api_oa, 4),
                "local_reason_correctness": lm["reason_correctness"],
                "api_reason_correctness": am["reason_correctness"],
                "delta_reason_correctness": round(
                    lm["reason_correctness"] - am["reason_correctness"], 4
                ),
                "local_reason_consistency": lm["reason_consistency"],
                "api_reason_consistency": am["reason_consistency"],
                "delta_reason_consistency": round(
                    lm["reason_consistency"] - am["reason_consistency"], 4
                ),
                "local_reason_grounding": lm["reason_grounding"],
                "api_reason_grounding": am["reason_grounding"],
                "delta_reason_grounding": round(
                    lm["reason_grounding"] - am["reason_grounding"], 4
                ),
            }
        )
        all_csv_rows.extend(local_judged)
        all_csv_rows.extend(api_judged)

        # per-user markdown
        md = [
            f"# Reason-path correctness — user {uid}",
            "",
            f"- aligned n = {len(aligned_local)} (same post_ids)",
            f"- judge = qwen3.7-plus · verb-only verbalization/path",
            f"- generated = {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "| side | OA | reason_correctness | reason_consistency | reason_grounding |",
            "|---|---:|---:|---:|---:|",
            f"| local_Flash | {local_oa} | {lm['reason_correctness']} | {lm['reason_consistency']} | {lm['reason_grounding']} |",
            f"| existing_API | {api_oa} | {am['reason_correctness']} | {am['reason_consistency']} | {am['reason_grounding']} |",
            f"| Δ (local−API) | {local_oa-api_oa:+.4f} | {lm['reason_correctness']-am['reason_correctness']:+.4f} | {lm['reason_consistency']-am['reason_consistency']:+.4f} | {lm['reason_grounding']-am['reason_grounding']:+.4f} |",
            "",
            "## Reading",
            "- reason_correctness: 归因动机/机制是否解释真实评论",
            "- reason_consistency: 理由与预测是否自洽",
            "- reason_grounding: 是否有理论/历史依据而非空话",
            "",
        ]
        (udir / "REASON_COMPARE.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    write_csv(OUT / "reason_correctness_all.csv", all_csv_rows)
    (OUT / "summary.json").write_text(
        json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# 推理归因路径正确性对比（local Flash vs existing API）",
        "",
        "对齐方式：按本地样本的 post_id 在 API 结果中取同帖；评测 verbalization/路径（verb-only）。",
        f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "| user | n | local OA | API OA | ΔOA | local correctness | API correctness | Δcorrectness | Δconsistency | Δgrounding |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summary_rows:
        lines.append(
            f"| {s['user_id']} | {s['n_aligned']} | {s['local_OA']} | {s['api_OA']} | {s['delta_OA']:+.4f} | "
            f"{s['local_reason_correctness']} | {s['api_reason_correctness']} | "
            f"{s['delta_reason_correctness']:+.4f} | {s['delta_reason_consistency']:+.4f} | "
            f"{s['delta_reason_grounding']:+.4f} |"
        )
    lines += [
        "",
        "## 结论提示",
        "- 若 Δcorrectness > 0：本地归因路径相对 API 更接近真实动机",
        "- OA 与 correctness 可能不同步（结果对 ≠ 理由对）",
        "",
        "## 文件",
        "- `reason_correctness_all.csv`：逐帖明细",
        "- `<uid>/REASON_COMPARE.md`：分用户报告",
        "- `<uid>/*_reason_judge_verbonly.jsonl`：原始打分",
        "",
    ]
    (OUT / "REASON_CORRECTNESS_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"\nExported to {OUT}", flush=True)


if __name__ == "__main__":
    main()
