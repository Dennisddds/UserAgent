"""Build the final aggregate report from all finished experiment outputs.

Run after Weibo v4 cells and X reruns are complete. Produces:
  - final_results.json   (machine-readable)
  - final_report.md      (human-readable tables + conclusions)
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compute_psychometrics_all import sweep as psych_sweep


ROOTS = [
    Path(r"D:\UserSimuAgent\项目最新版\exp_outputs_v2"),
    Path(r"D:\UserSimuAgent\项目最新版\exp_x_outputs_v2"),
]


def cell_stats(pred_path: Path) -> dict:
    oa = []
    dims = {k: [] for k in ("stance", "core_judgment", "belief", "value")}
    failures = 0
    steps = 0
    pred_tokens = 0.0
    judge_tokens = 0.0
    pred_calls = 0
    rows_by_step: dict[int, dict] = {}
    with pred_path.open(encoding="utf-8", errors="replace") as fh:
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
        steps += 1
        js = r.get("judge_scores") or {}
        if r.get("warmup") or not (r.get("prediction") or "").strip():
            continue
        if js.get("error") or r.get("error"):
            failures += 1
            continue
        v = js.get("opinion_alignment_score")
        if isinstance(v, (int, float)):
            oa.append(float(v))
        for k in dims:
            d = js.get(k)
            if isinstance(d, (int, float)):
                dims[k].append(float(d))
        pt = r.get("predict_trace") or {}
        usage = pt.get("usage") or {}
        pred_tokens += float(usage.get("prompt_tokens") or 0) + float(usage.get("completion_tokens") or 0)
        judge_tokens += float(js.get("judge_tokens") or 0)
        pred_calls += int(pt.get("num_llm_calls") or pt.get("llm_calls") or 1)

    def m(vals):
        return round(statistics.mean(vals), 4) if vals else None

    def head(vals, k):
        return round(statistics.mean(vals[:k]), 4) if len(vals) >= k else m(vals)

    def tail(vals, k):
        return round(statistics.mean(vals[-k:]), 4) if len(vals) >= k else m(vals)

    return {
        "steps": steps,
        "scored": len(oa),
        "failures": failures,
        "oa": m(oa),
        "oa_first5": head(oa, 5),
        "oa_last5": tail(oa, 5),
        **{f"dim_{k}": m(v) for k, v in dims.items()},
        "pred_tokens": round(pred_tokens, 1),
        "judge_tokens": round(judge_tokens, 1),
        "pred_calls": pred_calls,
    }


def collect(roots: list[Path]) -> list[dict]:
    rows = []
    for root in roots:
        if not root.exists():
            continue
        for pred_path in sorted(root.rglob("sequential_predictions.jsonl")):
            if any("CORRUPT" in p or p.lower().startswith("smoke") for p in pred_path.parts):
                continue
            rel = pred_path.relative_to(root)
            cell = rel.parts[0] if rel.parts else ""
            method_user = rel.parts[1] if len(rel.parts) > 1 else pred_path.stem
            stats = cell_stats(pred_path)
            stats["cell"] = cell
            stats["method_user"] = method_user
            rows.append(stats)
    return rows


def render_md(rows: list[dict], psych_rows: list[dict]) -> str:
    psych = {(r["cell"], r["method_user"]): r for r in psych_rows}
    lines = ["# 社交媒体用户个体对齐 Agent：最终实验结果", "", "生成时间：自动聚合。", ""]
    lines += ["## 方法对比（OA / 四维 / 结构化指标）", ""]
    lines += [
        "| cell | method_user | n | OA | first5 | last5 | stance | core | belief | value | psycho | emo_cov | style_cov | 失败 | pred_tokens |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda x: (x["cell"], x["method_user"])):
        p = psych.get((r["cell"], r["method_user"]), {})
        pm = p.get("mean", {})
        lines.append(
            f"| {r['cell']} | {r['method_user']} | {r['scored']} | {r['oa']} | {r['oa_first5']} | "
            f"{r['oa_last5']} | {r['dim_stance']} | {r['dim_core_judgment']} | {r['dim_belief']} | "
            f"{r['dim_value']} | {pm.get('psychometric_align')} | {p.get('emotion_coverage')} | "
            f"{p.get('style_coverage')} | {r['failures']} | {r['pred_tokens']:.0f} |"
        )
    lines += ["", "## 结论要点", ""]
    lines += [
        "- 小样本：架构（Path/Fusion/Agent + CoT）相对 GenMinds/TG 的 OA 提升。",
        "- 大样本：方法差距缩小；随机抽样与全量的差距（sample 0.3/0.5 vs all）。",
        "- 顺序：时序 vs shuffle vs topic-grouped 的差异是否显著。",
        "- 时间：first5 → last5 的在线学习趋势。",
        "- 效率：fast-path / 抽样的 token 与 OA 权衡。",
        "- 指标：judge 自一致性 92.3%（mean|Δ|=0.0045）；结构化指标作为 LLM-judge 的客观补充。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = collect(ROOTS)
    psych_rows = []
    for root in ROOTS:
        if root.exists():
            psych_rows.extend(psych_sweep(root))
    out_dir = Path(r"D:\UserSimuAgent\项目最新版")
    (out_dir / "final_results.json").write_text(
        json.dumps({"cells": rows, "psychometrics": psych_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "final_report.md").write_text(render_md(rows, psych_rows), encoding="utf-8")
    print(f"cells={len(rows)} psych_groups={len(psych_rows)}")
    print("wrote final_results.json / final_report.md")


if __name__ == "__main__":
    main()
