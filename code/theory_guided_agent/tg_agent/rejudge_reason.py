"""原因推断评测（reason-inference evaluation）。

对应 0720 会议：「调了给了原因的测试集去做评测，判断原因推断是不是对的」。

主意见对齐 benchmark 只评 stance/core/belief/value（结果对不对）。
本脚本评的是**过程**：CUV-TG 在 trace 里给出的推断理由
（verbalization + matched theory mechanism）是否解释了用户的真实动机。

用法：
    python -m tg_agent.rejudge_reason \
        --pred D:/UserAgent/outputs/benchmark_cuv_tg/CUV-TG_1989660417/predictions.jsonl \
        --sample 60
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

from tg_agent.benchmark_core import (
    LLMConfig,
    OpenAICompatClient,
    extract_context_and_gt,
)
from tg_agent.llm import load_env

ROOT = Path(__file__).resolve().parents[1]

REASON_JUDGE_SYSTEM = """你是严谨的「原因推断」评测员。
给定同一微博事件的【用户真实评论】、【模型预测评论】与【模型给出的推断理由】，
判断模型给出的理由是否正确解释了用户为什么这么想。从三个维度打分，每维 0 到 1：
- reason_correctness: 推断的动机/机制与真实评论背后动机是否一致
- reason_consistency: 推断理由与预测评论是否自洽（理由是否真的能推出该预测）
- reason_grounding: 推断理由是否有依据（理论机制/用户历史是否支撑，而非空泛套话）
只输出一个 JSON 对象，不要 markdown，不要解释。
格式：{"reason_correctness":0.0,"reason_consistency":0.0,"reason_grounding":0.0}
"""

REASON_KEYS = ["reason_correctness", "reason_consistency", "reason_grounding"]


def build_reason_user(context: str, gt: str, pred: str, reasoning: str) -> str:
    return (
        f"【事件上下文】\n{context}\n\n"
        f"【真实评论 ground_truth】\n{gt}\n\n"
        f"【预测评论 prediction】\n{pred}\n\n"
        f"【模型给出的推断理由】\n{reasoning}\n\n"
        "请打分。"
    )


def extract_reasoning(row: dict, verb_only: bool = False) -> str:
    tr = row.get("agent_trace") or {}
    # verb_only：只评模型陈述的理由本身（fusion 模式的 verbalization 即理由，
    # 因素分解是中间脚手架），避免脚手架文本稀释 grounding/consistency
    if verb_only:
        return str(tr.get("verbalization") or "")
    parts: list[str] = []
    factors = tr.get("factors") or []
    if factors:
        ftxt = "；".join(
            f"{f.get('id')}[{f.get('type')}] {str(f.get('text'))[:60]}"
            for f in factors[:4] if isinstance(f, dict)
        )
        parts.append(f"事件因素分解：{ftxt}")
    verb = tr.get("verbalization")
    if verb:
        label = "推理路径" if factors else "推断过程"
        parts.append(f"{label}：{verb}")
    for t in (tr.get("matched_theories") or [])[:3]:
        name = t.get("name") or t.get("id") or ""
        mech = t.get("mechanism") or ""
        parts.append(f"匹配理论：{name}——{mech}")
    coords = tr.get("activated_coordinates")
    if coords:
        parts.append(f"激活坐标：{coords}")
    return "\n".join(parts)


def parse_reason(text: str) -> dict[str, float]:
    import re

    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", t)
        if not m:
            return {k: 0.0 for k in REASON_KEYS} | {"parse_error": 1.0}
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return {k: 0.0 for k in REASON_KEYS} | {"parse_error": 1.0}
    out = {}
    for k in REASON_KEYS:
        try:
            out[k] = float(max(0.0, min(1.0, float(obj.get(k, 0.0)))))
        except (TypeError, ValueError):
            out[k] = 0.0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="predictions.jsonl (CUV-TG with agent_trace)")
    ap.add_argument("--sample", type=int, default=60, help="evenly-spaced sample size; 0=all")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="", help="output dir; default alongside --pred")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--verb-only", action="store_true", help="只评 verbalization（fusion 模式的理由本身），不含因素分解等脚手架")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    load_env(cfg["paths"]["env_file"])
    qwen_key = os.environ.get("QWEN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY", "")
    if not qwen_key:
        raise SystemExit("QWEN_API_KEY / DASHSCOPE_API_KEY missing")
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

    pred_path = Path(args.pred)
    rows = []
    for line in pred_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if extract_reasoning(r):
            rows.append(r)
    if args.sample and len(rows) > args.sample:
        step = len(rows) / args.sample
        rows = [rows[int(i * step)] for i in range(args.sample)]
    print(f"[reason-judge] {len(rows)} rows from {pred_path}", flush=True)

    def work(r: dict) -> dict:
        context, gt = extract_context_and_gt(r) if "ground_truth" not in r else (
            r.get("context") or "",
            r.get("ground_truth") or "",
        )
        reasoning = extract_reasoning(r, verb_only=args.verb_only)
        pred = r.get("prediction") or ""
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
            scores = {k: 0.0 for k in REASON_KEYS} | {"error": 1.0}
            print(f"[reason-judge] error: {e}", flush=True)
        out = dict(r)
        out["reason_scores"] = scores
        return out

    out_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, r) for r in rows]
        for i, fut in enumerate(as_completed(futs), 1):
            out_rows.append(fut.result())
            if i % 10 == 0 or i == len(futs):
                print(f"[reason-judge] {i}/{len(futs)}", flush=True)

    n = len(out_rows)
    means = {
        k: round(sum(r["reason_scores"].get(k, 0.0) for r in out_rows) / max(n, 1), 4)
        for k in REASON_KEYS
    }
    hi = sorted(out_rows, key=lambda r: -r["reason_scores"].get("reason_correctness", 0))[:3]
    lo = sorted(out_rows, key=lambda r: r["reason_scores"].get("reason_correctness", 0))[:3]

    out_dir = Path(args.out) if args.out else pred_path.parent
    suffix = "_verbonly" if args.verb_only else ""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"reason_judge{suffix}.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in out_rows) + "\n",
        encoding="utf-8",
    )

    def ex_block(r: dict) -> str:
        s = r["reason_scores"]
        return (
            f"- post_id={r.get('post_id')} correctness={s.get('reason_correctness'):.2f} "
            f"consistency={s.get('reason_consistency'):.2f} grounding={s.get('reason_grounding'):.2f}\n"
            f"  - 真实评论：{str(r.get('ground_truth'))[:120]}\n"
            f"  - 推断理由：{extract_reasoning(r, verb_only=args.verb_only)[:200]}"
        )

    report = "\n".join(
        [
            "# 原因推断评测（reason-inference）",
            "",
            f"- source: `{pred_path}`",
            f"- n: {n}（evenly-spaced sample）",
            f"- judge: qwen3.7-plus · generated {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "| metric | mean |",
            "|---|---:|",
            *[f"| {k} | {v:.4f} |" for k, v in means.items()],
            "",
            "reading: reason_correctness = 推断的动机/机制与真实动机一致程度；"
            "consistency = 理由与预测自洽；grounding = 理由有理论/历史依据而非套话。",
            "",
            "## 最好 3 例",
            *[ex_block(r) for r in hi],
            "",
            "## 最差 3 例",
            *[ex_block(r) for r in lo],
            "",
        ]
    )
    (out_dir / f"reason_judge_report{suffix}.md").write_text(report, encoding="utf-8")
    print(report, flush=True)


if __name__ == "__main__":
    main()
