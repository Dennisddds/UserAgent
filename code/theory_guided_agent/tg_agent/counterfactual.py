"""反事实探测（蓝图阶段四·反事实探测）：验证路径是因果的而非事后合理化。

对每条已完成的路径预测，把 salience 最高的因素消融掉
（告知模型「假设事件不涉及该因素」），重跑因素→路径→立场推导：
- stance 翻转 → 该因素有因果作用（路径敏感，好事）
- stance 不动 → 该因素只是装饰（事后合理化信号）
聚合指标 path_sensitivity = 翻转率 + 平均立场距离。

用法：
    python -m tg_agent.counterfactual \
        --pred outputs/.../CUV-Path_1989660417/predictions.jsonl --sample 20
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

from tg_agent.agent import _parse_json
from tg_agent.benchmark_core import extract_context_and_gt
from tg_agent.factors import EventFactor, ablate_stimulus
from tg_agent.llm import DeepSeekClient, load_env

ROOT = Path(__file__).resolve().parents[1]

STANCE_NUM = {"support": 1.0, "mixed": 0.0, "uncertain": 0.0, "oppose": -1.0}

_CF_SYSTEM = """你在扮演指定微博用户本人。给定一个事件（含反事实设定）和该用户的历史证据，
判断该用户在此【修改后的事件】下的立场。只输出 JSON：
{"stance":"support|oppose|mixed|uncertain","brief":"<=40字"}"""


def counterfactual_stance(
    llm: DeepSeekClient,
    *,
    identity: str,
    stimulus: str,
    factor: EventFactor,
    evidence_lines: list[str],
) -> dict[str, str]:
    cf_stimulus = ablate_stimulus(stimulus, factor)
    raw = llm.chat(
        [{"role": "system", "content": _CF_SYSTEM},
         {"role": "user", "content": (
             f"【身份】{identity}\n"
             f"【修改后的事件】\n{cf_stimulus[:600]}\n"
             f"【用户历史证据】\n" + ("\n".join(evidence_lines[:6]) or "(无)") + "\n请判断。"
         )}],
        temperature=0.2, max_tokens=300, disable_thinking=True,
    )
    obj = _parse_json(raw)
    return {
        "stance": str(obj.get("stance") or "uncertain"),
        "brief": str(obj.get("brief") or "")[:80],
    }


def probe_row(llm: DeepSeekClient, row: dict, *, max_factors: int = 2) -> dict:
    tr = row.get("agent_trace") or {}
    factors = [EventFactor.from_dict(f) for f in (tr.get("factors") or []) if isinstance(f, dict)]
    factors = sorted(factors, key=lambda f: -f.salience)[:max_factors]
    orig = str(row.get("stance") or tr.get("stance") or "uncertain")
    context = row.get("context") or extract_context_and_gt(row)[0]
    evidence_lines = [
        f"- {e.get('title', '')}: {(e.get('opinion') or '')[:120]}"
        for e in (tr.get("evidence_events") or [])[:6]
    ]
    identity = str(row.get("user_name") or row.get("user_id") or "")
    probes = []
    for f in factors:
        try:
            cf = counterfactual_stance(
                llm, identity=identity, stimulus=context, factor=f, evidence_lines=evidence_lines,
            )
        except Exception as e:  # noqa: BLE001
            cf = {"stance": "error", "brief": str(e)[:80]}
        cf_stance = cf["stance"] if cf["stance"] in STANCE_NUM else "uncertain"
        flip = (STANCE_NUM.get(orig, 0.0) > 0) != (STANCE_NUM.get(cf_stance, 0.0) > 0) \
            or (STANCE_NUM.get(orig, 0.0) < 0) != (STANCE_NUM.get(cf_stance, 0.0) < 0)
        if STANCE_NUM.get(orig, 0.0) == 0.0 and STANCE_NUM.get(cf_stance, 0.0) == 0.0:
            flip = False
        dist = abs(STANCE_NUM.get(orig, 0.0) - STANCE_NUM.get(cf_stance, 0.0)) / 2.0
        probes.append({
            "factor_id": f.id, "factor_type": f.type, "factor_text": f.text[:80],
            "salience": f.salience,
            "orig_stance": orig, "cf_stance": cf_stance,
            "flip": bool(flip), "stance_distance": round(dist, 3),
            "cf_brief": cf["brief"],
        })
    return {"post_id": row.get("post_id"), "orig_stance": orig, "probes": probes}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="predictions.jsonl (CUV-Path rows)")
    ap.add_argument("--sample", type=int, default=20, help="evenly-spaced sample; 0=all")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    load_env(cfg["paths"]["env_file"])
    llm = DeepSeekClient(model=cfg["llm"]["model"])

    pred_path = Path(args.pred)
    rows = []
    for line in pred_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if (r.get("agent_trace") or {}).get("factors"):
            rows.append(r)
    if args.sample and len(rows) > args.sample:
        step = len(rows) / args.sample
        rows = [rows[int(i * step)] for i in range(args.sample)]
    print(f"[counterfactual] {len(rows)} rows from {pred_path}", flush=True)

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(probe_row, llm, r) for r in rows]
        for i, fut in enumerate(as_completed(futs), 1):
            results.append(fut.result())
            if i % 5 == 0 or i == len(results):
                print(f"[counterfactual] {i}/{len(rows)}", flush=True)

    probes = [p for r in results for p in r["probes"]]
    n = max(1, len(probes))
    flip_rate = round(sum(1 for p in probes if p["flip"]) / n, 4)
    mean_dist = round(sum(p["stance_distance"] for p in probes) / n, 4)
    by_type: dict[str, list[bool]] = {}
    for p in probes:
        by_type.setdefault(p["factor_type"], []).append(p["flip"])
    type_table = {
        t: round(sum(1 for x in xs if x) / max(1, len(xs)), 3)
        for t, xs in sorted(by_type.items())
    }

    out_dir = Path(args.out) if args.out else pred_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "counterfactual.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n", encoding="utf-8",
    )
    report = "\n".join([
        "# 反事实探测：路径因果性验证",
        "",
        f"- source: `{pred_path}`",
        f"- rows: {len(results)} · probes: {len(probes)} · generated {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"| metric | value |",
        f"|---|---:|",
        f"| stance 翻转率（flip rate） | {flip_rate:.3f} |",
        f"| 平均立场距离（0-1） | {mean_dist:.3f} |",
        "",
        "reading: 消融某因素后 stance 翻转 → 该因素对结论有因果作用（路径是因果的）；",
        "翻转率过低 → 预测不依赖所声明的路径（事后合理化信号）。",
        "",
        "## 按因素类型的翻转率",
        *[f"| {t} | {v:.3f} |" for t, v in type_table.items()],
        "",
    ])
    (out_dir / "counterfactual_report.md").write_text(report, encoding="utf-8")
    print(report, flush=True)


if __name__ == "__main__":
    main()
