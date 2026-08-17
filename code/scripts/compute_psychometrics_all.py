"""Sweep all sequential_predictions.jsonl files and emit psychometric alignment means.

Read-only aggregation over experiment outputs; used for the structured-metric
supplement to the LLM-judge OA. Run again after all experiments finish.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from psychometrics_metrics import alignment_score, corpus_network_align


ROOTS = [
    Path(r"D:\UserSimuAgent\项目最新版\exp_outputs"),
    Path(r"D:\UserSimuAgent\项目最新版\exp_x_outputs"),
]


def sweep(root: Path) -> list[dict]:
    rows = []
    for pred_path in sorted(root.rglob("sequential_predictions.jsonl")):
        if any("CORRUPT" in p or p.lower().startswith("smoke") for p in pred_path.parts):
            continue
        rel = pred_path.relative_to(root)
        parts = rel.parts
        cell = parts[0] if parts else ""
        method_user = parts[1] if len(parts) > 1 else pred_path.stem
        n = 0
        acc = {k: 0.0 for k in ("emotion_align", "style_align", "network_align", "psychometric_align")}
        gt_corpus: list[str] = []
        pred_corpus: list[str] = []
        emo_signal = 0
        emo_signal_sum = 0.0
        sty_signal = 0
        sty_signal_sum = 0.0
        cov = {"emotion_coverage": 0.0, "style_coverage": 0.0}
        n_net = 0
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
            gt = r.get("ground_truth") or ""
            pred = r.get("prediction") or ""
            js = r.get("judge_scores") or {}
            if not gt or not pred or r.get("warmup") or js.get("error") or r.get("error"):
                continue
            gt_corpus.append(gt)
            pred_corpus.append(pred)
            s = alignment_score(gt, pred)
            for k in ("emotion_align", "style_align", "psychometric_align"):
                acc[k] += s.get(k, 0.0)
            cov["emotion_coverage"] += s.get("emotion_coverage", 0.0)
            cov["style_coverage"] += s.get("style_coverage", 0.0)
            if s["emotion_align"] < 1.0 or s["emotion_coverage"] > 0.0:
                emo_signal += 1
                emo_signal_sum += s["emotion_align"]
            if s["style_align"] < 1.0 or s["style_coverage"] > 0.0:
                sty_signal += 1
                sty_signal_sum += s["style_align"]
            net = s.get("network_align")
            if net is not None:
                acc["network_align"] += net
                n_net += 1
            n += 1
        if n:
            corpus_net = corpus_network_align(gt_corpus, pred_corpus)
            rows.append(
                {
                    "cell": cell,
                    "method_user": method_user,
                    "n": n,
                    "n_network_valid": n_net,
                    "corpus_network_cosine": corpus_net["cosine"],
                    "corpus_network_l1": corpus_net["l1_agreement"],
                    "emotion_coverage": round(cov["emotion_coverage"] / n, 4),
                    "style_coverage": round(cov["style_coverage"] / n, 4),
                    "emotion_align_signal_only": round(emo_signal_sum / emo_signal, 4) if emo_signal else None,
                    "style_align_signal_only": round(sty_signal_sum / sty_signal, 4) if sty_signal else None,
                    "mean": {
                        k: round(v / (n_net if k == "network_align" and n_net else n), 4)
                        for k, v in acc.items()
                    },
                }
            )
    return rows


def main() -> None:
    out_rows = []
    for root in ROOTS:
        if root.exists():
            out_rows.extend(sweep(root))
    out = Path(r"D:\UserSimuAgent\项目最新版\psychometrics_sweep.json")
    out.write_text(json.dumps(out_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out} with {len(out_rows)} (cell, method_user) groups")
    for row in out_rows:
        print(
            f"{row['cell']:20s} {row['method_user']:34s} n={row['n']:5d} "
            f"psycho={row['mean']['psychometric_align']:.3f} "
            f"emo={row['mean']['emotion_align']:.3f}(sig={row.get('emotion_align_signal_only') or 0:.3f},cov={row['emotion_coverage']:.2f}) "
            f"sty={row['mean']['style_align']:.3f}(cov={row['style_coverage']:.2f}) "
            f"net_corpus={row['corpus_network_cosine']:.3f}"
        )


if __name__ == "__main__":
    main()
