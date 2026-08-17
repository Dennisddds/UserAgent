"""Plot OA vs number of posts learned (online alignment curve).

Uses currently clean cells (deduped by step, warmup/error rows excluded).
Preliminary: big-user cells are mid gap-fill; the plot is regenerated in the
final report once everything completes.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def load_oa(pred_path: Path) -> list[float]:
    rows: dict[int, dict] = {}
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
                continue
            if step >= 0:
                rows[step] = r
    oa = []
    for step in sorted(rows):
        r = rows[step]
        if r.get("warmup"):
            continue
        js = r.get("judge_scores") or {}
        if js.get("error") or r.get("error"):
            continue
        v = js.get("opinion_alignment_score")
        if isinstance(v, (int, float)):
            oa.append(float(v))
    return oa


def load_oa_merged(*pred_paths: Path) -> list[float]:
    """Merge rows across runs by step; prefer valid (non-error) rows, then
    earlier paths among equal validity."""
    rows: dict[int, dict] = {}
    for pred_path in pred_paths:
        if not pred_path.exists():
            continue
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
                    continue
                if step < 0:
                    continue
                js = r.get("judge_scores") or {}
                valid = bool((r.get("prediction") or "").strip()) and not (js.get("error") or r.get("error"))
                old = rows.get(step)
                old_valid = False
                if old is not None:
                    ojs = old.get("judge_scores") or {}
                    old_valid = bool((old.get("prediction") or "").strip()) and not (ojs.get("error") or old.get("error"))
                if old is None or (valid and not old_valid):
                    rows[step] = r
    oa = []
    for step in sorted(rows):
        r = rows[step]
        if r.get("warmup"):
            continue
        js = r.get("judge_scores") or {}
        if js.get("error") or r.get("error"):
            continue
        v = js.get("opinion_alignment_score")
        if isinstance(v, (int, float)):
            oa.append(float(v))
    return oa


def rolling(vals: list[float], window: int) -> list[float]:
    out = []
    acc = 0.0
    for i, v in enumerate(vals):
        acc += v
        if i >= window:
            acc -= vals[i - window]
        out.append(acc / min(i + 1, window))
    return out


def panel(ax, series: dict[str, list[float]], title: str, window: int) -> None:
    for label, vals in series.items():
        x = list(range(1, len(vals) + 1))
        if len(vals) <= 1:
            continue
        ax.plot(x, rolling(vals, window), lw=1.8, label=f"{label} (n={len(vals)})")
        ax.plot(x, vals, lw=0.35, alpha=0.25)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("已学习的帖子数（步）")
    ax.set_ylabel("OA")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")


def main() -> None:
    v2 = Path(r"D:\UserSimuAgent\项目最新版\exp_outputs_v2")
    v1 = Path(r"D:\UserSimuAgent\项目最新版\exp_outputs")
    x1 = Path(r"D:\UserSimuAgent\项目最新版\exp_x_outputs")

    big_gm = load_oa_merged(
        v2 / "big_all_methods" / "seq-GenMinds_1989660417" / "sequential_predictions.jsonl",
        v1 / "big_all_methods" / "seq-GenMinds_1989660417" / "sequential_predictions.jsonl",
    )
    big_tg = load_oa_merged(
        v2 / "big_all_methods" / "seq-CUV-TG_1989660417" / "sequential_predictions.jsonl",
        v1 / "big_all_methods" / "seq-CUV-TG_1989660417" / "sequential_predictions.jsonl",
    )

    def x_series(uid: str) -> dict[str, list[float]]:
        return {
            m: load_oa(x1 / f"small_{uid}" / f"{m}_{uid}" / "sequential_predictions.jsonl")
            for m in ("seq-GenMinds", "seq-CUV-TG", "seq-CUV-Agent")
        }

    lilian = x_series("lilianweng")
    hinton = x_series("geoffreyhinton")

    small = {
        m: load_oa(v2 / "small_all_methods" / f"{m}_7463374646" / "sequential_predictions.jsonl")
        for m in ("seq-GenMinds", "seq-CUV-TG", "seq-CUV-Path", "seq-CUV-Fusion", "seq-CUV-Agent")
    }

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    panel(
        axes[0][0],
        {"GenMinds（合并 v1+v2，不含 sit 完全可比）": big_gm},
        "微博大用户 1989660417 · GenMinds（v1+v2 合并）",
        max(10, len(big_gm) // 20),
    )
    panel(
        axes[0][1],
        {"CUV-TG（合并 v1+v2，v1 为旧 sit）": big_tg},
        "微博大用户 1989660417 · CUV-TG（v1+v2 合并）",
        max(10, len(big_tg) // 20),
    )
    panel(axes[0][2], lilian, "X 用户 lilianweng · 三方法", max(5, len(next(iter(lilian.values()), [])) // 12))
    panel(axes[1][0], hinton, "X 用户 geoffreyhinton · 三方法", max(5, len(next(iter(hinton.values()), [])) // 12))
    panel(axes[1][1], small, "微博小用户 7463374646 · 五方法", 5)
    axes[1][2].axis("off")

    fig.suptitle("个体对齐在线学习曲线：OA 随已学习帖子数的变化（粗线=滑动平均，细线=单步）", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path(r"D:\UserSimuAgent\项目最新版\learning_curve_preliminary.png")
    fig.savefig(out, dpi=160)
    print("saved", out)
    print("big_gm n=", len(big_gm), "big_tg n=", len(big_tg))


if __name__ == "__main__":
    main()
