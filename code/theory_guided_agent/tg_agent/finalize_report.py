from __future__ import annotations

import json
import time
from pathlib import Path

ROOT = Path("d:/UserAgent/outputs/benchmark_cuv_tg")


def load_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    rows = []
    for p in sorted(ROOT.glob("*/metrics.json")):
        m = load_metrics(p)
        if m:
            rows.append(m)

    lines = [
        "# Benchmark: CUV-TG vs GenMinds",
        "",
        f"- generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "- protocol: DeepSeek-V4-Pro predict · Qwen3.7-Plus judge (`enable_thinking=false`)",
        "- metric: `opinion_alignment` = mean(stance, core_judgment, belief, value)",
        "- data: `outputs/weibo_user_{uid}/test.jsonl` (same split as paper_kg)",
        "- U memory: GenMinds banks (best prior method)",
        "",
        "## Results",
        "",
        "| user | method | opinion | stance | core | belief | value | coverage |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for m in sorted(rows, key=lambda x: (x.get("user_id", ""), x.get("method", ""))):
        b = m.get("benchmark") or {}
        lines.append(
            f"| {m.get('user_id')} | {m.get('method')} | "
            f"{float(b.get('opinion_alignment_score', 0)):.4f} | "
            f"{float(b.get('stance', 0)):.4f} | "
            f"{float(b.get('core_judgment', 0)):.4f} | "
            f"{float(b.get('belief', 0)):.4f} | "
            f"{float(b.get('value', 0)):.4f} | "
            f"{m.get('coverage', '')} |"
        )

    lines += [
        "",
        "## Notes",
        "",
        "- Large-user GenMinds scores here are **re-judged** with the current Qwen key on existing DeepSeek predictions.",
        "- CUV-TG uses sparse Theory matching + GenMinds retrieval (training-free); not the old prompt-dump TG.",
        "- Small-user n=10 rankings are unstable (prior paper_kg warning); treat as smoke.",
        "",
        "## Prior published reference",
        "",
        "| user | method | opinion |",
        "|---|---|---:|",
        "| 1989660417 | GenMinds (paper_kg) | 0.5982 |",
        "| 7463374646 | GenMinds (paper_kg) | 0.2850 |",
        "",
    ]
    out = ROOT / "comparison_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    (ROOT / "comparison_summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
