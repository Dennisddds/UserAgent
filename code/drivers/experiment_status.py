"""Print per-cell progress from prediction files (disambiguates same-method cells)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(r"D:\UserSimuAgent\项目最新版\exp_outputs_v2")


def main() -> None:
    if not ROOT.exists():
        return
    for cell in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        if "CORRUPT" in cell.name or cell.name.lower().startswith("smoke"):
            continue
        for method_dir in sorted(p for p in cell.iterdir() if p.is_dir()):
            pred = method_dir / "sequential_predictions.jsonl"
            if not pred.exists():
                continue
            n = 0
            fail = 0
            scored = 0
            with pred.open(encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    n += 1
                    js = r.get("judge_scores") or {}
                    if r.get("warmup"):
                        continue
                    if js.get("error") or r.get("error"):
                        fail += 1
                    elif (r.get("prediction") or "").strip():
                        scored += 1
            print(f"{cell.name:22s} {method_dir.name:34s} steps={n:5d} scored={scored:5d} fail={fail}")


if __name__ == "__main__":
    main()
