#!/usr/bin/env python3
"""Audit existing output directories and map them onto the meeting experiment matrix.

This script does NOT call any LLM. It reads `metrics.json` files and reports
which methods/users/settings exist and which required ablation cells are missing.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REQUIRED_METHODS = {
    "GenMinds",
    "seq-GenMinds",
    "CUV-TG",
    "seq-CUV-TG",
    "CUV-Path",
    "seq-CUV-Path",
    "CUV-Fusion",
    "seq-CUV-Fusion",
    "seq-CUV-Agent",
}

REQUIRED_USERS = {"1989660417", "7463374646"}


def _load_metrics(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("experiment_audit.csv"))
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(args.outputs_root.rglob("metrics.json")):
        m = _load_metrics(metrics_path)
        if not m:
            continue
        bench = m.get("benchmark") or {}
        rows.append(
            {
                "run_dir": str(metrics_path.parent),
                "user_id": m.get("user_id"),
                "method": m.get("method"),
                "benchmark_type": m.get("benchmark_type"),
                "protocol": m.get("protocol"),
                "predict_model": m.get("predict_model"),
                "judge_model": m.get("judge_model"),
                "num_events_total": m.get("num_events_total"),
                "num_steps": m.get("num_steps"),
                "warmup": m.get("warmup"),
                "num_scored": m.get("num_scored") or bench.get("n"),
                "oa": bench.get("opinion_alignment_score"),
                "stance": bench.get("stance"),
                "core_judgment": bench.get("core_judgment"),
                "belief": bench.get("belief"),
                "value": bench.get("value"),
                "first_5": (m.get("late_alignment") or {}).get("first_5"),
                "last_5": (m.get("late_alignment") or {}).get("last_5"),
                "last_10": (m.get("late_alignment") or {}).get("last_10"),
            }
        )

    if not rows:
        raise SystemExit("No metrics.json files found under " + str(args.outputs_root))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    present = {(str(r["method"]), str(r["user_id"])) for r in rows if r["method"] and r["user_id"]}
    missing_cells: list[str] = []
    for method in sorted(REQUIRED_METHODS):
        for user in sorted(REQUIRED_USERS):
            if (method, user) not in present:
                missing_cells.append(f"{method}/{user}")

    print("Wrote", args.out)
    print("Runs found:", len(rows))
    print("Methods found:", sorted({str(r["method"]) for r in rows if r["method"]}))
    print("Users found:", sorted({str(r["user_id"]) for r in rows if r["user_id"]}))
    print("Missing required cells:")
    print("\n".join(missing_cells) if missing_cells else "(none)")


if __name__ == "__main__":
    main()

