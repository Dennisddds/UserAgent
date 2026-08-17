#!/usr/bin/env python3
"""Audit LLM-judge stability from existing predictions.

If the same `post_id` has multiple judged rows, compute exact disagreement.
Otherwise, report score distribution and bootstrap CI to expose how much the
reported point estimate should be trusted.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _oa(row: dict[str, Any]) -> float | None:
    js = row.get("judge_scores") or {}
    if not js:
        return None
    try:
        return float(js.get("opinion_alignment_score"))
    except (TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=Path("judge_stability.json"))
    args = parser.parse_args()

    oa_by_file: dict[str, list[float]] = defaultdict(list)
    rows_by_post: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    files = sorted(args.outputs_root.rglob("sequential_predictions.jsonl"))
    files += sorted(args.outputs_root.rglob("predictions.jsonl"))

    for path in files:
        for row in _read_jsonl(path):
            oa = _oa(row)
            if oa is None:
                continue
            oa_by_file[str(path)].append(oa)
            key = (str(row.get("method") or ""), str(row.get("post_id") or ""))
            rows_by_post[key].append({"path": str(path), "oa": oa})

    exact_duplicates: list[dict[str, Any]] = []
    for (method, post_id), entries in rows_by_post.items():
        if len(entries) > 1:
            values = [e["oa"] for e in entries]
            exact_duplicates.append(
                {
                    "method": method,
                    "post_id": post_id,
                    "n": len(entries),
                    "min": min(values),
                    "max": max(values),
                    "range": round(max(values) - min(values), 4),
                    "entries": entries,
                }
            )

    all_oa = [v for vals in oa_by_file.values() for v in vals]
    summary: dict[str, Any] = {
        "files": len(oa_by_file),
        "scored_rows": len(all_oa),
        "exact_duplicate_groups": len(exact_duplicates),
        "max_exact_range": max((d["range"] for d in exact_duplicates), default=0.0),
    }
    if all_oa:
        mean = sum(all_oa) / len(all_oa)
        var = sum((x - mean) ** 2 for x in all_oa) / len(all_oa)
        summary["mean_oa"] = round(mean, 4)
        summary["std_oa"] = round(math.sqrt(var), 4)
        rng = random.Random(42)
        boot_means = []
        for _ in range(args.bootstrap):
            sample = [rng.choice(all_oa) for _ in all_oa]
            boot_means.append(sum(sample) / len(sample))
        boot_means.sort()
        summary["bootstrap_95ci"] = [
            round(boot_means[int(len(boot_means) * 0.025)], 4),
            round(boot_means[int(len(boot_means) * 0.975)], 4),
        ]
    summary["exact_duplicates"] = exact_duplicates[:20]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Wrote", args.out)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

