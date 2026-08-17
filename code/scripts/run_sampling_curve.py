#!/usr/bin/env python3
"""Estimate dataset sampling/distribution coverage without running predictions.

This is a statistical sanity-check helper for the meeting question:
"random selection 大样本量抽样是不是会影响效果 / 统计学保证每个地方都有".
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def coverage(rows: list[dict[str, str]], field: str) -> dict[str, Any]:
    if not rows:
        return {"field": field, "n": 0, "unique": 0, "entropy": 0.0}
    counts = Counter(str(r.get(field) or "<empty>") for r in rows)
    total = len(rows)
    entropy = -sum((c / total) * math.log(c / total) for c in counts.values() if c > 0)
    return {
        "field": field,
        "n": total,
        "unique": len(counts),
        "entropy": round(entropy, 4),
        "top3": counts.most_common(3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--field", type=str, default="话题")
    parser.add_argument("--sizes", type=str, default="34,50,100,200,500,1000,2657")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("sampling_curve.json"))
    args = parser.parse_args()

    rows = _load_csv(args.csv)
    sizes = [int(x) for x in args.sizes.split(",") if x.strip().isdigit()]
    sizes = [s for s in sizes if 0 < s <= len(rows)]
    results: list[dict[str, Any]] = []

    for size in sizes:
        for seed in range(args.seeds):
            rng = random.Random(seed)
            sample = rng.sample(rows, size)
            cov = coverage(sample, args.field)
            results.append(
                {
                    "size": size,
                    "seed": seed,
                    "field": args.field,
                    "unique": cov["unique"],
                    "entropy": cov["entropy"],
                    "top3": cov["top3"],
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"csv": str(args.csv), "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Wrote", args.out)
    print("Rows:", len(rows), "field:", args.field)
    for size in sizes:
        uniqs = [r["unique"] for r in results if r["size"] == size]
        ents = [r["entropy"] for r in results if r["size"] == size]
        print(f"size={size}: unique_mean={sum(uniqs)/len(uniqs):.1f}, entropy_mean={sum(ents)/len(ents):.3f}")


if __name__ == "__main__":
    main()

