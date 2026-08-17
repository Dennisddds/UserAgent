"""Expand theory domains + crawl + grounded enrich until coverage plateaus.

Operational 'complete enough' for Theory-Guided support:
- grounded structured cards grow with each round
- stop when a round adds < min_new_grounded AND no new crawl queries remain
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> int:
    print("\n>>>", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, cwd=str(ROOT))
    return int(p.returncode)


def _stats() -> dict:
    lib = ROOT / "data" / "theory_library"
    cards = []
    for line in (lib / "cards.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            cards.append(json.loads(line))
    grounded = [c for c in cards if c.get("grounded")]
    by_coord: dict[str, int] = {}
    for c in grounded:
        by_coord[c.get("coordinate") or "unknown"] = by_coord.get(c.get("coordinate") or "unknown", 0) + 1
    meta = json.loads((lib / "meta.json").read_text(encoding="utf-8"))
    return {
        "total": meta.get("num_cards", len(cards)),
        "grounded": len(grounded),
        "structured_grounded": sum(
            1 for c in grounded if (c.get("summary") or "") and c.get("propositions")
        ),
        "completed_queries": meta.get("completed_queries"),
        "coords_with_grounded": len(by_coord),
        "by_coord_grounded": dict(sorted(by_coord.items(), key=lambda x: (-x[1], x[0]))),
    }


def main() -> None:
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    enrich_batches = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    min_new = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    py = sys.executable

    prev_g = _stats()["grounded"]
    print(json.dumps({"start": _stats()}, ensure_ascii=False, indent=2), flush=True)

    for r in range(1, rounds + 1):
        print(f"\n======== EXPAND ROUND {r}/{rounds} ========", flush=True)
        rc = _run(
            [
                py,
                "-m",
                "tg_agent.cli",
                "bootstrap-theory",
                "--source",
                "crossref",
                "--per-query",
                "40",
                "--pages",
                "2",
            ]
        )
        if rc != 0:
            print("crawl failed", rc, flush=True)

        before = _stats()["grounded"]
        for b in range(1, enrich_batches + 1):
            print(f"\n--- enrich batch {b}/{enrich_batches} ---", flush=True)
            rc = _run(
                [
                    py,
                    "-m",
                    "tg_agent.cli",
                    "enrich-theory",
                    "--limit",
                    "40",
                    "--min-citations",
                    "50",
                ]
            )
            if rc != 0:
                print("enrich failed", rc, flush=True)
                break
            time.sleep(1)

        st = _stats()
        gained = st["grounded"] - before
        print(
            json.dumps({"round": r, "gained_grounded": gained, "stats": st}, ensure_ascii=False, indent=2),
            flush=True,
        )
        (ROOT / "data" / "theory_library" / "coverage_report.json").write_text(
            json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if gained < min_new:
            print(
                f"plateau: gained_grounded={gained} < {min_new}; stopping early",
                flush=True,
            )
            break

    final = _stats()
    print(
        json.dumps(
            {"done": True, "delta_grounded": final["grounded"] - prev_g, "final": final},
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
