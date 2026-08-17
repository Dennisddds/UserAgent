"""Early parallel gap-fill for cells damaged by the DeepSeek 402 outage.

Runs per-method --resume cells concurrently (5 workers). Safe to run while the
v4 driver is still finishing big_topic_grouped: all cells below read the
original events file and write only to their own method dirs.
"""

from __future__ import annotations

import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path

BASE = Path(r"D:\UserSimuAgent\项目最新版")
AGENT = BASE / "UserAgent" / "theory_guided_agent"
ROOT = BASE / "exp_outputs_v2"


def count_failures(cell: Path) -> int:
    total = 0
    for pred in cell.rglob("sequential_predictions.jsonl"):
        seen: dict[int, int] = {}
        with pred.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("warmup"):
                    continue
                if (r.get("judge_scores") or {}).get("error") or r.get("error"):
                    try:
                        step = int(r.get("step", -1))
                    except (TypeError, ValueError):
                        step = -1
                    seen[step] = 1
        total += len(seen)
    return total


def gap_fill(cell: str, methods: str, extra: list[str]) -> None:
    out = ROOT / cell
    cfg = AGENT / "config_win.yaml"
    for attempt in range(4):
        before = count_failures(out)
        if before == 0:
            print(f"[gf] {cell}: clean", flush=True)
            return
        r = subprocess.run(
            [
                sys.executable, "-m", "tg_agent.run_sequential",
                "--config", str(cfg),
                "--user", "1989660417",
                "--methods", methods,
                "--no-ensure-situational",
                "--out-root", str(out),
                "--resume",
                *extra,
            ],
            cwd=AGENT,
        )
        after = count_failures(out)
        print(f"[gf] {cell}/{methods}: pass {attempt+1} rc={r.returncode} fail {before} -> {after}", flush=True)
        if after == 0:
            return


def main() -> None:
    cells = [
        ("big_all_methods", "seq-GenMinds", []),
        ("big_all_methods", "seq-CUV-TG", []),
        ("big_all_methods", "seq-CUV-Path", []),
        ("big_all_methods", "seq-CUV-Fusion", []),
        ("big_all_methods", "seq-CUV-Agent", []),
        ("big_tg_fm", "seq-CUV-TG", ["--tg-failure-memory"]),
        ("big_agent_fast", "seq-CUV-Agent", ["--fast-path"]),
        ("big_agent_fm_off", "seq-CUV-Agent", ["--fm-mode", "off"]),
        ("big_sample_0.5", "seq-CUV-Agent", ["--window-sample-ratio", "0.5"]),
        ("big_sample_0.3", "seq-CUV-Agent", ["--window-sample-ratio", "0.3"]),
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        futs = [pool.submit(gap_fill, cell, methods, extra) for cell, methods, extra in cells]
        for fut in concurrent.futures.as_completed(futs):
            try:
                fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"[gf] raised {type(e).__name__}: {e}", flush=True)
    print("GAP_FILL_EARLY DONE", flush=True)


if __name__ == "__main__":
    main()
