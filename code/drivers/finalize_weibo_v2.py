"""Robust Weibo finalize (v2): per-method parallel gap-fill, then report.

Single-instance guarded by a pid marker. Gap-fills every method dir in
parallel (5 workers) with up to 8 passes and a wait between passes, so the
recurring DeepSeek 402 outages no longer force a give-up after 4 quick tries.
Order-variant cells run last with their specific events ordering.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(r"D:\UserSimuAgent\项目最新版")
SCRIPTS = BASE / "UserAgent" / "scripts"
AGENT = BASE / "UserAgent" / "theory_guided_agent"
ROOT = BASE / "exp_outputs_v2"
EVENTS = BASE / "UserAgent" / "outputs" / "weibo_user_1989660417" / "events_all.jsonl"
PID_FILE = BASE / "finalize_weibo.pid"


def acquire() -> None:
    if PID_FILE.exists():
        try:
            old = int(PID_FILE.read_text().strip())
        except Exception:
            old = 0
        if old:
            try:
                os.kill(old, 0)
                raise SystemExit(f"another finalize instance is alive (pid {old})")
            except OSError:
                pass
    PID_FILE.write_text(str(os.getpid()))


def count_failures(out: Path) -> int:
    total = 0
    for pred in out.rglob("sequential_predictions.jsonl"):
        if any("CORRUPT" in p or p.lower().startswith("smoke") for p in pred.parts):
            continue
        seen: set[int] = set()
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
                    seen.add(step)
        total += len(seen)
    return total


def gap_fill(cell: str, methods: str, extra: list[str]) -> None:
    out = ROOT / cell
    cfg = AGENT / "config_win.yaml"
    for attempt in range(8):
        before = count_failures(out)
        if before == 0:
            print(f"[fw] {cell}/{methods}: clean", flush=True)
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
        print(f"[fw] {cell}/{methods}: pass {attempt+1} rc={r.returncode} fail {before} -> {after}", flush=True)
        if after == 0:
            return
        time.sleep(60)
    print(f"[fw] {cell}/{methods}: GAVE UP after 8 passes", flush=True)


def parallel_gap_fill() -> None:
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
        futs = [pool.submit(gap_fill, c, m, e) for c, m, e in cells]
        for fut in concurrent.futures.as_completed(futs):
            try:
                fut.result()
            except Exception as ex:  # noqa: BLE001
                print(f"[fw] raised {type(ex).__name__}: {ex}", flush=True)


def order_variant(name: str, variant: str) -> None:
    backup = EVENTS.with_suffix(".jsonl.bak")
    rows = [json.loads(l) for l in backup.read_text(encoding="utf-8").splitlines() if l.strip()]
    if variant == "shuffle":
        random.Random(42).shuffle(rows)
    else:
        rows.sort(key=lambda r: "||".join(sorted(r.get("topics") or [])) or "zz_no_topic")
    EVENTS.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    try:
        gap_fill(name, "seq-CUV-Agent", [])
    finally:
        shutil.copyfile(backup, EVENTS)


def wait_for(path: Path, needle: str, label: str) -> None:
    print(f"[fw] waiting for {label}", flush=True)
    while True:
        if path.exists() and needle in path.read_text(encoding="utf-8", errors="replace"):
            print(f"[fw] {label} ready", flush=True)
            return
        time.sleep(30)


def sh(cmd: list[str]) -> None:
    print("[fw] $ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, cwd=BASE, check=False)


def main() -> None:
    acquire()
    wait_for(BASE / "experiments_v4.log", "ALL V4 EXPERIMENTS DONE", "weibo v4")
    parallel_gap_fill()
    order_variant("big_shuffle_42", "shuffle")
    order_variant("big_topic_grouped", "topic")

    sh([sys.executable, str(SCRIPTS / "judge_robustness.py"), "--self-consistency-n", "60"])
    sh([sys.executable, str(SCRIPTS / "compute_psychometrics_all.py")])
    sh([sys.executable, str(SCRIPTS / "audit_whitebox_consistency.py")])
    sh([sys.executable, str(SCRIPTS / "audit_time_cutoff.py")])
    sh([sys.executable, str(SCRIPTS / "build_final_report.py")])
    (BASE / "WEIBO_FINAL_COMPLETE.txt").write_text("done\n", encoding="utf-8")
    print("[fw] WEIBO_FINAL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
