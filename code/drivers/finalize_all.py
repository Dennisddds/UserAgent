"""Weibo-only final sweep + aggregate report.

Waits for the Weibo v4 marker, then:
  1. gap-fill sweep: rerun every cell with --resume (recovers judge-400 rows)
  2. judge robustness re-audit
  3. psychometric + white-box + time-cutoff audits
  4. final_report.md / final_results.json

The X pipeline (crawl -> sit -> prep -> experiments) is finalized separately by
finalize_x.py once the crawler completes.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(r"D:\UserSimuAgent\项目最新版")
SCRIPTS = BASE / "UserAgent" / "scripts"
AGENT = BASE / "UserAgent" / "theory_guided_agent"


def wait_for(path: Path, needle: str, label: str) -> None:
    print(f"[finalize] waiting for {label}", flush=True)
    while True:
        if path.exists():
            try:
                if needle in path.read_text(encoding="utf-8", errors="replace"):
                    print(f"[finalize] {label} ready", flush=True)
                    return
            except Exception:
                pass
        time.sleep(60)


def sh(cmd: list[str], cwd: Path | None = None) -> None:
    print("[finalize] $ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, cwd=cwd or BASE, check=False)


def gap_fill_cell(out_dir: Path, uid: str, methods: str, extra: list[str]) -> None:
    """Rerun a cell with --resume until no failed rows remain (max 4 passes)."""
    cfg = AGENT / "config_win.yaml"
    for attempt in range(4):
        before = count_failures(out_dir)
        if before == 0:
            return
        r = subprocess.run(
            [
                sys.executable, "-m", "tg_agent.run_sequential",
                "--config", str(cfg),
                "--user", uid,
                "--methods", methods,
                "--no-ensure-situational",
                "--out-root", str(out_dir),
                "--resume",
                *extra,
            ],
            cwd=AGENT,
        )
        if r.returncode == 0:
            after = count_failures(out_dir)
            print(f"[finalize] {out_dir.name}: pass {attempt+1} failures {before} -> {after}", flush=True)
            if after == 0:
                return


def count_failures(out_dir: Path) -> int:
    total = 0
    for pred in out_dir.rglob("sequential_predictions.jsonl"):
        if any("CORRUPT" in p or p.lower().startswith("smoke") for p in pred.parts):
            continue
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
                js = r.get("judge_scores") or {}
                if js.get("error") or r.get("error"):
                    try:
                        step = int(r.get("step", -1))
                    except (TypeError, ValueError):
                        step = -1
                    seen[step] = 1
        total += len(seen)
    return total


def gap_fill_parallel() -> None:
    import concurrent.futures

    weibo_root = BASE / "exp_outputs_v2"
    cells = [
        ("small_all_methods", "7463374646", "seq-GenMinds,seq-CUV-TG,seq-CUV-Path,seq-CUV-Fusion,seq-CUV-Agent", []),
        ("big_all_methods", "1989660417", "seq-GenMinds,seq-CUV-TG,seq-CUV-Path,seq-CUV-Fusion,seq-CUV-Agent", []),
        ("big_sample_0.3", "1989660417", "seq-CUV-Agent", ["--window-sample-ratio", "0.3"]),
        ("big_sample_0.5", "1989660417", "seq-CUV-Agent", ["--window-sample-ratio", "0.5"]),
        ("big_tg_fm", "1989660417", "seq-CUV-TG", ["--tg-failure-memory"]),
        ("big_agent_fm_off", "1989660417", "seq-CUV-Agent", ["--fm-mode", "off"]),
        ("big_agent_fast", "1989660417", "seq-CUV-Agent", ["--fast-path"]),
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futs = [pool.submit(gap_fill_cell, weibo_root / name, uid, methods, extra) for name, uid, methods, extra in cells]
        for fut in concurrent.futures.as_completed(futs):
            try:
                fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"[finalize] gap-fill raised {type(e).__name__}: {e}", flush=True)


def gap_fill_order_variant(name: str, variant: str) -> None:
    """Order-variant cells need their specific events ordering restored."""
    import json
    import random

    events = BASE / "UserAgent" / "outputs" / "weibo_user_1989660417" / "events_all.jsonl"
    backup = events.with_suffix(".jsonl.bak")
    rows = [json.loads(l) for l in backup.read_text(encoding="utf-8").splitlines() if l.strip()]
    if variant == "shuffle":
        random.Random(42).shuffle(rows)
    else:
        rows.sort(key=lambda r: "||".join(sorted(r.get("topics") or [])) or "zz_no_topic")
    events.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    gap_fill_cell(BASE / "exp_outputs_v2" / name, "1989660417", "seq-CUV-Agent", [])
    backup2 = backup.read_bytes()
    events.write_bytes(backup2)


def main() -> None:
    wait_for(BASE / "experiments_v4.log", "ALL V4 EXPERIMENTS DONE", "weibo v4")

    # 1) gap-fill the Weibo matrix
    gap_fill_parallel()
    gap_fill_order_variant("big_shuffle_42", "shuffle")
    gap_fill_order_variant("big_topic_grouped", "topic")

    # X cells were already run with --resume inside run_x_pipeline_v2.py.

    # 2) audits
    sh([sys.executable, str(SCRIPTS / "judge_robustness.py"), "--self-consistency-n", "60"])
    sh([sys.executable, str(SCRIPTS / "compute_psychometrics_all.py")])
    sh([sys.executable, str(SCRIPTS / "audit_whitebox_consistency.py")])
    sh([sys.executable, str(SCRIPTS / "audit_time_cutoff.py")])

    # 3) final report
    sh([sys.executable, str(SCRIPTS / "build_final_report.py")])
    (BASE / "WEIBO_FINAL_COMPLETE.txt").write_text("done\n", encoding="utf-8")
    print("[finalize] WEIBO_FINAL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
