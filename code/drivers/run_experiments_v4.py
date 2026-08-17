"""Parallel experiment driver, v4.

Same cells as v3 but runs independent cells concurrently (each cell keeps its
own out_dir / agent_state, and all state is isolated, so cross-cell parallelism
is safe). Order-variant cells still mutate the shared events file, so they run
sequentially after the parallel batches. Every cell uses --resume: a failed
step is gap-filled on restart instead of silently counted.
"""

from __future__ import annotations

import concurrent.futures
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"D:\UserSimuAgent\项目最新版\UserAgent\theory_guided_agent")
EVENTS = Path(r"D:\UserSimuAgent\项目最新版\UserAgent\outputs\weibo_user_1989660417\events_all.jsonl")
OUT = Path(r"D:\UserSimuAgent\项目最新版\exp_outputs_v2")

METHODS_ALL = "seq-GenMinds,seq-CUV-TG,seq-CUV-Path,seq-CUV-Fusion,seq-CUV-Agent"


def fix_truncated_tail(root: Path) -> None:
    """Repair a prediction file if a kill left a partial final JSON line."""
    for p in root.rglob("sequential_predictions.jsonl"):
        data = p.read_text(encoding="utf-8", errors="replace")
        stripped = data.rstrip("\n")
        if not stripped:
            continue
        idx = stripped.rfind("\n")
        last = stripped[idx + 1 :] if idx != -1 else stripped
        try:
            json.loads(last)
        except json.JSONDecodeError:
            if idx != -1:
                p.write_text(stripped[: idx + 1] + "\n", encoding="utf-8")
                print(f"[repair] truncated tail in {p.name}", flush=True)
            else:
                p.unlink()
                print(f"[repair] removed broken {p.name}", flush=True)


def run(cell: str, user: str, methods: str, extra: list[str]) -> None:
    out = OUT / cell
    cmd = [
        sys.executable,
        "-m",
        "tg_agent.run_sequential",
        "--config",
        str(ROOT / "config_win.yaml"),
        "--user",
        user,
        "--methods",
        methods,
        "--no-ensure-situational",
        "--out-root",
        str(out),
        "--resume",
        *extra,
    ]
    print(f"[cell] {cell} start", flush=True)
    for attempt in range(2):
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode == 0:
            print(f"[cell] {cell} done rc=0", flush=True)
            return
        print(f"[cell] {cell} attempt {attempt+1} rc={r.returncode}; retrying with resume", flush=True)
    print(f"[cell] {cell} FAILED after retries", flush=True)


def run_batch(cells: list[tuple[str, str, str, list[str]]], workers: int) -> None:
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(run, *cell) for cell in cells]
        for fut in concurrent.futures.as_completed(futs):
            try:
                fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"[batch] cell raised {type(e).__name__}: {e}", flush=True)


def make_shuffled(seed: int) -> None:
    rows = [json.loads(l) for l in EVENTS.read_text(encoding="utf-8").splitlines() if l.strip()]
    rng = random.Random(seed)
    rng.shuffle(rows)
    EVENTS.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")


def make_topic_grouped() -> None:
    rows = [json.loads(l) for l in EVENTS.read_text(encoding="utf-8").splitlines() if l.strip()]

    def key(r):
        return "||".join(sorted(r.get("topics") or [])) or "zz_no_topic"

    rows.sort(key=key)
    EVENTS.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    backup = EVENTS.with_suffix(".jsonl.bak")
    if not backup.exists():
        shutil.copyfile(EVENTS, backup)
    original = backup.read_bytes()
    if EVENTS.read_bytes() != original:
        raise SystemExit("events file differs from backup before starting; restore manually")
    fix_truncated_tail(OUT)

    batch1 = [
        ("small_all_methods", "7463374646", METHODS_ALL, []),
        ("big_all_methods", "1989660417", METHODS_ALL, []),
        ("big_sample_0.3", "1989660417", "seq-CUV-Agent", ["--window-sample-ratio", "0.3"]),
        ("big_sample_0.5", "1989660417", "seq-CUV-Agent", ["--window-sample-ratio", "0.5"]),
        ("big_tg_fm", "1989660417", "seq-CUV-TG", ["--tg-failure-memory"]),
    ]
    batch2 = [
        ("big_agent_fm_off", "1989660417", "seq-CUV-Agent", ["--fm-mode", "off"]),
        ("big_agent_fast", "1989660417", "seq-CUV-Agent", ["--fast-path"]),
    ]

    run_batch(batch1, workers=5)
    run_batch(batch2, workers=2)

    shutil.copyfile(backup, EVENTS)
    make_shuffled(42)
    run("big_shuffle_42", "1989660417", "seq-CUV-Agent", [])
    shutil.copyfile(backup, EVENTS)
    make_topic_grouped()
    run("big_topic_grouped", "1989660417", "seq-CUV-Agent", [])
    shutil.copyfile(backup, EVENTS)

    print("ALL V4 EXPERIMENTS DONE", flush=True)


if __name__ == "__main__":
    main()
