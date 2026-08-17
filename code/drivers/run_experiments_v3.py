"""Full Weibo experiment matrix, v3.

v3 changes vs v1:
  - runs on temporal-cutoff-cleaned situational envs (compliance 1.0)
  - writes to exp_outputs_v2 (v1 outputs kept as archive)
  - adds three ablation cells:
      big_tg_fm        TG + 错题本 (quantifies what FM adds on top of TG)
      big_agent_fm_off Agent with 错题本 fully off (no-failure-memory ablation)
      big_agent_fast   Agent with surprise-gated fast path (efficiency cell)
  - fresh outputs, no resume (fully reproducible from cleaned data)
"""

from __future__ import annotations

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
        *extra,
    ]
    print(f"[cell] {cell} start", flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    print(f"[cell] {cell} done rc={r.returncode}", flush=True)


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
    assert EVENTS.read_bytes() == original, "events file differs from backup before starting"

    run("small_all_methods", "7463374646", METHODS_ALL, [])
    run("big_all_methods", "1989660417", METHODS_ALL, [])
    run("big_sample_0.3", "1989660417", "seq-CUV-Agent", ["--window-sample-ratio", "0.3"])
    run("big_sample_0.5", "1989660417", "seq-CUV-Agent", ["--window-sample-ratio", "0.5"])
    run("big_tg_fm", "1989660417", "seq-CUV-TG", ["--tg-failure-memory"])
    run("big_agent_fm_off", "1989660417", "seq-CUV-Agent", ["--fm-mode", "off"])
    run("big_agent_fast", "1989660417", "seq-CUV-Agent", ["--fast-path"])

    shutil.copyfile(backup, EVENTS)
    make_shuffled(42)
    run("big_shuffle_42", "1989660417", "seq-CUV-Agent", [])
    shutil.copyfile(backup, EVENTS)
    make_topic_grouped()
    run("big_topic_grouped", "1989660417", "seq-CUV-Agent", [])
    shutil.copyfile(backup, EVENTS)

    print("ALL V3 EXPERIMENTS DONE", flush=True)


if __name__ == "__main__":
    main()
