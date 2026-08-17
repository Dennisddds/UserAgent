"""Improved-vs-baseline comparison driver (AgentX vs seq-CUV-Agent).

The baseline seq-CUV-Agent is unchanged; the improved variant seq-CUV-AgentX
adds the paper-informed layer (candidate branch simulation + DoG-style
multi-agent debate + reliability-guided rubric selection). This driver:
  - small Weibo user: AgentX (baseline already exists in exp_outputs_v2)
  - big Weibo user: AgentX (long background run)
  - three X users: fresh baseline Agent + AgentX under identical cleaned inputs
"""

from __future__ import annotations

import concurrent.futures
import shutil
import subprocess
import sys
import yaml
from pathlib import Path

BASE = Path(r"D:\UserSimuAgent\项目最新版")
AGENT = BASE / "UserAgent" / "theory_guided_agent"
OUT = BASE / "exp_outputs_improved"

X_USERS = ["geoffreyhinton", "johnschulman2", "lilianweng"]


def sh(cmd: list[str], cwd: Path | None = None) -> int:
    print("[imp] $ " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, cwd=cwd or BASE).returncode


def prep_x_sit(uid: str) -> None:
    """Cleaned sit copy into data/users (identical input for baseline & AgentX)."""
    src = BASE / "sit_env" / f"{uid}_situational_env.json"
    dst = AGENT / "data" / "users" / f"{uid}_situational_env.json"
    if not src.exists():
        print(f"[imp] {uid}: no sit env, skip", flush=True)
        return
    sh([sys.executable, str(BASE / "UserAgent" / "scripts" / "clean_sit_cutoff.py"), str(src)])
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def cfg_for(uid: str) -> Path:
    cfg = yaml.safe_load((AGENT / "config_win.yaml").read_text(encoding="utf-8"))
    cfg["paths"]["genminds"][uid] = str(AGENT / "outputs" / f"x_genminds_{uid}" / "memory_bank.json")
    cfg["paths"]["persona"][uid] = str(AGENT / "outputs" / f"x_user_{uid}" / "persona.json")
    cfg["paths"]["user_csv"][uid] = str(BASE / "x_weibo_csv" / f"{uid}.csv")
    p = AGENT / f"config_x_{uid}.yaml"
    p.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return p


def run_cell(cell: str, uid: str, methods: str, cfg: Path) -> None:
    for attempt in range(2):
        rc = sh(
            [
                sys.executable, "-m", "tg_agent.run_sequential",
                "--config", str(cfg),
                "--user", uid,
                "--methods", methods,
                "--no-ensure-situational",
                "--sit-suffix", "",
                "--out-root", str(OUT / cell),
                "--resume",
            ],
            cwd=AGENT,
        )
        if rc == 0:
            break


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    win_cfg = AGENT / "config_win.yaml"

    # 1) small Weibo user: quick, controlled comparison vs existing baseline
    run_cell("small_agentx", "7463374646", "seq-CUV-AgentX", win_cfg)
    print("[imp] small AgentX done", flush=True)

    # 2) big Weibo user + X users in parallel
    for uid in X_USERS:
        prep_x_sit(uid)
    cells = [("big_agentx", "1989660417", "seq-CUV-AgentX", win_cfg)]
    for uid in X_USERS:
        cfgp = cfg_for(uid)
        cells.append((f"x_{uid}_agent", uid, "seq-CUV-Agent", cfgp))
        cells.append((f"x_{uid}_agentx", uid, "seq-CUV-AgentX", cfgp))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futs = [pool.submit(run_cell, *c) for c in cells]
        for fut in concurrent.futures.as_completed(futs):
            try:
                fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"[imp] cell raised {type(e).__name__}: {e}", flush=True)
    print("IMPROVED COMPARISON DONE", flush=True)


if __name__ == "__main__":
    main()
