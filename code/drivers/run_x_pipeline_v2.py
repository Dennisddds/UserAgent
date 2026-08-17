"""Full X pipeline, v2: run after the crawl completes.

Waits for CRAWL_COMPLETE.txt and the first sit batch to finish, then:
  1. cleans all sit_env files for temporal-cutoff compliance
  2. builds missing per-user situational envs (all final users)
  3. re-runs x_prepare_pipeline.py
  4. re-runs the X experiments on full data with --resume
"""

from __future__ import annotations

import subprocess
import sys
import time
import yaml
from pathlib import Path

BASE = Path(r"D:\UserSimuAgent\项目最新版")
AGENT = BASE / "UserAgent" / "theory_guided_agent"
SIT_DIR = BASE / "sit_env"

# final non-politician KOL set
LARGE = ["elonmusk", "ggreenwald", "GaryMarcus", "nntaleb", "paulkrugman",
         "wangzhian8848", "ezraklein", "HuXijin_GT", "tegmark"]
SMALL = ["yoshua_bengio", "leahstokes", "mjs_DC", "FreedmanRach", "drmikeisraetel",
         "cdixon", "MiraMurati", "michaeljburry", "chipro", "DarioAmodei",
         "paulfchristiano", "geoffreyhinton", "lilianweng", "johnschulman2",
         "ArvindKrishna", "LisaSu", "HowardMarksBook", "sejnowski", "jensenhuang"]
ALL = LARGE + SMALL


def wait_for_marker(path: Path, needle: str, label: str) -> None:
    print(f"[x-v2] waiting for {label} ({needle!r})", flush=True)
    while True:
        if path.exists():
            try:
                if needle in path.read_text(encoding="utf-8", errors="replace"):
                    print(f"[x-v2] {label} ready", flush=True)
                    return
            except Exception:
                pass
        time.sleep(30)


def sh(cmd: list[str], cwd: Path | None = None) -> None:
    print("[x-v2] $ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, cwd=cwd or BASE, check=False)


def cfg_for(uid: str) -> Path:
    cfg = yaml.safe_load((AGENT / "config_win.yaml").read_text(encoding="utf-8"))
    cfg["paths"]["genminds"][uid] = str(AGENT / "outputs" / f"x_genminds_{uid}" / "memory_bank.json")
    cfg["paths"]["persona"][uid] = str(AGENT / "outputs" / f"x_user_{uid}" / "persona.json")
    cfg["paths"]["user_csv"][uid] = str(BASE / "x_weibo_csv" / f"{uid}.csv")
    p = AGENT / f"config_x_{uid}.yaml"
    p.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return p


def run_experiments(uid: str, methods: str, out: Path) -> None:
    cfgp = cfg_for(uid)
    sh(
        [
            sys.executable, "-m", "tg_agent.run_sequential",
            "--config", str(cfgp),
            "--user", uid,
            "--methods", methods,
            "--no-ensure-situational",
            "--sit-suffix", "",
            "--out-root", str(out),
            "--resume",
        ],
        cwd=AGENT,
    )


def main() -> None:
    wait_for_marker(BASE / "CRAWL_COMPLETE.txt", "done", "crawl")
    sit_batch_log = BASE / "sit_batch2.log"
    if sit_batch_log.exists() and "BATCH DONE" not in sit_batch_log.read_text(encoding="utf-8", errors="replace"):
        wait_for_marker(sit_batch_log, "BATCH DONE", "sit batch 1")

    # 1) clean temporal cutoff
    sh([sys.executable, str(BASE / "UserAgent" / "scripts" / "clean_sit_cutoff.py"), str(SIT_DIR)])

    # 2) build missing sit envs
    for h in ALL:
        out = SIT_DIR / f"{h}_situational_env.json"
        csv = BASE / "x_weibo_csv" / f"{h}.csv"
        if out.exists() or not csv.exists():
            print(f"[x-v2] sit: skip {h}", flush=True)
            continue
        for attempt in range(3):
            r = subprocess.run(
                [
                    sys.executable,
                    str(BASE / "UserAgent" / "scripts" / "x_situational_env.py"),
                    "--csv", str(csv),
                    "--user", h,
                    "--out", str(out),
                ],
                cwd=BASE,
            )
            if r.returncode == 0:
                break
            time.sleep(10)

    # 3) prepare pipeline inputs
    sh([sys.executable, str(BASE / "UserAgent" / "scripts" / "x_prepare_pipeline.py")])

    # 4) experiments on full data
    out_root = BASE / "exp_x_outputs_v2"
    out_root.mkdir(parents=True, exist_ok=True)
    for uid in SMALL:
        run_experiments(uid, "seq-GenMinds,seq-CUV-TG,seq-CUV-Agent", out_root / f"small_{uid}")
    for uid in LARGE:
        run_experiments(uid, "seq-CUV-TG,seq-CUV-Agent", out_root / f"large_{uid}")
    print("X V2 PIPELINE DONE", flush=True)


if __name__ == "__main__":
    main()
