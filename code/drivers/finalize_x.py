"""X-only final sweep + combined report, run after the X v2 pipeline finishes."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

BASE = Path(r"D:\UserSimuAgent\项目最新版")
SCRIPTS = BASE / "UserAgent" / "scripts"


def wait_for(path: Path, needle: str, label: str) -> None:
    print(f"[finalize-x] waiting for {label}", flush=True)
    while True:
        if path.exists():
            try:
                if needle in path.read_text(encoding="utf-8", errors="replace"):
                    print(f"[finalize-x] {label} ready", flush=True)
                    return
            except Exception:
                pass
        time.sleep(60)


def sh(cmd: list[str], cwd: Path | None = None) -> None:
    print("[finalize-x] $ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, cwd=cwd or BASE, check=False)


def main() -> None:
    wait_for(BASE / "x_pipeline_v2.log", "X V2 PIPELINE DONE", "x v2")
    sh([sys.executable, str(SCRIPTS / "judge_robustness.py"), "--self-consistency-n", "60"])
    sh([sys.executable, str(SCRIPTS / "compute_psychometrics_all.py")])
    sh([sys.executable, str(SCRIPTS / "audit_whitebox_consistency.py")])
    sh([sys.executable, str(SCRIPTS / "audit_time_cutoff.py")])
    sh([sys.executable, str(SCRIPTS / "build_final_report.py")])
    (BASE / "FINAL_COMPLETE.txt").write_text("done\n", encoding="utf-8")
    print("[finalize-x] FINAL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
