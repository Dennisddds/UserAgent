import subprocess, sys, time, os
from pathlib import Path

BASE = Path(r"D:\UserSimuAgent\项目最新版")
accounts = sys.argv[1].split(",") if len(sys.argv) > 1 else []
os.environ.setdefault("SIT_SEARCH_BACKEND", "duckduckgo")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

for h in accounts:
    csv = BASE / "x_weibo_csv" / f"{h}.csv"
    out = BASE / "sit_env" / f"{h}_situational_env.json"
    if not csv.exists():
        print(f"[skip] {h}: no csv", flush=True)
        continue
    print(f"[start] {h}", flush=True)
    cmd = [
        sys.executable,
        str(BASE / "UserAgent" / "scripts" / "x_situational_env.py"),
        "--csv", str(csv),
        "--user", h,
        "--out", str(out),
    ]
    ok = False
    for attempt in range(3):
        try:
            subprocess.run(cmd, check=True)
            ok = True
            break
        except subprocess.CalledProcessError as e:
            print(f"[fail] {h} attempt {attempt+1}: {e}", flush=True)
            time.sleep(8)
    if not ok:
        print(f"[giveup] {h}", flush=True)
    time.sleep(2)
print("BATCH DONE", flush=True)
