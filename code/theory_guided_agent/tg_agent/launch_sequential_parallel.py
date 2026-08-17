"""Launch seq-GenMinds and seq-CUV-TG in parallel for max API overlap."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="1989660417")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--sit-prefetch-workers", type=int, default=8)
    ap.add_argument("--sit-prefetch-ahead", type=int, default=16)
    ap.add_argument(
        "--out-root",
        default=str(Path("d:/UserAgent/outputs/benchmark_sequential_align")),
    )
    args = ap.parse_args()
    out = Path(args.out_root)
    out.mkdir(parents=True, exist_ok=True)

    common = [
        sys.executable,
        "-m",
        "tg_agent.run_sequential",
        "--user",
        args.user,
        "--warmup",
        str(args.warmup),
        "--resume",
        "--out-root",
        str(out),
        "--sit-prefetch-workers",
        str(args.sit_prefetch_workers),
        "--sit-prefetch-ahead",
        str(args.sit_prefetch_ahead),
    ]

    procs = []
    for method, log_name in [
        ("seq-GenMinds", f"{args.user}_genminds.log"),
        ("seq-CUV-TG", f"{args.user}_cuv.log"),
    ]:
        log_path = out / log_name
        cmd = common + ["--methods", method]
        print(f"launch {method} -> {log_path}", flush=True)
        log_f = open(log_path, "a", encoding="utf-8")
        p = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env={**dict(**{k: v for k, v in __import__("os").environ.items()}), "PYTHONIOENCODING": "utf-8"},
        )
        procs.append((method, p, log_f))

    codes = []
    for method, p, log_f in procs:
        code = p.wait()
        log_f.close()
        print(f"{method} exited {code}", flush=True)
        codes.append(code)
    raise SystemExit(0 if all(c == 0 for c in codes) else 1)


if __name__ == "__main__":
    main()
