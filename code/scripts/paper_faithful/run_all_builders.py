# -*- coding: utf-8 -*-
"""Run every per-paper builder for both users, with per-method isolation."""

import importlib
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

BUILDERS = [
    "build_cogkr", "build_cognet3", "build_cimplekg", "build_claimskg",
    "build_ddgcn", "build_semipergcn", "build_trignet", "build_kgrat",
    "build_cttn", "build_enm_senm", "build_sem",
    "build_rotdiff", "build_gorec", "build_cogigraph",
    "build_genminds", "build_lcg",
]
USERS = ["1989660417", "7463374646"]

if __name__ == "__main__":
    results = {}
    for name in BUILDERS:
        mod = importlib.import_module(name)
        for uid in USERS:
            key = f"{name}:{uid}"
            t0 = time.time()
            try:
                path = mod.build_user(uid)
                results[key] = {"ok": True, "path": path,
                                "seconds": round(time.time() - t0, 1)}
                print(f"[OK]   {key} ({results[key]['seconds']}s)")
            except Exception as exc:
                results[key] = {"ok": False, "error": str(exc)}
                print(f"[FAIL] {key}: {exc}")
                traceback.print_exc()
    out = Path(__file__).parent / "run_all_builders_result.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    n_ok = sum(1 for r in results.values() if r["ok"])
    print(f"\n{n_ok}/{len(results)} builds succeeded -> {out}")
