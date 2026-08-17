# -*- coding: utf-8 -*-
"""Build each paper method's memory bank independently for a Weibo user."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kg_paper_methods.common import load_user_bundle  # noqa: E402
from kg_paper_methods.builders.kg_reasoning import (  # noqa: E402
    build_claimskg,
    build_cimplekg,
    build_cogkr,
    build_cognet3,
)
from kg_paper_methods.builders.personality import (  # noqa: E402
    build_ddgcn,
    build_kgrat,
    build_semipergcn,
    build_trignet,
)
from kg_paper_methods.builders.stance_diffusion import (  # noqa: E402
    build_cogigraph,
    build_cttn,
    build_enm_senm,
    build_gorec,
    build_rotdiff,
    build_sem,
)
from kg_paper_methods.builders.cognitive import (  # noqa: E402
    build_cognimap,
    build_cognitive_maps_1977,
    build_genminds,
    build_lcg,
)

BUILDERS = {
    "cogkr": build_cogkr,
    "cognet3": build_cognet3,
    "cimplekg": build_cimplekg,
    "claimskg": build_claimskg,
    "ddgcn": build_ddgcn,
    "semipergcn": build_semipergcn,
    "trignet": build_trignet,
    "kgrat": build_kgrat,
    "cttn": build_cttn,
    "enm_senm": build_enm_senm,
    "sem": build_sem,
    "rotdiff": build_rotdiff,
    "gorec": build_gorec,
    "cogigraph": build_cogigraph,
    "genminds": build_genminds,
    "lcg": build_lcg,
    "cognimap": build_cognimap,
    "cognitive_maps_1977": build_cognitive_maps_1977,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user-id", required=True)
    ap.add_argument(
        "--methods",
        default=",".join(BUILDERS.keys()),
        help="comma-separated method keys",
    )
    args = ap.parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in BUILDERS]
    if unknown:
        raise SystemExit(f"unknown methods: {unknown}")

    bundle = load_user_bundle(args.user_id)
    print(f"user={args.user_id} events={len(bundle['events'])} methods={methods}")
    results = []
    for m in methods:
        t0 = time.time()
        try:
            out = BUILDERS[m](args.user_id, bundle)
            mb = json.loads((out / "memory_bank.json").read_text(encoding="utf-8"))
            n = len(mb.get("event_maps") or [])
            print(f"  OK {m}: maps={n} -> {out} ({time.time()-t0:.1f}s)")
            results.append({"method": m, "ok": True, "maps": n, "dir": str(out)})
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {m}: {e}")
            traceback.print_exc()
            results.append({"method": m, "ok": False, "error": str(e)})
    summary = Path("outputs") / f"rebuild_summary_{args.user_id}.json"
    # resolve relative to UserAgent root
    summary = ROOT.parent / "outputs" / f"rebuild_summary_{args.user_id}.json"
    # ROOT is scripts/, parent is UserAgent/
    summary = ROOT / "outputs" / f"rebuild_summary_{args.user_id}.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", summary)


if __name__ == "__main__":
    main()
