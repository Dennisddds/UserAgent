# -*- coding: utf-8 -*-
"""Unified paper-KG experiment pipeline for UserAgent.

Phases:
  1. Build all 16 rule-based paper KGs (from CognitiveMap faithful rebuilds)
  2. Validate bank structure
  3. Run predict+judge benchmark
  4. Report summary

Usage:
  python run_full_experiment.py --user 7463374646 --max-samples 10
  python run_full_experiment.py --user 1989660417 --build-only
  python run_full_experiment.py --user 1989660417 --benchmark-only --max-samples 100
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))  # for run_paper_kg_benchmark imports

ROOT = HERE.parents[1]  # UserAgent/
OUT = ROOT / "outputs"
TG = ROOT / "theory_guided_agent"
sys.path.insert(0, str(TG))

# Rule-based builders migrated from CognitiveMap
RULE_BUILDERS = [
    "build_cogkr",
    "build_cognet3",
    "build_cimplekg",
    "build_claimskg",
    "build_ddgcn",
    "build_semipergcn",
    "build_trignet",
    "build_kgrat",
    "build_cttn",
    "build_enm_senm",
    "build_sem",
    "build_rotdiff",
    "build_gorec",
    "build_cogigraph",
    "build_genminds",
    "build_lcg",
]

METHOD_ALIAS = {
    "build_cogkr": "cogkr",
    "build_cognet3": "cognet3",
    "build_cimplekg": "cimplekg",
    "build_claimskg": "claimskg",
    "build_ddgcn": "ddgcn",
    "build_semipergcn": "semipergcn",
    "build_trignet": "trignet",
    "build_kgrat": "kgrat",
    "build_cttn": "cttn",
    "build_enm_senm": "enm_senm",
    "build_sem": "sem",
    "build_rotdiff": "rotdiff",
    "build_gorec": "gorec",
    "build_cogigraph": "cogigraph",
    "build_genminds": "genminds",
    "build_lcg": "lcg",
}

USERS = ["1989660417", "7463374646"]


# ═══════════════════════════════════════════════════════════════════
# Phase 1: Build
# ═══════════════════════════════════════════════════════════════════

def build_all(uid: str, methods: list[str] | None = None) -> dict:
    """Build KGs for all rule-based methods. Returns {method_key: result_dict}."""
    names = methods or RULE_BUILDERS
    results = {}
    for name in names:
        key = METHOD_ALIAS[name]
        t0 = time.time()
        try:
            mod = importlib.import_module(name)
            path = mod.build_user(uid)
            results[key] = {
                "ok": True,
                "path": path,
                "seconds": round(time.time() - t0, 1),
            }
            print(f"  [OK]   {key:20s} ({results[key]['seconds']:.1f}s)")
        except Exception as exc:
            results[key] = {"ok": False, "error": str(exc)[:200]}
            print(f"  [FAIL] {key:20s}  {exc}")
            traceback.print_exc()
    return results


# ═══════════════════════════════════════════════════════════════════
# Phase 2: Validate
# ═══════════════════════════════════════════════════════════════════

def validate_one(uid: str, method_key: str) -> dict | None:
    """Structural sanity checks on one memory bank."""
    bank_path = OUT / f"weibo_kg_{method_key}_{uid}" / "memory_bank.json"
    if not bank_path.exists():
        return {"ok": False, "error": "memory_bank.json not found"}

    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    issues = []

    # Required top-level keys
    for k in ("method", "paper_ref", "static_map", "event_maps",
              "retrieval_index", "stats", "method_extras"):
        if k not in bank:
            issues.append(f"missing key: {k}")

    ems = bank.get("event_maps") or []
    if len(ems) == 0:
        issues.append("zero event_maps")

    empty_3d = sum(1 for m in ems if not m.get("feature_3d_triples"))
    total_triples = sum(len(m.get("feature_3d_triples") or []) for m in ems)
    avg_t = total_triples / max(len(ems), 1)

    ri = bank.get("retrieval_index") or {}
    vecs = ri.get("vectors") or []
    if len(vecs) != len(ems):
        issues.append(f"vector/map mismatch: {len(vecs)} vs {len(ems)}")

    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "num_maps": len(ems),
        "empty_3d": empty_3d,
        "avg_triples": round(avg_t, 1),
        "retriever": (bank.get("method_extras") or {}).get("retriever", "default"),
    }


def validate_all(uid: str, methods: list[str] | None = None) -> dict:
    """Validate all built banks."""
    method_keys = [METHOD_ALIAS[m] for m in (methods or RULE_BUILDERS)]
    results = {}
    for mk in method_keys:
        v = validate_one(uid, mk)
        if v:
            results[mk] = v
            status = "OK" if v["ok"] else f"ISSUES: {v['issues']}"
            print(f"  [{status}] {mk:20s} maps={v['num_maps']} "
                  f"empty3d={v['empty_3d']} avg_t={v['avg_triples']} "
                  f"retriever={v['retriever']}")
        else:
            results[mk] = {"ok": False, "error": "validation failed"}
            print(f"  [FAIL] {mk}")
    return results


# ═══════════════════════════════════════════════════════════════════
# Phase 3: Benchmark
# ═══════════════════════════════════════════════════════════════════

def run_benchmark_for_user(uid: str, max_samples: int = 0,
                           predict_conc: int = 4) -> dict:
    """Run the paper-KG predict+judge benchmark using UserAgent's infrastructure."""
    sys.path.insert(0, str(HERE.parent))  # scripts/
    from run_paper_kg_benchmark import run_method

    # Discover available banks
    available = []
    for d in sorted(OUT.iterdir()):
        if not d.is_dir() or not d.name.startswith("weibo_kg_"):
            continue
        if not d.name.endswith(f"_{uid}"):
            continue
        mb = d / "memory_bank.json"
        if mb.exists():
            method_key = d.name.replace("weibo_kg_", "").replace(f"_{uid}", "")
            available.append(method_key)

    if not available:
        print("  No memory banks found!")
        return {}

    test_path = OUT / f"weibo_user_{uid}" / "test.jsonl"
    samples = []
    with open(test_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    total_samples = len(samples)
    limit = min(max_samples, total_samples) if max_samples and max_samples > 0 else 0

    print(f"  Methods: {len(available)}  Samples: {total_samples}  Limit: {limit or 'all'}")

    summary = {}
    for method_key in sorted(available):
        t0 = time.time()
        try:
            metrics = run_method(
                uid, method_key,
                limit=limit,
                top_k=5,
                predict_conc=predict_conc,
                resume=True,
            )
            score = (metrics.get("benchmark") or {}).get("opinion_alignment_score", "N/A")
            summary[method_key] = metrics.get("benchmark") or {}
            print(f"  [{method_key:20s}] OA={score}  "
                  f"n={metrics.get('n','?')}  "
                  f"({time.time()-t0:.0f}s)")
        except Exception as exc:
            print(f"  [FAIL] {method_key}: {exc}")
            traceback.print_exc()
            summary[method_key] = {"error": str(exc)[:200]}

    return summary


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True, help="Weibo user ID")
    ap.add_argument("--methods", default="all",
                    help="Comma-separated builder module names, or 'all'")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--benchmark-only", action="store_true")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--max-samples", type=int, default=0,
                    help="Max test samples (0=all)")
    ap.add_argument("--predict-conc", type=int, default=4)
    args = ap.parse_args()

    methods = None if args.methods == "all" else args.methods.split(",")
    uid = args.user
    do_build = not args.benchmark_only and not args.validate_only
    do_validate = not args.build_only and not args.benchmark_only
    do_benchmark = not args.build_only and not args.validate_only

    t_total = time.time()
    report = {"user": uid, "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}

    # Phase 1: Build
    if do_build:
        print(f"\n{'='*60}\nPhase 1: BUILD ({uid})\n{'='*60}")
        report["build"] = build_all(uid, methods)
        n_ok = sum(1 for r in report["build"].values() if r.get("ok"))
        print(f"Build: {n_ok}/{len(report['build'])} succeeded")

    # Phase 2: Validate
    if do_validate:
        print(f"\n{'='*60}\nPhase 2: VALIDATE ({uid})\n{'='*60}")
        report["validate"] = validate_all(uid, methods)
        n_ok = sum(1 for r in report["validate"].values() if r.get("ok"))
        print(f"Validate: {n_ok}/{len(report['validate'])} clean")

    # Phase 3: Benchmark
    if do_benchmark:
        print(f"\n{'='*60}\nPhase 3: BENCHMARK ({uid})\n{'='*60}")
        report["benchmark"] = run_benchmark_for_user(
            uid,
            max_samples=args.max_samples,
            predict_conc=args.predict_conc,
        )

    # Final report
    report["elapsed_total"] = round(time.time() - t_total, 1)
    report_path = OUT / f"paper_kg_full_experiment_{uid}.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{'='*60}")
    print(f"Full experiment done ({report['elapsed_total']:.0f}s)")
    print(f"Report -> {report_path}")

    # Print scoreboard
    bench = report.get("benchmark") or {}
    if bench:
        print("\nScoreboard (opinion_alignment_score):")
        for mk, bm in sorted(bench.items(),
                             key=lambda x: (x[1].get("opinion_alignment_score") or 0),
                             reverse=True):
            score = bm.get("opinion_alignment_score", "N/A")
            if isinstance(score, (int, float)):
                print(f"  {mk:25s}  {score:.4f}")
            else:
                print(f"  {mk:25s}  {score}")


if __name__ == "__main__":
    main()
