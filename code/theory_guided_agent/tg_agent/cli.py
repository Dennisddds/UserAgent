from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

# allow `python -m tg_agent.cli` from package root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tg_agent.agent import CUVAgent
from tg_agent.genminds import GenMindsMemory
from tg_agent.llm import DeepSeekClient, load_env
from tg_agent.theory_lib import TheoryLibrary


def _load_cfg(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve(cfg: dict, key: str) -> Path:
    p = Path(cfg["paths"][key])
    if not p.is_absolute():
        p = ROOT / p
    return p


def build_llm(cfg: dict) -> DeepSeekClient:
    """Build OpenAI-compatible client from config + env (local vLLM or cloud)."""
    load_env(cfg["paths"]["env_file"])
    llm_cfg = cfg.get("llm") or {}
    # Prefer explicit env (local serve switch) over config defaults.
    model = (
        os.environ.get("LLM_MODEL")
        or llm_cfg.get("model")
        or "DeepSeek-V4-Flash"
    )
    base_url = (
        os.environ.get("LLM_BASE_URL")
        or llm_cfg.get("base_url")
        or "http://127.0.0.1:8001/v1"
    )
    api_key = (
        os.environ.get("LLM_API_KEY")
        or llm_cfg.get("api_key")
        or "EMPTY"
    )
    env_think = os.environ.get("LLM_ENABLE_THINKING", "").strip().lower()
    if env_think in {"1", "true", "yes", "on"}:
        enable_thinking = True
    elif env_think in {"0", "false", "no", "off"}:
        enable_thinking = False
    else:
        enable_thinking = bool(llm_cfg.get("enable_thinking", True))
    return DeepSeekClient(
        api_key=api_key,
        base_url=base_url,
        model=str(model),
        enable_thinking=enable_thinking,
        reasoning_effort=llm_cfg.get("reasoning_effort")
        or os.environ.get("LLM_REASONING_EFFORT")
        or "high",
    )


def build_agent(cfg: dict, user_id: str) -> CUVAgent:
    load_env(cfg["paths"]["env_file"])
    mem = GenMindsMemory(
        cfg["paths"]["genminds"][user_id],
        persona_path=cfg["paths"]["persona"].get(user_id),
    )
    theories = TheoryLibrary(
        _resolve(cfg, "theory_seed"),
        _resolve(cfg, "theory_library"),
    )
    llm = build_llm(cfg)
    return CUVAgent(
        user_id,
        mem,
        theories,
        llm,
        state_dir=_resolve(cfg, "user_state"),
        top_k_events=cfg["retrieval"]["top_k_events"],
        top_k_theories=cfg["retrieval"]["top_k_theories"],
        max_motifs=cfg["retrieval"]["max_motifs"],
        evolve_lr=cfg["loop"]["evolve_lr"],
    )


def cmd_bootstrap_theory(cfg: dict, args: argparse.Namespace) -> None:
    load_env(cfg["paths"]["env_file"])
    lib = TheoryLibrary(_resolve(cfg, "theory_seed"), _resolve(cfg, "theory_library"))
    crawl = cfg["theory_crawl"]
    per = args.per_query or crawl["per_query"]
    pages = args.pages or crawl.get("pages", 1)
    mailto = crawl.get("mailto", "")
    domains = crawl.get("domains_file")
    if domains and not args.legacy_queries:
        domains_path = Path(domains)
        if not domains_path.is_absolute():
            domains_path = ROOT / domains_path
        n1 = lib.crawl_from_domains(
            domains_path, per_query=per, pages=pages, mailto=mailto, source=args.source
        )
        queries = crawl.get("seed_queries") or []
    else:
        queries = crawl["seed_queries"]
        n1 = lib.crawl_openalex(queries, per_query=per, pages=pages, mailto=mailto)
    n2 = 0
    if args.with_serper:
        n2 = lib.crawl_serper(queries or crawl.get("seed_queries") or [], per_query=min(5, per))
    print(
        json.dumps(
            {
                "source": getattr(args, "source", "openalex"),
                "added": n1,
                "serper_added": n2,
                "total_cards": len(lib.cards),
                "completed_queries": len(lib.completed_queries),
                "by_coordinate": json.loads(
                    (_resolve(cfg, "theory_library") / "meta.json").read_text(encoding="utf-8")
                ).get("by_coordinate"),
                "by_source": json.loads(
                    (_resolve(cfg, "theory_library") / "meta.json").read_text(encoding="utf-8")
                ).get("by_source"),
                "library": str(_resolve(cfg, "theory_library")),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_smoke(cfg: dict, args: argparse.Namespace) -> None:
    stimulus = args.stimulus
    results = {}
    uids = list(cfg["paths"]["genminds"].keys())
    if getattr(args, "user", None):
        uids = [args.user]
    for uid in uids:
        agent = build_agent(cfg, uid)
        out = agent.predict(stimulus)
        results[uid] = out.to_dict()
        out_path = _resolve(cfg, "user_state") / f"{uid}_smoke.json"
        out_path.write_text(json.dumps(out.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n=== user {uid} ===")
        print("stance:", out.stance)
        print("opinion:", out.predicted_opinion[:400])
        print("coords:", out.activated_coordinates)
        print("theories:", [t["name"][:60] for t in out.matched_theories])
        wb = (out.c_trace or {}).get("white_box") or {}
        reasoning = (out.c_trace or {}).get("model_reasoning") or ""
        print("white_box.has_reasoning:", wb.get("has_reasoning"), "chars:", wb.get("reasoning_chars"))
        if reasoning:
            print("--- RTWI CoT (head) ---")
            print(reasoning[:600])
            print("--- RTWI CoT (tail) ---")
            print(reasoning[-600:])
        print("saved:", out_path)


def cmd_enrich_theory(cfg: dict, args: argparse.Namespace) -> None:
    """Distill structured theory fields into thin crawled cards (retrieval support)."""
    from tg_agent.enrich_theories import main as enrich_main

    argv = [
        "--config",
        str(cfg.get("_config_path") or ROOT / "config.yaml"),
        "--limit",
        str(args.limit),
        "--min-citations",
        str(args.min_citations),
    ]
    if args.no_distill:
        argv.append("--no-distill")
    if args.no_fetch_abstracts:
        argv.append("--no-fetch-abstracts")
    if args.all_thin:
        argv.append("--all-thin")
    if getattr(args, "reset_ungrounded", False):
        argv.append("--reset-ungrounded")
    enrich_main(argv)


def cmd_build_env(cfg: dict, args: argparse.Namespace) -> None:
    """Build communication/psych/social environment profile for a user."""
    from tg_agent.user_env import build_user_environment, save_env

    uid = args.user
    mem = GenMindsMemory(
        cfg["paths"]["genminds"][uid],
        persona_path=cfg["paths"]["persona"].get(uid),
    )
    csv_path = None
    csvs = (cfg.get("paths") or {}).get("user_csv") or {}
    if uid in csvs:
        csv_path = Path(csvs[uid])
    elif Path(cfg["paths"]["root"]) / f"{uid}.csv":
        cand = Path(cfg["paths"]["root"]) / f"{uid}.csv"
        if cand.exists():
            csv_path = cand
    profile = build_user_environment(
        uid,
        persona=mem.persona,
        memory_static=mem.static,
        csv_path=csv_path,
    )
    out = save_env(_resolve(cfg, "user_state"), profile)
    print(
        json.dumps(
            {
                "saved": str(out),
                "top_coords": list(profile["coordinate_weights"].items())[:12],
                "comm_coords": list(profile["communication"]["coordinate_weights"].items())[:6],
                "psych_coords": list(profile["psychological"]["coordinate_weights"].items())[:6],
                "social_coords": list(profile["social"]["coordinate_weights"].items())[:6],
                "top_topics": profile["social"].get("top_topics", [])[:8],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_build_situational_env(cfg: dict, args: argparse.Namespace) -> None:
    """Recover external 3D environment at posting time (not user-trait extraction)."""
    from tg_agent.llm import DeepSeekClient, load_env as _load_env
    from tg_agent.situational_env import build_situational_envs

    uid = args.user
    _load_env(cfg["paths"]["env_file"])
    csvs = (cfg.get("paths") or {}).get("user_csv") or {}
    csv_path = Path(csvs.get(uid) or (Path(cfg["paths"]["root"]) / f"{uid}.csv"))
    if not csv_path.exists():
        raise SystemExit(f"csv not found: {csv_path}")
    out = _resolve(cfg, "user_state") / f"{uid}_situational_env{getattr(args, 'sit_suffix', '_weibo_ai')}.json"
    llm = DeepSeekClient(model=cfg["llm"]["model"])
    retrieval = getattr(args, "retrieval", "weibo_ai")
    if retrieval in {"weibo_ai", "both"} and not os.environ.get("WEIBO_COOKIE", "").strip():
        raise SystemExit("WEIBO_COOKIE missing — set logged-in cookie before 智搜 retrieval")
    priority: set[str] = set()
    if getattr(args, "priority_eval", False):
        pred = Path(cfg["paths"]["root"]) / "outputs" / f"weibo_kg_genminds_{uid}" / "predictions.jsonl"
        if pred.exists():
            for line in pred.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                d = json.loads(line)
                if d.get("post_id"):
                    priority.add(str(d["post_id"]))
            print(f"loaded priority_eval ids={len(priority)} from {pred}", flush=True)
    payload = build_situational_envs(
        user_id=uid,
        csv_path=csv_path,
        env_file=cfg["paths"]["env_file"],
        out_path=out,
        llm=llm,
        limit=args.limit,
        priority_post_ids=priority or None,
        dedupe_slots=not getattr(args, "no_dedupe", False),
        retrieval=retrieval,
    )
    print(
        json.dumps(
            {
                "saved": str(out),
                "num_posts": payload.get("num_posts"),
                "num_slots": payload.get("num_slots"),
                "sample": [
                    {
                        "date": r.get("date"),
                        "topic": (r.get("topic") or "")[:40],
                        "summary": r.get("summary"),
                        "coords": r.get("theory_coordinates"),
                    }
                    for r in (payload.get("records") or [])[:3]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_loop(cfg: dict, args: argparse.Namespace) -> None:
    agent = build_agent(cfg, args.user)
    outs = agent.loop(
        args.stimulus,
        max_iterations=args.iters or cfg["loop"]["max_iterations"],
        auto_feedback=args.feedback,
    )
    path = _resolve(cfg, "user_state") / f"{args.user}_loop.json"
    path.write_text(
        json.dumps([o.to_dict() for o in outs], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"iterations": len(outs), "saved": str(path)}, ensure_ascii=False, indent=2))
    print(outs[-1].predicted_opinion[:500])
    print("--- verbalization ---")
    print(outs[-1].verbalization[:800])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="tg_agent")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("bootstrap-theory")
    p1.add_argument("--per-query", type=int, default=0, help="OpenAlex results per page (max 200)")
    p1.add_argument("--pages", type=int, default=0, help="Cursor pages per query")
    p1.add_argument(
        "--legacy-queries",
        action="store_true",
        help="Use config seed_queries instead of theory_domains.yaml",
    )
    p1.add_argument(
        "--source",
        default="crossref",
        choices=["openalex", "crossref", "s2"],
        help="Paper API backend (default: crossref; OpenAlex is often rate-limited)",
    )
    p1.add_argument(
        "--with-serper",
        action="store_true",
        help="Also crawl Serper web snippets (optional, lower quality)",
    )

    p2 = sub.add_parser("smoke")
    p2.add_argument(
        "--stimulus",
        default="外媒称中国经济即将崩溃并配了夸张图表。请预测该用户会如何回应，并解释机制。",
    )
    p2.add_argument("--user", default="", help="optional single user id; default=all")

    p3 = sub.add_parser("loop")
    p3.add_argument("--user", required=True)
    p3.add_argument("--stimulus", required=True)
    p3.add_argument("--iters", type=int, default=0)
    p3.add_argument("--feedback", default="解释方向对，但要更贴合该用户历史表达风格。")

    p4 = sub.add_parser(
        "enrich-theory",
        help="Fetch paper abstracts + distill ONLY from abstract text (grounded)",
    )
    p4.add_argument("--limit", type=int, default=40)
    p4.add_argument("--min-citations", type=int, default=80)
    p4.add_argument("--no-distill", action="store_true")
    p4.add_argument("--no-fetch-abstracts", action="store_true")
    p4.add_argument(
        "--all-thin",
        action="store_true",
        help="skip social-science relevance filter",
    )
    p4.add_argument(
        "--reset-ungrounded",
        action="store_true",
        help="clear distilled fields not backed by a paper abstract",
    )

    p5 = sub.add_parser(
        "build-env",
        help="(legacy) user-trait coordinate prior from persona — NOT situational context",
    )
    p5.add_argument("--user", required=True)

    p6 = sub.add_parser(
        "build-situational-env",
        help="Recover external communication/psych/social context at each post's time via web search",
    )
    p6.add_argument("--user", required=True)
    p6.add_argument("--limit", type=int, default=0, help="max authored posts (0=all)")
    p6.add_argument(
        "--priority-eval",
        action="store_true",
        help="Process prediction-set post_ids first (recommended for large user)",
    )
    p6.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Disable day+topic slot reuse",
    )
    p6.add_argument(
        "--retrieval",
        default="weibo_ai",
        choices=["weibo_ai", "serper", "both"],
        help="presentation evidence backend (default: weibo_ai 智搜)",
    )
    p6.add_argument(
        "--sit-suffix",
        default="_weibo_ai",
        help="output file suffix: {uid}_situational_env{suffix}.json",
    )

    args = parser.parse_args(argv)
    cfg_path = Path(args.config)
    cfg = _load_cfg(cfg_path)
    cfg["_config_path"] = str(cfg_path)
    if args.cmd == "bootstrap-theory":
        cmd_bootstrap_theory(cfg, args)
    elif args.cmd == "smoke":
        cmd_smoke(cfg, args)
    elif args.cmd == "loop":
        cmd_loop(cfg, args)
    elif args.cmd == "enrich-theory":
        cmd_enrich_theory(cfg, args)
    elif args.cmd == "build-env":
        cmd_build_env(cfg, args)
    elif args.cmd == "build-situational-env":
        cmd_build_situational_env(cfg, args)
    else:
        raise SystemExit(f"unknown cmd {args.cmd}")


if __name__ == "__main__":
    main()
