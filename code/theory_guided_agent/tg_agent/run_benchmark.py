from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tg_agent.agent import CUVAgent
from tg_agent.benchmark_core import (
    LLMConfig,
    OpenAICompatClient,
    aggregate_metrics,
    build_judge_user,
    extract_context_and_gt,
    JUDGE_SYSTEM,
    map_parallel,
    parse_judge,
)
from tg_agent.genminds import GenMindsMemory
from tg_agent.llm import DeepSeekClient, load_env
from tg_agent.memory_layers import detect_emotions, emotion_alignment
from tg_agent.path_agent import PathAgent
from tg_agent.theory_lib import TheoryLibrary
from tg_agent.user_actions import build_source_profile


def _load_jsonl(path: Path, limit: int = 0) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        pid = str(d.get("post_id") or d.get("index"))
        if d.get("prediction") and d.get("judge_scores"):
            done.add(pid)
    return done


def stratified_subset(
    samples: list[dict],
    *,
    ratio: float,
    seed: int = 42,
    gap_hours: float = 1.0,
    large_min: int = 5,
) -> list[dict]:
    """按话题时间窗分层抽样：大窗/小窗按比例保留，控制评测代价（重点保小窗）。"""
    from tg_agent.temporal_windows import build_topic_windows, stratified_sample_windows

    prepared = []
    for i, s in enumerate(samples):
        meta = s.get("metadata") if isinstance(s.get("metadata"), dict) else {}
        prepared.append(
            {
                "topic": str(s.get("topic") or meta.get("topic") or "untopic")[:80],
                "date": str(s.get("timestamp") or meta.get("date") or ""),
                "post_id": str(s.get("post_id") or f"idx{i}"),
            }
        )
    wins = build_topic_windows(prepared, gap_hours=gap_hours)
    keep = stratified_sample_windows(wins, ratio=ratio, seed=seed, large_min=large_min)
    keep_ids = {pid for w in keep for pid in w.post_ids}
    return [
        s for i, s in enumerate(samples) if str(s.get("post_id") or f"idx{i}") in keep_ids
    ]


def genminds_predict(memory: GenMindsMemory, llm: DeepSeekClient, sample: dict, top_k: int = 5) -> tuple[str, list, list]:
    context, _ = extract_context_and_gt(sample)
    retrieved = memory.retrieve(context, top_k=top_k)
    evidence = "\n".join(
        f"- {e.event_title}: {e.user_opinion or e.text[:160]}" for e in retrieved
    )
    u = memory.u_snapshot(max_motifs=6)
    system = (
        "你正在扮演下方【身份/人设】指定的微博账号本人发短评。"
        "严格贴合其人设表达特征、历史立场与风格；你就是该账号本人，不是旁观者。"
        "只输出一条简短原创微博评论正文，不要解释，不要前缀标签。"
    )
    user = f"""【身份/人设】
{memory.identity_block()}

用户历史信念样例：{u.get('beliefs', [])[:6]}
表达风格：{u.get('communication', [])[:6]}
相关历史：
{evidence}

事件：
{context}

请以该账号本人声口发表一条简短原创微博评论："""
    pred = llm.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.4,
        max_tokens=400,
        disable_thinking=True,
    ).strip()
    # strip accidental quotes/labels
    if pred.startswith("WEIBO:"):
        pred = pred[6:].strip()
    return pred, [e.map_id for e in retrieved], [e.score for e in retrieved]


def cuv_predict(agent: CUVAgent, sample: dict) -> tuple[str, dict]:
    context, _ = extract_context_and_gt(sample)
    meta = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    out = agent.predict(
        context,
        post_id=str(sample.get("post_id") or ""),
        bid=str(meta.get("bid") or ""),
        date=str(sample.get("timestamp") or meta.get("date") or ""),
        topic=str(sample.get("topic") or meta.get("topic") or ""),
    )
    return out.predicted_opinion.strip(), out.to_dict()


def judge_one(judge: OpenAICompatClient, context: str, gt: str, pred: str) -> dict:
    if not pred.strip():
        return {
            "stance": 0.0,
            "core_judgment": 0.0,
            "belief": 0.0,
            "value": 0.0,
            "opinion_alignment_score": 0.0,
        }
    raw = judge.chat(
        [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": build_judge_user(context, gt, pred)},
        ],
        temperature=0.0,
        max_tokens=300,
    )
    return parse_judge(raw)


def run_method(
    *,
    method: str,
    user_id: str,
    samples: list[dict],
    out_dir: Path,
    predict_fn,
    judge: OpenAICompatClient,
    predict_workers: int,
    judge_workers: int,
    resume: bool,
    emotion_eval: bool = False,
    scoring: dict | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    done = _load_done_ids(pred_path) if resume else set()
    if not resume and pred_path.exists():
        pred_path.unlink()

    pending = []
    for i, s in enumerate(samples):
        pid = str(s.get("post_id") or i)
        if pid in done:
            continue
        pending.append((i, s))

    print(f"[{method}/{user_id}] pending={len(pending)} done={len(done)}", flush=True)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _pred_task(item):
        i, s = item
        context, gt = extract_context_and_gt(s)
        pid = str(s.get("post_id") or i)
        try:
            if method == "GenMinds":
                pred, map_ids, scores = predict_fn(s)
                extra = {
                    "retrieved_map_ids": map_ids,
                    "retrieved_scores": scores,
                    "agent_trace": None,
                }
            else:
                pred, trace = predict_fn(s)
                extra = {
                    "retrieved_map_ids": [e["map_id"] for e in trace.get("evidence_events", [])],
                    "retrieved_scores": [e["score"] for e in trace.get("evidence_events", [])],
                    "stance": trace.get("stance"),
                    "agent_trace": {
                        "matched_theories": trace.get("matched_theories"),
                        "verbalization": trace.get("verbalization"),
                        "activated_coordinates": trace.get("activated_coordinates"),
                        "c_trace": trace.get("c_trace"),
                        "caveats": trace.get("caveats"),
                        # CUV-Path 扩展字段（旧方法为 None，不影响）
                        "factors": trace.get("factors"),
                        "paths": trace.get("paths"),
                        "emotion_probs": trace.get("emotion_probs"),
                        "confidence": trace.get("confidence"),
                        "low_evidence": trace.get("low_evidence"),
                        "skeptic": trace.get("skeptic"),
                    },
                }
        except Exception as e:  # noqa: BLE001
            pred, extra = "", {
                "error": str(e),
                "retrieved_map_ids": [],
                "retrieved_scores": [],
            }
        return {
            "index": i,
            "post_id": pid,
            "user_id": user_id,
            "topic": s.get("topic"),
            "ground_truth": gt,
            "context": context,
            "method": method,
            "prediction": pred,
            **extra,
        }

    predicted_rows = []
    if pending:
        with ThreadPoolExecutor(max_workers=predict_workers) as ex:
            futs = {ex.submit(_pred_task, p): p[0] for p in pending}
            done_n = 0
            for fut in as_completed(futs):
                row = fut.result()
                predicted_rows.append(row)
                done_n += 1
                if done_n % 10 == 0 or done_n == len(pending):
                    print(f"[{method}/{user_id} predict] {done_n}/{len(pending)}", flush=True)

    def jwork(row):
        try:
            scores = judge_one(
                judge, row["context"], row["ground_truth"], row.get("prediction") or ""
            )
        except Exception as e:  # noqa: BLE001
            scores = {
                "stance": 0.0,
                "core_judgment": 0.0,
                "belief": 0.0,
                "value": 0.0,
                "opinion_alignment_score": 0.0,
                "error": str(e),
            }
        row = dict(row)
        if emotion_eval and not scores.get("error"):
            tr = row.get("agent_trace") or {}
            pred_probs = tr.get("emotion_probs") or {}
            try:
                gt_probs = detect_emotions(judge, row.get("ground_truth") or "")
                row["gt_emotion_probs"] = gt_probs
                if pred_probs:
                    scores["emotion_alignment"] = emotion_alignment(pred_probs, gt_probs)
            except Exception:  # noqa: BLE001
                pass
        row["judge_scores"] = scores
        return row

    if predicted_rows:
        with ThreadPoolExecutor(max_workers=judge_workers) as ex:
            futs = {ex.submit(jwork, r): r["post_id"] for r in predicted_rows}
            done_n = 0
            for fut in as_completed(futs):
                row = fut.result()
                _append_jsonl(pred_path, row)
                done_n += 1
                if done_n % 10 == 0 or done_n == len(predicted_rows):
                    print(f"[{method}/{user_id} judge] {done_n}/{len(predicted_rows)}", flush=True)

    # aggregate all rows on disk
    all_rows = []
    if pred_path.exists():
        for line in pred_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                all_rows.append(json.loads(line))
    agg = aggregate_metrics(
        all_rows,
        weights=(scoring or {}).get("result_weights") or None,
        reason_weight=float((scoring or {}).get("reason_weight") or 0.0),
    )
    metrics = {
        "user_id": user_id,
        "method": method,
        "predict_model": "deepseek/deepseek-v4-pro",
        "judge_model": "dashscope/qwen3.7-plus",
        "benchmark_type": "opinion_alignment",
        "num_samples": len(all_rows),
        "benchmark": {
            "stance": agg.get("stance", 0.0),
            "core_judgment": agg.get("core_judgment", 0.0),
            "belief": agg.get("belief", 0.0),
            "value": agg.get("value", 0.0),
            "opinion_alignment_score": agg.get("opinion_alignment_score", 0.0),
        },
        "coverage": f"{agg.get('n', 0)}/{len(samples)}",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    for extra_k in ("weighted_result", "reason_correctness", "composite"):
        if extra_k in agg:
            metrics["benchmark"][extra_k] = agg[extra_k]
    emo_rows = [
        r for r in all_rows
        if (r.get("judge_scores") or {}).get("emotion_alignment") is not None
    ]
    if emo_rows:
        metrics["benchmark"]["emotion_alignment"] = round(
            sum(float(r["judge_scores"]["emotion_alignment"]) for r in emo_rows)
            / len(emo_rows),
            4,
        )
    conf_rows = [
        r for r in all_rows
        if ((r.get("agent_trace") or {}).get("confidence") or 0) > 0
    ]
    if conf_rows:
        metrics["mean_confidence"] = round(
            sum(float((r.get("agent_trace") or {}).get("confidence") or 0) for r in conf_rows)
            / len(conf_rows),
            4,
        )
        metrics["low_evidence_rate"] = round(
            sum(1 for r in conf_rows if (r.get("agent_trace") or {}).get("low_evidence"))
            / len(conf_rows),
            4,
        )
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--user", action="append", default=[])
    ap.add_argument("--methods", default="GenMinds,CUV-TG")
    ap.add_argument("--limit", type=int, default=0, help="0=all")
    ap.add_argument("--predict-workers", type=int, default=4)
    ap.add_argument("--judge-workers", type=int, default=6)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--test-path",
        default="",
        help="override test.jsonl path (single --user only); default outputs/weibo_user_{uid}/test.jsonl",
    )
    ap.add_argument(
        "--out-root",
        default=str(Path("d:/UserAgent/outputs/benchmark_cuv_tg_situational_pilot")),
    )
    ap.add_argument(
        "--weak-match-threshold",
        type=float,
        default=0.40,
        help="CUV-TG: below this top theory score → GenMinds fallback",
    )
    ap.add_argument(
        "--no-situational",
        action="store_true",
        help="CUV-TG: disable situational env gate (reproduce pre-gate static protocol)",
    )
    ap.add_argument(
        "--emotion-eval",
        action="store_true",
        help="detect ground-truth emotions and score emotion_alignment (needs emotion_probs in trace)",
    )
    ap.add_argument(
        "--no-skeptic",
        action="store_true",
        help="CUV-Path: disable the skeptic (反证质疑) step",
    )
    ap.add_argument(
        "--stratify-ratio",
        type=float,
        default=0.0,
        help=">0 时按话题时间窗分层抽样（0 = 用 config sampling.stratify_ratio 或全量）",
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    load_env(cfg["paths"]["env_file"])
    # also allow process env overrides
    qwen_key = os.environ.get("QWEN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY", "")
    if not qwen_key:
        raise SystemExit("QWEN_API_KEY / DASHSCOPE_API_KEY missing")

    deepseek = DeepSeekClient(model=cfg["llm"]["model"])
    # China DashScope compatible endpoint (sk- keys typically bind here)
    judge = OpenAICompatClient(
        LLMConfig(
            api_key=qwen_key,
            base_url=os.environ.get(
                "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
            ),
            model=os.environ.get("QWEN_MODEL", "qwen3.7-plus"),
            disable_thinking=True,
        )
    )

    users = args.user or list(cfg["paths"]["genminds"].keys())
    if args.test_path and len(users) != 1:
        raise SystemExit("--test-path requires exactly one --user")
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    out_root = Path(args.out_root)
    summary = []
    scoring = cfg.get("scoring") or {}
    path_tuning = dict(cfg.get("path_agent") or {})
    path_tuning["fast_path"] = cfg.get("fast_path") or {}
    fm_budget = cfg.get("failure_memory") or {}
    samp_cfg = cfg.get("sampling") or {}
    stratify_ratio = args.stratify_ratio or float(samp_cfg.get("stratify_ratio") or 0.0)

    for uid in users:
        test_path = (
            Path(args.test_path)
            if args.test_path
            else Path(f"d:/UserAgent/outputs/weibo_user_{uid}/test.jsonl")
        )
        samples = _load_jsonl(test_path, limit=args.limit)
        if stratify_ratio > 0:
            before = len(samples)
            samples = stratified_subset(
                samples,
                ratio=stratify_ratio,
                seed=int(samp_cfg.get("seed") or 42),
                gap_hours=float((cfg.get("temporal_windows") or {}).get("gap_hours") or 1.0),
            )
            print(f"[{uid}] stratified: {before} -> {len(samples)} samples", flush=True)
        memory = GenMindsMemory(
            cfg["paths"]["genminds"][uid],
            persona_path=cfg["paths"]["persona"].get(uid),
        )
        theories = TheoryLibrary(
            ROOT / cfg["paths"]["theory_seed"]
            if not Path(cfg["paths"]["theory_seed"]).is_absolute()
            else Path(cfg["paths"]["theory_seed"]),
            ROOT / cfg["paths"]["theory_library"]
            if not Path(cfg["paths"]["theory_library"]).is_absolute()
            else Path(cfg["paths"]["theory_library"]),
        )
        agent = CUVAgent(
            uid,
            memory,
            theories,
            deepseek,
            state_dir=ROOT / "data" / "users",
            top_k_events=cfg["retrieval"]["top_k_events"],
            top_k_theories=cfg["retrieval"]["top_k_theories"],
            max_motifs=cfg["retrieval"]["max_motifs"],
            use_situational=not args.no_situational,
            weak_match_threshold=args.weak_match_threshold,
        )
        path_agent: PathAgent | None = None
        if any(m in {"CUV-Path", "path"} for m in methods):
            csv_path = (cfg["paths"].get("user_csv") or {}).get(uid)
            source_profile = (
                build_source_profile(csv_path) if csv_path else {"available": False}
            )
            path_agent = PathAgent(
                uid,
                memory,
                theories,
                deepseek,
                state_dir=ROOT / "data" / "users",
                top_k_events=cfg["retrieval"]["top_k_events"],
                top_k_theories=cfg["retrieval"]["top_k_theories"],
                max_motifs=cfg["retrieval"]["max_motifs"],
                use_situational=not args.no_situational,
                weak_match_threshold=args.weak_match_threshold,
                use_skeptic=not args.no_skeptic,
                source_profile=source_profile,
                tuning=path_tuning,
                failure_memory_budget=fm_budget,
            )

        for method in methods:
            if method == "GenMinds":

                def pfn(s, mem=memory, llm=deepseek):
                    return genminds_predict(mem, llm, s, top_k=5)

            elif method in {"CUV-TG", "TheoryGuided-CUV", "tg_cuv"}:

                def pfn(s, ag=agent):
                    return cuv_predict(ag, s)

            elif method in {"CUV-Path", "path"}:
                assert path_agent is not None

                def pfn(s, ag=path_agent):
                    return cuv_predict(ag, s)

            else:
                raise SystemExit(f"unknown method {method}")

            metrics = run_method(
                method=method,
                user_id=uid,
                samples=samples,
                out_dir=out_root / f"{method.replace('/', '_')}_{uid}",
                predict_fn=pfn,
                judge=judge,
                predict_workers=args.predict_workers,
                judge_workers=args.judge_workers,
                resume=args.resume,
                emotion_eval=args.emotion_eval,
                scoring=scoring,
            )
            summary.append(metrics)

    report = out_root / "comparison_report.md"
    has_weighted = any("weighted_result" in (m.get("benchmark") or {}) for m in summary)
    has_composite = any("composite" in (m.get("benchmark") or {}) for m in summary)
    extra = (["weighted"] if has_weighted else []) + (["composite"] if has_composite else [])
    lines = [
        "# CUV-TG vs GenMinds Benchmark",
        "",
        f"- generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "- predict: DeepSeek deepseek-v4-pro",
        "- judge: DashScope qwen3.7-plus (thinking off)",
        "- metric: opinion_alignment = mean(stance, core_judgment, belief, value)"
        + ("; weighted_result/composite 见 config scoring" if extra else ""),
        "",
        "| user | method | opinion |"
        + "".join(f" {c} |" for c in extra)
        + " stance | core | belief | value | coverage |",
        "|---|---|---:|" + "---:|" * len(extra) + "---:|---:|---:|---:|---|",
    ]
    for m in summary:
        b = m["benchmark"]
        row = (
            f"| {m['user_id']} | {m['method']} | {b['opinion_alignment_score']:.4f} | "
        )
        if has_weighted:
            row += f"{b.get('weighted_result', 0):.4f} | "
        if has_composite:
            row += f"{b.get('composite', 0):.4f} | "
        row += (
            f"{b['stance']:.4f} | {b['core_judgment']:.4f} | {b['belief']:.4f} | "
            f"{b['value']:.4f} | {m['coverage']} |"
        )
        lines.append(row)
    # also include published GenMinds reference
    lines += [
        "",
        "## Reference (prior paper_kg run, same protocol)",
        "",
        "| user | method | opinion | note |",
        "|---|---|---:|---|",
        "| 1989660417 | GenMinds (published) | 0.5982 | outputs/weibo_kg_genminds_1989660417 |",
        "| 7463374646 | GenMinds (published) | 0.2850 | n=10 sparse |",
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")
    (out_root / "comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {report}", flush=True)


if __name__ == "__main__":
    main()
