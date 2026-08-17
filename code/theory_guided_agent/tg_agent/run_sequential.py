from __future__ import annotations

"""Sequential temporal alignment experiment.

Protocol (same for small/large users; replaces train/test split route):
  for each post t in timestamp order:
    1) (CUV) ensure situational 3D env for this post if missing
    2) predict with memory built from posts < t
    3) judge vs ground-truth opinion
    4) ingest the real post into memory
    5) (CUV) evolve theory weights from judge feedback

Agent grows record-by-record and should align better over time.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tg_agent.agent import CUVAgent
from tg_agent.benchmark_core import (
    JUDGE_SYSTEM,
    LLMConfig,
    OpenAICompatClient,
    aggregate_metrics,
    build_judge_user,
    parse_judge,
)
from tg_agent.genminds import GenMindsMemory
from tg_agent.llm import DeepSeekClient, load_env
from tg_agent.memory_layers import detect_emotions, emotion_alignment
from tg_agent.path_agent import PathAgent, PathOutput
from tg_agent.situational_env import (
    ensure_situational_for_post,
    event_to_post,
    load_situational_store,
)
from tg_agent.theory_lib import TheoryLibrary
from tg_agent.user_actions import build_source_profile


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_chrono_events(uid: str, *, root: str | Path | None = None) -> list[dict[str, Any]]:
    """Load chronological events; prefer Linux layout, keep Windows legacy paths."""
    root = Path(root) if root else Path("/root/autodl-tmp/UserAgent")
    candidates = [
        root / f"outputs/weibo_user_{uid}/events_all.jsonl",
        root / f"New/outputs/New-outputs/weibo_user_{uid}/events_all.jsonl",
        Path(f"d:/CognitiveMap/outputs/weibo_user_{uid}/events_all.jsonl"),
        Path(f"d:/UserAgent/outputs/weibo_user_{uid}/events_all.jsonl"),
    ]
    cm = next((p for p in candidates if p.exists()), None)
    if cm is None:
        raise SystemExit(
            f"missing events_all for {uid}; tried: "
            + ", ".join(str(p) for p in candidates)
        )
    events = _load_jsonl(cm)
    events = [e for e in events if e.get("user_opinion") or e.get("raw_text")]
    events.sort(key=lambda e: float(e.get("timestamp") or 0.0))
    return events


def stimulus_from_event(e: dict[str, Any]) -> str:
    topics = e.get("topics") or []
    topic = ""
    if topics:
        topic = str(topics[0])
    elif e.get("topic_hashtag"):
        topic = str(e["topic_hashtag"]).strip("#")
    title = str(e.get("event_title") or "").strip()
    summary = str(e.get("event_summary") or "").strip()
    parts = []
    if topic:
        parts.append(f"微博热议话题：#{topic}#")
    if title:
        parts.append(f"事件标题：{title}")
    if summary:
        parts.append(f"事件摘要：{summary}")
    parts.append("请以该用户身份对该事件发表一条简短原创微博评论。")
    return "\n".join(parts)


def ground_truth(e: dict[str, Any]) -> str:
    return str(e.get("user_opinion") or e.get("raw_text") or "").strip()


def build_units(events: list[dict[str, Any]], gap_hours: float) -> list[list[int]]:
    """时间窗聚合：同话题 gap_hours 内连续发帖合并为一个 data point（事件索引组）。

    窗按首帖时间全局排序，chrono 协议不破；同窗多帖共享一次预测/评判。
    """
    from tg_agent.temporal_windows import build_topic_windows

    prepared = []
    for i, e in enumerate(events):
        topic = str((e.get("topics") or [""])[0] or e.get("event_title") or "untopic")[:80]
        prepared.append(
            {
                "topic": topic,
                "date": str(e.get("timestamp") or e.get("date") or ""),
                "post_id": str(e.get("post_id") or f"idx{i}"),
                "_idx": i,
            }
        )
    wins = build_topic_windows(prepared, gap_hours=gap_hours)
    id2idx = {p["post_id"]: p["_idx"] for p in prepared}
    units = [sorted(id2idx[pid] for pid in w.post_ids if pid in id2idx) for w in wins]
    units = [u for u in units if u]
    units.sort(key=lambda g: min(float(events[i].get("timestamp") or 0.0) for i in g))
    return units


def sample_units(
    units: list[list[int]],
    *,
    ratio: float,
    seed: int = 42,
    large_min: int = 5,
    skip_below: int = 0,
) -> set[int]:
    """分层采样验证：大窗/小窗按比例抽；未抽中的步仅 ingest 不预测（省调用）。"""
    import random

    rng = random.Random(seed)
    large = [i for i, g in enumerate(units) if len(g) >= large_min and i >= skip_below]
    small = [i for i, g in enumerate(units) if len(g) < large_min and i >= skip_below]
    keep: set[int] = set()
    for bucket in (large, small):
        k = max(1 if bucket else 0, int(round(len(bucket) * ratio)))
        if bucket:
            keep.update(rng.sample(bucket, min(k, len(bucket))))
    return keep


def combined_context(events: list[dict[str, Any]], idxs: list[int]) -> str:
    base = stimulus_from_event(events[idxs[0]])
    if len(idxs) > 1:
        base += f"\n（注：该话题在短时间内有 {len(idxs)} 条连续讨论，请预测用户在这一时段的整体回应。）"
    return base


def combined_gt(events: list[dict[str, Any]], idxs: list[int]) -> str:
    return "\n".join(t for t in (ground_truth(events[i]) for i in idxs) if t)


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


def genminds_predict(memory: GenMindsMemory, llm: DeepSeekClient, stimulus: str, top_k: int = 5) -> str:
    retrieved = memory.retrieve(stimulus, top_k=top_k)
    evidence = "\n".join(
        f"- {e.event_title}: {e.user_opinion or e.text[:160]}" for e in retrieved
    ) or "- （尚无历史事件，仅依据人格先验）"
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
已观察历史条数：{u.get('num_events', 0)}
相关历史：
{evidence}

事件：
{stimulus}

请以该账号本人声口发表一条简短原创微博评论："""
    pred = llm.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.4,
        max_tokens=400,
        disable_thinking=True,
    ).strip()
    if pred.startswith("WEIBO:"):
        pred = pred[6:].strip()
    return pred


def rolling_mean(vals: list[float], window: int) -> list[float]:
    out: list[float] = []
    for i in range(len(vals)):
        lo = max(0, i + 1 - window)
        chunk = vals[lo : i + 1]
        out.append(sum(chunk) / max(1, len(chunk)))
    return out


def _reload_agent_situational(agent: CUVAgent, sit_path: Path) -> None:
    if sit_path.exists():
        agent.situational_store = load_situational_store(sit_path)


def run_method(
    *,
    method: str,
    uid: str,
    events: list[dict[str, Any]],
    memory_template: GenMindsMemory,
    theories: TheoryLibrary | None,
    deepseek: DeepSeekClient,
    judge: OpenAICompatClient,
    out_dir: Path,
    warmup: int,
    evolve_threshold: float,
    weak_match_threshold: float,
    resume: bool,
    ensure_situational: bool,
    env_file: str,
    canonical_sit_path: Path,
    sit_prefetch_workers: int = 0,
    sit_prefetch_ahead: int = 0,
    retrieval: str = "weibo_ai",
    source_profile: dict[str, Any] | None = None,
    units: list[list[int]] | None = None,
    sampled_steps: set[int] | None = None,
    scoring: dict[str, Any] | None = None,
    path_tuning: dict[str, Any] | None = None,
    fm_budget: dict[str, Any] | None = None,
    agent_cfg: dict[str, Any] | None = None,
    tg_failure_memory: bool = False,
) -> dict[str, Any]:
    from concurrent.futures import ThreadPoolExecutor

    is_cuv = method in {"seq-CUV-TG", "seq-CUV-Path", "seq-CUV-Fusion", "seq-CUV-Agent", "seq-CUV-AgentX"}
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "sequential_predictions.jsonl"
    run_state = out_dir / "agent_state"
    run_state.mkdir(parents=True, exist_ok=True)
    sit_path = canonical_sit_path
    run_sit = run_state / f"{uid}_situational_env.json"
    n_steps = len(units) if units is not None else len(events)

    def unit_idxs(step: int) -> list[int]:
        return units[step] if units is not None else [step]

    start_step = 0
    existing_rows: list[dict[str, Any]] = []
    done_steps: set[int] = set()
    from tg_agent.loop_agent import CheckpointStore

    ckpt = CheckpointStore(pred_path, n_steps, resume)
    start_step = ckpt.start_step
    existing_rows = ckpt.existing_rows
    done_steps = ckpt.done_steps
    consecutive_llm_failures = 0

    memory = memory_template.clone_for_sequential(keep_static=True)
    for u in range(start_step):
        for i in unit_idxs(u):
            memory.ingest_event(events[i])

    agent: CUVAgent | None = None
    if method == "seq-CUV-TG":
        assert theories is not None
        if sit_path.exists() and not run_sit.exists():
            run_sit.write_text(sit_path.read_text(encoding="utf-8"), encoding="utf-8")
        agent = CUVAgent(
            uid,
            memory,
            theories,
            deepseek,
            state_dir=run_state,
            use_situational=True,
            weak_match_threshold=weak_match_threshold,
        )
        if tg_failure_memory:
            from .failure_memory import FailureMemory

            agent.failure_memory = FailureMemory.load(agent.state_dir, uid)
            agent.failure_memory.enabled = True
        if not resume or not agent._weights_path().exists():
            agent.weights = {}
            agent.save_weights()
        _reload_agent_situational(agent, sit_path if sit_path.exists() else run_sit)
    elif method in {"seq-CUV-Path", "seq-CUV-Fusion"}:
        assert theories is not None
        if sit_path.exists() and not run_sit.exists():
            run_sit.write_text(sit_path.read_text(encoding="utf-8"), encoding="utf-8")
        # fusion：v1 自由生成 + v2 图工作流（图推理产物即理由，不强制因果链）
        ptuning = dict(path_tuning or {})
        if method == "seq-CUV-Fusion":
            ptuning["path_mode"] = "fusion"
        agent = PathAgent(
            uid,
            memory,
            theories,
            deepseek,
            state_dir=run_state,
            use_situational=True,
            weak_match_threshold=weak_match_threshold,
            source_profile=source_profile or {"available": False},
            tuning=ptuning,
            failure_memory_budget=fm_budget,
        )
        if not resume or not agent._weights_path().exists():
            agent.weights = {}
            agent.save_weights()
        _reload_agent_situational(agent, sit_path if sit_path.exists() else run_sit)
    elif method == "seq-CUV-Agent":
        assert theories is not None
        if sit_path.exists() and not run_sit.exists():
            run_sit.write_text(sit_path.read_text(encoding="utf-8"), encoding="utf-8")
        # Graph Agent：LLM 持工具自主完成 Agentic RAG（LangGraph 循环，fusion 哲学）
        from tg_agent.loop_agent import build_graph_agent

        ptuning = dict(path_tuning or {})
        agent = build_graph_agent(
            uid=uid,
            memory=memory,
            theories=theories,
            deepseek=deepseek,
            run_state=run_state,
            weak_match_threshold=weak_match_threshold,
            source_profile=source_profile,
            path_tuning=ptuning,
            fm_budget=fm_budget,
            agent_cfg=agent_cfg,
        )
        if not resume or not agent._weights_path().exists():
            agent.weights = {}
            agent.save_weights()
        _reload_agent_situational(agent, sit_path if sit_path.exists() else run_sit)
    elif method == "seq-CUV-AgentX":
        assert theories is not None
        if sit_path.exists() and not run_sit.exists():
            run_sit.write_text(sit_path.read_text(encoding="utf-8"), encoding="utf-8")
        # Graph AgentX：基线 Agent + 论文机制改进层（分支推演/MAD辩论/可靠性评分）
        from tg_agent.loop_agent import build_graph_agent

        ptuning = dict(path_tuning or {})
        agent = build_graph_agent(
            uid=uid,
            memory=memory,
            theories=theories,
            deepseek=deepseek,
            run_state=run_state,
            weak_match_threshold=weak_match_threshold,
            source_profile=source_profile,
            path_tuning=ptuning,
            fm_budget=fm_budget,
            agent_cfg=agent_cfg,
            agentx=True,
        )
        if not resume or not agent._weights_path().exists():
            agent.weights = {}
            agent.save_weights()
        _reload_agent_situational(agent, sit_path if sit_path.exists() else run_sit)

    scored_rows: list[dict[str, Any]] = [
        r
        for r in existing_rows
        if r.get("judge_scores")
        and not r.get("warmup")
        and not (r.get("judge_scores") or {}).get("error")
        and not r.get("error")
        and (r.get("prediction") or "").strip()
    ]
    oa_series: list[float] = [
        float((r.get("judge_scores") or {}).get("opinion_alignment_score") or 0.0)
        for r in scored_rows
    ]

    # Prefetch is OFF by default: env+predict must stay strictly chronological.
    # Enabling workers would build future posts' envs before their turn.
    prefetch_pool: ThreadPoolExecutor | None = None
    prefetch_futs: dict[int, Any] = {}
    if is_cuv and ensure_situational and sit_prefetch_workers > 0:
        prefetch_pool = ThreadPoolExecutor(max_workers=sit_prefetch_workers)

        def _prefetch_one(step: int) -> None:
            for idx in unit_idxs(step):
                ensure_situational_for_post(
                    user_id=uid,
                    post=event_to_post(events[idx]),
                    out_path=sit_path,
                    llm=deepseek,
                    env_file=env_file,
                    retrieval=retrieval,
                )

        def _pump_prefetch(around: int) -> None:
            assert prefetch_pool is not None
            for j in range(around, min(n_steps, around + sit_prefetch_ahead + 1)):
                if j in prefetch_futs:
                    continue
                prefetch_futs[j] = prefetch_pool.submit(_prefetch_one, j)

        _pump_prefetch(start_step)
    else:

        def _pump_prefetch(around: int) -> None:
            return

    print(
        f"[{method}/{uid}] sequential n={n_steps} (events={len(events)}, "
        f"units={'off' if units is None else 'on'}) warmup={warmup} "
        f"resume_from={start_step} ensure_sit={ensure_situational} "
        f"sit_prefetch_workers={sit_prefetch_workers} retrieval={retrieval} "
        f"sampled={len(sampled_steps) if sampled_steps is not None else 'all'}",
        flush=True,
    )

    # GT 情绪检测缓存（语义缓存理念：同文本不重复调 judge LLM）
    emo_cache_path = run_state / "gt_emotion_cache.json"
    emo_cache: dict[str, Any] = {}
    if emo_cache_path.exists():
        try:
            emo_cache = json.loads(emo_cache_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            emo_cache = {}

    def _gt_emotions(gt_text: str) -> dict[str, float]:
        key = hashlib.sha1(gt_text.strip().encode("utf-8")).hexdigest()[:16]
        if key not in emo_cache:
            emo_cache[key] = detect_emotions(judge, gt_text)
            emo_cache_path.write_text(
                json.dumps(emo_cache, ensure_ascii=False), encoding="utf-8"
            )
        return emo_cache[key]

    try:
        for step in range(start_step, n_steps):
            idxs = unit_idxs(step)
            e = events[idxs[0]]
            if step in done_steps:
                # gap-fill resume: 已完成的步只按时间序吞进记忆，不重复预测
                for i in idxs:
                    memory.ingest_event(events[i])
                if agent is not None:
                    agent.memory = memory
                continue
            pid = str(e.get("post_id") or step)
            bid = str(e.get("bid") or e.get("post_id") or "")
            context = combined_context(events, idxs)
            gt = combined_gt(events, idxs)
            is_warmup = step < warmup or (
                sampled_steps is not None and step not in sampled_steps
            )
            _pump_prefetch(step + 1)

            row: dict[str, Any] = {
                "step": step,
                "post_id": pid,
                "post_ids": [str(events[i].get("post_id") or i) for i in idxs],
                "unit_size": len(idxs),
                "bid": bid,
                "timestamp": e.get("timestamp"),
                "topic": (e.get("topics") or [None])[0],
                "num_events_before": len(memory.events),
                "warmup": is_warmup,
                "ground_truth": gt,
                "context": context,
                "method": method,
            }

            out = None
            if is_warmup:
                if is_cuv and ensure_situational:
                    try:
                        # wait for prefetch of this step if any
                        fut = prefetch_futs.get(step)
                        if fut is not None:
                            fut.result()
                        rec: dict[str, Any] = {}
                        for i in idxs:
                            rec = ensure_situational_for_post(
                                user_id=uid,
                                post=event_to_post(events[i]),
                                out_path=sit_path,
                                llm=deepseek,
                                env_file=env_file,
                                retrieval=retrieval,
                            )
                        row["situational_built"] = True
                        row["theory_coordinates"] = rec.get("theory_coordinates") or []
                        if agent is not None:
                            _reload_agent_situational(agent, sit_path)
                    except Exception as ex:  # noqa: BLE001
                        row["situational_error"] = str(ex)
                row["prediction"] = ""
                row["judge_scores"] = None
                row["skipped_reason"] = (
                    "warmup_ingest_only"
                    if step < warmup
                    else "stratified_sample_skip"
                )
            else:
                try:
                    if is_cuv and ensure_situational:
                        fut = prefetch_futs.get(step)
                        if fut is not None:
                            fut.result()
                        rec = {}
                        for i in idxs:
                            rec = ensure_situational_for_post(
                                user_id=uid,
                                post=event_to_post(events[i]),
                                out_path=sit_path,
                                llm=deepseek,
                                env_file=env_file,
                                retrieval=retrieval,
                            )
                        row["theory_coordinates"] = rec.get("theory_coordinates") or []
                        if agent is not None:
                            _reload_agent_situational(agent, sit_path)

                    if method == "seq-GenMinds":
                        pred = genminds_predict(memory, deepseek, context)
                        trace = {"mode": "genminds", "num_events": len(memory.events)}
                    else:
                        assert agent is not None
                        out = agent.predict(
                            context,
                            post_id=pid,
                            bid=bid,
                            date="",
                            topic=str((e.get("topics") or [""])[0] or ""),
                        )
                        pred = out.predicted_opinion.strip()
                        trace = out.c_trace or {}
                        row["agent_trace"] = {
                            "matched_theories": out.matched_theories,
                            "evidence_events": getattr(out, "evidence_events", []),
                            "verbalization": out.verbalization,
                            "activated_coordinates": out.activated_coordinates,
                            "c_trace": out.c_trace,
                            "caveats": out.caveats,
                        }
                        if isinstance(out, PathOutput):
                            row["stance"] = out.stance
                            row["agent_trace"].update({
                                "factors": out.factors,
                                "paths": out.paths,
                                "emotion_probs": out.emotion_probs,
                                "confidence": out.confidence,
                                "low_evidence": out.low_evidence,
                                "skeptic": out.skeptic,
                            })

                    try:
                        scores = judge_one(judge, context, gt, pred)
                    except Exception as jex:
                        # Qwen content-moderation 400 (or any judge outage): keep the
                        # prediction and fall back to a DeepSeek judge so the step is
                        # not lost. Record the fallback in judge_scores for audit.
                        raw = deepseek.chat(
                            [
                                {"role": "system", "content": JUDGE_SYSTEM},
                                {"role": "user", "content": build_judge_user(context, gt, pred)},
                            ],
                            temperature=0.0,
                            max_tokens=500,
                            disable_thinking=True,
                        )
                        scores = parse_judge(raw)
                        scores["judge_fallback"] = f"deepseek_after_{type(jex).__name__}"
                        if scores.get("error"):
                            raise jex
                    row["prediction"] = pred
                    # 情绪对齐：真实评论情绪分布 vs 预测情绪分布
                    gt_emotion_probs = None
                    if isinstance(out, PathOutput):
                        try:
                            gt_emotion_probs = _gt_emotions(gt)
                            scores["emotion_alignment"] = emotion_alignment(
                                out.emotion_probs,
                                gt_emotion_probs,
                            )
                            row["gt_emotion_probs"] = gt_emotion_probs
                        except Exception:  # noqa: BLE001
                            pass
                    row["judge_scores"] = scores
                    row["predict_trace"] = trace
                    consecutive_llm_failures = 0
                    oa = float(scores.get("opinion_alignment_score") or 0.0)
                    oa_series.append(oa)
                    scored_rows.append(row)

                    if (
                        is_cuv
                        and agent is not None
                        and out is not None
                        and out.matched_theories
                    ):
                        if isinstance(out, PathOutput) and isinstance(agent, PathAgent):
                            ev_log = agent.evolve_attributed(
                                out,
                                gt=gt,
                                oa=oa,
                                gt_emotion_probs=gt_emotion_probs,
                                topic=str((e.get("topics") or [""])[0] or ""),
                                threshold=evolve_threshold,
                            )
                            row["evolved"] = {
                                "helpful": ev_log.get("helpful"),
                                "oa": oa,
                                "attribution": ev_log.get("attribution"),
                            }
                        else:
                            ev_log = agent.evolve(
                                out,
                                feedback=f"judge_oa={oa:.3f}",
                                oa=oa,
                                judge_scores=scores,
                            )
                            row["evolved"] = {
                                "helpful": ev_log.get("helpful"),
                                "oa": oa,
                                "thinking_error": ev_log.get("thinking_error"),
                                "thinking_quality": ev_log.get("thinking_quality"),
                                "repair_action": ev_log.get("repair_action"),
                            }
                except Exception as ex:  # noqa: BLE001
                    # 失败行（如 LLM 402/400）不计入 oa_series/scored_rows，
                    # 也不视为已完成——下次 --resume 会 gap-fill 重跑该步
                    row["prediction"] = ""
                    row["judge_scores"] = {
                        "stance": 0.0,
                        "core_judgment": 0.0,
                        "belief": 0.0,
                        "value": 0.0,
                        "opinion_alignment_score": 0.0,
                        "error": str(ex),
                    }
                    row["error"] = str(ex)
                    consecutive_llm_failures += 1
                    print(f"[{method}/{uid}] step {step+1} FAILED: {ex}", flush=True)
                    if consecutive_llm_failures >= 8:
                        print(
                            f"[{method}/{uid}] FAILFAST: {consecutive_llm_failures} consecutive "
                            "LLM failures (balance/outage). Stopping; resume will gap-fill.",
                            flush=True,
                        )
                        sys.exit(2)

            for i in idxs:
                memory.ingest_event(events[i])
            if agent is not None:
                agent.memory = memory

            _append_jsonl(pred_path, row)
            if (step + 1) % 5 == 0 or step + 1 == n_steps:
                recent = oa_series[-5:] if oa_series else []
                avg = sum(recent) / len(recent) if recent else float("nan")
                print(
                    f"[{method}/{uid}] step {step+1}/{n_steps} "
                    f"mem={len(memory.events)} scored={len(oa_series)} recent5_oa={avg:.3f}",
                    flush=True,
                )
                _write_metrics(
                    out_dir,
                    uid=uid,
                    method=method,
                    events_n=len(events),
                    n_steps=n_steps,
                    warmup=warmup,
                    oa_series=oa_series,
                    scored_rows=scored_rows,
                    scoring=scoring,
                )
    finally:
        if prefetch_pool is not None:
            prefetch_pool.shutdown(wait=False, cancel_futures=True)

    return _write_metrics(
        out_dir,
        uid=uid,
        method=method,
        events_n=len(events),
        n_steps=n_steps,
        warmup=warmup,
        oa_series=oa_series,
        scored_rows=scored_rows,
        scoring=scoring,
    )


def _write_metrics(
    out_dir: Path,
    *,
    uid: str,
    method: str,
    events_n: int,
    warmup: int,
    oa_series: list[float],
    scored_rows: list[dict[str, Any]],
    n_steps: int = 0,
    scoring: dict[str, Any] | None = None,
) -> dict[str, Any]:
    window = 5
    roll = rolling_mean(oa_series, window)
    weights = (scoring or {}).get("result_weights") or None
    reason_weight = float((scoring or {}).get("reason_weight") or 0.0)
    # 成本护栏：每计分步的 predict LLM 调用数（fast_path/genminds 记 1，缺省 0）
    def _row_llm_calls(r: dict[str, Any]) -> int:
        trace = r.get("predict_trace") or {}
        n = trace.get("num_llm_calls")
        if n is not None:
            return int(n)
        return 1 if trace.get("mode") in {"fast_path", "genminds"} else 0

    total_llm_calls = sum(_row_llm_calls(r) for r in scored_rows)
    metrics = {
        "user_id": uid,
        "method": method,
        "protocol": "sequential_temporal_construct",
        "num_events_total": events_n,
        "num_steps": n_steps or events_n,
        "warmup": warmup,
        "num_scored": len(oa_series),
        "predict_model": f"local/{os.environ.get('LLM_MODEL') or 'DeepSeek-V4-Flash'}",
        "judge_model": "dashscope/qwen3.7-plus",
        "benchmark_type": "opinion_alignment",
        "benchmark": aggregate_metrics(
            scored_rows, weights=weights, reason_weight=reason_weight
        )
        if scored_rows
        else {},
        "late_alignment": {
            "last_5": round(sum(oa_series[-5:]) / max(1, len(oa_series[-5:])), 4)
            if oa_series
            else 0.0,
            "last_10": round(sum(oa_series[-10:]) / max(1, len(oa_series[-10:])), 4)
            if oa_series
            else 0.0,
            "first_5": round(sum(oa_series[:5]) / max(1, len(oa_series[:5])), 4)
            if oa_series
            else 0.0,
        },
        "rolling_window": window,
        "oa_series": [round(x, 4) for x in oa_series],
        "oa_rolling": [round(x, 4) for x in roll],
        "llm_calls": {
            "total": total_llm_calls,
            "mean_per_scored": round(total_llm_calls / max(1, len(scored_rows)), 3),
        },
        "fast_path": {
            "scored": sum(
                1
                for r in scored_rows
                if (r.get("predict_trace") or {}).get("mode") == "fast_path"
            ),
            "rate": round(
                sum(
                    1
                    for r in scored_rows
                    if (r.get("predict_trace") or {}).get("mode") == "fast_path"
                )
                / max(1, len(scored_rows)),
                4,
            ),
        },
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--user", default="7463374646")
    ap.add_argument("--methods", default="seq-GenMinds,seq-CUV-TG")
    ap.add_argument("--warmup", type=int, default=5, help="ingest-only first N posts")
    ap.add_argument("--limit", type=int, default=0, help="0=all chronological events")
    ap.add_argument("--evolve-threshold", type=float, default=0.5)
    ap.add_argument("--weak-match-threshold", type=float, default=0.40)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--ensure-situational",
        action="store_true",
        default=True,
        help="for seq-CUV-TG: build situational 3D env on-demand per post",
    )
    ap.add_argument("--no-ensure-situational", action="store_true")
    ap.add_argument(
        "--sit-prefetch-workers",
        type=int,
        default=0,
        help="0=strict chrono (default). >0 prefetches future sit envs (breaks pure temporal generation).",
    )
    ap.add_argument("--sit-prefetch-ahead", type=int, default=0)
    ap.add_argument(
        "--retrieval",
        default="weibo_ai",
        choices=["weibo_ai", "serper", "both"],
        help="presentation evidence backend for pathway env (default: weibo_ai / 智搜)",
    )
    ap.add_argument(
        "--sit-suffix",
        default="_weibo_ai",
        help="suffix for situational env file, e.g. {uid}_situational_env{suffix}.json",
    )
    ap.add_argument(
        "--out-root",
        default=str(Path("d:/UserAgent/outputs/benchmark_sequential_weibo_ai")),
    )
    ap.add_argument(
        "--aggregate-windows",
        action="store_true",
        help="同话题 gap_hours 内连续发帖合并为一个 data point（也可在 config temporal_windows.enabled 开启）",
    )
    ap.add_argument(
        "--gap-hours",
        type=float,
        default=0.0,
        help="时间窗间隔（小时）；0 = 用 config temporal_windows.gap_hours（默认 1.0）",
    )
    ap.add_argument(
        "--window-sample-ratio",
        type=float,
        default=0.0,
        help=">0 时按窗口分层采样验证，未抽中的步仅 ingest（0 = 用 config 或全量）",
    )
    ap.add_argument(
        "--fast-path",
        action="store_true",
        help="覆盖 config：开启 surprise 门控快慢通路（routine 帖单调用预测）",
    )
    ap.add_argument(
        "--tg-failure-memory",
        action="store_true",
        help="把错题本挂到 seq-CUV-TG（TG+FM 消融单元）",
    )
    ap.add_argument(
        "--fm-mode",
        choices=["full", "notes", "off"],
        default="full",
        help="错题本 ablation：full=完整（含权重修复+撤销防线）；notes=只学 strategy_note 软建议；off=整体关闭",
    )
    ap.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="0=all；>0 时只跑前 warmup+N 个 step（在窗口聚合之后切片，便宜 smoke；区别于 --limit 在聚合前切事件）",
    )
    ap.add_argument(
        "--self-restart",
        action="store_true",
        default=None,
        help="进程内看门狗：崩溃后按检查点自动重启（seq-CUV-Agent 默认开；吸收 watchdog bat 职责）",
    )
    ap.add_argument("--no-self-restart", action="store_true")
    args = ap.parse_args()
    ensure_sit = args.ensure_situational and not args.no_ensure_situational

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    load_env(cfg["paths"]["env_file"])
    qwen_key = os.environ.get("QWEN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY", "")
    if not qwen_key:
        raise SystemExit("QWEN_API_KEY / DASHSCOPE_API_KEY missing")
    if (
        args.retrieval in {"weibo_ai", "both"}
        and ensure_sit
        and any("CUV" in m or "TG" in m for m in args.methods.split(","))
        and not os.environ.get("WEIBO_COOKIE", "").strip()
    ):
        raise SystemExit(
            "WEIBO_COOKIE missing. Set a logged-in Weibo cookie before running 智搜 retrieval:\n"
            "  $env:WEIBO_COOKIE = '<cookie string from browser>'\n"
            "Do not commit the cookie."
        )

    from tg_agent.cli import build_llm

    deepseek = build_llm(cfg)
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

    uid = args.user
    events = load_chrono_events(uid, root=cfg["paths"].get("root"))
    if args.limit:
        events = events[: args.limit]

    # 0728 会议改造：时间窗聚合 + 分层采样 + 图工作流超参 + 评测加权
    tw = cfg.get("temporal_windows") or {}
    scoring = cfg.get("scoring") or {}
    path_tuning = dict(cfg.get("path_agent") or {})
    path_tuning["fast_path"] = dict(cfg.get("fast_path") or {})
    if args.fast_path:
        path_tuning["fast_path"]["enabled"] = True
    fm_budget = dict(cfg.get("failure_memory") or {})
    # 0729 错题本 ablation：off=整体关闭；notes=只学 strategy_note 软建议不动权重
    if args.fm_mode == "off":
        fm_budget["enabled"] = False
    elif args.fm_mode == "notes":
        fm_budget["weight_repairs_enabled"] = False
    aggregate = bool(tw.get("enabled")) or args.aggregate_windows
    gap_hours = args.gap_hours or float(tw.get("gap_hours") or 1.0)
    units = build_units(events, gap_hours=gap_hours) if aggregate else None
    sample_ratio = args.window_sample_ratio or float(tw.get("sample_ratio") or 0.0)
    sampled_steps: set[int] | None = None
    if sample_ratio > 0:
        base_units = units if units is not None else [[i] for i in range(len(events))]
        sampled_steps = sample_units(
            base_units,
            ratio=sample_ratio,
            seed=int(tw.get("sample_seed") or 42),
            large_min=int(tw.get("large_window_min") or 5),
            skip_below=args.warmup,
        )
        print(
            f"[{uid}] stratified sampling: {len(sampled_steps)}/{len(base_units)} steps scored "
            f"(ratio={sample_ratio})",
            flush=True,
        )
    if units is not None:
        n_posts = sum(len(g) for g in units)
        print(
            f"[{uid}] window aggregation: {len(events)} events -> {len(units)} data points "
            f"(gap={gap_hours}h, merged={n_posts - len(units)})",
            flush=True,
        )
    # --max-steps：warmup 之后再取 N 步（聚合之后切片，smoke 用）
    if args.max_steps:
        keep = args.warmup + args.max_steps
        if units is not None:
            units = units[:keep]
        else:
            events = events[:keep]
        print(f"[{uid}] max-steps: keep first {keep} steps", flush=True)

    memory_template = GenMindsMemory(
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

    out_root = Path(args.out_root)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    summary = []
    sit_path = ROOT / "data" / "users" / f"{uid}_situational_env{args.sit_suffix}.json"
    csv_path = (cfg["paths"].get("user_csv") or {}).get(uid)
    source_profile = build_source_profile(csv_path) if csv_path else {"available": False}

    # Loop Agent 运行时配置（config agent/loop 段）
    agent_cfg = dict(cfg.get("agent") or {})
    loop_cfg = dict(cfg.get("loop") or {})
    max_restarts = int(loop_cfg.get("max_supervisor_restarts", 20))
    restart_backoff = float(loop_cfg.get("supervisor_backoff_s", 10))

    for method in methods:
        # seq-CUV-Agent 默认开进程内看门狗；--self-restart / --no-self-restart 覆盖
        self_restart = bool(args.self_restart) and not args.no_self_restart
        if args.self_restart is None and not args.no_self_restart:
            self_restart = method in {"seq-CUV-Agent", "seq-CUV-AgentX"}

        def _run(m: str = method) -> dict[str, Any]:
            return run_method(
                method=m,
                uid=uid,
                events=events,
                memory_template=memory_template,
                theories=theories if ("CUV" in m or "TG" in m) else None,
                deepseek=deepseek,
                judge=judge,
                out_dir=out_root / f"{m}_{uid}",
                warmup=args.warmup,
                evolve_threshold=args.evolve_threshold,
                weak_match_threshold=args.weak_match_threshold,
                resume=args.resume,
                ensure_situational=ensure_sit and ("CUV" in m or "TG" in m),
                env_file=cfg["paths"]["env_file"],
                canonical_sit_path=sit_path,
                sit_prefetch_workers=args.sit_prefetch_workers,
                sit_prefetch_ahead=args.sit_prefetch_ahead,
                retrieval=args.retrieval,
                source_profile=source_profile,
                units=units,
                sampled_steps=sampled_steps,
                scoring=scoring,
                path_tuning=path_tuning,
                fm_budget=fm_budget,
                agent_cfg=agent_cfg,
                tg_failure_memory=args.tg_failure_memory,
            )

        if self_restart:
            from tg_agent.loop_agent import run_with_supervision

            metrics = run_with_supervision(
                _run,
                max_restarts=max_restarts,
                backoff_s=restart_backoff,
                log_prefix=f"[{method}/{uid}]",
            )
        else:
            metrics = _run()
        summary.append(metrics)
        print(
            json.dumps(
                {k: metrics[k] for k in metrics if k not in {"oa_series", "oa_rolling"}},
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )

    report = out_root / f"comparison_report_{uid}.md"
    has_weighted = any("weighted_result" in (m.get("benchmark") or {}) for m in summary)
    has_composite = any("composite" in (m.get("benchmark") or {}) for m in summary)
    extra = (["weighted"] if has_weighted else []) + (["composite"] if has_composite else [])
    lines = [
        "# Sequential Temporal Alignment",
        "",
        f"- user: {uid}",
        "- protocol: chrono construct (no train/test split)"
        + ("; window-aggregated data points" if units is not None else ""),
        f"- warmup: {args.warmup}",
        f"- ensure_situational: {ensure_sit}",
        f"- generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "| method | overall |"
        + "".join(f" {c} |" for c in extra)
        + " first5 | last5 | last10 | n_scored |",
        "|---|---:|" + "---:|" * len(extra) + "---:|---:|---:|---:|",
    ]
    for m in summary:
        b = m.get("benchmark") or {}
        late = m.get("late_alignment") or {}
        row = f"| {m['method']} | {b.get('opinion_alignment_score', 0):.4f} | "
        if has_weighted:
            row += f"{b.get('weighted_result', 0):.4f} | "
        if has_composite:
            row += f"{b.get('composite', 0):.4f} | "
        row += (
            f"{late.get('first_5', 0):.4f} | {late.get('last_5', 0):.4f} | "
            f"{late.get('last_10', 0):.4f} | {m.get('num_scored', 0)} |"
        )
        lines.append(row)
    report.write_text("\n".join(lines), encoding="utf-8")
    (out_root / f"comparison_summary_{uid}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {report}", flush=True)


if __name__ == "__main__":
    main()
