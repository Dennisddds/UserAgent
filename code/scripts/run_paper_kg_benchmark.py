# -*- coding: utf-8 -*-
"""Paper-KG predict + judge benchmark (retrieve from rebuilt memory banks)."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]  # UserAgent/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TG = ROOT / "theory_guided_agent"
if str(TG) not in sys.path:
    sys.path.insert(0, str(TG))

from tg_agent.benchmark_core import (  # noqa: E402
    JUDGE_SYSTEM,
    LLMConfig,
    OpenAICompatClient,
    aggregate_metrics,
    build_judge_user,
    extract_context_and_gt,
    parse_judge,
)
from tg_agent.llm import DeepSeekClient, load_env  # noqa: E402

OUT = ROOT / "outputs"
DIM = 512


def encode_text(text: str, dim: int = DIM) -> list[float]:
    import hashlib

    s = "".join(str(text).split()).lower()
    v = [0.0] * dim
    for j in range(max(0, len(s) - 2)):
        h = int(hashlib.md5(s[j : j + 3].encode("utf-8")).hexdigest(), 16)
        v[h % dim] += 1.0 if (h >> 8) & 1 else -1.0
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n else v


def cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def load_jsonl(path: Path, limit: int = 0) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


class MemoryBank:
    def __init__(self, path: Path):
        self.bank = json.loads(path.read_text(encoding="utf-8"))
        self.method = self.bank.get("method", "")
        self.paper_ref = self.bank.get("paper_ref", "")
        self.events = self.bank.get("event_maps") or []
        self.static = self.bank.get("static_map") or {}
        ri = self.bank.get("retrieval_index") or {}
        self.texts = ri.get("texts") or []
        self.vectors = ri.get("vectors") or []
        self.retriever = (self.bank.get("method_extras") or {}).get("retriever", "default")
        if len(self.vectors) != len(self.events):
            # rebuild vectors if mismatched
            self.texts, self.vectors = [], []
            for m in self.events:
                t = (m.get("feature_2d_text") or "") + " || " + (m.get("feature_3d_text") or "")
                self.texts.append(t)
                self.vectors.append(encode_text(t))

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[dict, float]]:
        qv = encode_text(query)
        scored = []
        for i, v in enumerate(self.vectors):
            s = cos(qv, v)
            # retriever variants
            if self.retriever == "temporal":
                ts = float(self.events[i].get("timestamp") or 0.0)
                s = 0.85 * s + 0.15 * (ts / (ts + 1.0))
            elif self.retriever == "signed":
                pol = float(self.events[i].get("polarity") or 0.0)
                s = s + 0.05 * abs(pol)
            elif self.retriever == "multihop":
                hops = float(self.events[i].get("cognitive_hops") or 0.0)
                s = s + 0.02 * hops
            scored.append((i, s))
        scored.sort(key=lambda x: -x[1])
        out = []
        for i, s in scored[:top_k]:
            out.append((self.events[i], float(s)))
        return out


PREDICT_SYSTEM = (
    "你是社交媒体用户行为模拟器。根据用户历史认知图谱/记忆片段，"
    "以该用户口吻对给定事件发表一条简短原创微博评论。"
    "只输出评论正文，不要解释。"
)


def build_predict_prompt(sample: dict, retrieved: list[tuple[dict, float]], static: dict) -> str:
    context, _ = extract_context_and_gt(sample)
    lines = ["【用户稳定信念】"]
    for b in (static.get("beliefs") or [])[:6]:
        lines.append(f"- {b}")
    lines.append("【检索到的历史认知图谱片段】")
    for i, (m, score) in enumerate(retrieved, 1):
        lines.append(
            f"{i}. (score={score:.3f}) title={m.get('event_title','')}; "
            f"opinion={str(m.get('user_opinion') or '')[:80]}; "
            f"triples={str(m.get('feature_3d_text') or '')[:180]}"
        )
    lines.append("【当前事件】")
    lines.append(context)
    lines.append("请以该用户身份发表一条简短原创微博评论：")
    return "\n".join(lines)


def make_predict_client() -> DeepSeekClient:
    env = ROOT / "agentic-harness-engineering" / ".env"
    load_env(env)
    new_env = ROOT / "New" / ".env.local"
    if new_env.exists():
        load_env(new_env)
    api_key = (
        os.environ.get("PAPER_KG_LLM_API_KEY")
        or os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("LLM_API_KEY")
        or "EMPTY"
    )
    base_url = (
        os.environ.get("PAPER_KG_LLM_BASE_URL")
        or os.environ.get("DEEPSEEK_BASE_URL")
        or "https://api.deepseek.com"
    )
    model = (
        os.environ.get("PAPER_KG_LLM_MODEL")
        or os.environ.get("DEEPSEEK_MODEL")
        or "deepseek-chat"
    )
    # Avoid accidental local vLLM URL when running paper-KG on Windows without serve
    if "127.0.0.1" in base_url or "localhost" in base_url:
        if os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("PAPER_KG_LLM_API_KEY"):
            base_url = os.environ.get("PAPER_KG_LLM_BASE_URL") or "https://api.deepseek.com"
            model = os.environ.get("PAPER_KG_LLM_MODEL") or "deepseek-chat"
            api_key = (
                os.environ.get("PAPER_KG_LLM_API_KEY")
                or os.environ.get("DEEPSEEK_API_KEY")
                or api_key
            )
    return DeepSeekClient(
        api_key=api_key,
        base_url=base_url,
        model=model,
        enable_thinking=False,
    )


def make_judge_client() -> OpenAICompatClient:
    env = ROOT / "agentic-harness-engineering" / ".env"
    load_env(env)
    api_key = (
        os.environ.get("DASHSCOPE_API_KEY")
        or os.environ.get("QWEN_API_KEY")
        or "EMPTY"
    )
    base_url = (
        os.environ.get("QWEN_BASE_URL")
        or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    model = os.environ.get("QWEN_MODEL") or "qwen-plus"
    return OpenAICompatClient(
        LLMConfig(api_key=api_key, base_url=base_url, model=model, disable_thinking=True)
    )


def run_method(
    user_id: str,
    method_key: str,
    *,
    limit: int = 0,
    top_k: int = 5,
    predict_conc: int = 4,
    resume: bool = True,
) -> dict[str, Any]:
    kg_dir = OUT / f"weibo_kg_{method_key}_{user_id}"
    mb_path = kg_dir / "memory_bank.json"
    if not mb_path.exists():
        raise FileNotFoundError(mb_path)
    mem = MemoryBank(mb_path)
    test_path = OUT / f"weibo_user_{user_id}" / "test.jsonl"
    samples = load_jsonl(test_path, limit=limit)
    pred_path = kg_dir / "predictions.jsonl"
    if resume and pred_path.exists():
        # fresh run for rebuilt graphs: if rebuild newer than preds, wipe
        if mb_path.stat().st_mtime > pred_path.stat().st_mtime:
            pred_path.unlink()
    done = set()
    rows_existing = []
    if resume and pred_path.exists():
        for line in pred_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            rows_existing.append(d)
            if d.get("prediction") and d.get("judge_scores"):
                done.add(str(d.get("post_id")))

    llm = make_predict_client()
    judge = make_judge_client()
    print(
        f"[{mem.method}] samples={len(samples)} maps={len(mem.events)} "
        f"done={len(done)} predict_conc={predict_conc}"
    )

    def predict_one(idx_sample: tuple[int, dict]) -> dict:
        idx, sample = idx_sample
        pid = str(sample.get("post_id") or idx)
        context, gt = extract_context_and_gt(sample)
        retrieved = mem.retrieve(context, top_k=top_k)
        prompt = build_predict_prompt(sample, retrieved, mem.static)
        pred = llm.chat(
            [{"role": "system", "content": PREDICT_SYSTEM}, {"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=400,
            disable_thinking=True,
        )
        content = (pred or "").strip()
        return {
            "index": idx,
            "post_id": pid,
            "user_id": user_id,
            "topic": sample.get("topic") or "",
            "ground_truth": gt,
            "context": context,
            "method": mem.method,
            "paper_ref": mem.paper_ref,
            "retrieved_map_ids": [m.get("map_id") for m, _ in retrieved],
            "retrieved_scores": [round(s, 4) for _, s in retrieved],
            "prediction": content,
            "judge_scores": None,
        }

    todo = [(i, s) for i, s in enumerate(samples) if str(s.get("post_id") or i) not in done]
    t0 = time.time()
    new_rows = []
    if todo:
        with ThreadPoolExecutor(max_workers=predict_conc) as ex:
            futs = {ex.submit(predict_one, item): item[0] for item in todo}
            for n, fut in enumerate(as_completed(futs), 1):
                row = fut.result()
                new_rows.append(row)
                if n % 20 == 0 or n == len(todo):
                    print(f"  predict {n}/{len(todo)}")
    print(f"  predict done in {time.time()-t0:.1f}s")

    # judge
    t1 = time.time()
    to_judge = [r for r in new_rows if r.get("prediction") and not r.get("judge_scores")]
    # also rejudge existing without scores
    for r in rows_existing:
        if r.get("prediction") and not r.get("judge_scores"):
            to_judge.append(r)

    def judge_one(row: dict) -> dict:
        user = build_judge_user(row["context"], row["ground_truth"], row["prediction"])
        raw = judge.chat(
            [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=500,
        )
        scores = parse_judge(raw if isinstance(raw, str) else (raw.get("content") or ""))
        row["judge_scores"] = scores
        return row

    judged = []
    if to_judge:
        with ThreadPoolExecutor(max_workers=max(2, predict_conc)) as ex:
            futs = [ex.submit(judge_one, r) for r in to_judge]
            for n, fut in enumerate(as_completed(futs), 1):
                judged.append(fut.result())
                if n % 20 == 0 or n == len(to_judge):
                    print(f"  judge {n}/{len(to_judge)}")
    print(f"  judge done in {time.time()-t1:.1f}s")

    # merge write
    by_pid = {str(r.get("post_id")): r for r in rows_existing}
    for r in judged:
        by_pid[str(r.get("post_id"))] = r
    for r in new_rows:
        pid = str(r.get("post_id"))
        if pid not in by_pid or not by_pid[pid].get("judge_scores"):
            by_pid[pid] = r
    # keep sample order
    ordered = []
    for i, s in enumerate(samples):
        pid = str(s.get("post_id") or i)
        if pid in by_pid:
            ordered.append(by_pid[pid])
    with pred_path.open("w", encoding="utf-8") as f:
        for r in ordered:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    metrics = aggregate_metrics(ordered)
    metrics_out = {"method": mem.method, "paper_ref": mem.paper_ref, "benchmark": metrics, "n": len(ordered)}
    (kg_dir / "metrics.json").write_text(
        json.dumps(metrics_out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics_out, ensure_ascii=False, indent=2))
    return metrics_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user-id", required=True)
    ap.add_argument("--methods", required=True, help="comma-separated method keys")
    ap.add_argument("--limit", type=int, default=0, help="0=all test samples; large user cap 1000")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--predict-conc", type=int, default=4)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    limit = args.limit
    if args.user_id == "1989660417" and (limit == 0 or limit > 1000):
        # user requested large-sample cap at 1000 (test has 676)
        limit = min(1000, limit) if limit else 1000
        # but test only has 676; keep 0 to mean all, with note
        limit = 0  # all available test (<=676 < 1000)
    all_metrics = []
    for m in methods:
        try:
            met = run_method(
                args.user_id,
                m,
                limit=limit,
                top_k=args.top_k,
                predict_conc=args.predict_conc,
                resume=not args.no_resume,
            )
            all_metrics.append({"user_id": args.user_id, "method_key": m, **met})
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {m}: {e}")
            all_metrics.append({"user_id": args.user_id, "method_key": m, "error": str(e)})
    out = OUT / f"paper_kg_rerun_summary_{args.user_id}.json"
    out.write_text(json.dumps(all_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
