#!/usr/bin/env python3
"""Audit the opinionatedness of crawled X accounts with an LLM judge.

For each crawled user, sample a deterministic subset of posts and ask DeepSeek
(configurable) to score each post on a 0-1 "opinionatedness" scale plus a
coarse stance/topic tag. Outputs a per-account aggregate report.

This supports the crawl requirement that selected KOLs are "观点鲜明" rather
than neutral feed content.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import requests


def load_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def sample_rows(rows: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    if len(rows) <= n:
        return rows
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(rows)), n))
    return [rows[i] for i in idx]


def judge_post(client: dict[str, Any], text: str) -> dict[str, Any]:
    prompt = (
        "你是社交媒体内容分析器。下面是一条用户帖子。请判断它的「观点鲜明度」。\n"
        "定义：0 = 完全中立、客观事实陈述、没有任何价值判断或立场；"
        "1 = 立场极其鲜明、有明确的态度、价值判断、批评或主张。\n"
        "只输出一个 JSON 对象，不要任何其他文字，格式如下：\n"
        '{"opinionatedness": 0.0, "stance": "pro|anti|neutral", "topic": "简短主题"}\n\n'
        f"帖子：\n{text}\n"
    )
    for attempt in range(3):
        try:
            resp = requests.post(
                client["base_url"] + "/chat/completions",
                headers={
                    "Authorization": f"Bearer {client['api_key']}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": client["model"],
                    "messages": [
                    {"role": "system", "content": "You are a strict, concise social-media content analyst."},
                    {"role": "user", "content": prompt},
                    ],
                    "temperature": 0,
                    "max_tokens": 80,
                },
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            raw = data["choices"][0]["message"]["content"].strip()
            raw = raw.strip().lstrip("```json").rstrip("```").strip()
            start, end = raw.find("{"), raw.rfind("}")
            if start != -1 and end != -1:
                raw = raw[start:end + 1]
            obj = json.loads(raw)
            obj["opinionatedness"] = float(obj.get("opinionatedness", 0.0))
            return obj
        except Exception:
            time.sleep(1.0)
    return {"opinionatedness": None, "stance": None, "topic": None, "error": "judge_failed"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crawl-dir", type=Path, default=Path("x_crawl_outputs"))
    parser.add_argument("--env", type=Path,
                        default=Path(r"D:\UserSimuAgent\UserAgent\agentic-harness-engineering\.env"))
    parser.add_argument("--samples-per-user", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("opinionation_audit.json"))
    parser.add_argument("--provider", choices=["deepseek", "qwen"], default="deepseek")
    args = parser.parse_args()

    env = load_env(args.env)
    if args.provider == "qwen":
        api_key = env.get("QWEN_API_KEY", "")
        base_url = env.get("QWEN_BASE_URL", "").rstrip("/")
        model = env.get("QWEN_MODEL", "qwen3.7-plus")
    else:
        api_key = env.get("LLM_API_KEY", "")
        base_url = env.get("LLM_BASE_URL", "").rstrip("/")
        model = env.get("LLM_MODEL", "deepseek-v4-pro")

    if not api_key or not base_url:
        raise SystemExit("Missing API key/base URL in env file")

    client = {"api_key": api_key, "base_url": base_url, "model": model}

    files = sorted(args.crawl_dir.glob("user_*.jsonl"))
    report: dict[str, Any] = {}
    for path in files:
        handle = path.stem[len("user_"):]
        rows = read_jsonl(path)
        if not rows:
            continue
        sample = sample_rows(rows, args.samples_per_user, seed=42)
        scores = []
        for row in sample:
            text = (row.get("text") or "").strip()
            if not text:
                continue
            res = judge_post(client, text)
            res["status_id"] = row.get("status_id")
            res["handle"] = handle
            scores.append(res)
        vals = [s["opinionatedness"] for s in scores if isinstance(s.get("opinionatedness"), (int, float))]
        if vals:
            report[handle] = {
                "sampled": len(sample),
                "judged": len(scores),
                "mean_opinionatedness": round(sum(vals) / len(vals), 4),
                "median_opinionatedness": round(sorted(vals)[len(vals) // 2], 4),
                "min": round(min(vals), 4),
                "max": round(max(vals), 4),
                "n_strong_opinionated_ge_0.8": sum(1 for v in vals if v >= 0.8),
            }
        else:
            report[handle] = {"sampled": len(sample), "judged": 0, "error": "no_valid_scores"}
        print(f"[{handle}] mean={report[handle].get('mean_opinionatedness')}", flush=True)

    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
