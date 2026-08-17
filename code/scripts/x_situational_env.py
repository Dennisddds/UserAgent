#!/usr/bin/env python3
"""Build personal-reception-pathway / situational environments for crawled X users.

Mirrors the Weibo pipeline (theory_guided_agent/tg_agent/situational_env.py) but
uses Serper Google search instead of Weibo 智搜, because X/Twitter events are
global news. For every post it retrieves contemporaneous evidence restricted to
<= posting time, then a DeepSeek judge organizes it into the same JSON schema
(observed_pathway / propagation / environment / theory_coordinates / summary).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(r"D:\UserSimuAgent\项目最新版\UserAgent\theory_guided_agent")
sys.path.insert(0, str(ROOT))

from tg_agent.llm import DeepSeekClient, load_env  # noqa: E402
from tg_agent.situational_env import (  # noqa: E402
    STRUCT_SYSTEM,
    _date_key,
    _topic_of,
    load_authored_posts,
    observed_pathway_from_post,
    serper_search,
    structure_env,
    _normalize_structured,
    _load_existing,
)

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")


def ddgs_search(query: str, *, max_results: int = 5) -> list[dict]:
    """Free, keyless, local search via DuckDuckGo (works without Docker/WSL)."""
    from ddgs import DDGS
    hits: list[dict] = []
    with DDGS() as d:
        try:
            for item in d.news(query, max_results=max_results):
                hits.append({
                    "title": item.get("title") or "",
                    "snippet": (item.get("body") or "")[:400],
                    "link": item.get("url") or "",
                    "date": item.get("date") or "",
                })
        except Exception:
            pass
        if len(hits) < 3:
            for item in d.text(query, max_results=max_results):
                hits.append({
                    "title": item.get("title") or "",
                    "snippet": (item.get("body") or "")[:400],
                    "link": item.get("href") or item.get("url") or "",
                    "date": "",
                })
    return hits


def tavily_search(query: str, *, max_results: int = 5) -> list[dict]:
    payload = json.dumps({
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": "basic",
        "max_results": max_results,
    }).encode()
    req = urllib.request.Request(
        "https://api.tavily.com/search",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode())
    hits = []
    for item in data.get("results") or []:
        hits.append({
            "title": item.get("title") or "",
            "snippet": (item.get("content") or "")[:400],
            "link": item.get("url") or "",
            "date": item.get("published_date") or "",
        })
    return hits


SEARCH_BACKEND = os.environ.get("SIT_SEARCH_BACKEND", "duckduckgo")


def extract_topic(llm: DeepSeekClient, text: str) -> str:
    """LLM-extract a concise, searchable event phrase from an X post."""
    prompt = (
        "从下面这条社交媒体帖子中提取一个可用于新闻检索的「事件/议题短语」（中英文均可，"
        "10 字以内或 6 词以内，去掉人称和语气词，只要核心事件）。"
        "只输出该短语本身，不要任何解释或引号。\n\n"
        f"帖子：{text[:600]}"
    )
    try:
        out = llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=40,
            disable_thinking=True,
        ).strip().strip('"').strip()
        out = re.sub(r"^[：:]\s*", "", out)
        return (out or "")[:80]
    except Exception:
        return ""


def build_x_queries(post: dict) -> list[str]:
    """Contemporaneous presentation evidence for a global/X event."""
    topic = post.get("_event_topic") or _topic_of(post)
    dk = _date_key(post.get("date") or "")
    before = f"before:{dk}" if dk else ""
    topic_clean = topic.replace(",", " ").strip() or "news"
    return [
        f"{topic_clean} {dk} news".strip(),
        f"{topic_clean} {dk} Twitter X reaction".strip(),
        f"{topic_clean} {dk} 事件 背景 报道".strip(),
        f"{topic_clean} {dk}".strip(),
    ]


def retrieve_x_evidence(post: dict) -> tuple[list[dict], list[str], dict]:
    topic = post.get("_event_topic") or _topic_of(post)
    dk = _date_key(post.get("date") or "")
    queries = build_x_queries(post)
    hits: list[dict] = []
    seen: set[str] = set()
    for q in queries:
        try:
            searcher = ddgs_search if SEARCH_BACKEND == "duckduckgo" else tavily_search
            for h in searcher(q, max_results=5):
                key = h.get("link") or h.get("title") or ""
                if not key or key in seen:
                    continue
                hd = _date_key(h.get("date") or "")
                if dk and hd and hd > dk:
                    continue
                seen.add(key)
                h = dict(h)
                h["source"] = "tavily"
                hits.append(h)
        except Exception as e:  # noqa: BLE001
            print(f"  tavily fail {q!r}: {str(e)[:120]}", flush=True)
    meta = {
        "retrieval": SEARCH_BACKEND,
        "temporal_cutoff": dk,
        "causal_window": "as_of_or_before_posting_time",
        "topic": topic,
        "hits": len(hits),
    }
    return hits, queries, meta


def build_one(llm: DeepSeekClient, post: dict, hits: list, queries: list, meta: dict) -> dict:
    structured = structure_env(llm, post, hits)
    return _normalize_structured(post, structured, hits, queries, meta)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--user", type=str, required=True)
    ap.add_argument("--env", type=Path,
                    default=Path(r"D:\UserSimuAgent\UserAgent\agentic-harness-engineering\.env"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", type=str, default="deepseek-v4-pro")
    args = ap.parse_args()

    load_env(args.env)
    if not os.environ.get("SERPER_API_KEY", "").strip():
        raise SystemExit("SERPER_API_KEY missing")

    posts = load_authored_posts(args.csv, limit=0)
    if args.limit:
        posts = posts[: args.limit]

    llm = DeepSeekClient(model=args.model)
    payload = _load_existing(args.out)
    payload["user_id"] = args.user
    payload["kind"] = "personal_reception_pathway"
    payload["retrieval"] = "tavily"
    done = {str(r.get("post_id")) for r in payload.get("records") or []}
    todo = [p for p in posts if p["post_id"] not in done]
    print(f"[{args.user}] posts={len(posts)} done={len(done)} todo={len(todo)}", flush=True)

    topic_cache: dict[str, str] = {}
    for i, post in enumerate(todo, 1):
        key = (post.get("text") or "")[:80]
        if key not in topic_cache:
            t = extract_topic(llm, post.get("text") or "")
            topic_cache[key] = t or _topic_of(post)
        post["_event_topic"] = topic_cache[key]
        hits, queries, meta = retrieve_x_evidence(post)
        body = build_one(llm, post, hits, queries, meta)
        topic = post["_event_topic"]
        rec = {
            "user_id": args.user,
            "post_id": post["post_id"],
            "bid": post["bid"],
            "date": post["date"],
            "topic": topic,
            "text": post["text"],
            "slot_key": f"{_date_key(post['date'])}||{topic}",
            **body,
            "kind": "personal_reception_pathway",
            "reused_slot": False,
        }
        payload.setdefault("records", []).append(rec)
        payload["num_posts"] = len(payload["records"])
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"[{i}/{len(todo)}] {post['date'][:16]} topic={topic[:30]} "
            f"hits={len(hits)} coords={(body.get('theory_coordinates') or [])[:3]}",
            flush=True,
        )

    print(f"[{args.user}] DONE records={len(payload.get('records') or [])} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
