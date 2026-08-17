#!/usr/bin/env python3
"""Convert crawled X users into the Agent pipeline's events_all.jsonl format.

Mapping from X JSONL (x_crawl_outputs/user_<handle>.jsonl) to the Weibo-style
event schema used by theory_guided_agent.run_sequential.load_chrono_events:
post_id/bid/timestamp/raw_text/topics/user_opinion/event_title/event_summary.
Also copies the prebuilt situational env into data/users/.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path


def parse_dt(value: str | None) -> float:
    if not value:
        return 0.0
    for fmt in ("%a %b %d %H:%M:%S %z %Y", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt).timestamp()
        except Exception:
            continue
    try:
        return datetime.fromisoformat(value).timestamp()
    except Exception:
        return 0.0


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def sit_topics(handle: str, status_ids: set[str]) -> dict[str, str]:
    p = Path(r"D:\UserSimuAgent\项目最新版\sit_env") / f"{handle}_situational_env.json"
    out: dict[str, str] = {}
    if not p.exists():
        return out
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return out
    for rec in data.get("records") or []:
        pid = str(rec.get("post_id") or rec.get("bid") or "")
        topic = rec.get("topic") or ""
        if pid in status_ids and topic:
            out[pid] = topic
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--handle", required=True)
    ap.add_argument("--crawl-dir", type=Path, default=Path(r"D:\UserSimuAgent\项目最新版\x_crawl_outputs"))
    ap.add_argument("--out-root", type=Path, default=Path(r"D:\UserSimuAgent\项目最新版\UserAgent\theory_guided_agent\outputs"))
    ap.add_argument("--user-state", type=Path, default=Path(r"D:\UserSimuAgent\项目最新版\UserAgent\theory_guided_agent\data\users"))
    args = ap.parse_args()

    rows = read_jsonl(args.crawl_dir / f"user_{args.handle}.jsonl")
    if not rows:
        raise SystemExit(f"no crawled rows for {args.handle}")
    rows.sort(key=lambda r: parse_dt(r.get("created_at")))
    status_ids = {str(r.get("status_id") or "") for r in rows}
    topics = sit_topics(args.handle, status_ids)

    events = []
    for r in rows:
        text = (r.get("text") or "").strip()
        if not text:
            continue
        sid = str(r.get("status_id") or "")
        topic = topics.get(sid, "")
        topics_list = [topic] if topic else []
        events.append({
            "post_id": sid,
            "bid": sid,
            "user_id": args.handle,
            "user_name": r.get("handle") or args.handle,
            "timestamp": parse_dt(r.get("created_at")),
            "raw_text": text,
            "tool": "X",
            "likes": r.get("like_count"),
            "topic_hashtag": topic or None,
            "event_title": topic or text[:40],
            "event_summary": topic or text[:120],
            "entities": [],
            "topics": topics_list,
            "user_opinion": text,
            "stance_keywords": [],
            "is_event_driven": bool(topic),
            "static_belief": None,
            "extraction_method": "x_direct",
            "extraction_error": None,
        })

    out_dir = args.out_root / f"x_user_{args.handle}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "events_all.jsonl").open("w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"[{args.handle}] wrote {len(events)} events -> {out_dir / 'events_all.jsonl'}")

    # copy sit env into user_state
    src = Path(r"D:\UserSimuAgent\项目最新版\sit_env") / f"{args.handle}_situational_env.json"
    if src.exists():
        args.user_state.mkdir(parents=True, exist_ok=True)
        dst = args.user_state / f"{args.handle}_situational_env.json"
        shutil.copyfile(src, dst)
        print(f"[{args.handle}] copied sit env -> {dst}")


if __name__ == "__main__":
    main()
