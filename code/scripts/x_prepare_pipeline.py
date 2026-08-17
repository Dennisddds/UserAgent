#!/usr/bin/env python3
"""Prepare crawled X users for the Agent pipeline.

For each handle:
  - events -> outputs/weibo_user_<handle>/events_all.jsonl (load_chrono_events convention)
  - empty GenMinds bank -> outputs/x_genminds_<handle>/memory_bank.json
  - persona -> outputs/x_user_<handle>/persona.json (profile + post-derived)
  - sit env -> data/users/<handle>_situational_env.json
"""

from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

BASE = Path(r"D:\UserSimuAgent\项目最新版")
CRAWL = BASE / "x_crawl_outputs"
SIT = BASE / "sit_env"
AGENT = BASE / "UserAgent" / "theory_guided_agent"


def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def parse_dt(v: str | None) -> float:
    if not v:
        return 0.0
    for fmt in ("%a %b %d %H:%M:%S %z %Y", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(v, fmt).timestamp()
        except Exception:
            continue
    return 0.0


def events(rows: list[dict], handle: str, topics: dict[str, str]) -> list[dict]:
    out = []
    for r in rows:
        text = (r.get("text") or "").strip()
        if not text:
            continue
        sid = str(r.get("status_id") or "")
        topic = topics.get(sid, "")
        out.append({
            "post_id": sid, "bid": sid,
            "user_id": handle, "user_name": handle,
            "timestamp": parse_dt(r.get("created_at")),
            "raw_text": text, "tool": "X", "likes": r.get("like_count"),
            "topic_hashtag": topic or None,
            "event_title": topic or text[:40],
            "event_summary": topic or text[:120],
            "entities": [], "topics": [topic] if topic else [],
            "user_opinion": text, "stance_keywords": [],
            "is_event_driven": bool(topic), "static_belief": None,
            "extraction_method": "x_direct", "extraction_error": None,
        })
    out.sort(key=lambda e: e["timestamp"])
    return out


def persona(rows: list[dict], profile: dict, topics: list[str], handle: str) -> dict:
    texts = [(r.get("text") or "") for r in rows if (r.get("text") or "").strip()]
    tags = Counter()
    emo = Counter()
    for t in texts:
        for m in re.findall(r"#([^\s#]+)", t):
            tags[m.lower()] += 1
    interests = [t for t, _ in tags.most_common(10)] or topics[:10]
    likes = [r.get("like_count") or 0 for r in rows]
    return {
        "analysis": (
            f"{profile.get('name') or handle} is an opinionated commentator. "
            f"Bio: {profile.get('bio') or ''}"
        ),
        "demographics": {
            "followers": profile.get("followers"),
            "posts_total": profile.get("posts"),
        },
        "interests": interests,
        "values": topics[:10],
        "communication": [t[:120] for t in texts[:8]],
        "statistics": [
            f"crawled_posts={len(texts)}",
            f"avg_likes={sum(likes)/max(len(likes),1):.1f}",
        ],
    }


def main() -> None:
    handles = [p.stem[5:] for p in CRAWL.glob("user_*.jsonl")]
    prepared = []
    for h in sorted(handles):
        rows = read_jsonl(CRAWL / f"user_{h}.jsonl")
        if not rows:
            continue
        profile = {}
        pf = CRAWL / f"user_{h}.profile.json"
        if pf.exists():
            try:
                profile = json.loads(pf.read_text(encoding="utf-8"))
            except Exception:
                pass
        # topics from sit env
        topics_map: dict[str, str] = {}
        sit = SIT / f"{h}_situational_env.json"
        if sit.exists():
            try:
                d = json.loads(sit.read_text(encoding="utf-8"))
                for rec in d.get("records") or []:
                    pid = str(rec.get("post_id") or rec.get("bid") or "")
                    if rec.get("topic"):
                        topics_map[pid] = rec["topic"]
            except Exception:
                pass
        ev = events(rows, h, topics_map)
        if not ev:
            continue
        # events
        edir = BASE / "UserAgent" / "outputs" / f"weibo_user_{h}"
        edir.mkdir(parents=True, exist_ok=True)
        (edir / "events_all.jsonl").write_text(
            "\n".join(json.dumps(e, ensure_ascii=False) for e in ev) + "\n",
            encoding="utf-8",
        )
        # empty genminds bank
        gdir = AGENT / "outputs" / f"x_genminds_{h}"
        gdir.mkdir(parents=True, exist_ok=True)
        (gdir / "memory_bank.json").write_text(
            json.dumps({"static_map": {}, "event_maps": []}, ensure_ascii=False),
            encoding="utf-8",
        )
        # persona
        pdir = AGENT / "outputs" / f"x_user_{h}"
        pdir.mkdir(parents=True, exist_ok=True)
        (pdir / "persona.json").write_text(
            json.dumps(persona(rows, profile, list(topics_map.values())[:10], h), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        # sit env
        if sit.exists():
            (AGENT / "data" / "users").mkdir(parents=True, exist_ok=True)
            shutil.copyfile(sit, AGENT / "data" / "users" / f"{h}_situational_env.json")
        print(f"[{h}] prepared: {len(ev)} events", flush=True)
        prepared.append(h)
    print("PREPARED:", ",".join(prepared), flush=True)


if __name__ == "__main__":
    main()
