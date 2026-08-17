#!/usr/bin/env python3
"""Convert crawled X posts to the Weibo-export CSV format (1989660417.csv).

The reference Weibo export has 39 columns. X has no exact equivalents for
media/geo/source fields, so those are left empty; the fields that matter for
user simulation (text, timestamps, counts, topics, mentions, originality) are
mapped from the crawled X JSONL.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path


HEADER = [
    "id", "bid", "正文", "头条文章url", "原始图片url", "视频url",
    "Live Photo视频url", "位置", "日期", "工具", "点赞数", "评论数", "转发数",
    "话题", "@用户", "完整日期", "是否编辑过", "编辑次数", "是否原创",
    "源用户id", "源用户昵称", "源微博id", "源微博bid", "源微博正文",
    "源微博头条文章url", "源微博原始图片url", "源微博视频url",
    "源微博Live Photo视频url", "源微博位置", "源微博日期", "源微博工具",
    "源微博点赞数", "源微博评论数", "源微博转发数", "源微博话题",
    "源微博@用户", "源微博完整日期", "源微博是否编辑过", "源微博编辑次数",
]


def parse_twitter_dt(value: str | None):
    if not value:
        return None
    try:
        return datetime.strptime(value, "%a %b %d %H:%M:%S %z %Y")
    except Exception:
        try:
            return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S")
        except Exception:
            return None


def extract_hashtags(text: str) -> str:
    tags = re.findall(r"#([^\s#]+)", text or "")
    return " ".join(tags)


def extract_mentions(text: str) -> str:
    mentions = re.findall(r"@([A-Za-z0-9_]+)", text or "")
    return " ".join(mentions)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def to_weibo_row(row: dict) -> dict:
    text = row.get("text") or ""
    dt = parse_twitter_dt(row.get("created_at"))
    iso = dt.strftime("%Y-%m-%dT%H:%M:%S") if dt else ""
    full = dt.strftime("%Y-%m-%d %H:%M:%S") if dt else ""
    status_id = str(row.get("status_id") or "")

    out = {h: "" for h in HEADER}
    out["id"] = status_id
    out["bid"] = status_id
    out["正文"] = text
    out["日期"] = iso
    out["完整日期"] = full
    out["点赞数"] = "" if row.get("like_count") is None else str(row["like_count"])
    out["评论数"] = "" if row.get("reply_count") is None else str(row["reply_count"])
    out["转发数"] = "" if row.get("repost_count") is None else str(row["repost_count"])
    out["话题"] = extract_hashtags(text)
    out["@用户"] = extract_mentions(text)
    out["是否编辑过"] = "False"
    out["编辑次数"] = "0"
    out["是否原创"] = "True"
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Match the reference: BOM + utf-8-sig, comma delimiter.
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accounts", type=str, required=True,
                        help="Comma-separated handles to convert")
    parser.add_argument("--crawl-dir", type=Path, default=Path("x_crawl_outputs"))
    parser.add_argument("--out-dir", type=Path, default=Path("x_weibo_csv"))
    args = parser.parse_args()

    handles = [h.strip().lstrip("@") for h in args.accounts.split(",") if h.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for handle in handles:
        rows = read_jsonl(args.crawl_dir / f"user_{handle}.jsonl")
        if not rows:
            print(f"[{handle}] no crawled posts; skipping", flush=True)
            continue
        weibo_rows = [to_weibo_row(r) for r in rows]
        weibo_rows.sort(key=lambda r: r["日期"], reverse=True)
        out_path = args.out_dir / f"{handle}.csv"
        write_csv(out_path, weibo_rows)
        print(f"[{handle}] wrote {len(weibo_rows)} rows -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
