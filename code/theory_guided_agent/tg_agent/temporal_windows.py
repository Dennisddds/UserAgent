"""Temporal topic windows — 会议：一段时间内相同话题相互影响更大.

把用户时间线切成 data point（默认按话题簇 + 时间间隔），用于：
  - 顺序评测分层采样
  - 检索时优先同窗证据
  - 小代价模拟「用户若干小时」行为块
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class TimeWindow:
    topic: str
    start: str
    end: str
    post_ids: list[str] = field(default_factory=list)
    n: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_ts(s: str) -> float:
    s = (s or "").strip()
    if not s:
        return 0.0
    # epoch 秒（events_all.jsonl 的 timestamp 字段）
    try:
        v = float(s)
        if v > 1e8:
            return v
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d %H:%M", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s[:19], fmt).timestamp()
        except ValueError:
            continue
    return 0.0


def build_topic_windows(
    events: list[dict[str, Any]],
    *,
    gap_hours: float = 6.0,
    topic_key: str = "topic",
    time_key: str = "date",
    id_key: str = "post_id",
) -> list[TimeWindow]:
    """同一话题内，相邻发帖间隔 > gap_hours 则切新窗（默认约 6–7 小时用户会话块）。"""
    # group by topic then sort by time
    by_topic: dict[str, list[dict[str, Any]]] = {}
    for e in events:
        topic = str(e.get(topic_key) or e.get("event_title") or "untopic")[:80]
        by_topic.setdefault(topic, []).append(e)

    windows: list[TimeWindow] = []
    gap_sec = gap_hours * 3600.0
    for topic, items in by_topic.items():
        items = sorted(items, key=lambda x: _parse_ts(str(x.get(time_key) or "")))
        cur: TimeWindow | None = None
        prev_ts = 0.0
        for e in items:
            ts = _parse_ts(str(e.get(time_key) or ""))
            pid = str(e.get(id_key) or e.get("map_id") or "")
            date_s = str(e.get(time_key) or "")
            if cur is None or (prev_ts and ts and (ts - prev_ts) > gap_sec):
                if cur is not None:
                    cur.n = len(cur.post_ids)
                    windows.append(cur)
                cur = TimeWindow(topic=topic, start=date_s, end=date_s, post_ids=[pid] if pid else [])
            else:
                if pid:
                    cur.post_ids.append(pid)
                cur.end = date_s or cur.end
            prev_ts = ts or prev_ts
        if cur is not None:
            cur.n = len(cur.post_ids)
            windows.append(cur)
    return windows


def stratified_sample_windows(
    windows: list[TimeWindow],
    *,
    ratio: float = 0.2,
    min_n: int = 1,
    seed: int = 42,
    large_min: int = 5,
) -> list[TimeWindow]:
    """按窗口大小分层：大窗/小窗按比例采样，控制评测代价。"""
    import random

    rng = random.Random(seed)
    large = [w for w in windows if w.n >= large_min]
    small = [w for w in windows if w.n < large_min]
    out: list[TimeWindow] = []
    for bucket in (large, small):
        k = max(min_n if bucket else 0, int(round(len(bucket) * ratio)))
        if k and bucket:
            out.extend(rng.sample(bucket, min(k, len(bucket))))
    return out
