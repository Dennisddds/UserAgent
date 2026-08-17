"""信源影响建模（蓝图阶段二）：从用户 CSV 提取转发/提及行为，构建信源画像。

数据现状（1989660417.csv）：2720 条中 target_retweet 13 条、@提及 17 条、
图片 180 条。转发/互动稀疏 → 信源画像是弱信号，只作 prompt 侧证据，
不进入权重主回路。点赞/关注数据缺失（需额外抓取，暂 blocked）。
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path
from typing import Any


def build_source_profile(csv_path: str | Path, *, top_k: int = 10) -> dict[str, Any]:
    """从用户微博 CSV 构建信源影响画像：
    - repost_sources: 转发过的源用户（常互动/信任信源代理）
    - mentions: @过的账号
    - repost_topics: 转发内容的关键词（信源议题偏好代理）
    """
    p = Path(csv_path)
    if not p.exists():
        return {"available": False}
    repost_sources: Counter[str] = Counter()
    mentions: Counter[str] = Counter()
    repost_texts: list[str] = []
    n_reposts = 0
    text = None
    for enc in ("utf-8-sig", "gb18030"):
        try:
            text = p.read_text(encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        return {"available": False, "error": "encoding"}
    import io
    with io.StringIO(text) as f:
        for row in csv.DictReader(f):
            if (row.get("记录类型") or "") == "target_retweet":
                n_reposts += 1
                src = (row.get("源用户昵称") or "").strip()
                if src:
                    repost_sources[src] += 1
                txt = (row.get("源微博正文") or "").strip()
                if txt:
                    repost_texts.append(txt[:120])
            # @用户 列直接存昵称（无 @ 前缀，可能多个用空格/逗号分隔）
            cell = (row.get("@用户") or "").strip()
            if cell:
                for name in re.split(r"[,，;；\s]+", cell):
                    name = name.strip().lstrip("@")
                    if name and len(name) <= 30:
                        mentions[name] += 1
            # 正文里的 @提及
            for m in re.findall(r"@([\w一-鿿\-·]{1,30})", row.get("正文") or ""):
                mentions[m] += 1
    return {
        "available": True,
        "n_reposts": n_reposts,
        "repost_sources": repost_sources.most_common(top_k),
        "mentions": mentions.most_common(top_k),
        "repost_samples": repost_texts[:5],
    }


def format_source_block(profile: dict[str, Any]) -> str:
    if not profile.get("available"):
        return "(无信源行为数据)"
    lines = []
    rs = profile.get("repost_sources") or []
    if rs:
        lines.append("转发过的信源（信任代理）：" + "、".join(f"{n}×{c}" for n, c in rs[:6]))
    ms = profile.get("mentions") or []
    if ms:
        lines.append("常互动账号（@）：" + "、".join(f"{n}×{c}" for n, c in ms[:6]))
    if not lines:
        lines.append("(该用户转发/互动记录稀疏，信源信号弱)")
    return "\n".join(lines)
