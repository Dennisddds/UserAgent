"""Weibo 智搜 (AI Search) retrieval for event background + link evidence.

Wraps the logic from ``weibo-ai-search/scripts/fetch_weibo_ai_search.py``
into an importable helper that returns presentation evidence hits.
Requires ``WEIBO_COOKIE`` (logged-in) in the environment.
"""

from __future__ import annotations

import html as html_lib
import json
import os
import re
import time
import urllib.parse
from typing import Any

import requests
from lxml import html

AI_ENDPOINT = "https://ai.s.weibo.com/api/wis/show.json"


def clean_msg(msg: str) -> str:
    text = re.sub(r"<think>.*?</think>\s*", "", msg, flags=re.S)
    text = re.sub(r"```wbCustomBlock.*?```", "", text, flags=re.S)
    text = re.sub(r"<media-block>.*?</media-block>", "", text, flags=re.S)
    text = re.sub(r"<[^>]+>", "", text)
    text = html_lib.unescape(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _add_cookie(session: requests.Session, cookie: str) -> None:
    for part in cookie.split(";"):
        if "=" not in part:
            continue
        key, value = part.strip().split("=", 1)
        for domain in [".weibo.com", ".s.weibo.com", ".ai.s.weibo.com"]:
            session.cookies.set(key, value, domain=domain)


def parse_date_key(date_str: str) -> str:
    """Normalize to YYYY-MM-DD if possible."""
    m = re.search(r"(20\d{2})[\/\-年]?(\d{1,2})[\/\-月]?(\d{1,2})?", date_str or "")
    if not m:
        return ""
    y, mo, d = m.group(1), m.group(2), m.group(3) or "1"
    return f"{y}-{int(mo):02d}-{int(d):02d}"


def build_aisearch_topic(topic: str, date_key: str = "", post_datetime: str = "") -> str:
    """Temporal 智搜 query: event development + propagation STRICTLY as-of posting time.

    Must not solicit post-hoc retrospectives after the user's post.
    """
    t = (topic or "").replace("#", "").strip() or "微博"
    t = t.split(",")[0].strip()[:60]
    dk = date_key if (date_key and re.match(r"20\d{2}-\d{2}-\d{2}", date_key)) else parse_date_key(post_datetime)
    if dk:
        y, m, d = dk.split("-")
        # Explicit causal cutoff for alignment validity
        return (
            f"{t} 截至{y}年{int(m)}月{int(d)}日之前的事件发展与微博传播路径"
            f"（只写该日及以前已发生的信息、热搜/话题呈现与转发链；"
            f"禁止{y}年{int(m)}月{int(d)}日之后的后续进展、结局或事后复盘）"
        )
    return f"{t} 发帖当时可见的事件发展与传播路径（禁止事后信息）"


def _parse_hit_date(s: str) -> str:
    return parse_date_key(s or "")


def filter_hits_before_cutoff(
    hits: list[dict[str, str]],
    cutoff_date: str,
) -> tuple[list[dict[str, str]], int]:
    """Drop evidence whose explicit date is after the posting day (keep undated)."""
    if not cutoff_date:
        return hits, 0
    kept: list[dict[str, str]] = []
    dropped = 0
    for h in hits:
        hd = _parse_hit_date(h.get("date") or "")
        if hd and hd > cutoff_date:
            dropped += 1
            continue
        # Scrub snippets that loudly announce later years than cutoff year
        snip = h.get("snippet") or ""
        title = h.get("title") or ""
        cy = int(cutoff_date[:4])
        later_years = re.findall(r"(20[2-3]\d)", snip + " " + title)
        if any(int(y) > cy for y in later_years) and h.get("source") == "weibo_ai_msg":
            # soft-keep but mark; structure LLM must respect cutoff
            h = dict(h)
            h["temporal_warning"] = "may_contain_post_cutoff_years"
        kept.append(h)
    return kept, dropped


def fetch_weibo_ai_search(
    topic: str,
    *,
    cookie: str | None = None,
    max_loops: int = 12,
    sleep_seconds: float = 1.5,
) -> dict[str, Any]:
    cookie = (cookie if cookie is not None else os.environ.get("WEIBO_COOKIE", "")).strip()
    if not cookie:
        raise RuntimeError("WEIBO_COOKIE missing — logged-in Weibo cookie required for 智搜")

    encoded = urllib.parse.quote(topic)
    page_url = f"https://s.weibo.com/aisearch?q={encoded}&Refer=aisearch_aisearch"
    session = requests.Session()
    _add_cookie(session, cookie)

    base_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36"
        ),
        "Referer": page_url,
        "Origin": "https://s.weibo.com",
        "X-Requested-With": "XMLHttpRequest",
    }

    page = session.get(page_url, headers={**base_headers, "Accept": "text/html"}, timeout=30)
    page.raise_for_status()
    doc = html.fromstring(page.text)
    nodes = doc.xpath('//*[contains(concat(" ", normalize-space(@class), " "), " zhishou_tab ")]')
    attrs = dict(nodes[0].attrib) if nodes else {}

    request_id = int(time.time())
    request_time = 0
    model = ""
    last_msg = ""
    final: dict[str, Any] = {}

    for loop_num in range(1, max_loops + 1):
        params: dict[str, Any] = {
            "query": topic,
            "content_type": "loop",
            "request_id": request_id,
            "request_time": request_time,
            "search_source": "default_init",
            "sid": "pc_search",
            "vstyle": "1",
            "cot": "1",
            "speed": "full",
            "loop_num": loop_num,
        }
        if attrs.get("data-pageid"):
            params["page_id"] = attrs["data-pageid"]
        if attrs.get("data-queryid"):
            params["query_id"] = attrs["data-queryid"]
        if model:
            params["model"] = model

        response = session.post(
            AI_ENDPOINT,
            headers={
                **base_headers,
                "Accept": "application/json, text/plain, */*",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data=params,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        final = data

        if request_time <= 0:
            request_time = data.get("current_time") or 0
        model = data.get("model") or model
        msg = data.get("msg") or ""

        if msg and msg == last_msg and loop_num >= 3:
            break
        if data.get("chat_status_stage") == 4 and len(msg) > 0:
            break
        last_msg = msg
        time.sleep(sleep_seconds)

    return final


def weibo_ai_to_hits(data: dict[str, Any], *, topic: str = "") -> list[dict[str, str]]:
    """Convert 智搜 JSON into evidence hits used by pathway structuring."""
    hits: list[dict[str, str]] = []
    clean = clean_msg(data.get("msg") or "")
    if clean:
        # chunk background into snippet-sized evidence pieces
        chunks = [c.strip() for c in re.split(r"\n{2,}", clean) if c.strip()]
        for i, chunk in enumerate(chunks[:4]):
            hits.append(
                {
                    "title": f"微博智搜背景[{i+1}]: {topic or data.get('display_query') or ''}"[:120],
                    "snippet": chunk[:500],
                    "link": "https://s.weibo.com/aisearch",
                    "date": "",
                    "source": "weibo_ai_msg",
                }
            )

    for item in data.get("link_list") or []:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or item.get("link") or item.get("scheme") or "").strip()
        title = str(item.get("title") or item.get("name") or item.get("text") or "").strip()
        snippet = str(
            item.get("desc") or item.get("summary") or item.get("content") or item.get("text") or ""
        ).strip()
        if not url and not title:
            continue
        hits.append(
            {
                "title": title[:200] or url,
                "snippet": snippet[:400],
                "link": url,
                "date": str(item.get("date") or item.get("time") or ""),
                "source": "weibo_ai_link",
            }
        )

    tips_raw = data.get("reliable_tips")
    if tips_raw:
        try:
            tips = json.loads(tips_raw) if isinstance(tips_raw, str) else tips_raw
        except Exception:  # noqa: BLE001
            tips = {"raw": tips_raw}
        hits.append(
            {
                "title": "微博智搜信源构成 reliable_tips",
                "snippet": json.dumps(tips, ensure_ascii=False)[:500],
                "link": "https://s.weibo.com/aisearch",
                "date": "",
                "source": "weibo_ai_reliable_tips",
            }
        )
    return hits


def retrieve_weibo_ai_evidence(
    topic: str,
    date_key: str = "",
    *,
    post_datetime: str = "",
    cookie: str | None = None,
    max_loops: int = 12,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    dk = date_key if (date_key and re.match(r"20\d{2}-\d{2}-\d{2}", date_key)) else parse_date_key(post_datetime)
    q = build_aisearch_topic(topic, dk, post_datetime=post_datetime)
    data = fetch_weibo_ai_search(q, cookie=cookie, max_loops=max_loops)
    hits = weibo_ai_to_hits(data, topic=q)
    hits, n_drop = filter_hits_before_cutoff(hits, dk)
    meta = {
        "query": q,
        "display_query": data.get("display_query") or data.get("query") or q,
        "chat_status_stage": data.get("chat_status_stage"),
        "status_stage": data.get("status_stage"),
        "reference_num": data.get("reference_num"),
        "link_count": len(data.get("link_list") or []),
        "msg_clean": clean_msg(data.get("msg") or "")[:2000],
        "retrieval": "weibo_ai_search",
        "temporal_cutoff": dk,
        "dropped_post_cutoff_hits": n_drop,
        "causal_window": "as_of_or_before_posting_time",
    }
    return hits, meta
