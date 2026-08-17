from __future__ import annotations

"""Personal reception pathway / 3D environment at posting time.

NOT inferred user personality, and NOT free-form 'public mood' speculation.

For each post we reconstruct what can be *observed or retrieved* about:
  1) information sources the user could have received
  2) how the event propagated to them (hashtag / retweet source / @ / topic page)
  3) how it was presented (headlines, frames, media form) when it reached the feed

Primary evidence:
  - CSV-native pathway fields (源微博, @用户, 话题/hashtag, 工具, URLs)
  - Contemporaneous Weibo/search snippets about THAT topic's presentation that day
LLM may only organize evidence; it must not invent sources or unseen pathways.
"""

import csv
import io
import json
import os
import re
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

from tg_agent.llm import DeepSeekClient, load_env

_SIT_FILE_LOCK = threading.Lock()

STRUCT_SYSTEM = """你是传播路径取证员，不是舆论评论员。
任务：根据【用户发帖时的可观察路径字段】+【检索到的、截止发帖时刻的信息源片段】，
还原该用户发帖前可能接触到的真实信息环境与传播路径。

时间因果硬约束（对齐实验有效性前提）：
- 只允许使用「发帖时间及之前」已发生的事件发展、传播路径、热搜/话题呈现。
- 严禁使用发帖之后的后续进展、结局、事后复盘、事后统计或后见之明。
- 检索片段若混入发帖日之后的内容，必须忽略，并在 evidence_gaps 写 post_cutoff_leakage_ignored。
- CSV 源微博若标注时间晚于发帖时间，视为不可用上游。

禁止臆造未出现在证据里的信源、传播链或网民情绪。

只输出 JSON：
{
  "observed_pathway": {
    "entry_channel": "从路径字段归纳：话题标签页/转发源博/热搜词/@提及/原创跟帖 等",
    "upstream_source": "若有源微博且不晚于发帖：作者+源博摘要；否则写 none",
    "device_context": "发帖工具等设备侧线索",
    "mentions": ["@到的账号"]
  },
  "information_sources": [
    {"title":"来自检索的标题","url":"链接","snippet":"原句摘要","role":"how presented / corroboration","as_of":"不晚于发帖日"}
  ],
  "propagation": {
    "path_to_user": "发帖前可证据支持的传播到用户路径（如：话题#X#页面呈现 → 用户跟帖）。证据不足写 unknown",
    "presentation_form": "到达时的呈现方式：话题标签/视频/图文/转发正文/标题党等（须有证据且不晚于发帖）",
    "salient_cues_in_feed": ["从源博或检索标题/摘要中抽出的可见线索，禁止自造与事后信息"]
  },
  "environment": {
    "communication": {
      "platform_climate": "仅基于发帖前证据：该议题在微博/媒体上如何被呈现与转发",
      "information_flow": "证据中的信息从哪类源流出、何种框架被置顶",
      "cues": ["短关键词"]
    },
    "psychological": {
      "public_mood": "仅当发帖前证据明确出现情绪词/对立框架时才写；否则写 evidence_insufficient",
      "salient_frames": ["发帖前证据中出现的框架"],
      "cues": []
    },
    "social": {
      "event_backdrop": "发帖前证据支持的事件/制度背景；无则 evidence_insufficient",
      "actors_groups": ["证据中出现的行动者"],
      "cues": []
    }
  },
  "theory_coordinates": ["2-6 个坐标 id，必须能被上述路径/呈现证据支撑"],
  "summary": "80字内：发帖前经由什么通道、看到怎样呈现、再发帖（不得编造或使用事后信息）",
  "evidence_gaps": ["缺失的关键证据，或已忽略的事后泄漏"]
}
可用坐标 id：risk_perception,trust,identity_threat,fairness,technology_threat,uncertainty_reduction,motivated_reasoning,social_identity,framing,agenda_setting,spiral_of_silence,inoculation,misinformation,moral_foundations,affective_polarization,selective_exposure,narrative_persuasion,cognitive_dissonance,prospect_theory,cultural_cognition,source_credibility,collective_action,third_person_effect,hostile_media,opinion_leadership,dual_process,face_culture,public_opinion_china,organizational_behavior,impression_management,developmental_media,parasocial,uses_gratifications,social_capital,macro_social_theory,public_sphere,network_society,habitus_capital,cultivation,priming,diffusion_innovation,cmc_theory,echo_chamber,algorithmic_curation,online_disinhibition,privacy_calculus,crisis_communication,health_communication,social_comparison,media_dependency,digital_divide,system_justification,terror_management,cancel_culture

硬约束：
1) information_sources 只能来自给定检索片段（可改写 title/snippet，不可发明 URL），且须不晚于发帖时间。
2) 没有证据就写 evidence_insufficient / unknown，不要脑补“网民普遍愤怒”。
3) 三维 environment 是对发帖前证据的归类，不是对用户性格的推断，也不是事后叙事。
"""


def _decode_csv(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("gb18030", errors="ignore")


def load_authored_posts(csv_path: Path, limit: int = 0) -> list[dict[str, str]]:
    text = _decode_csv(csv_path)
    rows = list(csv.DictReader(io.StringIO(text)))
    out: list[dict[str, str]] = []
    for r in rows:
        if r.get("是否目标用户作者") == "否":
            continue
        if "target_authored" not in (r.get("记录类型") or "") and r.get("是否目标用户作者") != "是":
            if r.get("是否原创") != "TRUE" and r.get("是否原创") != "True":
                # still keep retweets authored by target
                if "target_retweet" not in (r.get("记录类型") or ""):
                    continue
        date = (r.get("完整日期") or r.get("日期") or "").strip()
        body = (r.get("正文") or "").strip()
        topic = (r.get("话题") or "").strip()
        if not body and not topic and not (r.get("源微博正文") or "").strip():
            continue
        mentions = (r.get("@用户") or "").strip()
        out.append(
            {
                "post_id": str(r.get("id") or r.get("bid") or len(out)),
                "bid": str(r.get("bid") or ""),
                "date": date,
                "topic": topic,
                "text": body[:500],
                "record_type": str(r.get("记录类型") or ""),
                "is_original": str(r.get("是否原创") or ""),
                "tool": str(r.get("工具") or ""),
                "mentions": mentions,
                "article_url": str(r.get("头条文章url") or ""),
                "image_url": str(r.get("原始图片url") or ""),
                "video_url": str(r.get("视频url") or ""),
                "source_user_id": str(r.get("源用户id") or ""),
                "source_user_name": str(r.get("源用户昵称") or ""),
                "source_weibo_id": str(r.get("源微博id") or r.get("源微博bid") or ""),
                "source_text": str(r.get("源微博正文") or "")[:800],
                "source_topic": str(r.get("源微博话题") or ""),
                "source_date": str(r.get("源微博完整日期") or r.get("源微博日期") or ""),
                "source_tool": str(r.get("源微博工具") or ""),
            }
        )
        if limit and len(out) >= limit:
            break
    return out


def observed_pathway_from_post(post: dict[str, str]) -> dict[str, Any]:
    """CSV-grounded pathway cues — no LLM. Upstream later than post is dropped."""
    hashtag = _topic_of(post)
    src_name = (post.get("source_user_name") or "").strip()
    src_text = (post.get("source_text") or "").strip()
    mentions = [m for m in re.split(r"[,，\s]+", post.get("mentions") or "") if m]
    post_dk = _date_key(post.get("date") or "")
    src_dk = _date_key(post.get("source_date") or "")
    upstream_usable = bool(src_text or src_name)
    if upstream_usable and post_dk and src_dk and src_dk > post_dk:
        upstream_usable = False  # post-cutoff upstream would leak future info
    entry = []
    if hashtag:
        entry.append(f"weibo_topic_hashtag:#{hashtag}#")
    if upstream_usable:
        entry.append("retweet_or_quote_of_upstream_weibo")
    if mentions:
        entry.append("mention_directed")
    if (post.get("article_url") or "").strip():
        entry.append("headline_article_link")
    if (post.get("video_url") or "").strip():
        entry.append("video_attached")
    if not entry:
        entry.append("original_post_without_explicit_upstream")
    return {
        "entry_channel": " + ".join(entry),
        "hashtag": hashtag,
        "temporal_cutoff": post_dk,
        "upstream_source": (
            {
                "user_id": post.get("source_user_id") or "",
                "user_name": src_name,
                "weibo_id": post.get("source_weibo_id") or "",
                "text": src_text[:400],
                "topic": post.get("source_topic") or "",
                "date": post.get("source_date") or "",
            }
            if upstream_usable
            else None
        ),
        "device_context": post.get("tool") or "",
        "mentions": mentions,
        "media": {
            "article_url": post.get("article_url") or "",
            "image_url": (post.get("image_url") or "")[:120],
            "video_url": post.get("video_url") or "",
        },
        "record_type": post.get("record_type") or "",
        "is_original": post.get("is_original") or "",
    }


def serper_search(query: str, *, num: int = 6) -> list[dict[str, str]]:
    key = os.environ.get("SERPER_API_KEY", "")
    if not key or key.startswith("your_"):
        return []
    data = json.dumps({"q": query, "num": num, "gl": "cn", "hl": "zh-cn"}).encode()
    req = urllib.request.Request(
        "https://google.serper.dev/search",
        data=data,
        headers={"X-API-KEY": key, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    hits = []
    for item in body.get("organic") or []:
        hits.append(
            {
                "title": item.get("title") or "",
                "snippet": item.get("snippet") or "",
                "link": item.get("link") or "",
                "date": item.get("date") or "",
            }
        )
    return hits


def retrieve_presentation_evidence(
    post: dict[str, str],
    *,
    backend: str = "weibo_ai",
) -> tuple[list[dict[str, str]], list[str], dict[str, Any]]:
    """Retrieve how the event was presented BEFORE the post time.

    backend:
      - weibo_ai: only 智搜 (default; needs WEIBO_COOKIE)
      - serper: only Serper Google
      - both: 智搜 first, Serper only if 智搜 yields no hits
    Returns (hits, queries, meta).
    """
    backend = (backend or os.environ.get("SIT_RETRIEVAL", "weibo_ai")).strip().lower()
    topic = _topic_of(post)
    dk = _date_key(post.get("date") or "")
    hits: list[dict[str, str]] = []
    queries: list[str] = []
    meta: dict[str, Any] = {
        "retrieval": backend,
        "temporal_cutoff": dk,
        "causal_window": "as_of_or_before_posting_time",
    }

    if backend in {"weibo_ai", "both", "weibo", "aisearch"}:
        try:
            from tg_agent.weibo_ai_search import retrieve_weibo_ai_evidence

            w_hits, w_meta = retrieve_weibo_ai_evidence(
                topic, dk, post_datetime=str(post.get("date") or "")
            )
            hits.extend(w_hits)
            queries.append(str(w_meta.get("query") or topic))
            meta["weibo_ai"] = w_meta
            print(
                f"weibo_ai ok q={w_meta.get('query')!r} hits={len(w_hits)} "
                f"links={w_meta.get('link_count')} stage={w_meta.get('chat_status_stage')} "
                f"cutoff={dk} drop_post={w_meta.get('dropped_post_cutoff_hits')}",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001
            meta["weibo_ai_error"] = str(e)
            print(f"weibo_ai fail: {e}", flush=True)

    need_serper = backend == "serper" or (backend == "both" and not hits)
    if need_serper:
        qs = build_queries(post)
        queries = qs if not queries else queries + qs
        seen: set[str] = set(h.get("link") or h.get("title") or "" for h in hits)
        for q in qs:
            try:
                for h in serper_search(q, num=5):
                    key = h.get("link") or h.get("title") or ""
                    if not key or key in seen:
                        continue
                    # drop clearly post-cutoff organic dates
                    hd = _date_key(h.get("date") or "")
                    if dk and hd and hd > dk:
                        continue
                    seen.add(key)
                    h = dict(h)
                    h["source"] = "serper"
                    hits.append(h)
            except Exception as e:  # noqa: BLE001
                print(f"serper fail: {e}", flush=True)
            time.sleep(0.25)
        meta["serper_hits"] = len([h for h in hits if h.get("source") == "serper"])

    return hits, queries or [topic], meta


def _date_key(date_str: str) -> str:
    m = re.search(r"(20\d{2})[\/\-年]?(\d{1,2})[\/\-月]?(\d{1,2})?", date_str)
    if not m:
        return date_str[:10]
    y, mo, d = m.group(1), m.group(2), m.group(3) or "1"
    return f"{y}-{int(mo):02d}-{int(d):02d}"


def _source_date(src: dict[str, Any]) -> str | None:
    """Best-effort publication/cutoff date of one information source."""
    for key in ("as_of", "date", "published", "pub_date", "time"):
        v = src.get(key)
        if v:
            dk = _date_key(str(v))
            if re.match(r"20\d{2}-\d{2}-\d{2}$", dk):
                return dk
    # Weibo pipeline stores the date inside the `role` narrative.
    for key in ("role", "snippet", "title"):
        v = src.get(key)
        if v:
            dk = _date_key(str(v))
            if re.match(r"20\d{2}-\d{2}-\d{2}$", dk):
                return dk
    return None


def _enforce_temporal_cutoff(
    post: dict[str, Any], sources: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Drop sources published after the post; flag sources whose date is unknown."""
    post_dk = _date_key(str(post.get("date") or ""))
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for src in sources or []:
        if not isinstance(src, dict):
            continue
        sd = _source_date(src)
        if sd is None:
            src = dict(src)
            src["date_unknown"] = True
            kept.append(src)
        elif post_dk and sd > post_dk:
            dropped.append(
                {
                    "title": src.get("title") or "",
                    "url": src.get("url") or src.get("link") or "",
                    "date": sd,
                    "role": "removed_later_than_cutoff",
                }
            )
        else:
            kept.append(src)
    return kept, dropped


def _topic_of(post: dict[str, str]) -> str:
    topic = (post.get("topic") or "").strip()
    if not topic:
        hm = re.search(r"#([^#]+)#", post.get("text") or "")
        topic = hm.group(1).strip() if hm else (post.get("text") or "")[:24]
    topic = topic.split(",")[0].strip()
    return topic[:80]


def env_slot_key(post: dict[str, str]) -> str:
    return f"{_date_key(post.get('date') or '')}||{_topic_of(post)}"


def build_queries(post: dict[str, str]) -> list[str]:
    """Retrieve how the topic/event was presented on channels BEFORE the post."""
    dk = _date_key(post.get("date") or "")
    topic = _topic_of(post).replace(",", " ").strip() or "微博"
    src_name = (post.get("source_user_name") or "").strip()
    # before:YYYY-MM-DD biases search engines toward pre-cutoff pages when supported
    before = f"before:{dk}" if dk else ""
    qs = [
        f"site:weibo.com {topic} {dk} {before}".strip(),
        f"{topic} {dk} 微博 热搜 传播 {before}".strip(),
        f"{topic} 截至{dk} 微博话题 呈现".strip(),
        f"{topic} {dk} 发酵 OR 热议 {before}".strip(),
    ]
    if src_name:
        qs.insert(0, f"{src_name} {topic} {dk} 微博 {before}".strip())
    return qs


def structure_env(llm: DeepSeekClient, post: dict[str, str], hits: list[dict[str, str]]) -> dict[str, Any]:
    pathway = observed_pathway_from_post(post)
    cutoff = _date_key(post.get("date") or "")
    evidence = "\n".join(
        f"- [{h.get('date','')}] {h.get('title','')}\n  url: {h.get('link','')}\n  snippet: {h.get('snippet','')}"
        + (f"\n  temporal_warning: {h.get('temporal_warning')}" if h.get("temporal_warning") else "")
        for h in hits[:10]
    ) or "- （无检索片段）"
    upstream = pathway.get("upstream_source")
    upstream_txt = (
        json.dumps(upstream, ensure_ascii=False)
        if upstream
        else "none（CSV 无源微博字段）"
    )
    user = (
        f"发帖时间: {post.get('date')}\n"
        f"时间截止(cutoff): {cutoff} —— 只允许使用该日及以前已发生的信息；禁止事后进展\n"
        f"用户正文: {post.get('text')}\n"
        f"话题字段: {post.get('topic')}\n"
        f"记录类型/是否原创: {post.get('record_type')} / {post.get('is_original')}\n"
        f"工具: {post.get('tool')}\n"
        f"@用户: {post.get('mentions')}\n"
        f"CSV可观察路径: {json.dumps(pathway, ensure_ascii=False)}\n"
        f"上游源微博: {upstream_txt}\n\n"
        f"截止发帖前检索到的信息源/呈现片段（只能用这些且须不晚于 cutoff，禁止编造）:\n{evidence}\n"
    )
    raw = llm.chat(
        [{"role": "system", "content": STRUCT_SYSTEM}, {"role": "user", "content": user}],
        temperature=0.1,
        max_tokens=1600,
        disable_thinking=True,
    ).strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    obj = _parse_json_obj(raw)
    if not obj:
        raw2 = llm.chat(
            [
                {"role": "system", "content": STRUCT_SYSTEM},
                {
                    "role": "user",
                    "content": user
                    + "\n\n上一轮 JSON 解析失败。请只输出一个合法 JSON 对象，不要额外文字；无证据处写 evidence_insufficient。",
                },
            ],
            temperature=0.0,
            max_tokens=1400,
            disable_thinking=True,
        ).strip()
        if raw2.startswith("```"):
            raw2 = re.sub(r"^```(?:json)?\s*", "", raw2)
            raw2 = re.sub(r"\s*```$", "", raw2)
        obj = _parse_json_obj(raw2)
    if not obj:
        obj = {
            "observed_pathway": pathway,
            "information_sources": [],
            "propagation": {
                "path_to_user": "unknown",
                "presentation_form": "evidence_insufficient",
                "salient_cues_in_feed": [],
            },
            "environment": {
                "communication": {
                    "platform_climate": "evidence_insufficient",
                    "information_flow": "",
                    "cues": [],
                },
                "psychological": {
                    "public_mood": "evidence_insufficient",
                    "salient_frames": [],
                    "cues": [],
                },
                "social": {
                    "event_backdrop": "evidence_insufficient",
                    "actors_groups": [],
                    "cues": [],
                },
            },
            "theory_coordinates": ["framing", "agenda_setting"] if _topic_of(post) else [],
            "summary": "模型输出无法解析；仅保留 CSV 可观察路径",
            "evidence_gaps": ["llm_parse_failed"],
        }
    # always keep CSV pathway as ground truth overlay
    obj["observed_pathway_csv"] = pathway
    if not obj.get("observed_pathway"):
        obj["observed_pathway"] = pathway
    # attach raw hits for audit
    obj.setdefault(
        "information_sources",
        [
            {
                "title": h.get("title"),
                "url": h.get("link"),
                "snippet": h.get("snippet"),
                "date": h.get("date"),
                "role": "retrieved",
            }
            for h in hits[:8]
        ],
    )
    # Hard temporal enforcement: the LLM is instructed to respect the cutoff but
    # may still emit an as_of later than the post. Filter those out here.
    kept, dropped = _enforce_temporal_cutoff(post, obj.get("information_sources") or [])
    obj["information_sources"] = kept
    gaps = list(obj.get("evidence_gaps") or [])
    if dropped:
        gaps.append(f"removed_{len(dropped)}_sources_later_than_cutoff")
        obj.setdefault("removed_sources", []).extend(dropped)
    unknown = sum(1 for s in kept if s.get("date_unknown"))
    if unknown:
        gaps.append(f"kept_{unknown}_sources_with_unknown_date")
    obj["evidence_gaps"] = gaps
    return obj


def _parse_json_obj(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        pass
    # first balanced object
    start = raw.find("{")
    if start < 0:
        return {}
    depth = 0
    for i in range(start, len(raw)):
        ch = raw[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                chunk = raw[start : i + 1]
                try:
                    obj = json.loads(chunk)
                    return obj if isinstance(obj, dict) else {}
                except json.JSONDecodeError:
                    return {}
    return {}


def _load_existing(out_path: Path) -> dict[str, Any]:
    if not out_path.exists():
        return {
            "user_id": "",
            "kind": "situational_3d_environment",
            "definition": (
                "External communication/psychological/social context at posting time; "
                "not extracted user personality."
            ),
            "records": [],
            "slots": {},
        }
    data = json.loads(out_path.read_text(encoding="utf-8"))
    data.setdefault("records", [])
    data.setdefault("slots", {})
    # rebuild slots from records if missing
    if not data["slots"]:
        for r in data["records"]:
            sk = r.get("slot_key") or env_slot_key(r)
            if sk not in data["slots"]:
                data["slots"][sk] = {
                    "slot_key": sk,
                    "environment": r.get("environment"),
                    "theory_coordinates": r.get("theory_coordinates"),
                    "summary": r.get("summary"),
                    "web_evidence": r.get("web_evidence"),
                    "search_queries": r.get("search_queries"),
                }
    return data


def load_situational_store(path: str | Path) -> dict[str, Any]:
    """Load situational 3D env file and index by post_id / bid / slot."""
    path = Path(path)
    if not path.exists():
        return {"user_id": "", "records": [], "by_post_id": {}, "by_bid": {}, "by_slot": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    by_post: dict[str, dict[str, Any]] = {}
    by_bid: dict[str, dict[str, Any]] = {}
    by_slot: dict[str, dict[str, Any]] = {}
    for r in data.get("records") or []:
        pid = str(r.get("post_id") or "").strip()
        bid = str(r.get("bid") or "").strip()
        if pid:
            by_post[pid] = r
        if bid:
            by_bid[bid] = r
        sk = r.get("slot_key") or env_slot_key(
            {"date": str(r.get("date") or ""), "topic": str(r.get("topic") or ""), "text": str(r.get("text") or "")}
        )
        by_slot.setdefault(sk, r)
    return {
        "user_id": str(data.get("user_id") or ""),
        "kind": data.get("kind") or "situational_3d_environment",
        "records": data.get("records") or [],
        "by_post_id": by_post,
        "by_bid": by_bid,
        "by_slot": by_slot,
        "path": str(path),
    }


def resolve_situational(
    store: dict[str, Any] | None,
    *,
    post_id: str | None = None,
    bid: str | None = None,
    date: str | None = None,
    topic: str | None = None,
    text: str | None = None,
) -> dict[str, Any] | None:
    if not store:
        return None
    pid = str(post_id or "").strip()
    if pid and pid in (store.get("by_post_id") or {}):
        return store["by_post_id"][pid]
    b = str(bid or "").strip()
    if b and b in (store.get("by_bid") or {}):
        return store["by_bid"][b]
    if date or topic or text:
        sk = env_slot_key({"date": date or "", "topic": topic or "", "text": text or ""})
        hit = (store.get("by_slot") or {}).get(sk)
        if hit:
            return hit
    return None


def situational_env_weights(
    record: dict[str, Any] | None,
    *,
    boost: float = 1.85,
) -> dict[str, float]:
    """Coordinate priors from situational theory_coordinates (not trait env).

    theory_lib soft-downweights coords absent from this map (≈0.72) when the map is non-empty.
    """
    if not record:
        return {}
    coords = [str(c) for c in (record.get("theory_coordinates") or []) if c]
    return {c: float(boost) for c in coords}


def format_situational_block(record: dict[str, Any] | None) -> str:
    if not record:
        return "(no personal reception pathway for this post)"
    path = record.get("observed_pathway_csv") or record.get("observed_pathway") or {}
    prop = record.get("propagation") or {}
    env = record.get("environment") or {}
    if not env and isinstance(record.get("communication"), dict):
        # legacy flat schema
        env = {
            "communication": record.get("communication") or {},
            "psychological": record.get("psychological") or {},
            "social": record.get("social") or {},
        }
    comm = env.get("communication") or {}
    psych = env.get("psychological") or {}
    social = env.get("social") or {}
    coords = record.get("theory_coordinates") or []
    sources = record.get("information_sources") or record.get("web_evidence") or []
    src_lines = []
    for s in sources[:5]:
        if isinstance(s, dict):
            src_lines.append(
                f"- {s.get('title') or ''} | {s.get('url') or s.get('link') or ''} | "
                f"{(s.get('snippet') or '')[:100]}"
            )
    upstream = path.get("upstream_source") if isinstance(path, dict) else None
    up_txt = "none"
    if isinstance(upstream, dict) and (upstream.get("text") or upstream.get("user_name")):
        up_txt = f"{upstream.get('user_name','')}: {(upstream.get('text') or '')[:160]}"
    gaps = record.get("evidence_gaps") or []
    return (
        f"summary: {record.get('summary') or ''}\n"
        f"observed_entry: {path.get('entry_channel') if isinstance(path, dict) else ''}\n"
        f"hashtag: {path.get('hashtag') if isinstance(path, dict) else ''}\n"
        f"upstream_source: {up_txt}\n"
        f"propagation_path: {prop.get('path_to_user', '')}\n"
        f"presentation_form: {prop.get('presentation_form', '')}\n"
        f"feed_cues: {', '.join((prop.get('salient_cues_in_feed') or [])[:8])}\n"
        f"retrieved_sources:\n{chr(10).join(src_lines) or '- (none)'}\n"
        f"communication(evidence-bound): {comm.get('platform_climate', '')}\n"
        f"psychological(evidence-bound): {psych.get('public_mood', '')}\n"
        f"social(evidence-bound): {social.get('event_backdrop', '')}\n"
        f"theory_coordinates: {', '.join(coords[:8])}\n"
        f"evidence_gaps: {', '.join(gaps[:6])}"
    )


def _normalize_structured(
    post: dict[str, str],
    structured: dict[str, Any],
    hits: list[dict[str, str]],
    queries: list[str],
    retrieval_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Unify LLM output + CSV pathway into one record body."""
    pathway = observed_pathway_from_post(post)
    env = structured.get("environment")
    if not isinstance(env, dict):
        # legacy / flat
        env = {
            "communication": structured.get("communication") or {},
            "psychological": structured.get("psychological") or {},
            "social": structured.get("social") or {},
        }
    sources = structured.get("information_sources")
    if not isinstance(sources, list) or not sources:
        sources = [
            {
                "title": h.get("title"),
                "url": h.get("link"),
                "snippet": h.get("snippet"),
                "date": h.get("date"),
                "role": "retrieved",
                "source": h.get("source") or "retrieved",
            }
            for h in hits[:8]
        ]
    kept, dropped = _enforce_temporal_cutoff(post, sources)
    gaps = list(structured.get("evidence_gaps") or [])
    if dropped:
        gaps.append(f"removed_{len(dropped)}_sources_later_than_cutoff")
    unknown = sum(1 for s in kept if s.get("date_unknown"))
    if unknown:
        gaps.append(f"kept_{unknown}_sources_with_unknown_date")
    meta = retrieval_meta or {}
    return {
        "observed_pathway_csv": pathway,
        "observed_pathway": structured.get("observed_pathway") or pathway,
        "propagation": structured.get("propagation") or {},
        "information_sources": kept,
        "environment": env,
        "theory_coordinates": structured.get("theory_coordinates") or [],
        "summary": structured.get("summary") or "",
        "evidence_gaps": gaps,
        "removed_sources": dropped,
        "search_queries": queries,
        "web_evidence": hits[:10],
        "retrieval_meta": meta,
        "kind": "personal_reception_pathway",
    }


def build_situational_envs(
    *,
    user_id: str,
    csv_path: Path,
    env_file: str,
    out_path: Path,
    llm: DeepSeekClient,
    limit: int = 0,
    sleep_s: float = 0.8,
    priority_post_ids: set[str] | None = None,
    dedupe_slots: bool = True,
    retrieval: str = "weibo_ai",
) -> dict[str, Any]:
    load_env(env_file)
    posts = load_authored_posts(csv_path, limit=0)
    if priority_post_ids:
        pri = [p for p in posts if p["post_id"] in priority_post_ids or p["bid"] in priority_post_ids]
        rest = [p for p in posts if p["post_id"] not in priority_post_ids and p["bid"] not in priority_post_ids]
        posts = pri + rest
        print(f"priority_posts={len(pri)} rest={len(rest)}", flush=True)
    if limit:
        posts = posts[:limit]

    payload = _load_existing(out_path)
    payload["user_id"] = user_id
    payload["kind"] = "personal_reception_pathway"
    payload["definition"] = (
        "Evidence-bound personal reception pathway at posting time: "
        "CSV upstream/hashtag/@ + Weibo 智搜 (or Serper) presentation sources. "
        "Built one post/slot at a time in chronological order — not batch-inferred."
    )
    payload["retrieval"] = retrieval
    done_posts = {str(r.get("post_id")) for r in payload["records"]}
    slots: dict[str, Any] = dict(payload.get("slots") or {})

    todo_posts = [p for p in posts if p["post_id"] not in done_posts]
    print(
        f"resume: done_posts={len(done_posts)} slots={len(slots)} todo={len(todo_posts)} "
        f"retrieval={retrieval}",
        flush=True,
    )

    for i, post in enumerate(todo_posts, 1):
        sk = env_slot_key(post)
        if dedupe_slots and sk in slots:
            slot = slots[sk]
            body = {k: slot.get(k) for k in (
                "observed_pathway_csv", "observed_pathway", "propagation",
                "information_sources", "environment", "theory_coordinates",
                "summary", "evidence_gaps", "search_queries", "web_evidence",
                "retrieval_meta", "kind",
            )}
            body["observed_pathway_csv"] = observed_pathway_from_post(post)
            rec = {
                "user_id": user_id,
                "post_id": post["post_id"],
                "bid": post["bid"],
                "date": post["date"],
                "topic": _topic_of(post),
                "text": post["text"],
                "slot_key": sk,
                **{k: v for k, v in body.items() if v is not None},
                "kind": "personal_reception_pathway",
                "reused_slot": True,
            }
        else:
            hits, queries, rmeta = retrieve_presentation_evidence(post, backend=retrieval)
            structured = structure_env(llm, post, hits)
            body = _normalize_structured(post, structured, hits, queries, rmeta)
            slot = {"slot_key": sk, "date_key": _date_key(post["date"]), "topic": _topic_of(post), **body}
            slots[sk] = slot
            rec = {
                "user_id": user_id,
                "post_id": post["post_id"],
                "bid": post["bid"],
                "date": post["date"],
                "topic": _topic_of(post),
                "text": post["text"],
                "slot_key": sk,
                **body,
                "reused_slot": False,
            }
            time.sleep(sleep_s)

        payload["records"].append(rec)
        payload["slots"] = slots
        payload["num_posts"] = len(payload["records"])
        payload["num_slots"] = len(slots)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"[{i}/{len(todo_posts)}] posts={payload['num_posts']} slots={len(slots)} "
            f"{post['date'][:16]} reuse={rec.get('reused_slot')} "
            f"entry={(rec.get('observed_pathway_csv') or {}).get('entry_channel','')[:40]} "
            f"coords={(rec.get('theory_coordinates') or [])[:4]} "
            f"sum={(rec.get('summary') or '')[:36]}",
            flush=True,
        )
    return payload


def event_to_post(event: dict[str, Any], csv_row: dict[str, str] | None = None) -> dict[str, str]:
    """Convert event (+ optional CSV row) into pathway-aware post fields."""
    ts = float(event.get("timestamp") or 0.0)
    if ts > 0:
        date = time.strftime("%Y/%m/%d %H:%M", time.localtime(ts))
    else:
        date = str(event.get("date") or "")
    topics = event.get("topics") or []
    topic = str(topics[0]) if topics else str(event.get("topic_hashtag") or "").strip("#")
    text = str(
        event.get("raw_text") or event.get("user_opinion") or event.get("event_summary") or ""
    )
    pid = str(event.get("post_id") or event.get("bid") or "").strip()
    bid = str(event.get("bid") or event.get("post_id") or "").strip()
    base = {
        "post_id": pid,
        "bid": bid,
        "date": date,
        "topic": topic,
        "text": text[:500],
        "record_type": "",
        "is_original": "",
        "tool": str(event.get("tool") or ""),
        "mentions": "",
        "article_url": "",
        "image_url": "",
        "video_url": "",
        "source_user_id": "",
        "source_user_name": "",
        "source_weibo_id": "",
        "source_text": "",
        "source_topic": "",
        "source_date": "",
        "source_tool": "",
    }
    if csv_row:
        for k in base:
            if k in {"post_id", "bid"}:
                continue
            if csv_row.get(k):
                base[k] = csv_row[k]
        if csv_row.get("date"):
            base["date"] = csv_row["date"]
        if csv_row.get("topic"):
            base["topic"] = csv_row["topic"]
        if csv_row.get("text"):
            base["text"] = csv_row["text"][:500]
    return base


def ensure_situational_for_post(
    *,
    user_id: str,
    post: dict[str, str],
    out_path: Path,
    llm: DeepSeekClient,
    env_file: str | None = None,
    sleep_s: float = 0.35,
    retrieval: str = "weibo_ai",
) -> dict[str, Any]:
    """Build or reuse one post's personal reception pathway (chrono / on-demand).

    Always one post at a time — never batch-generate future posts' envs here.
    """
    if env_file:
        load_env(env_file)
    out_path = Path(out_path)
    pid = str(post.get("post_id") or "").strip()
    bid = str(post.get("bid") or "").strip()
    retrieval = (retrieval or os.environ.get("SIT_RETRIEVAL", "weibo_ai")).strip()

    with _SIT_FILE_LOCK:
        payload = _load_existing(out_path)
        payload["user_id"] = user_id
        payload["kind"] = "personal_reception_pathway"
        payload["retrieval"] = retrieval
        for r in payload.get("records") or []:
            if pid and str(r.get("post_id") or "").strip() == pid:
                return r
            if bid and str(r.get("bid") or "").strip() == bid:
                return r
        slots: dict[str, Any] = dict(payload.get("slots") or {})
        sk = env_slot_key(post)
        if sk in slots:
            slot = slots[sk]
            body = {k: slot.get(k) for k in (
                "observed_pathway_csv", "observed_pathway", "propagation",
                "information_sources", "environment", "theory_coordinates",
                "summary", "evidence_gaps", "search_queries", "web_evidence",
                "retrieval_meta", "kind",
            )}
            body["observed_pathway_csv"] = observed_pathway_from_post(post)
            rec = {
                "user_id": user_id,
                "post_id": pid or bid,
                "bid": bid or pid,
                "date": post.get("date") or "",
                "topic": _topic_of(post),
                "text": post.get("text") or "",
                "slot_key": sk,
                **{k: v for k, v in body.items() if v is not None},
                "kind": "personal_reception_pathway",
                "reused_slot": True,
            }
            payload.setdefault("records", []).append(rec)
            payload["slots"] = slots
            payload["num_posts"] = len(payload["records"])
            payload["num_slots"] = len(slots)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            return rec

    # Retrieve + structure OUTSIDE the file lock so chronological callers aren't blocked,
    # and we never prefetch unrelated future posts from this function.
    hits, queries, rmeta = retrieve_presentation_evidence(post, backend=retrieval)
    structured = structure_env(llm, post, hits)
    body = _normalize_structured(post, structured, hits, queries, rmeta)
    time.sleep(sleep_s)

    with _SIT_FILE_LOCK:
        payload = _load_existing(out_path)
        payload["user_id"] = user_id
        payload["kind"] = "personal_reception_pathway"
        payload["retrieval"] = retrieval
        for r in payload.get("records") or []:
            if pid and str(r.get("post_id") or "").strip() == pid:
                return r
            if bid and str(r.get("bid") or "").strip() == bid:
                return r
        slots = dict(payload.get("slots") or {})
        sk = env_slot_key(post)
        slots[sk] = {"slot_key": sk, "date_key": _date_key(post.get("date") or ""), "topic": _topic_of(post), **body}
        rec = {
            "user_id": user_id,
            "post_id": pid or bid,
            "bid": bid or pid,
            "date": post.get("date") or "",
            "topic": _topic_of(post),
            "text": post.get("text") or "",
            "slot_key": sk,
            **body,
            "reused_slot": False,
        }
        payload.setdefault("records", []).append(rec)
        payload["slots"] = slots
        payload["num_posts"] = len(payload["records"])
        payload["num_slots"] = len(slots)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return rec
