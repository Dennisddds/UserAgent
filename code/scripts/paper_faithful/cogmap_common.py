# -*- coding: utf-8 -*-
"""Shared infrastructure for per-paper KG builders.

This module contains ONLY method-agnostic infrastructure:
  * data loading (train post ids, extracted events, persona)
  * the verified retrieval vectorizer (identical to the original pipeline)
  * generic Chinese lexical helpers (sentiment/emotion word lists, tokenization)
  * memory-bank assembly / backup / writing

No method-specific graph logic lives here: every paper's construction is
implemented in its own module and never reads another method's output.
"""

import hashlib
import json
import math
import re
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # UserAgent/
OUT = ROOT / "outputs"
DIM = 512
USERS = ["1989660417", "7463374646"]

# ----------------------------------------------------------------------------
# Vectorizer (reverse-engineered from the original index; verified cos~=1.0)
# ----------------------------------------------------------------------------

def encode_text(text, dim=DIM):
    s = "".join(text.split()).lower()
    v = [0.0] * dim
    for j in range(len(s) - 2):
        h = int(hashlib.md5(s[j:j + 3].encode("utf-8")).hexdigest(), 16)
        v[h % dim] += 1.0 if (h >> 8) & 1 else -1.0
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n else v

# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------

def load_events(uid):
    """Train-split events in train order (post_id join, no test leakage)."""
    order = []
    with open(OUT / f"weibo_user_{uid}" / "train.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                order.append(json.loads(line)["post_id"])
    ev_by_id = {}
    with open(OUT / f"weibo_user_{uid}" / "events_all.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ev = json.loads(line)
                ev_by_id[ev["post_id"]] = ev
    seen = set()
    events = []
    for pid in order:
        if pid in ev_by_id and pid not in seen:
            events.append(ev_by_id[pid])
            seen.add(pid)
    return events


def load_persona(uid):
    return json.loads((OUT / f"weibo_user_{uid}" / "persona.json").read_text(encoding="utf-8"))

# ----------------------------------------------------------------------------
# Generic Chinese lexical helpers
# ----------------------------------------------------------------------------

POS_WORDS = ("支持|赞|点赞|赞赏|赞扬|佩服|敬佩|尊重|骄傲|自豪|欣慰|欣赏|了不起|好事|进步|"
             "正能量|太牛|致敬|期望|希望|相信|乐见|喝彩|叫好|感动|温暖|振奋|可贵|难得|"
             "爱国|正义|公正|合理|值得肯定|积极|向好|靠谱|真诚|勇敢|祝贺|恭喜|优秀|伟大")
NEG_WORDS = ("批评|质疑|反对|谴责|愤怒|气愤|失望|担忧|忧虑|遗憾|可耻|可悲|荒唐|荒谬|歪风|"
             "恶劣|错误|悲剧|愚蠢|无耻|丑陋|虚伪|双标|霸凌|挑衅|抹黑|造谣|欺骗|不公|"
             "不满|警惕|反感|痛心|寒心|讽刺|嘲讽|不该|不应|过分|离谱|危险|威胁|谎言")
_POS = re.compile(POS_WORDS)
_NEG = re.compile(NEG_WORDS)


def sentiment(text):
    """-1 / 0 / +1 lexicon polarity."""
    if not text:
        return 0
    p, n = len(_POS.findall(text)), len(_NEG.findall(text))
    return 1 if p > n else (-1 if n > p else 0)


def sentiment_score(text):
    """Continuous polarity in [-1, 1]."""
    if not text:
        return 0.0
    p, n = len(_POS.findall(text)), len(_NEG.findall(text))
    return (p - n) / (p + n) if (p + n) else 0.0


# Plutchik's 8 basic emotions (Chinese lexicon)
PLUTCHIK = {
    "anger": "愤怒|气愤|恼火|怒|可恨|激怒|抗议|谴责|痛斥",
    "fear": "恐惧|害怕|恐慌|担心|担忧|忧虑|不安|警惕|风险|威胁",
    "sadness": "悲伤|悲痛|痛心|遗憾|哀悼|惋惜|难过|心酸|凄凉",
    "joy": "高兴|喜悦|开心|欣慰|喜讯|可喜|祝贺|快乐|欢乐|振奋",
    "disgust": "厌恶|恶心|反感|鄙视|唾弃|可耻|丑陋|肮脏|无耻",
    "surprise": "惊讶|震惊|意外|没想到|出乎意料|吃惊|惊人|竟然",
    "trust": "信任|相信|信心|可靠|靠谱|支持|拥护|依靠|放心",
    "anticipation": "期待|期望|希望|盼望|展望|预计|即将|未来|前景",
}
_PLUTCHIK_RE = {k: re.compile(v) for k, v in PLUTCHIK.items()}


def emotion_counts(text):
    return {k: len(rx.findall(text or "")) for k, rx in _PLUTCHIK_RE.items()}


def emotion_level(count, total_chars):
    """Map raw counts to the 4-level scheme none/low/medium/high."""
    if count == 0:
        return "none"
    dens = count / max(total_chars, 1) * 100
    if dens < 0.5:
        return "low"
    return "medium" if dens < 1.5 else "high"


_MENTION = re.compile(r"@([\w\u4e00-\u9fff·\-]{2,20})")
_REPOST = re.compile(r"//@([\w\u4e00-\u9fff·\-]{2,20})")
_URL = re.compile(r"https?://\S+|t\.cn/\S+")


def mentions(text):
    """All @-mentions (repost sources included)."""
    return _MENTION.findall(text or "")


def repost_chain(text):
    """Repost chain, earliest source last: '//@A: ... //@B: ...' -> [A, B]."""
    return _REPOST.findall(text or "")


def strip_urls(text):
    return _URL.sub("", text or "")


def tokens(text):
    import jieba
    return [w for w in jieba.cut(strip_urls(text or "")) if w.strip() and len(w) >= 2
            and not re.match(r"^[\W\d_]+$", w)]

# ----------------------------------------------------------------------------
# Standard event-map skeleton and bank assembly
# ----------------------------------------------------------------------------

def base_map(ev):
    """Method-agnostic descriptive fields shared by every bank (same schema
    as the original pipeline)."""
    title = ev.get("event_title") or ""
    summary = ev.get("event_summary") or ""
    topics = [t for t in (ev.get("topics") or []) if t]
    f2d = f"{title} {summary} " + " ".join(topics)
    return {
        "map_id": hashlib.md5((ev["post_id"] + title).encode("utf-8")).hexdigest()[:12],
        "post_id": ev["post_id"],
        "event_title": title,
        "event_summary": summary,
        "entities": [e for e in (ev.get("entities") or []) if e],
        "topics": topics,
        "user_opinion": ev.get("user_opinion") or "",
        "stance_keywords": [s for s in (ev.get("stance_keywords") or []) if s],
        "feature_2d_text": f2d.strip(),
        "timestamp": ev.get("timestamp"),
        "polarity": sentiment_score((ev.get("user_opinion") or "") + " "
                                    + " ".join(ev.get("stance_keywords") or [])),
    }


def entity_stance_agg(events):
    agg = defaultdict(Counter)
    for ev in events:
        for e in ev.get("entities") or []:
            for s in ev.get("stance_keywords") or []:
                agg[e][s] += 1
    return {e: [{"stance": s, "count": c} for s, c in cnt.most_common(3)]
            for e, cnt in agg.items()}


def static_beliefs(events):
    out = []
    for ev in events:
        b = ev.get("static_belief")
        if b and b not in out:
            out.append(b)
    return out


def assemble_bank(uid, method, paper_ref, analogy, event_maps, static_extra,
                  method_extras, events):
    persona = load_persona(uid)
    static_map = {
        "beliefs": static_beliefs(events),
        "persona_values": persona.get("values") or [],
        "persona_interests": persona.get("interests") or [],
        "communication": persona.get("communication") or [],
        "entity_stance": entity_stance_agg(events),
    }
    static_map.update(static_extra or {})
    texts = [m["feature_2d_text"] + " || " + m["feature_3d_text"] for m in event_maps]
    ents = {e for ev in events for e in (ev.get("entities") or [])}
    bank = {
        "method": method,
        "paper_ref": paper_ref,
        "analogy": analogy,
        "static_map": static_map,
        "event_maps": event_maps,
        "retrieval_index": {"dim": DIM, "texts": texts,
                            "vectors": [encode_text(t) for t in texts]},
        "stats": {
            "num_train_posts": len(events),
            "num_event_maps": len(event_maps),
            "num_static_beliefs": len(static_map["beliefs"]),
            "num_entities": len(ents),
        },
        "method_extras": method_extras or {},
    }
    return bank


def write_bank(uid, dir_key, bank, rebuild_note):
    kg_dir = OUT / f"weibo_kg_{dir_key}_{uid}"
    kg_dir.mkdir(parents=True, exist_ok=True)
    mb = kg_dir / "memory_bank.json"
    if mb.exists():
        backup = kg_dir / "memory_bank_before_fix.json"
        if not backup.exists():
            shutil.copy2(mb, backup)
    mb.write_text(json.dumps(bank, ensure_ascii=False), encoding="utf-8")
    meta_path = kg_dir / "build_meta.json"
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    meta.update({
        "user_id": uid,
        "method": bank["method"],
        "paper_ref": bank["paper_ref"],
        "stats": bank["stats"],
        "memory_bank": str(mb),
        "retriever": bank["method_extras"].get("retriever", "default"),
        "rebuilt_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rebuild_note": rebuild_note,
    })
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(mb)
