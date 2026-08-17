# -*- coding: utf-8 -*-
"""Shared I/O + retrieval hashing only. No method-specific graph logic lives here."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

DIM = 512
ROOT = Path(__file__).resolve().parents[2]  # UserAgent/
OUT = ROOT / "outputs"


def encode_text(text: str, dim: int = DIM) -> list[float]:
    """Original paper-KG vectorizer: char-3gram of lowercased whitespace-stripped text."""
    s = "".join(str(text).split()).lower()
    v = [0.0] * dim
    for j in range(max(0, len(s) - 2)):
        h = int(hashlib.md5(s[j : j + 3].encode("utf-8")).hexdigest(), 16)
        v[h % dim] += 1.0 if (h >> 8) & 1 else -1.0
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n else v


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_user_bundle(user_id: str) -> dict[str, Any]:
    udir = OUT / f"weibo_user_{user_id}"
    events = load_jsonl(udir / "events_all.jsonl")
    train = load_jsonl(udir / "train.jsonl")
    persona = json.loads((udir / "persona.json").read_text(encoding="utf-8"))
    train_ids = {str(r.get("post_id")) for r in train}
    # Prefer train events in chronological order; fall back to all events.
    by_id = {str(e.get("post_id")): e for e in events}
    ordered = [by_id[pid] for pid in [str(r.get("post_id")) for r in train] if pid in by_id]
    if not ordered:
        ordered = sorted(events, key=lambda e: float(e.get("timestamp") or 0.0))
    return {
        "user_id": user_id,
        "events": ordered,
        "all_events": events,
        "train": train,
        "train_ids": train_ids,
        "persona": persona,
        "udir": udir,
    }


def short_id(seed: str) -> str:
    return hashlib.md5(seed.encode("utf-8")).hexdigest()[:12]


def feature_2d(ev: dict[str, Any]) -> str:
    title = str(ev.get("event_title") or "").strip()
    summary = str(ev.get("event_summary") or "").strip()
    topics = " ".join(str(t) for t in (ev.get("topics") or []) if t)
    return " ".join(x for x in [title, summary, topics] if x)


def polarity_from_text(*parts: str) -> float:
    text = " ".join(p for p in parts if p)
    pos = len(re.findall(r"支持|赞|期望|希望|相信|爱国|正能量|欣慰|进步|公正|应该", text))
    neg = len(re.findall(r"批评|质疑|反对|谴责|愤怒|失望|担忧|可耻|错误|歪风|双标|不满", text))
    if pos == neg == 0:
        return 0.0
    return round((pos - neg) / max(pos + neg, 1), 3)


def base_event_fields(ev: dict[str, Any]) -> dict[str, Any]:
    return {
        "map_id": short_id(str(ev.get("post_id") or uuid.uuid4())),
        "post_id": str(ev.get("post_id") or ""),
        "event_title": ev.get("event_title") or "",
        "event_summary": ev.get("event_summary") or "",
        "entities": list(ev.get("entities") or []),
        "topics": list(ev.get("topics") or []),
        "user_opinion": ev.get("user_opinion") or "",
        "stance_keywords": list(ev.get("stance_keywords") or []),
        "feature_2d_text": feature_2d(ev),
        "timestamp": float(ev.get("timestamp") or 0.0),
        "polarity": polarity_from_text(
            str(ev.get("user_opinion") or ""),
            " ".join(ev.get("stance_keywords") or []),
        ),
    }


def build_static_from_persona(
    persona: dict[str, Any],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    values = list(persona.get("values") or [])
    interests = list(persona.get("interests") or [])
    communication = list(persona.get("communication") or [])
    beliefs = [v for v in values if isinstance(v, str) and v.strip()][:16]
    if not beliefs:
        beliefs = [
            str(e.get("user_opinion") or "")[:80]
            for e in events
            if e.get("user_opinion")
        ][:16]
    entity_stance: dict[str, list[dict[str, Any]]] = {}
    for e in events:
        ents = e.get("entities") or []
        kws = [str(k) for k in (e.get("stance_keywords") or []) if k]
        if not ents or not kws:
            continue
        for ent in ents:
            bucket = entity_stance.setdefault(str(ent), [])
            # aggregate stance keyword counts per entity
            counts = {x["stance"]: x["count"] for x in bucket}
            for kw in kws:
                counts[kw] = counts.get(kw, 0) + 1
            entity_stance[str(ent)] = [
                {"stance": k, "count": v}
                for k, v in sorted(counts.items(), key=lambda kv: -kv[1])[:8]
            ]
    return {
        "beliefs": beliefs,
        "persona_values": values[:8] if values else beliefs[:8],
        "persona_interests": interests[:8],
        "communication": communication[:8],
        "entity_stance": entity_stance,
    }


def finalize_maps(event_maps: list[dict[str, Any]]) -> tuple[list[str], list[list[float]]]:
    texts: list[str] = []
    vectors: list[list[float]] = []
    for m in event_maps:
        triples = m.get("feature_3d_triples") or []
        f3d = " ; ".join(str(t) for t in triples)
        m["feature_3d_text"] = f3d
        text = (m.get("feature_2d_text") or "") + " || " + f3d
        texts.append(text)
        vectors.append(encode_text(text))
    return texts, vectors


def write_memory_bank(
    *,
    user_id: str,
    method_key: str,
    method_name: str,
    paper_ref: str,
    analogy: dict[str, Any],
    static_map: dict[str, Any],
    event_maps: list[dict[str, Any]],
    method_extras: dict[str, Any],
) -> Path:
    out_dir = OUT / f"weibo_kg_{method_key}_{user_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    mb_path = out_dir / "memory_bank.json"
    if mb_path.exists():
        bak = out_dir / "memory_bank_before_rebuild.json"
        if not bak.exists():
            shutil.copy2(mb_path, bak)

    texts, vectors = finalize_maps(event_maps)
    bank = {
        "method": method_name,
        "paper_ref": paper_ref,
        "analogy": analogy,
        "static_map": static_map,
        "event_maps": event_maps,
        "retrieval_index": {"dim": DIM, "texts": texts, "vectors": vectors},
        "stats": {
            "num_train_posts": len(event_maps),
            "num_event_maps": len(event_maps),
            "num_static_beliefs": len(static_map.get("beliefs") or []),
            "num_entities": len(static_map.get("entity_stance") or {}),
        },
        "method_extras": {
            **method_extras,
            "rebuilt_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "rebuild_note": "paper-faithful independent rebuild (no cross-method reuse)",
        },
    }
    mb_path.write_text(json.dumps(bank, ensure_ascii=False), encoding="utf-8")
    meta = {
        "user_id": user_id,
        "method": method_name,
        "paper_ref": paper_ref,
        "stats": bank["stats"],
        "memory_bank": str(mb_path),
        "retriever": method_extras.get("retriever", "default"),
        "rebuilt_at": bank["method_extras"]["rebuilt_at"],
    }
    (out_dir / "build_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out_dir


# Lightweight Chinese causal patterns used ONLY by GenMinds / CognitiveMaps-style
# builders that need textual causal extraction. Imported explicitly by those modules.
CAUSAL_INC = (
    "提高|提升|增强|增加|促进|推动|推进|加强|扩大|巩固|有助于|有利于|带动|激发|保障|支撑"
)
CAUSAL_DEC = (
    "降低|削弱|减少|损害|破坏|阻碍|妨碍|抑制|打击|危害|拖累|威胁|恶化|不利于"
)
CAUSAL_NEUT = "导致|造成|引发|引起|带来|促使|使得"

_SENT_SPLIT = re.compile(r"[。！？!?；;\n]+")


def extract_signed_causals(text: str) -> list[tuple[str, str, int]]:
    """Return (cause, effect, sign) from Chinese causal frames."""
    out: list[tuple[str, str, int]] = []
    if not text:
        return out
    C = r"[^，,。！？!?；;：:]"
    for sent in _SENT_SPLIT.split(text):
        sent = sent.strip()
        if len(sent) < 6:
            continue
        if re.search(r"哪是|难道|岂能|怎么可能|不会导致|不能导致", sent):
            continue
        m = re.search(rf"(?:因为|由于)({C}{{2,24}})[，,](?:所以|因此|因而)?({C}{{2,30}})", sent)
        if m:
            out.append((m.group(1), m.group(2), 1))
        m = re.search(rf"({C}{{2,24}})[，,](?:所以|因此|因而|从而|于是)({C}{{2,30}})", sent)
        if m:
            out.append((m.group(1), m.group(2), 1))
        for verbs, sign in ((CAUSAL_INC, 1), (CAUSAL_DEC, -1)):
            for mm in re.finditer(rf"({C}{{2,24}})(?:会|将)?(?:{verbs})(?:了|着)?({C}{{2,24}})", sent):
                out.append((mm.group(1), mm.group(2), sign))
        for mm in re.finditer(rf"({C}{{2,24}})(?:会|将)?(?:{CAUSAL_NEUT})(?:了|着)?({C}{{2,24}})", sent):
            out.append((mm.group(1), mm.group(2), 1))
        m = re.search(rf"只有({C}{{2,20}})才(?:能|会)?({C}{{2,24}})", sent)
        if m:
            out.append((m.group(1), m.group(2), 1))
        m = re.search(rf"越({C}{{1,12}})[，,]?越({C}{{1,16}})", sent)
        if m:
            out.append((m.group(1), m.group(2), 1))
    cleaned: list[tuple[str, str, int]] = []
    for a, b, s in out:
        a = re.sub(r"\s+", "", a)[:24]
        b = re.sub(r"\s+", "", b)[:24]
        if len(a) >= 2 and len(b) >= 2 and a != b:
            cleaned.append((a, b, s))
    return cleaned
