#!/usr/bin/env python3
"""Live GenMinds knowledge-graph server.

Watches sequential_predictions.jsonl. Each new row:
  - advances chrono progress / OA metrics
  - rebuilds the USER knowledge graph from GenMinds memory
    (ego = account identity, entities/topics from events ingested so far)
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
HTML_PATH = ROOT / "live_graph.html"

_lock = threading.Lock()
_state: dict[str, Any] = {
    "ready": False,
    "source": "",
    "lines_read": 0,
    "updated_at": "",
    "snapshot": {},
}
_subs: dict[int, threading.Event] = {}
_sub_payload: dict[int, str] = {}
_next_sub = 1

_memory: dict[str, Any] = {}
_identity: dict[str, str] = {}
_sorted_events: list[dict[str, Any]] = []


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def load_memory(path: Path) -> None:
    global _memory, _identity, _sorted_events
    import re

    _memory = json.loads(path.read_text(encoding="utf-8"))
    uid = ""
    m = re.search(r"(\d{5,})", str(path))
    if m:
        uid = m.group(1)
    name = ""
    if uid:
        dp = Path(f"d:/UserAgent/outputs/weibo_user_{uid}/data_preview.json")
        if dp.exists():
            d = json.loads(dp.read_text(encoding="utf-8"))
            name = str(d.get("user_name") or "").strip()
            uid = str(d.get("user_id") or uid)
    if not name and uid == "1989660417":
        name = "胡锡进"
    _identity = {"user_id": uid, "user_name": name or uid}

    events = list(_memory.get("event_maps") or [])

    def _ts(e: dict[str, Any]) -> float:
        t = e.get("timestamp")
        if t is not None:
            try:
                return float(t)
            except Exception:  # noqa: BLE001
                pass
        return 0.0

    _sorted_events = sorted(events, key=_ts)
    print(
        f"memory loaded events={len(_sorted_events)} identity={_identity} path={path}",
        flush=True,
    )


def build_kg(upto_ts: float | None, upto_n: int) -> dict[str, Any]:
    """Ego-centric knowledge graph from GenMinds events up to time/count."""
    ego = _identity.get("user_name") or _identity.get("user_id") or "user"
    static = _memory.get("static_map") or {}
    beliefs = list(static.get("beliefs") or [])[:8]

    if upto_ts and upto_ts > 0:
        evs = [e for e in _sorted_events if float(e.get("timestamp") or 0) <= upto_ts + 1e-6]
    else:
        evs = _sorted_events[: max(0, upto_n)]

    ent_count: Counter[str] = Counter()
    topic_count: Counter[str] = Counter()
    pair: Counter[tuple[str, str]] = Counter()
    stance_hit: dict[str, str] = {}

    for e in evs:
        ents = [str(x) for x in (e.get("entities") or []) if str(x).strip()]
        topics = [str(x).strip("#") for x in (e.get("topics") or []) if str(x).strip()]
        # drop ego self-noise aliases that duplicate identity
        ents = [x for x in ents if x not in {ego, "老胡", "胡老师"} or x == ego]
        for x in ents:
            ent_count[x] += 1
        for t in topics:
            topic_count[t] += 1
        nodes_e = ents[:8]
        for i, a in enumerate(nodes_e):
            for b in nodes_e[i + 1 :]:
                pair[tuple(sorted((a, b)))] += 1
            # link entity to ego
            pair[tuple(sorted((ego, a)))] += 1
        for t in topics[:4]:
            pair[tuple(sorted((ego, f"#{t}")))] += 1
            topic_count[f"#{t}"] += 0  # ensure key shape
        # light stance cue from keywords
        for kw in (e.get("stance_keywords") or [])[:2]:
            for a in ents[:3]:
                stance_hit.setdefault(a, str(kw)[:12])

    # seed important static entities if early / sparse
    es = static.get("entity_stance") or {}
    for name, stances in list(es.items())[:40]:
        if name in {ego, "老胡"}:
            continue
        if ent_count[name] == 0 and isinstance(stances, list) and stances:
            # only include high-count static entities as weak prior nodes
            c = sum(int(s.get("count") or 0) for s in stances if isinstance(s, dict))
            if c >= 8:
                ent_count[name] += max(1, c // 20)
                pair[tuple(sorted((ego, name)))] += 1
                stance_hit[name] = str(stances[0].get("stance") or "")[:12]

    # top nodes
    top_ent = [e for e, _ in ent_count.most_common(18) if e != ego]
    top_topic = [t for t, _ in topic_count.most_common(8)]
    node_ids = [ego] + top_ent + [t if t.startswith("#") else f"#{t}" for t in top_topic]
    # unique preserve
    seen: set[str] = set()
    nodes_out = []
    for i, nid in enumerate(node_ids):
        if nid in seen:
            continue
        seen.add(nid)
        kind = "ego" if nid == ego else ("topic" if nid.startswith("#") else "entity")
        cnt = ent_count.get(nid, 0) or topic_count.get(nid.lstrip("#"), 0) or topic_count.get(nid, 0)
        nodes_out.append(
            {
                "id": nid,
                "kind": kind,
                "count": int(cnt) if nid != ego else max(len(evs), 1),
                "stance": stance_hit.get(nid, ""),
            }
        )

    id_set = {n["id"] for n in nodes_out}
    links = []
    for (a, b), w in pair.most_common(50):
        if a in id_set and b in id_set and a != b:
            links.append({"source": a, "target": b, "weight": int(w)})

    return {
        "ego": ego,
        "identity": _identity,
        "num_events_in_window": len(evs),
        "beliefs": beliefs,
        "nodes": nodes_out,
        "links": links,
        "latest_event": {
            "title": (evs[-1].get("event_title") if evs else "") or "",
            "opinion": ((evs[-1].get("user_opinion") if evs else "") or "")[:160],
            "entities": (evs[-1].get("entities") if evs else []) or [],
            "topics": (evs[-1].get("topics") if evs else []) or [],
        }
        if evs
        else None,
    }


def build_snapshot(rows: list[dict[str, Any]], *, source: str, total_target: int) -> dict[str, Any]:
    mode_count: Counter[str] = Counter()
    oa_series: list[float] = []
    steps: list[dict[str, Any]] = []

    for r in rows:
        at = r.get("agent_trace") or {}
        js = r.get("judge_scores") or {}
        mode = (at.get("c_trace") or {}).get("mode") or (
            "warmup" if r.get("warmup") else "unknown"
        )
        mode_count[mode] += 1
        oa = js.get("opinion_alignment_score") if js else None
        if oa is not None and not r.get("warmup"):
            oa_series.append(float(oa))
        steps.append(
            {
                "step": r.get("step"),
                "post_id": r.get("post_id"),
                "topic": r.get("topic") or "",
                "warmup": bool(r.get("warmup")),
                "mode": mode,
                "oa": float(oa) if oa is not None else None,
                "prediction": (r.get("prediction") or "")[:240],
                "verbalization": (at.get("verbalization") or "")[:280],
                "timestamp": r.get("timestamp"),
            }
        )

    latest = steps[-1] if steps else None
    upto_ts = float((latest or {}).get("timestamp") or 0) or None
    upto_n = len(steps)
    kg = build_kg(upto_ts, upto_n)

    scored = len(oa_series)
    overall = sum(oa_series) / scored if scored else None
    window = oa_series[-20:]
    rolling = sum(window) / len(window) if window else None

    return {
        "ready": True,
        "kind": "genminds_knowledge_graph",
        "source": source,
        "updated_at": _now(),
        "progress": {
            "lines": len(rows),
            "total_target": total_target,
            "pct": round(100.0 * len(rows) / total_target, 2) if total_target else None,
            "scored": scored,
            "warmup": int(mode_count.get("warmup", 0)),
            "kg_events": kg.get("num_events_in_window"),
        },
        "metrics": {
            "overall_oa": round(overall, 4) if overall is not None else None,
            "recent20_oa": round(rolling, 4) if rolling is not None else None,
            "last_oa": round(oa_series[-1], 4) if oa_series else None,
            "modes": dict(mode_count),
        },
        "oa_series": [round(x, 4) for x in oa_series],
        "kg": kg,
        "latest": latest,
        "recent": list(reversed([s for s in steps if not s["warmup"]][-10:] or steps[-8:])),
    }


def _notify(snapshot: dict[str, Any]) -> None:
    payload = "data: " + json.dumps(snapshot, ensure_ascii=False) + "\n\n"
    with _lock:
        for sid, ev in _subs.items():
            _sub_payload[sid] = payload
            ev.set()


def load_complete_rows(path: Path) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    if not raw:
        return []
    if not raw.endswith(b"\n"):
        cut = raw.rfind(b"\n")
        if cut < 0:
            return []
        raw = raw[: cut + 1]
    rows: list[dict[str, Any]] = []
    for line in raw.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def ingest_file(path: Path, total_target: int, poll_s: float = 0.8) -> None:
    last_n = -1
    while True:
        try:
            if path.exists():
                rows = load_complete_rows(path)
                if len(rows) != last_n:
                    last_n = len(rows)
                    snap = build_snapshot(rows, source=str(path), total_target=total_target)
                    with _lock:
                        _state["ready"] = True
                        _state["source"] = str(path)
                        _state["lines_read"] = len(rows)
                        _state["updated_at"] = snap["updated_at"]
                        _state["snapshot"] = snap
                    _notify(snap)
                    latest = snap.get("latest") or {}
                    kg = snap.get("kg") or {}
                    print(
                        f"[{snap['updated_at']}] kg-update lines={len(rows)}/{total_target} "
                        f"kg_events={kg.get('num_events_in_window')} "
                        f"nodes={len((kg.get('nodes') or []))} "
                        f"ego={kg.get('ego')} step={latest.get('step')}",
                        flush=True,
                    )
        except Exception as e:  # noqa: BLE001
            print(f"watcher error: {e}", flush=True)
        time.sleep(poll_s)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")

    def do_GET(self) -> None:  # noqa: N802
        global _next_sub
        path = urlparse(self.path).path

        if path in {"/", "/index.html", "/live_graph.html"}:
            body = HTML_PATH.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self._cors()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/api/snapshot":
            with _lock:
                snap = _state.get("snapshot") or {
                    "ready": False,
                    "message": "waiting for predictions...",
                    "updated_at": _now(),
                }
                payload = json.dumps(snap, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self._cors()
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        if path == "/api/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Connection", "keep-alive")
            self._cors()
            self.end_headers()
            ev = threading.Event()
            with _lock:
                sid = _next_sub
                _next_sub += 1
                _subs[sid] = ev
                snap = _state.get("snapshot")
                if snap:
                    msg = ("data: " + json.dumps(snap, ensure_ascii=False) + "\n\n").encode("utf-8")
                else:
                    msg = b'data: {"ready":false,"message":"waiting..."}\n\n'
            try:
                self.wfile.write(msg)
                self.wfile.flush()
                while True:
                    if not ev.wait(timeout=20):
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                        continue
                    ev.clear()
                    with _lock:
                        payload = _sub_payload.get(sid)
                    if payload:
                        self.wfile.write(payload.encode("utf-8"))
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                return
            finally:
                with _lock:
                    _subs.pop(sid, None)
                    _sub_payload.pop(sid, None)
            return

        self.send_response(404)
        self.end_headers()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--watch",
        default=str(
            Path(
                "d:/UserAgent/outputs/benchmark_sequential_weibo_ai/"
                "seq-CUV-TG_1989660417/sequential_predictions.jsonl"
            )
        ),
    )
    ap.add_argument(
        "--memory",
        default=str(Path("d:/UserAgent/outputs/weibo_kg_genminds_1989660417/memory_bank.json")),
    )
    ap.add_argument("--total", type=int, default=2707)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--poll", type=float, default=0.8)
    args = ap.parse_args()

    if not HTML_PATH.exists():
        raise SystemExit(f"missing {HTML_PATH}")
    load_memory(Path(args.memory))

    threading.Thread(
        target=ingest_file,
        args=(Path(args.watch), args.total, args.poll),
        daemon=True,
    ).start()

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Live KG → http://{args.host}:{args.port}/", flush=True)
    print(f"watching {args.watch}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
