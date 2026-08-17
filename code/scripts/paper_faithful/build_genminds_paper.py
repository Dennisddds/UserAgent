# -*- coding: utf-8 -*-
"""Belief-graph construction per GenMinds, "Simulating Society Requires
Simulating Thought" (NeurIPS 2025), section 5.1.

Self-contained on purpose: no graph logic, cache or helper module is shared
with any other method builder. Input is the user's raw weibo JSON.

Pipeline implemented from the paper:
  Structured Thought Capture - semi-structured interviews adaptively conducted
      by an LLM ("why do you support X?", "what does Y influence?") whose
      answers are parsed into a directed acyclic graph of the participant's
      belief structure; each node is a concept, each edge a directional causal
      relation carrying confidence and polarity scores.
  Shared Knowledge - cognitive motifs, the minimal causal reasoning units
      ("Transparency -> Crime rate -> Public safety"), aggregated across
      interviews into a topology of commonly held belief structures and
      represented as a symbolic causal graph (CBN).
  Inference via Symbolic-Neural Hybrid Graph Simulation - forward inference
      over the belief graph: given an intervention do(X = high), probabilistic
      updates propagate downstream posteriors and final stances.
  Be Aware of Unknown - weakly supported and isolated nodes are flagged rather
      than silently completed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]  # UserAgent/
sys.path.insert(0, str(ROOT / "theory_guided_agent"))

from tg_agent.llm import DeepSeekClient, load_env  # noqa: E402

OUT = ROOT / "outputs"
METHOD_KEY = "genminds"
METHOD_NAME = "GenMinds"
PAPER_REF = "GenMinds: Simulating Society Requires Simulating Thought (NeurIPS 2025)"
DIM = 512


def encode_text(text: str, dim: int = DIM) -> list[float]:
    s = "".join(str(text).split()).lower()
    v = [0.0] * dim
    for j in range(max(0, len(s) - 2)):
        h = int(hashlib.md5(s[j : j + 3].encode("utf-8")).hexdigest(), 16)
        v[h % dim] += 1.0 if (h >> 8) & 1 else -1.0
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n else v


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_interview_material(
    user_id: str,
    raw_json: Path,
    *,
    include_all_events: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Interview turns are seeded by the participant's own training posts.

    Test-split posts exist in the raw crawl and are deliberately skipped.
    """
    udir = OUT / f"weibo_user_{user_id}"
    all_events = load_jsonl(udir / "events_all.jsonl")
    if include_all_events:
        source_rows = sorted(all_events, key=lambda e: float(e.get("timestamp") or 0.0))
    else:
        source_rows = load_jsonl(udir / "train.jsonl")
    source_ids = [str(r.get("post_id")) for r in source_rows]
    events = {str(e.get("post_id")): e for e in all_events}
    persona = json.loads((udir / "persona.json").read_text(encoding="utf-8"))
    raw = json.loads(raw_json.read_text(encoding="utf-8"))
    raw_by_bid = {str(w.get("bid")): w for w in raw.get("weibo", [])}

    turns = []
    for pid in source_ids:
        ev = events.get(pid, {})
        w = raw_by_bid.get(pid)
        text = (w.get("text") if w else "") or str(ev.get("raw_text") or "")
        text = re.sub(r"https?://\S+", "", text).strip()
        if len(text) < 10:
            continue
        turns.append(
            {
                "post_id": pid,
                "text": text,
                "event": ev,
                "created_at": (w or {}).get("full_created_at") or "",
            }
        )
    return turns, persona


# --------------------------------------------------------------------------
# Structured Thought Capture: adaptive semi-structured interview -> DAG
# --------------------------------------------------------------------------
INTERVIEW_SYSTEM = """你是 GenMinds 框架里的访谈者兼解析器（NeurIPS 2025, Simulating Society
Requires Simulating Thought, 5.1 节）。给你一段受访者本人的公开表述，你要完成三件事：

一、半结构化访谈（由你自适应提出问题）。围绕这段表述，提出 1-3 个论文规定形式的追问：
   "你为什么支持/反对 X？" 或 "Y 会影响什么？"。追问要贴着这段表述里真正出现的概念，
   不要问表述里没有的东西。
二、以受访者第一人称回答这些追问，回答必须是"日常语言的因果解释"，且只能依据这段表述本身
   的立场和理由，不许引入表述之外的新事实。
三、把回答解析成信念结构：
   nodes  概念节点，如"国家利益""舆论监督""民众安全感"。必须是可增减的概念，不是事件或人名。
   edges  有向因果边，每条给出：
          polarity 极性："+" 表示 src 增加使 dst 增加，"-" 表示 src 增加使 dst 减少；
          confidence 置信度 0-1，反映受访者在这段表述里对该因果关系的笃定程度。
   motifs 认知母题（cognitive motif），即最小因果推理单元，形如
          "A -> B -> C"（链式，type=chain）或 "B <- A -> C"（分叉，type=fork）。
          母题必须由上面 edges 里已有的边组成。
   emphasis 受访者在这段表述中的强调程度 0-1（用了"一定""必须""绝不"等强调语气则高）。

信念图必须是有向无环图：不要输出会形成环的边。

只输出 JSON，不要解释、不要 markdown 代码块。格式：
{"interview":[{"q":"...","a":"..."}],
 "nodes":["概念1","概念2"],
 "edges":[{"src":"概念1","dst":"概念2","polarity":"+","confidence":0.8,"quote":"依据"}],
 "motifs":[{"path":["概念1","概念2","概念3"],"type":"chain"}],
 "emphasis":0.6}

若这段表述没有任何因果解释，nodes / edges / motifs 返回空数组。"""


def interview_one(llm: DeepSeekClient, turn: dict[str, Any]) -> dict[str, Any]:
    user = f"受访者本人表述（{turn['created_at']}）：\n{turn['text'][:2400]}"
    raw = llm.chat(
        [{"role": "system", "content": INTERVIEW_SYSTEM}, {"role": "user", "content": user}],
        temperature=0.0,
        max_tokens=1400,
        disable_thinking=True,
    )
    txt = (raw or "").strip()
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return {"interview": [], "nodes": [], "edges": [], "motifs": [], "emphasis": 0.0}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"interview": [], "nodes": [], "edges": [], "motifs": [], "emphasis": 0.0}


def norm_concept(name: str) -> str:
    s = re.sub(r"[\s\"'“”‘’《》()（）\[\]【】,，。.、;；:：!！?？]", "", str(name))
    return s.strip()[:24]


# --------------------------------------------------------------------------
# Composing the Causal Belief Network
# --------------------------------------------------------------------------
def enforce_dag(
    edges: dict[tuple[str, str], dict[str, Any]]
) -> tuple[dict[tuple[str, str], dict[str, Any]], list[dict[str, Any]]]:
    """Belief structure is a DAG (5.1). Cycle-closing edges are dropped
    lowest-confidence first and recorded as removed."""
    kept = dict(edges)
    removed: list[dict[str, Any]] = []

    def find_cycle() -> list[str] | None:
        adj: dict[str, list[str]] = {}
        for (a, b) in kept:
            adj.setdefault(a, []).append(b)
        color: dict[str, int] = {}
        stack: list[str] = []

        def dfs(u: str) -> list[str] | None:
            color[u] = 1
            stack.append(u)
            for v in adj.get(u, []):
                c = color.get(v, 0)
                if c == 1:
                    return stack[stack.index(v) :] + [v]
                if c == 0:
                    r = dfs(v)
                    if r:
                        return r
            stack.pop()
            color[u] = 2
            return None

        for n in list(adj.keys()):
            if color.get(n, 0) == 0:
                r = dfs(n)
                if r:
                    return r
        return None

    while True:
        cyc = find_cycle()
        if not cyc:
            break
        ring = [(cyc[i], cyc[i + 1]) for i in range(len(cyc) - 1)]
        ring = [e for e in ring if e in kept]
        if not ring:
            break
        weakest = min(ring, key=lambda e: (kept[e]["confidence"], kept[e]["support"]))
        rec = kept.pop(weakest)
        removed.append(
            {
                "src": weakest[0],
                "dst": weakest[1],
                "confidence": rec["confidence"],
                "reason": "cycle-closing edge removed to keep belief graph acyclic",
            }
        )
    return kept, removed


def topo_order(nodes: list[str], edges: dict[tuple[str, str], dict[str, Any]]) -> list[str]:
    indeg = {n: 0 for n in nodes}
    adj: dict[str, list[str]] = {n: [] for n in nodes}
    for (a, b) in edges:
        adj[a].append(b)
        indeg[b] += 1
    queue = sorted([n for n in nodes if indeg[n] == 0])
    order: list[str] = []
    while queue:
        u = queue.pop(0)
        order.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
    order.extend([n for n in nodes if n not in set(order)])
    return order


def propagate(
    order: list[str],
    parents: dict[str, list[tuple[str, int, float]]],
    priors: dict[str, float],
    clamp: dict[str, float] | None = None,
) -> dict[str, float]:
    """Forward inference over the belief graph: a node's posterior is its prior
    shifted by the signed, confidence-weighted state of its parents."""
    clamp = clamp or {}
    post: dict[str, float] = {}
    for n in order:
        if n in clamp:
            post[n] = clamp[n]
            continue
        ps = parents.get(n, [])
        if not ps:
            post[n] = priors.get(n, 0.5)
            continue
        num = 0.0
        den = 0.0
        for p, pol, conf in ps:
            num += conf * pol * (post.get(p, priors.get(p, 0.5)) - 0.5)
            den += conf
        shift = (num / den) if den else 0.0
        post[n] = min(0.99, max(0.01, priors.get(n, 0.5) + shift))
    return post


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user-id", required=True)
    ap.add_argument("--raw-json", required=True)
    ap.add_argument("--conc", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--include-all-events",
        action="store_true",
        help="Code every chronological event for online sequential ingestion. "
        "The sequential runner still reveals each coding only after its prediction.",
    )
    args = ap.parse_args()

    load_env(ROOT / "agentic-harness-engineering" / ".env")
    load_env(ROOT / "New" / ".env.local")
    llm = DeepSeekClient(
        api_key=os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("LLM_API_KEY"),
        base_url=os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com",
        model=os.environ.get("DEEPSEEK_MODEL") or "deepseek-chat",
        enable_thinking=False,
    )

    turns, persona = load_interview_material(
        args.user_id,
        Path(args.raw_json),
        include_all_events=args.include_all_events,
    )
    if args.limit:
        turns = turns[: args.limit]
    out_dir = OUT / f"weibo_kg_{METHOD_KEY}_{args.user_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "interview_cache_genminds.jsonl"

    cache: dict[str, dict[str, Any]] = {}
    if cache_path.exists():
        for row in load_jsonl(cache_path):
            cache[str(row.get("post_id"))] = row.get("parse") or {}
    todo = [t for t in turns if t["post_id"] not in cache]
    print(f"[GenMinds] turns={len(turns)} cached={len(cache)} to_interview={len(todo)}")

    lock = threading.Lock()
    if todo:
        t0 = time.time()
        with cache_path.open("a", encoding="utf-8") as cf, ThreadPoolExecutor(
            max_workers=args.conc
        ) as ex:
            futs = {ex.submit(interview_one, llm, t): t for t in todo}
            for n, fut in enumerate(as_completed(futs), 1):
                t = futs[fut]
                try:
                    parse = fut.result()
                except Exception as e:  # noqa: BLE001
                    print(f"  interview failed {t['post_id']}: {e}")
                    parse = {"interview": [], "nodes": [], "edges": [], "motifs": []}
                with lock:
                    cache[t["post_id"]] = parse
                    cf.write(
                        json.dumps({"post_id": t["post_id"], "parse": parse}, ensure_ascii=False)
                        + "\n"
                    )
                    cf.flush()
                if n % 50 == 0 or n == len(todo):
                    print(f"  interviewed {n}/{len(todo)}  {time.time()-t0:.0f}s")

    # ---- aggregate motifs into the participant's belief graph -------------
    edge_acc: dict[tuple[str, str], dict[str, Any]] = {}
    node_support: dict[str, int] = {}
    motif_counts: dict[str, dict[str, Any]] = {}
    event_maps: list[dict[str, Any]] = []

    for turn in turns:
        parse = cache.get(turn["post_id"]) or {}
        emphasis = parse.get("emphasis")
        try:
            emphasis = float(emphasis)
        except (TypeError, ValueError):
            emphasis = 0.0
        emphasis = min(1.0, max(0.0, emphasis))

        local_edges: list[dict[str, Any]] = []
        for e in parse.get("edges") or []:
            src = norm_concept(e.get("src") or "")
            dst = norm_concept(e.get("dst") or "")
            if len(src) < 2 or len(dst) < 2 or src == dst:
                continue
            pol = -1 if str(e.get("polarity")).strip().startswith("-") else 1
            try:
                conf = float(e.get("confidence"))
            except (TypeError, ValueError):
                conf = 0.5
            conf = min(1.0, max(0.05, conf))
            rec = edge_acc.setdefault(
                (src, dst),
                {"pos": 0, "neg": 0, "support": 0, "conf_sum": 0.0, "emph_sum": 0.0, "quotes": []},
            )
            rec["support"] += 1
            rec["pos" if pol > 0 else "neg"] += 1
            rec["conf_sum"] += conf
            rec["emph_sum"] += emphasis
            if e.get("quote") and len(rec["quotes"]) < 3:
                rec["quotes"].append(str(e.get("quote"))[:80])
            node_support[src] = node_support.get(src, 0) + 1
            node_support[dst] = node_support.get(dst, 0) + 1
            local_edges.append(
                {"src": src, "dst": dst, "polarity": "+" if pol > 0 else "-", "confidence": round(conf, 3)}
            )

        local_motifs: list[dict[str, Any]] = []
        for mo in parse.get("motifs") or []:
            path = [norm_concept(x) for x in (mo.get("path") or []) if norm_concept(x)]
            if len(path) < 3:
                continue
            mtype = "fork" if str(mo.get("type")) == "fork" else "chain"
            arrow = " -> " if mtype == "chain" else " <- | -> "
            key = arrow.join(path[:3]) if mtype == "chain" else f"{path[1]} <- {path[0]} -> {path[2]}"
            rec = motif_counts.setdefault(key, {"motif": key, "type": mtype, "count": 0, "nodes": path[:3]})
            rec["count"] += 1
            local_motifs.append({"motif": key, "type": mtype})

        for nd in parse.get("nodes") or []:
            c = norm_concept(nd)
            if len(c) >= 2:
                node_support.setdefault(c, 0)

        ev = turn["event"]
        triples = [
            f"{e['src']} --({e['polarity']}{e['confidence']})--> {e['dst']}" for e in local_edges
        ]
        pol_val = 0.0
        if local_edges:
            pol_val = round(
                sum(1 if e["polarity"] == "+" else -1 for e in local_edges) / len(local_edges), 3
            )
        hops = max([len(m.get("path") or []) - 1 for m in (parse.get("motifs") or [])] or [0])
        event_maps.append(
            {
                "map_id": hashlib.md5(turn["post_id"].encode("utf-8")).hexdigest()[:12],
                "post_id": turn["post_id"],
                "event_title": ev.get("event_title") or "",
                "event_summary": ev.get("event_summary") or "",
                "entities": list(ev.get("entities") or []),
                "topics": list(ev.get("topics") or []),
                "user_opinion": ev.get("user_opinion") or "",
                "stance_keywords": list(ev.get("stance_keywords") or []),
                "timestamp": float(ev.get("timestamp") or 0.0),
                "feature_2d_text": " ".join(
                    x
                    for x in [
                        str(ev.get("event_title") or "").strip(),
                        str(ev.get("event_summary") or "").strip(),
                        " ".join(str(t) for t in (ev.get("topics") or []) if t),
                    ]
                    if x
                ),
                "interview_qa": (parse.get("interview") or [])[:3],
                "belief_edges": local_edges,
                "cognitive_motifs": local_motifs,
                "emphasis": round(emphasis, 3),
                "cognitive_hops": hops,
                "feature_3d_triples": triples,
                "polarity": pol_val,
            }
        )

    # confidence from motif density and respondent emphasis (5.1, Step 2)
    max_support = max([r["support"] for r in edge_acc.values()] or [1])
    edges: dict[tuple[str, str], dict[str, Any]] = {}
    for key, rec in edge_acc.items():
        pol = 1 if rec["pos"] >= rec["neg"] else -1
        mean_conf = rec["conf_sum"] / rec["support"]
        density = math.log1p(rec["support"]) / math.log1p(max_support)
        emph = rec["emph_sum"] / rec["support"]
        conf = 0.5 * mean_conf + 0.35 * density + 0.15 * emph
        edges[key] = {
            "polarity": pol,
            "confidence": round(min(0.99, max(0.05, conf)), 4),
            "support": rec["support"],
            "agreement": round(max(rec["pos"], rec["neg"]) / rec["support"], 3),
            "quotes": rec["quotes"],
        }

    edges, removed_edges = enforce_dag(edges)
    nodes = sorted({n for k in edges for n in k} | set(node_support.keys()))
    order = topo_order(nodes, edges)

    parents: dict[str, list[tuple[str, int, float]]] = {}
    children: dict[str, list[str]] = {}
    for (a, b), rec in edges.items():
        parents.setdefault(b, []).append((a, rec["polarity"], rec["confidence"]))
        children.setdefault(a, []).append(b)

    # priors: how strongly the participant asserts the concept at all
    max_sup_node = max(node_support.values() or [1])
    priors = {
        n: round(0.5 + 0.4 * (math.log1p(node_support.get(n, 0)) / math.log1p(max_sup_node)), 4)
        for n in nodes
    }
    baseline = propagate(order, parents, priors)

    # do-calculus interventions on the most connected concepts
    ranked = sorted(nodes, key=lambda n: -(len(children.get(n, [])) + len(parents.get(n, []))))
    interventions = []
    for n in ranked[:8]:
        if not children.get(n):
            continue
        high = propagate(order, parents, priors, clamp={n: 0.95})
        low = propagate(order, parents, priors, clamp={n: 0.05})
        shifted = sorted(
            (
                {
                    "node": d,
                    "baseline": round(baseline[d], 3),
                    "do_high": round(high[d], 3),
                    "do_low": round(low[d], 3),
                    "delta": round(high[d] - baseline[d], 3),
                }
                for d in nodes
                if d != n and abs(high[d] - baseline[d]) > 0.01
            ),
            key=lambda r: -abs(r["delta"]),
        )[:10]
        if shifted:
            interventions.append({"do": f"do({n} = high)", "node": n, "downstream": shifted})
        if len(interventions) >= 5:
            break

    weak_nodes = [
        n for n in nodes if node_support.get(n, 0) <= 1 and not children.get(n) and not parents.get(n)
    ]
    isolated = [n for n in nodes if not children.get(n) and not parents.get(n)]
    weak_edges = [
        {"src": a, "dst": b, "confidence": r["confidence"]}
        for (a, b), r in edges.items()
        if r["confidence"] < 0.25
    ]

    node_table = [
        {
            "concept": n,
            "prior": priors[n],
            "posterior": round(baseline.get(n, priors[n]), 4),
            "in_degree": len(parents.get(n, [])),
            "out_degree": len(children.get(n, [])),
            "support": node_support.get(n, 0),
            "uncertain": n in set(isolated),
        }
        for n in nodes
    ]
    node_table.sort(key=lambda r: -(r["in_degree"] + r["out_degree"]))

    edge_table = [
        {
            "src": a,
            "dst": b,
            "polarity": "+" if r["polarity"] > 0 else "-",
            "confidence": r["confidence"],
            "support": r["support"],
            "agreement": r["agreement"],
            "quotes": r["quotes"],
        }
        for (a, b), r in sorted(edges.items(), key=lambda kv: -kv[1]["confidence"])
    ]

    motif_table = sorted(motif_counts.values(), key=lambda r: -r["count"])

    static_map = {
        "beliefs": [
            f"{r['src']} --({r['polarity']}{r['confidence']})--> {r['dst']}" for r in edge_table[:16]
        ],
        "persona_values": list(persona.get("values") or [])[:8],
        "persona_interests": list(persona.get("interests") or [])[:8],
        "communication": list(persona.get("communication") or [])[:8],
        "entity_stance": {},
        "belief_graph": {
            "is_dag": True,
            "num_nodes": len(nodes),
            "num_edges": len(edges),
            "nodes": node_table[:600],
            "edges": edge_table[:2000],
            "topological_order": order[:200],
            "removed_cycle_edges": removed_edges[:50],
        },
        "cognitive_motifs": motif_table[:300],
        "motif_topology": {
            "num_distinct_motifs": len(motif_table),
            "num_chain_motifs": sum(1 for m in motif_table if m["type"] == "chain"),
            "num_fork_motifs": sum(1 for m in motif_table if m["type"] == "fork"),
            "top_motifs": [m["motif"] for m in motif_table[:20]],
        },
        "causal_bayesian_network": {
            "priors": {n: priors[n] for n in ranked[:200]},
            "baseline_posteriors": {n: round(baseline[n], 4) for n in ranked[:200]},
            "cpt": "signed confidence-weighted parent mixture, P(n) = prior_n + sum_p conf*pol*(P(p)-0.5)/sum conf",
        },
        "interventions": interventions,
        "uncertainty": {
            "isolated_nodes": isolated[:100],
            "weakly_supported_nodes": weak_nodes[:100],
            "weakly_supported_edges": weak_edges[:100],
            "note": "Be Aware of Unknown: weak or isolated dependencies are surfaced, not completed",
        },
    }

    entity_stance: dict[str, list[dict[str, Any]]] = {}
    for em in event_maps:
        kws = [str(k) for k in em["stance_keywords"] if k]
        if not kws:
            continue
        for ent in em["entities"]:
            bucket = {x["stance"]: x["count"] for x in entity_stance.get(str(ent), [])}
            for kw in kws:
                bucket[kw] = bucket.get(kw, 0) + 1
            entity_stance[str(ent)] = [
                {"stance": k, "count": v}
                for k, v in sorted(bucket.items(), key=lambda kv: -kv[1])[:8]
            ]
    static_map["entity_stance"] = entity_stance

    texts, vectors = [], []
    for em in event_maps:
        f3d = " ; ".join(em["feature_3d_triples"])
        em["feature_3d_text"] = f3d
        t = em["feature_2d_text"] + " || " + f3d
        texts.append(t)
        vectors.append(encode_text(t))

    mb_path = out_dir / "memory_bank.json"
    if mb_path.exists():
        shutil.copy2(mb_path, out_dir / "memory_bank_before_raw_rebuild.json")
    bank = {
        "method": METHOD_NAME,
        "paper_ref": PAPER_REF,
        "analogy": {
            "interview_turn": "one weibo post seeds one adaptive semi-structured interview turn",
            "node": "concept in the participant's belief structure",
            "edge": "directional causal relation with polarity and confidence",
            "motif": "minimal causal reasoning unit, chain or fork over three concepts",
        },
        "static_map": static_map,
        "event_maps": event_maps,
        "retrieval_index": {"dim": DIM, "texts": texts, "vectors": vectors},
        "stats": {
            "num_train_posts": len(event_maps),
            "num_event_maps": len(event_maps),
            "num_static_beliefs": len(static_map["beliefs"]),
            "num_entities": len(entity_stance),
        },
        "method_extras": {
            "retriever": "multihop",
            "source": (
                "raw weibo json, all chronological events; codings are intended "
                "for reveal-after-predict online ingestion"
                if args.include_all_events
                else "raw weibo json, train split only (test posts excluded)"
            ),
            "interviewer": "LLM-conducted adaptive semi-structured interview per 5.1",
            "rebuilt_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "rebuild_note": "independent GenMinds rebuild; no logic or cache shared with other methods",
        },
    }
    mb_path.write_text(json.dumps(bank, ensure_ascii=False), encoding="utf-8")
    (out_dir / "build_meta.json").write_text(
        json.dumps(
            {
                "user_id": args.user_id,
                "method": METHOD_NAME,
                "paper_ref": PAPER_REF,
                "stats": bank["stats"],
                "memory_bank": str(mb_path),
                "retriever": "multihop",
                "rebuilt_at": bank["method_extras"]["rebuilt_at"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "nodes": len(nodes),
                "edges": len(edges),
                "removed_cycle_edges": len(removed_edges),
                "distinct_motifs": len(motif_table),
                "chain_motifs": static_map["motif_topology"]["num_chain_motifs"],
                "fork_motifs": static_map["motif_topology"]["num_fork_motifs"],
                "interventions": len(interventions),
                "isolated_nodes": len(isolated),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("wrote", mb_path)


if __name__ == "__main__":
    main()
