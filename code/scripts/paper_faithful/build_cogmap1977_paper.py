# -*- coding: utf-8 -*-
"""Cognitive map construction per Hart (1977), "Cognitive Maps of Three Latin
American Policy Makers", World Politics 30(1): 115-140.

Self-contained on purpose: this file shares no graph logic, no cache and no
helper module with any other method builder. Input is the user's raw weibo
JSON; each post is treated as one "document" for the Axelrod/Wrightson
documentary coding method described in section B of the paper.

Coding steps implemented verbatim from p.118:
  1. identify the conceptual variables the author emphasized (underline key
     words, label with capital letters; two-letter code = page letter +
     sequence letter on that page)
  2. list the conceptual variables, rewording kept as close as possible to the
     original
  3. identify the causal relationships explicitly stated *or implied*
  4. look for direct correspondences between variables and merge them

Comparison measures implemented from p.119-121:
  - typology: utility / goal / policy / peripheral (3 varieties)
  - path signs (positive path = even number of negative assertions)
  - path-balance (parallel paths between a pair of variables share a sign)
  - policy consistency (all paths from a policy variable to utility balanced)
  - causal cycles
  - density = n / (m * (m - 1))
  - variable frequency = variables per word of document
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
METHOD_KEY = "cognitive_maps_1977"
METHOD_NAME = "CognitiveMaps1977"
PAPER_REF = "Hart, Cognitive Maps of Three Latin American Policy Makers (World Politics 1977)"
DIM = 512


# --------------------------------------------------------------------------
# retrieval hashing (identical formula across all benchmark methods so that
# retrieval quality differences come from the graph, not from the vectorizer)
# --------------------------------------------------------------------------
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


def load_documents(
    user_id: str,
    raw_json: Path,
    *,
    include_all_events: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Documents to be coded = training posts, text taken from the raw crawl.

    Test-split posts are never touched: the raw file contains them and using
    them would leak the evaluation set.
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

    docs = []
    for pid in source_ids:
        ev = events.get(pid, {})
        w = raw_by_bid.get(pid)
        text = (w.get("text") if w else "") or str(ev.get("raw_text") or "")
        text = re.sub(r"https?://\S+", "", text).strip()
        if len(text) < 10:
            continue
        docs.append(
            {
                "post_id": pid,
                "text": text,
                "words": len(text),  # Chinese: characters are the word unit
                "event": ev,
                "from_raw": w is not None,
                "created_at": (w or {}).get("full_created_at") or "",
            }
        )
    return docs, persona


# --------------------------------------------------------------------------
# step 1-4: documentary coding of one document
# --------------------------------------------------------------------------
CODER_SYSTEM = """你是政治学文献编码员，严格执行 Axelrod / Wrightson 的认知图谱"文献编码法"
（Hart 1977, World Politics）。对给定的一篇文档（一条微博全文），按论文规定的四个步骤编码：

第一步 识别作者强调的"概念变量"（conceptual variables）：划出关键词。概念变量必须是可以
     增减、有程度变化的概念（如"国家主权""对外国技术的依赖""管理效率"），不是事件、人名或口号。
第二步 列出概念变量，措辞尽量贴近原文用词，不要抽象成大词。
第三步 识别作者"明确陈述或隐含"的因果关系。只有两类符号：
     "+" 表示原因变量增加会使结果变量增加（正因果断言）；
     "-" 表示原因变量增加会使结果变量减少（负因果断言）。
     必须是作者本人相信的因果信念，不要编造文档里没有的关系。
第四步 找出互为"直接对应"（同义、只是措辞不同）的变量对。

另外按论文 C 节的类型学要求标注两件事：
  evaluations：作者对某个变量取值做出正面/负面评价的，记录该变量、评价符号，以及该评价所指向
     的效用主体（beneficiary，即"谁的福祉"，如"中国""中国民众""国际社会"）。论文规定：只要作者
     对某变量取值做正面评价，就可假定该变量对相应效用变量有直接正效应。
  policy_candidates：作者认为可由政府控制或操纵（susceptible to control or manipulation
     by his government）的变量序号。

只输出 JSON，不要解释、不要 markdown 代码块。格式：
{"variables":[{"seq":1,"label":"变量措辞","key_words":"原文关键词"}],
 "causal_statements":[{"cause":1,"effect":2,"sign":"+","evidence":"原文依据","implied":false}],
 "evaluations":[{"variable":1,"sign":"+","beneficiary":"中国"}],
 "policy_candidates":[1],
 "correspondences":[[2,5]]}

若文档没有任何因果信念，variables 与 causal_statements 都返回空数组。"""


def code_document(llm: DeepSeekClient, doc: dict[str, Any]) -> dict[str, Any]:
    user = f"文档（{doc['created_at']}）：\n{doc['text'][:2400]}"
    raw = llm.chat(
        [{"role": "system", "content": CODER_SYSTEM}, {"role": "user", "content": user}],
        temperature=0.0,
        max_tokens=1400,
        disable_thinking=True,
    )
    txt = (raw or "").strip()
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return {"variables": [], "causal_statements": []}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"variables": [], "causal_statements": []}


# --------------------------------------------------------------------------
# two-letter codes: first letter = "page" (document), second = sequence on page
# --------------------------------------------------------------------------
_ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def page_code(idx: int) -> str:
    s = ""
    n = idx
    while True:
        s = _ALPHA[n % 26] + s
        n = n // 26 - 1
        if n < 0:
            break
    return s


def seq_code(idx: int) -> str:
    return page_code(idx)


def norm_label(label: str) -> str:
    s = re.sub(r"[\s\"'“”‘’《》()（）\[\]【】,，。.、;；:：!！?？]", "", str(label))
    return s.strip()[:40]


_BENEFICIARY_ALIASES = (
    (r"^(中华人民共和国|我国|国家|中国大陆|大陆|祖国)$", "中国"),
    (r"^(中国人民|中国民众|中国老百姓|老百姓|民众|人民|普通民众|公众|中国公众)$", "中国民众"),
    (r"^(国际社会|世界|全球|各国)$", "国际社会"),
    (r"^(中国政府|政府|官方)$", "中国政府"),
)


def canon_beneficiary(name: str) -> str:
    """Step 4 applied to utility variables: the same welfare entity stated in
    different words is one variable."""
    s = re.sub(r"(的利益|的福祉|的整体利益|的国家利益|利益|福祉)$", "", norm_label(name))
    for pat, canon in _BENEFICIARY_ALIASES:
        if re.match(pat, s):
            return canon
    return s


# --------------------------------------------------------------------------
# graph measures, section C of the paper
# --------------------------------------------------------------------------
def enumerate_paths(
    adj: dict[str, list[tuple[str, int]]],
    src: str,
    targets: set[str],
    max_len: int = 6,
    max_paths: int = 400,
    max_expansions: int = 20000,
) -> list[tuple[list[str], int]]:
    """Causal paths: follow the direction of effects without passing through
    any variable more than once (paper p.119)."""
    found: list[tuple[list[str], int]] = []
    stack: list[tuple[str, list[str], int]] = [(src, [src], 1)]
    expansions = 0
    while stack and len(found) < max_paths and expansions < max_expansions:
        expansions += 1
        node, path, sign = stack.pop()
        if len(path) > max_len:
            continue
        for nxt, s in adj.get(node, []):
            if nxt in path:
                continue
            np_, ns = path + [nxt], sign * s
            if nxt in targets:
                found.append((np_, ns))
            stack.append((nxt, np_, ns))
    return found


def path_balance(adj: dict[str, list[tuple[str, int]]], nodes: list[str]) -> dict[str, Any]:
    """Degree to which parallel paths between a pair of variables share a sign."""
    pairs_multi = 0
    pairs_balanced = 0
    unbalanced_pairs: list[dict[str, Any]] = []
    node_set = set(nodes)
    for src in nodes:
        paths = enumerate_paths(adj, src, node_set, max_len=5, max_paths=300)
        by_target: dict[str, set[int]] = {}
        for p, s in paths:
            by_target.setdefault(p[-1], set()).add(s)
        for dst, signs in by_target.items():
            n_paths = sum(1 for p, _ in paths if p[-1] == dst)
            if n_paths < 2:
                continue
            pairs_multi += 1
            if len(signs) == 1:
                pairs_balanced += 1
            elif len(unbalanced_pairs) < 40:
                unbalanced_pairs.append({"from": src, "to": dst})
    ratio = pairs_balanced / pairs_multi if pairs_multi else 1.0
    return {
        "pairs_with_parallel_paths": pairs_multi,
        "balanced_pairs": pairs_balanced,
        "path_balance": round(ratio, 4),
        "unbalanced_pairs_sample": unbalanced_pairs,
    }


def policy_consistency(
    adj: dict[str, list[tuple[str, int]]],
    policy_vars: list[str],
    utility_vars: set[str],
) -> dict[str, Any]:
    """A policy choice is consistent when every direct or indirect effect on
    utility carries the same sign, i.e. the paths to utility are balanced."""
    checked = 0
    consistent = 0
    detail: list[dict[str, Any]] = []
    for pv in policy_vars:
        paths = enumerate_paths(adj, pv, utility_vars, max_len=6, max_paths=300)
        if not paths:
            continue
        checked += 1
        signs = {s for _, s in paths}
        ok = len(signs) == 1
        consistent += 1 if ok else 0
        if len(detail) < 60:
            detail.append(
                {
                    "policy_variable": pv,
                    "n_paths_to_utility": len(paths),
                    "consistent": ok,
                    "path_sign": (list(signs)[0] if ok else 0),
                }
            )
    return {
        "policy_vars_linked_to_utility": checked,
        "consistent_policy_vars": consistent,
        "policy_consistency": round(consistent / checked, 4) if checked else 0.0,
        "detail": detail,
    }


def find_cycles(adj: dict[str, list[tuple[str, int]]], limit: int = 50) -> list[list[str]]:
    """A causal cycle exists when the terminal variable of a path affects the
    initial variable of that path."""
    cycles: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    color: dict[str, int] = {}
    stack: list[str] = []

    def dfs(u: str) -> None:
        if len(cycles) >= limit:
            return
        color[u] = 1
        stack.append(u)
        for v, _ in adj.get(u, []):
            if len(cycles) >= limit:
                break
            c = color.get(v, 0)
            if c == 1:
                cyc = stack[stack.index(v) :]
                key = tuple(sorted(cyc))
                if key not in seen:
                    seen.add(key)
                    cycles.append(cyc + [v])
            elif c == 0:
                dfs(v)
        stack.pop()
        color[u] = 2

    for n in list(adj.keys()):
        if color.get(n, 0) == 0:
            dfs(n)
    return cycles


def classify_variables(
    nodes: list[str],
    adj: dict[str, list[tuple[str, int]]],
    radj: dict[str, list[tuple[str, int]]],
    utility_vars: set[str],
    primary_utility: str,
    policy_marked: set[str],
) -> dict[str, str]:
    """Typology from p.119.

    utility    - the welfare variables themselves
    goal       - directly affect the (primary) utility variable, no mediation
    policy     - designated by the author as controllable by his government
    peripheral - everything else; three varieties:
                 1 affected by policy but does not affect policy
                 2 affects policy but is not affected by policy
                 3 both affects and is affected by policy
    """
    types: dict[str, str] = {}
    direct_to_primary = {u for u, _ in radj.get(primary_utility, [])}
    for n in nodes:
        if n in utility_vars:
            types[n] = "utility"
        elif n in direct_to_primary:
            types[n] = "goal"
        elif n in policy_marked:
            types[n] = "policy"
    policy_set = {n for n, t in types.items() if t == "policy"}

    def reaches(start: str, targets: set[str]) -> bool:
        seen = {start}
        stack = [start]
        while stack:
            u = stack.pop()
            for v, _ in adj.get(u, []):
                if v in targets:
                    return True
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        return False

    def reached_by(start: str, sources: set[str]) -> bool:
        seen = {start}
        stack = [start]
        while stack:
            u = stack.pop()
            for v, _ in radj.get(u, []):
                if v in sources:
                    return True
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        return False

    for n in nodes:
        if n in types:
            continue
        affects = reaches(n, policy_set)
        affected = reached_by(n, policy_set)
        if affected and not affects:
            types[n] = "peripheral_1"
        elif affects and not affected:
            types[n] = "peripheral_2"
        elif affects and affected:
            types[n] = "peripheral_3"
        else:
            types[n] = "peripheral_1"
    return types


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

    docs, persona = load_documents(
        args.user_id,
        Path(args.raw_json),
        include_all_events=args.include_all_events,
    )
    if args.limit:
        docs = docs[: args.limit]
    out_dir = OUT / f"weibo_kg_{METHOD_KEY}_{args.user_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "coding_cache_1977.jsonl"

    cache: dict[str, dict[str, Any]] = {}
    if cache_path.exists():
        for row in load_jsonl(cache_path):
            cache[str(row.get("post_id"))] = row.get("coding") or {}
    todo = [d for d in docs if d["post_id"] not in cache]
    print(f"[1977] documents={len(docs)} cached={len(cache)} to_code={len(todo)}")

    lock = threading.Lock()
    if todo:
        t0 = time.time()
        with cache_path.open("a", encoding="utf-8") as cf, ThreadPoolExecutor(
            max_workers=args.conc
        ) as ex:
            futs = {ex.submit(code_document, llm, d): d for d in todo}
            for n, fut in enumerate(as_completed(futs), 1):
                d = futs[fut]
                try:
                    coding = fut.result()
                except Exception as e:  # noqa: BLE001
                    print(f"  coding failed {d['post_id']}: {e}")
                    coding = {"variables": [], "causal_statements": []}
                with lock:
                    cache[d["post_id"]] = coding
                    cf.write(
                        json.dumps({"post_id": d["post_id"], "coding": coding}, ensure_ascii=False)
                        + "\n"
                    )
                    cf.flush()
                if n % 50 == 0 or n == len(todo):
                    print(f"  coded {n}/{len(todo)}  {time.time()-t0:.0f}s")

    # ---- assemble the map -------------------------------------------------
    label_to_code: dict[str, str] = {}
    code_to_label: dict[str, str] = {}
    code_docs: dict[str, set[str]] = {}
    assertions: dict[tuple[str, str], dict[str, Any]] = {}
    utility_counts: dict[str, int] = {}
    policy_marked: set[str] = set()
    evaluation_log: list[dict[str, Any]] = []
    event_maps: list[dict[str, Any]] = []
    total_words = 0

    for di, doc in enumerate(docs):
        coding = cache.get(doc["post_id"]) or {}
        total_words += doc["words"]
        variables = coding.get("variables") or []
        # step 4 (within document): merge direct correspondences
        alias: dict[int, int] = {}
        for pair in coding.get("correspondences") or []:
            if isinstance(pair, list) and len(pair) == 2:
                try:
                    a, b = int(pair[0]), int(pair[1])
                except (TypeError, ValueError):
                    continue
                alias[max(a, b)] = min(a, b)

        seq_to_code: dict[int, str] = {}
        local_vars: list[dict[str, str]] = []
        for vi, v in enumerate(variables):
            try:
                seq = int(v.get("seq", vi + 1))
            except (TypeError, ValueError):
                seq = vi + 1
            seq = alias.get(seq, seq)
            label = norm_label(v.get("label") or "")
            if len(label) < 2:
                continue
            # step 4 (across documents): identical wording is one variable
            if label in label_to_code:
                code = label_to_code[label]
            else:
                code = page_code(di) + seq_code(vi)
                while code in code_to_label:
                    code += "X"
                label_to_code[label] = code
                code_to_label[code] = label
            seq_to_code[seq] = code
            code_docs.setdefault(code, set()).add(doc["post_id"])
            local_vars.append({"code": code, "label": label, "key_words": str(v.get("key_words") or "")})

        local_edges: list[dict[str, Any]] = []
        for st in coding.get("causal_statements") or []:
            try:
                c = seq_to_code.get(alias.get(int(st.get("cause")), int(st.get("cause"))))
                e = seq_to_code.get(alias.get(int(st.get("effect")), int(st.get("effect"))))
            except (TypeError, ValueError):
                continue
            if not c or not e or c == e:
                continue
            sign = -1 if str(st.get("sign")).strip().startswith("-") else 1
            key = (c, e)
            rec = assertions.setdefault(
                key, {"sign": sign, "count": 0, "evidence": [], "implied": 0}
            )
            rec["count"] += 1
            if bool(st.get("implied")):
                rec["implied"] += 1
            if len(rec["evidence"]) < 3 and st.get("evidence"):
                rec["evidence"].append(str(st.get("evidence"))[:80])
            local_edges.append(
                {
                    "cause": c,
                    "effect": e,
                    "sign": "+" if sign > 0 else "-",
                    "implied": bool(st.get("implied")),
                }
            )

        # evaluations -> direct effect on a utility variable (paper p.119)
        local_evals: list[dict[str, Any]] = []
        for ev in coding.get("evaluations") or []:
            try:
                vcode = seq_to_code.get(alias.get(int(ev.get("variable")), int(ev.get("variable"))))
            except (TypeError, ValueError):
                continue
            if not vcode:
                continue
            ben = canon_beneficiary(ev.get("beneficiary") or "")
            if len(ben) < 2:
                continue
            ucode = "U" + ben[:8]
            code_to_label.setdefault(ucode, f"{ben}的效用")
            utility_counts[ucode] = utility_counts.get(ucode, 0) + 1
            sign = -1 if str(ev.get("sign")).strip().startswith("-") else 1
            rec = assertions.setdefault(
                (vcode, ucode), {"sign": sign, "count": 0, "evidence": [], "implied": 0}
            )
            rec["count"] += 1
            rec["from_evaluation"] = True
            local_evals.append({"variable": vcode, "sign": "+" if sign > 0 else "-", "utility": ucode})
            local_edges.append(
                {"cause": vcode, "effect": ucode, "sign": "+" if sign > 0 else "-", "implied": True}
            )

        for pc in coding.get("policy_candidates") or []:
            try:
                code = seq_to_code.get(alias.get(int(pc), int(pc)))
            except (TypeError, ValueError):
                continue
            if code:
                policy_marked.add(code)

        evaluation_log.extend(local_evals)
        ev_meta = doc["event"]
        triples = [
            f"{e['cause']}({code_to_label.get(e['cause'],'')}) --({e['sign']})--> "
            f"{e['effect']}({code_to_label.get(e['effect'],'')})"
            for e in local_edges
        ]
        pol = 0.0
        if local_edges:
            pol = round(
                sum(1 if e["sign"] == "+" else -1 for e in local_edges) / len(local_edges), 3
            )
        event_maps.append(
            {
                "map_id": hashlib.md5(doc["post_id"].encode("utf-8")).hexdigest()[:12],
                "post_id": doc["post_id"],
                "event_title": ev_meta.get("event_title") or "",
                "event_summary": ev_meta.get("event_summary") or "",
                "entities": list(ev_meta.get("entities") or []),
                "topics": list(ev_meta.get("topics") or []),
                "user_opinion": ev_meta.get("user_opinion") or "",
                "stance_keywords": list(ev_meta.get("stance_keywords") or []),
                "timestamp": float(ev_meta.get("timestamp") or 0.0),
                "feature_2d_text": " ".join(
                    x
                    for x in [
                        str(ev_meta.get("event_title") or "").strip(),
                        str(ev_meta.get("event_summary") or "").strip(),
                        " ".join(str(t) for t in (ev_meta.get("topics") or []) if t),
                    ]
                    if x
                ),
                "document_words": doc["words"],
                "coded_variables": local_vars,
                "causal_assertions": local_edges,
                "evaluations": local_evals,
                "feature_3d_triples": triples,
                "polarity": pol,
                "variable_frequency": round(len(local_vars) / doc["words"], 5) if doc["words"] else 0.0,
            }
        )

    # ---- global signed digraph -------------------------------------------
    nodes = sorted({c for k in assertions for c in k})
    adj: dict[str, list[tuple[str, int]]] = {}
    radj: dict[str, list[tuple[str, int]]] = {}
    for (c, e), rec in assertions.items():
        adj.setdefault(c, []).append((e, rec["sign"]))
        radj.setdefault(e, []).append((c, rec["sign"]))

    utility_vars = {u for u in utility_counts if u in nodes}
    primary_utility = max(utility_counts, key=lambda k: utility_counts[k]) if utility_counts else ""
    types = classify_variables(nodes, adj, radj, utility_vars, primary_utility, policy_marked)

    m = len(nodes)
    n_beliefs = len(assertions)
    density = n_beliefs / (m * (m - 1)) if m > 1 else 0.0
    pb = path_balance(adj, nodes)
    policy_vars = [n for n, t in types.items() if t == "policy"]
    pc = policy_consistency(adj, policy_vars, utility_vars)
    cycles = find_cycles(adj)

    type_counts: dict[str, int] = {}
    for t in types.values():
        type_counts[t] = type_counts.get(t, 0) + 1

    var_table = [
        {
            "code": c,
            "label": code_to_label.get(c, c),
            "type": types.get(c, "peripheral_1"),
            "in_degree": len(radj.get(c, [])),
            "out_degree": len(adj.get(c, [])),
            "documents": len(code_docs.get(c, ())),
        }
        for c in nodes
    ]
    var_table.sort(key=lambda r: -(r["in_degree"] + r["out_degree"]))

    assertion_table = [
        {
            "cause": c,
            "cause_label": code_to_label.get(c, c),
            "effect": e,
            "effect_label": code_to_label.get(e, e),
            "sign": "+" if rec["sign"] > 0 else "-",
            "count": rec["count"],
            "implied": rec["implied"],
            "evidence": rec["evidence"],
        }
        for (c, e), rec in sorted(assertions.items(), key=lambda kv: -kv[1]["count"])
    ]

    static_map = {
        "beliefs": [
            f"{r['cause_label']} --({r['sign']})--> {r['effect_label']}"
            for r in assertion_table[:16]
        ],
        "persona_values": list(persona.get("values") or [])[:8],
        "persona_interests": list(persona.get("interests") or [])[:8],
        "communication": list(persona.get("communication") or [])[:8],
        "entity_stance": {},
        "coding_method": "Axelrod/Wrightson documentary coding, Hart 1977 steps 1-4",
        "utility_variables": [
            {"code": u, "label": code_to_label.get(u, u), "evaluations": utility_counts[u]}
            for u in sorted(utility_counts, key=lambda k: -utility_counts[k])
        ],
        "primary_utility_variable": primary_utility,
        "variable_list": var_table[:600],
        "causal_assertions": assertion_table[:2000],
        "variable_typology": type_counts,
        "path_balance": pb,
        "policy_consistency": pc,
        "causal_cycles": {
            "count": len(cycles),
            "sample": [" -> ".join(c) for c in cycles[:20]],
        },
        "density": round(density, 6),
        "density_formula": "n / (m * (m - 1))",
        "n_beliefs": n_beliefs,
        "n_variables": m,
        "variable_frequency": round(m / total_words, 6) if total_words else 0.0,
        "total_words": total_words,
    }

    entity_stance: dict[str, list[dict[str, Any]]] = {}
    for em in event_maps:
        for ent in em["entities"]:
            kws = [str(k) for k in em["stance_keywords"] if k]
            if not kws:
                continue
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
            "document": "one weibo post = one coded document",
            "conceptual_variable": "concept the author emphasizes, two-letter code",
            "causal_assertion": "signed (+/-) causal belief stated or implied",
            "utility_variable": "welfare entity the author positively evaluates toward",
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
            "retriever": "signed",
            "source": (
                "raw weibo json, all chronological events; codings are intended "
                "for reveal-after-predict online ingestion"
                if args.include_all_events
                else "raw weibo json, train split only (test posts excluded)"
            ),
            "coder": "LLM coder executing the paper's four coding steps",
            "rebuilt_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "rebuild_note": "independent Hart-1977 rebuild; no logic or cache shared with other methods",
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
                "retriever": "signed",
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
                "variables_m": m,
                "beliefs_n": n_beliefs,
                "density": static_map["density"],
                "variable_frequency": static_map["variable_frequency"],
                "typology": type_counts,
                "path_balance": pb["path_balance"],
                "policy_consistency": pc["policy_consistency"],
                "cycles": len(cycles),
                "utility_variables": [u["code"] for u in static_map["utility_variables"][:5]],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("wrote", mb_path)


if __name__ == "__main__":
    main()
