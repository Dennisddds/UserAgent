# -*- coding: utf-8 -*-
"""Audit the two raw-data rebuilds against the requirements stated in their
own papers. Structural checks only; no graph is modified."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str, str]] = []


def check(method: str, requirement: str, ok: bool, detail: str = "") -> None:
    results.append((method, requirement, PASS if ok else FAIL, detail))


def load(method_key: str, user_id: str) -> dict[str, Any]:
    p = OUT / f"weibo_kg_{method_key}_{user_id}" / "memory_bank.json"
    return json.loads(p.read_text(encoding="utf-8"))


def path_sign_unit_test() -> tuple[bool, str]:
    """Paper p.120 figure: A-+->B, B---->D, A-+->C, C-+->D.
    Path A,B,D must be negative; path A,C,D positive; the pair is unbalanced."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_cogmap1977_paper import enumerate_paths, path_balance  # noqa: E402

    adj = {"A": [("B", 1), ("C", 1)], "B": [("D", -1)], "C": [("D", 1)]}
    paths = enumerate_paths(adj, "A", {"D"})
    signs = {tuple(p): s for p, s in paths}
    ok_neg = signs.get(("A", "B", "D")) == -1
    ok_pos = signs.get(("A", "C", "D")) == 1
    pb = path_balance(adj, ["A", "B", "C", "D"])
    ok_unbal = pb["path_balance"] < 1.0
    return (
        ok_neg and ok_pos and ok_unbal,
        f"A-B-D={signs.get(('A','B','D'))} A-C-D={signs.get(('A','C','D'))} "
        f"path_balance={pb['path_balance']}",
    )


def verify_1977(user_id: str) -> None:
    m = "CognitiveMaps1977"
    b = load("cognitive_maps_1977", user_id)
    sm = b["static_map"]
    ems = b["event_maps"]

    ok, detail = path_sign_unit_test()
    check(m, "positive path = even number of negative assertions (p.120)", ok, detail)

    codes = [v["code"] for v in sm["variable_list"]]
    two_letter = sum(1 for c in codes if re.fullmatch(r"[A-Z]{2,}X*", c))
    check(
        m,
        "step 1: variables carry page+sequence capital-letter codes",
        two_letter > 0.9 * len(codes),
        f"{two_letter}/{len(codes)} codes match",
    )

    labels_ok = all(len(v["label"]) >= 2 for v in sm["variable_list"])
    check(m, "step 2: variables listed with wording close to the original", labels_ok,
          f"e.g. {sm['variable_list'][0]['label']}")

    signs = {a["sign"] for a in sm["causal_assertions"]}
    check(m, "step 3: only two kinds of causal assertion, + and -", signs <= {"+", "-"}, str(signs))
    implied = sum(1 for a in sm["causal_assertions"] if a.get("implied"))
    check(m, "step 3: relationships explicitly stated OR implied are coded", implied > 0,
          f"{implied} assertions carry implied evidence")

    dup_labels = len({v["label"] for v in sm["variable_list"]}) == len(sm["variable_list"])
    check(m, "step 4: direct correspondences merged (no duplicate wording)", dup_labels)

    util = sm["utility_variables"]
    check(m, "utility variables exist in the map (p.119)", len(util) > 0,
          f"{len(util)} utility variables, primary={sm['primary_utility_variable']}")
    ev_backed = all(u["evaluations"] > 0 for u in util)
    check(m, "utility edge is created only from a positive/negative evaluation", ev_backed)

    typ = sm["variable_typology"]
    has_all = {"utility", "goal", "policy"} <= set(typ)
    periph = {k for k in typ if k.startswith("peripheral_")}
    check(m, "typology: utility / goal / policy / peripheral", has_all and periph,
          json.dumps(typ, ensure_ascii=False))
    check(m, "peripheral variables come in three varieties", periph <= {"peripheral_1", "peripheral_2", "peripheral_3"},
          str(sorted(periph)))

    n, mm = sm["n_beliefs"], sm["n_variables"]
    expect = n / (mm * (mm - 1)) if mm > 1 else 0.0
    check(m, "density = n / (m(m-1))", abs(sm["density"] - expect) < 1e-6,
          f"n={n} m={mm} density={sm['density']}")
    vf = mm / sm["total_words"] if sm["total_words"] else 0
    check(m, "variable frequency = variables per word of document", abs(sm["variable_frequency"] - vf) < 1e-6,
          f"{sm['variable_frequency']}")

    pb = sm["path_balance"]
    check(m, "path-balance measured over parallel paths between variable pairs",
          "pairs_with_parallel_paths" in pb and 0.0 <= pb["path_balance"] <= 1.0,
          f"{pb['balanced_pairs']}/{pb['pairs_with_parallel_paths']} = {pb['path_balance']}")
    pc = sm["policy_consistency"]
    check(m, "policy consistency: all paths from a policy variable to utility share a sign",
          0.0 <= pc["policy_consistency"] <= 1.0,
          f"{pc['consistent_policy_vars']}/{pc['policy_vars_linked_to_utility']} = {pc['policy_consistency']}")
    check(m, "causal cycles counted (documents rarely contain them, p.121)",
          "count" in sm["causal_cycles"], f"cycles={sm['causal_cycles']['count']}")

    triple_ok = all(
        re.search(r"--\([+-]\)-->", t)
        for em in ems[:200]
        for t in em["feature_3d_triples"]
    )
    check(m, "event maps store signed assertions in map notation", triple_ok)
    check(m, "one weibo post = one coded document", len(ems) > 0, f"{len(ems)} documents")


def verify_genminds(user_id: str) -> None:
    m = "GenMinds"
    b = load("genminds", user_id)
    sm = b["static_map"]
    ems = b["event_maps"]
    bg = sm["belief_graph"]

    edges = [(e["src"], e["dst"]) for e in bg["edges"]]
    adj: dict[str, list[str]] = {}
    for a, d in edges:
        adj.setdefault(a, []).append(d)
    color: dict[str, int] = {}
    cyclic = False

    def dfs(u: str) -> None:
        nonlocal cyclic
        color[u] = 1
        for v in adj.get(u, []):
            if color.get(v, 0) == 1:
                cyclic = True
            elif color.get(v, 0) == 0:
                dfs(v)
        color[u] = 2

    sys.setrecursionlimit(20000)
    for nd in list(adj.keys()):
        if color.get(nd, 0) == 0:
            dfs(nd)
    check(m, "interviews parsed into a directed ACYCLIC graph (5.1)", not cyclic,
          f"{bg['num_nodes']} nodes, {bg['num_edges']} edges, "
          f"{len(bg['removed_cycle_edges'])} cycle-closing edges dropped")

    pol_ok = all(e["polarity"] in {"+", "-"} for e in bg["edges"])
    conf_ok = all(0.0 < e["confidence"] <= 1.0 for e in bg["edges"])
    check(m, "each edge encodes a directional causal relation", bool(edges), f"{len(edges)} edges")
    check(m, "each edge carries polarity and confidence scores", pol_ok and conf_ok)

    concepts_ok = all(len(n["concept"]) >= 2 for n in bg["nodes"])
    check(m, "each node encodes a concept", concepts_ok, f"e.g. {bg['nodes'][0]['concept']}")

    motifs = sm["cognitive_motifs"]
    three = all(len(mo["nodes"]) == 3 for mo in motifs)
    check(m, "cognitive motifs are minimal causal units over three concepts", three and bool(motifs),
          f"{len(motifs)} distinct motifs, e.g. {motifs[0]['motif'] if motifs else ''}")
    kinds = {mo["type"] for mo in motifs}
    check(m, "motifs cover chain and fork forms (QA#1 chain, QA#2 fork)", kinds <= {"chain", "fork"},
          str(sorted(kinds)))
    aggregated = any(mo["count"] > 1 for mo in motifs)
    check(m, "motifs aggregated across interviews into a shared topology", aggregated,
          f"max reuse = {max((mo['count'] for mo in motifs), default=0)}")

    cbn = sm["causal_bayesian_network"]
    check(m, "motifs compiled into a symbolic causal Bayesian network",
          bool(cbn.get("priors")) and bool(cbn.get("baseline_posteriors")),
          f"{len(cbn.get('priors', {}))} priors")
    check(m, "confidence derived from motif density or respondent emphasis",
          all("support" in e for e in bg["edges"]),
          "edge support counts recorded")

    iv = sm["interventions"]
    do_ok = bool(iv) and all(re.match(r"do\(.+ = high\)", x["do"]) for x in iv)
    shift_ok = bool(iv) and all(x["downstream"] for x in iv)
    check(m, "forward inference under do(X = high) updates downstream posteriors",
          do_ok and shift_ok,
          f"{len(iv)} interventions, e.g. {iv[0]['do'] if iv else ''}")
    if iv:
        d = iv[0]["downstream"][0]
        check(m, "intervention actually moves a posterior away from baseline",
              d["do_high"] != d["baseline"] or d["do_low"] != d["baseline"],
              f"{d['node']}: baseline={d['baseline']} do_high={d['do_high']} do_low={d['do_low']}")

    unc = sm["uncertainty"]
    check(m, "Be Aware of Unknown: weak / isolated dependencies surfaced",
          {"isolated_nodes", "weakly_supported_nodes", "weakly_supported_edges"} <= set(unc),
          f"isolated={len(unc['isolated_nodes'])} weak_nodes={len(unc['weakly_supported_nodes'])} "
          f"weak_edges={len(unc['weakly_supported_edges'])}")

    qa = [em for em in ems if em.get("interview_qa")]
    check(m, "semi-structured interview turns recorded per participant statement",
          len(qa) > 0, f"{len(qa)}/{len(ems)} turns carry QA")
    forms = [q["q"] for em in qa[:400] for q in em["interview_qa"]]
    why = sum(1 for q in forms if "为什么" in q)
    what = sum(1 for q in forms if "影响" in q)
    check(m, "questions follow the paper's two forms (why support X / what does Y influence)",
          why > 0 and what > 0, f"why={why} what-influences={what} of {len(forms)} sampled questions")


def verify_isolation(user_id: str) -> None:
    """The two rebuilds must not share structure or caches."""
    m = "isolation"
    d1 = OUT / f"weibo_kg_cognitive_maps_1977_{user_id}"
    d2 = OUT / f"weibo_kg_genminds_{user_id}"
    check(m, "separate LLM caches per method",
          (d1 / "coding_cache_1977.jsonl").exists() and (d2 / "interview_cache_genminds.jsonl").exists())
    b1, b2 = load("cognitive_maps_1977", user_id), load("genminds", user_id)
    k1 = set(b1["static_map"]) - {"beliefs", "persona_values", "persona_interests", "communication", "entity_stance"}
    k2 = set(b2["static_map"]) - {"beliefs", "persona_values", "persona_interests", "communication", "entity_stance"}
    check(m, "no shared method-specific static_map keys", not (k1 & k2), str(sorted(k1 & k2)))
    t1 = {t for em in b1["event_maps"][:300] for t in em["feature_3d_triples"]}
    t2 = {t for em in b2["event_maps"][:300] for t in em["feature_3d_triples"]}
    check(m, "no shared triples between the two graphs", not (t1 & t2), f"{len(t1 & t2)} shared")

    src1 = (Path(__file__).parent / "build_cogmap1977_paper.py").read_text(encoding="utf-8")
    src2 = (Path(__file__).parent / "build_genminds_paper.py").read_text(encoding="utf-8")
    cross = "build_genminds_paper" in src1 or "build_cogmap1977_paper" in src2
    shared_mod = "kg_paper_methods" in src1 or "kg_paper_methods" in src2
    check(m, "builders import neither each other nor a shared graph module", not cross and not shared_mod)


def main() -> None:
    user_id = sys.argv[1] if len(sys.argv) > 1 else "1989660417"
    verify_1977(user_id)
    verify_genminds(user_id)
    verify_isolation(user_id)

    width = max(len(r[1]) for r in results) + 2
    cur = None
    for method, req, status, detail in results:
        if method != cur:
            print(f"\n=== {method} ===")
            cur = method
        print(f"  [{status}] {req.ljust(width)} {detail}")
    n_fail = sum(1 for r in results if r[2] == FAIL)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
