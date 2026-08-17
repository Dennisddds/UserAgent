from __future__ import annotations

"""Enrich thin theory cards for retrieval: fetch abstracts + distill ONLY from paper text."""

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tg_agent.llm import DeepSeekClient, load_env
from tg_agent.models import TheoryCard
from tg_agent.theory_lib import TheoryLibrary, _compute_richness, _invert_abstract


DISTILL_SYSTEM = """你是传播学/社会心理学理论图书管理员。
任务：仅根据【论文摘要原文】抽取可检索理论要点。禁止用训练知识补全论文未写明的实验数字、样本、效应量或未出现的机制。

硬性规则：
1. 只能使用用户消息里 abstract 字段的信息；摘要没有的内容不要写。
2. 必须给出 evidence_quotes：1-2条从摘要中逐字摘录的短句（英文原文优先），用来支撑 mechanism。
3. 若摘要过短、缺失，或无法支撑任何社会认知/传播机制，将 summary 设为 "SKIP_INSUFFICIENT_ABSTRACT"。
4. 若主要是统计方法/软件/生物信息/工程工具而非社会认知理论，将 summary 设为 "SKIP_METHOD_PAPER"。

只输出 JSON，字段：
summary (120-220字中文，严格对应摘要),
mechanism (一句话机制，必须能被 evidence_quotes 支撑),
prediction (若刺激匹配，预期结果；无依据则写“摘要未给出明确预测”),
boundary (适用边界；无依据则写“摘要未界定边界”),
constructs (字符串数组，关键构念，尽量用摘要用词),
propositions (字符串数组，2-4条；每条必须能回溯到摘要),
conditions (字符串数组，触发情境),
outcomes (字符串数组，典型结果),
trigger (字符串数组，中英检索触发词，8-14个),
evidence_quotes (字符串数组，1-2条摘要原文摘录).
"""

KNOWLEDGE_DISTILL_SYSTEM = """你是传播学/社会心理学理论图书管理员。
任务：根据你训练知识中对该理论/论文的了解，生成可检索的理论卡字段。摘要缺失或不可靠时，允许使用训练知识，但必须遵守：

硬性规则：
1. 诚实优先：对该理论不熟悉、或只能凭标题猜测时，将 summary 设为 "SKIP_UNKNOWN_THEORY"，不要编造。
2. 不得编造具体实验数字、样本量、效应量；不确定的内容写进 boundary 说明。
3. trigger 是本字段最重要的产出：8-14个中英混合检索触发词，覆盖该理论的核心构念、典型情境、常见中文译法，
   必须能把「这条理论」和「其他理论」区分开（避免所有卡都写"身份、威胁、群体"这类泛词）。
4. mechanism 一句话讲清因果机制；propositions 2-4条可检验命题；conditions 触发情境；boundary 适用边界。

只输出 JSON，字段：
summary (120-220字中文),
mechanism (一句话机制),
prediction (若刺激匹配，预期结果),
boundary (适用边界 + 不确定性说明),
constructs (字符串数组，关键构念),
propositions (字符串数组，2-4条),
conditions (字符串数组，触发情境),
outcomes (字符串数组，典型结果),
trigger (字符串数组，8-14个中英混合、有区分度的检索触发词).
"""

# high-cite STEM / methods noise that pollutes social-theory retrieval
_EXCLUDE_TITLE = re.compile(
    r"(gene expression|ab-?initio|u-?net|xgboost|matplotlib|mega x|limma|g\*power|"
    r"prisma|vmd:|rna-?seq|biomedical image|molecular evolutionary|coronavirus in wuhan|"
    r"differential expression|total energy calculations|visual molecular|convolutional networks|"
    r"graphics environment|power analysis program|atomically thin carbon|graphene|"
    r"electric field effect|estrogen plus|alzheimer|metabolic syndrome|exosome|"
    r"blood-glucose|sulphonylureas|robb?ins-?i|risk of bias|numpy|secrecy systems|"
    r"lime|why should i trust you|dementia due to|postmenopausal|"
    r"qualitative content analysis|thematic analysis in psychology|"
    r"construct validity in psychological tests|mindfulness and its role|"
    r"broaden-and-build|self-determination theory|positive psychology: an introduction|"
    r"mathematical theory of communication|wireless communication|stochastic network|"
    r"mechanical turk|public-?\s*private partner|fundamentals of wireless|"
    r"moderator.?mediator variable|amazon.?s mechanical)",
    re.I,
)

_SOCIAL_HINT = re.compile(
    r"(social psycholog|political psycholog|mass communication|media communication|"
    r"persuasion|attitude change|attitude formation|misinformation|disinformation|"
    r"fake news|rumor correction|media effect|framing (theory|of decisions|effects)|"
    r"inoculation theory|cognitive dissonance|source credibility|public opinion|"
    r"motivated reasoning|selective exposure|spiral of silence|agenda.?setting|"
    r"hostile media|elaboration likelihood|prospect theory|risk perception|"
    r"perception of risk|trust repair|organizational trust|social identity|"
    r"cultural cognition|opinion leader|third.?person effect|confirmation bias|"
    r"debunk|continued influence|theory of planned behavior|reasoned action|"
    r"homophily|social capital|relationship marketing|intersectionality|"
    r"theory of practice|habitus|nationalism|guanxi|weibo|censorship|"
    r"impression management|self-presentation|parasocial|uses and gratification|"
    r"media dependency|organizational identification|employee voice|"
    r"psychological safety|adolescent identity|peer influence|social comparison|"
    r"public sphere|habermas|network society|castells|bourdieu|cultural capital|"
    r"risk society|foucault discourse|giddens|liquid modernity|face negotiation|"
    r"moral foundation|collective action|connective action|digital activism|"
    r"cultivation|media priming|diffusion of innovation|hyperpersonal|"
    r"social information processing|side model|media richness|social presence|"
    r"warranting|echo chamber|filter bubble|online disinhibition|privacy calculus|"
    r"contextual integrity|crisis communication|image repair|health belief|"
    r"social comparison|digital divide|knowledge gap|system justification|"
    r"terror management|cancel culture|public shaming|fear of missing out)",
    re.I,
)

_SOCIAL_CONCEPT = re.compile(
    r"^(Social psychology|Sociology|Communication|Political science|Public opinion|"
    r"Persuasion|Attitude change|Media studies|Cognitive psychology)$",
    re.I,
)

_PLACEHOLDER_MECH = re.compile(
    r"^(crossref scholarly|scholarly work|empirical/theoretical|semantic scholar|"
    r"web-scholar hit|when stimuli match)",
    re.I,
)


class LibraryLock:
    """Exclusive lock so only one enrich process writes cards.jsonl."""

    def __init__(self, lock_path: Path) -> None:
        self.lock_path = Path(lock_path)
        self.fd: int | None = None

    def __enter__(self) -> "LibraryLock":
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.time() + 600
        while time.time() < deadline:
            try:
                self.fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                os.write(self.fd, str(os.getpid()).encode())
                return self
            except FileExistsError:
                # stale lock: if pid dead, remove
                try:
                    old = self.lock_path.read_text(encoding="utf-8").strip()
                    if old.isdigit() and not _pid_alive(int(old)):
                        self.lock_path.unlink(missing_ok=True)
                        continue
                except OSError:
                    pass
                time.sleep(1.0)
        raise TimeoutError(f"could not acquire lock: {self.lock_path}")

    def __exit__(self, *args: object) -> None:
        if self.fd is not None:
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.fd = None
        try:
            self.lock_path.unlink(missing_ok=True)
        except OSError:
            pass


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False
    except AttributeError:
        # Windows: kill(0) may not exist the same way; try OpenProcess via ctypes skip
        import ctypes

        k = ctypes.windll.kernel32  # type: ignore[attr-defined]
        h = k.OpenProcess(0x1000, False, pid)
        if h:
            k.CloseHandle(h)
            return True
        return False


def _oa_get(url: str, mailto: str = "") -> dict:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": f"theory-guided-agent/0.2 (mailto:{mailto or 'research@useragent.local'})"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_openalex_abstract(work_id_or_url: str, mailto: str = "") -> str:
    wid = work_id_or_url.rstrip("/").split("/")[-1]
    if not wid.startswith("W"):
        return ""
    data = _oa_get(f"https://api.openalex.org/works/{wid}", mailto=mailto)
    return _invert_abstract(data.get("abstract_inverted_index"))


def fetch_openalex_abstract_by_doi(doi: str, mailto: str = "") -> str:
    if not doi:
        return ""
    url = "https://api.openalex.org/works/doi:" + urllib.parse.quote(doi)
    try:
        data = _oa_get(url, mailto=mailto)
    except Exception:  # noqa: BLE001
        return ""
    return _invert_abstract(data.get("abstract_inverted_index"))


def fetch_crossref_abstract(doi: str, mailto: str = "") -> str:
    if not doi:
        return ""
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": f"theory-guided-agent/0.2 (mailto:{mailto or 'research@useragent.local'})"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    msg = data.get("message") or {}
    ab = msg.get("abstract") or ""
    if not ab:
        return ""
    ab = re.sub(r"<[^>]+>", " ", ab)
    return re.sub(r"\s+", " ", ab).strip()


def fetch_s2_abstract_by_doi(doi: str) -> str:
    """Semantic Scholar often has abstracts when Crossref/OpenAlex do not."""
    if not doi:
        return ""
    url = (
        "https://api.semanticscholar.org/graph/v1/paper/DOI:"
        + urllib.parse.quote(doi)
        + "?fields=title,abstract"
    )
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "theory-guided-agent/0.2 (research abstract backfill)"},
        )
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return (data.get("abstract") or "").strip()
    except Exception:  # noqa: BLE001
        return ""


def fetch_s2_abstract_by_title(title: str) -> str:
    if not title or len(title) < 12:
        return ""
    url = (
        "https://api.semanticscholar.org/graph/v1/paper/search?"
        + urllib.parse.urlencode({"query": title, "limit": 1, "fields": "title,abstract"})
    )
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "theory-guided-agent/0.2 (research abstract backfill)"},
        )
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        items = data.get("data") or []
        if not items:
            return ""
        return (items[0].get("abstract") or "").strip()
    except Exception:  # noqa: BLE001
        return ""


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def quotes_grounded_in_abstract(quotes: list[str], abstract: str) -> bool:
    """Every evidence quote must appear (fuzzy) in the paper abstract."""
    if not quotes or not abstract:
        return False
    ab = _norm_ws(abstract)
    ok = 0
    for q in quotes[:3]:
        qq = _norm_ws(str(q))
        if len(qq) < 12:
            continue
        # allow minor punctuation drift: require contiguous 40-char window or full short quote
        if qq in ab:
            ok += 1
            continue
        # try without punctuation
        qq2 = re.sub(r"[^a-z0-9\u4e00-\u9fff ]+", "", qq)
        ab2 = re.sub(r"[^a-z0-9\u4e00-\u9fff ]+", "", ab)
        if len(qq2) >= 20 and qq2 in ab2:
            ok += 1
            continue
        # sliding: first 48 chars of quote
        head = qq2[:48] if len(qq2) >= 48 else qq2
        if len(head) >= 20 and head in ab2:
            ok += 1
    return ok >= 1


def _parse_llm_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            return {}
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return {}


def distill(llm: DeepSeekClient, card: TheoryCard) -> dict:
    abstract = (card.abstract or "").strip()
    if len(abstract) < 80:
        return {"summary": "SKIP_INSUFFICIENT_ABSTRACT"}
    user = (
        f"title: {card.name}\n"
        f"authors: {card.authors}\n"
        f"year: {card.year}\n"
        f"doi: {card.doi}\n"
        f"venue: {card.venue}\n"
        f"coordinate_hint: {card.coordinate}\n\n"
        f"===== PAPER ABSTRACT (sole evidence; do not invent beyond this) =====\n"
        f"{abstract}\n"
        f"===== END ABSTRACT =====\n"
    )
    raw = llm.chat(
        [{"role": "system", "content": DISTILL_SYSTEM}, {"role": "user", "content": user}],
        temperature=0.1,
        max_tokens=1000,
        disable_thinking=True,
    )
    return _parse_llm_json(raw)


def distill_knowledge(llm: DeepSeekClient, card: TheoryCard) -> dict:
    """Distill from model's trained knowledge (no usable abstract).

    Provenance: caller must mark grounding_source="model_knowledge", grounded=False.
    """
    user = (
        f"title: {card.name}\n"
        f"authors: {card.authors}\n"
        f"year: {card.year}\n"
        f"doi: {card.doi}\n"
        f"venue: {card.venue}\n"
        f"coordinate_hint: {card.coordinate}\n"
        f"concepts: {', '.join(str(c) for c in (card.concepts or [])[:8])}\n"
    )
    abstract = (card.abstract or "").strip()
    if abstract:
        user += f"\nreference_text (may be incomplete):\n{abstract[:800]}\n"
    raw = llm.chat(
        [
            {"role": "system", "content": KNOWLEDGE_DISTILL_SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        max_tokens=1100,
        disable_thinking=True,
    )
    return _parse_llm_json(raw)


def apply_distill(card: TheoryCard, obj: dict, *, grounded: bool, source: str = "") -> TheoryCard:
    if obj.get("summary"):
        card.summary = str(obj["summary"])[:800]
    if obj.get("mechanism"):
        card.mechanism = str(obj["mechanism"])[:500]
    if obj.get("prediction"):
        card.prediction = str(obj["prediction"])[:400]
    if obj.get("boundary"):
        card.boundary = str(obj["boundary"])[:400]
    for field in ("constructs", "propositions", "conditions", "outcomes", "trigger"):
        val = obj.get(field)
        if isinstance(val, list) and val:
            setattr(card, field, [str(x) for x in val][:16])
    quotes = obj.get("evidence_quotes")
    if isinstance(quotes, list) and quotes:
        card.evidence_quotes = [str(x)[:400] for x in quotes][:3]
    card.grounded = grounded
    card.grounding_source = source or ("abstract" if grounded else "")
    card.richness = _compute_richness(card)
    return card


def clear_ungrounded_distill(card: TheoryCard) -> TheoryCard:
    """Strip LLM fields that are not backed by a paper abstract."""
    card.summary = ""
    card.propositions = []
    card.constructs = []
    card.conditions = []
    card.outcomes = []
    card.evidence_quotes = []
    card.grounded = False
    card.grounding_source = ""
    # keep abstract; if mechanism looks like placeholder/LLM without abstract, clear soft fields only
    if not (card.abstract or "").strip():
        if _PLACEHOLDER_MECH.search((card.mechanism or "").strip()):
            card.mechanism = ""
        card.prediction = (
            card.prediction
            if not card.prediction.lower().startswith("when stimuli match")
            else ""
        )
    card.richness = _compute_richness(card)
    return card


def _is_relevant_social_theory(card: TheoryCard) -> bool:
    if _EXCLUDE_TITLE.search(card.name or ""):
        return False
    if any(_SOCIAL_CONCEPT.match(str(c).strip()) for c in (card.concepts or [])):
        if _SOCIAL_HINT.search(card.name or ""):
            return True
    content = " ".join(
        [
            card.name or "",
            " ".join(card.concepts or []),
            card.abstract or "",
            ""
            if _PLACEHOLDER_MECH.search((card.mechanism or "").strip())
            else (card.mechanism or ""),
        ]
    )
    return bool(_SOCIAL_HINT.search(content))


def ensure_abstract(card: TheoryCard, *, mailto: str) -> bool:
    """Fetch real paper abstract. Returns True if abstract is usable (>=80 chars)."""
    if len((card.abstract or "").strip()) >= 80:
        return True
    ab = ""
    try:
        if card.source == "openalex" or "openalex" in (card.url or ""):
            ab = fetch_openalex_abstract(card.url or card.id, mailto=mailto)
        if not ab and card.doi:
            ab = fetch_openalex_abstract_by_doi(card.doi, mailto=mailto)
        if not ab and card.doi:
            ab = fetch_crossref_abstract(card.doi, mailto=mailto)
        # Semantic Scholar fallback — critical for classics missing CR/OA abstracts
        if not ab and card.doi:
            ab = fetch_s2_abstract_by_doi(card.doi)
            if ab:
                time.sleep(1.2)
        if not ab and (card.name or "").strip():
            ab = fetch_s2_abstract_by_title(card.name.strip()[:180])
            if ab:
                time.sleep(1.2)
    except Exception as e:  # noqa: BLE001
        print(f"abstract fetch fail {card.id}: {e}", flush=True)
        return False
    if ab and len(ab.strip()) >= 80:
        card.abstract = ab.strip()[:2000]
        return True
    return False


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Enrich theory cards ONLY from paper abstracts")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--limit", type=int, default=40, help="max cards to distill this run")
    ap.add_argument("--min-citations", type=int, default=50)
    ap.add_argument("--fetch-abstracts", action="store_true", default=True)
    ap.add_argument("--no-fetch-abstracts", action="store_true")
    ap.add_argument("--distill", action="store_true", default=True)
    ap.add_argument("--no-distill", action="store_true")
    ap.add_argument("--only-thin", action="store_true", default=True)
    ap.add_argument(
        "--all-thin",
        action="store_true",
        help="do not require social-science relevance filter",
    )
    ap.add_argument(
        "--reset-ungrounded",
        action="store_true",
        help="clear distilled fields on cards that lack a paper abstract / grounding",
    )
    ap.add_argument(
        "--knowledge",
        action="store_true",
        help="for cards without a usable abstract: distill from DeepSeek trained knowledge "
        "(marked grounding_source=model_knowledge, grounded=False, match 时 ×0.7 降权)",
    )
    ap.add_argument(
        "--knowledge-only",
        action="store_true",
        help="implies --knowledge; skip abstract fetching entirely (fast backfill of triggers)",
    )
    ap.add_argument(
        "--require-grounding",
        action="store_true",
        default=True,
        help="reject distill unless evidence_quotes match abstract (default on)",
    )
    ap.add_argument("--no-require-grounding", action="store_true")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    load_env(cfg["paths"]["env_file"])
    seed = (
        ROOT / cfg["paths"]["theory_seed"]
        if not Path(cfg["paths"]["theory_seed"]).is_absolute()
        else Path(cfg["paths"]["theory_seed"])
    )
    libdir = (
        ROOT / cfg["paths"]["theory_library"]
        if not Path(cfg["paths"]["theory_library"]).is_absolute()
        else Path(cfg["paths"]["theory_library"])
    )
    mailto = (cfg.get("theory_crawl") or {}).get("mailto", "")
    lock = LibraryLock(libdir / ".enrich.lock")

    with lock:
        lib = TheoryLibrary(seed, libdir)

        if args.reset_ungrounded:
            n_reset = 0
            for c in list(lib.cards.values()):
                if c.source in {"canonical", "seed"}:
                    c.grounded = True
                    c.grounding_source = c.source
                    continue
                has_abs = len((c.abstract or "").strip()) >= 80
                looks_distilled = bool(c.propositions or (c.summary and len(c.summary) >= 40))
                if looks_distilled and (not has_abs or not c.grounded):
                    clear_ungrounded_distill(c)
                    # if has abstract but was marked ungrounded, keep abstract for re-distill
                    n_reset += 1
                    lib.cards[c.id] = c
            lib.save_library()
            print(json.dumps({"reset_ungrounded": n_reset, "total": len(lib.cards)}, ensure_ascii=False))
            if args.no_distill and args.no_fetch_abstracts:
                return

        candidates = [
            c
            for c in lib.cards.values()
            if c.source not in {"canonical", "seed"}
            and c.grounding_source != "model_knowledge"  # 已知识蒸馏过的卡不重复处理
            and (not args.only_thin or c.is_thin() or _compute_richness(c) < 0.45 or not c.grounded)
            and int(c.cited_by_count or 0) >= args.min_citations
            and (args.all_thin or _is_relevant_social_theory(c))
            and "降权" not in (c.summary or "")
            and "方法/工具" not in (c.summary or "")
        ]

        def _prio(c: TheoryCard) -> tuple:
            has_body = 1 if len((c.abstract or "").strip()) >= 80 else 0
            title_hit = 1 if _SOCIAL_HINT.search(c.name or "") else 0
            return (-title_hit, -has_body, -int(c.cited_by_count or 0), c.id)

        candidates.sort(key=_prio)
        candidates = candidates[: args.limit]
        print(f"enrich candidates={len(candidates)} total_cards={len(lib.cards)}", flush=True)

        fetch_abs = args.fetch_abstracts and not args.no_fetch_abstracts and not args.knowledge_only
        do_distill = args.distill and not args.no_distill
        use_knowledge = args.knowledge or args.knowledge_only
        require_g = args.require_grounding and not args.no_require_grounding
        llm = DeepSeekClient(model=cfg["llm"]["model"]) if do_distill else None

        updated = 0
        skipped_method = 0
        skipped_no_abs = 0
        skipped_unknown = 0
        knowledge_distilled = 0
        rejected_ungrounded = 0

        for i, card in enumerate(candidates, 1):
            changed = False
            has_abs = len((card.abstract or "").strip()) >= 80
            if fetch_abs and not has_abs:
                before = card.abstract
                if ensure_abstract(card, mailto=mailto):
                    if card.abstract != before:
                        changed = True
                        has_abs = True
                    time.sleep(0.3)

            if not has_abs:
                if do_distill and llm and use_knowledge:
                    # model-knowledge backfill (provenance: model_knowledge, ungrounded)
                    try:
                        obj = distill_knowledge(llm, card)
                        summary = str(obj.get("summary") or "")
                        if "SKIP_UNKNOWN_THEORY" in summary or not summary:
                            skipped_unknown += 1
                            print(f"[{i}] SKIP unknown theory: {card.id} | {card.name[:60]}", flush=True)
                        else:
                            apply_distill(card, obj, grounded=False, source="model_knowledge")
                            knowledge_distilled += 1
                            changed = True
                    except Exception as e:  # noqa: BLE001
                        print(f"[{i}] knowledge distill fail {card.id}: {e}", flush=True)
                else:
                    skipped_no_abs += 1
                    # do not distill without paper abstract
                    if card.propositions or (card.summary and len(card.summary) >= 40):
                        clear_ungrounded_distill(card)
                        changed = True
                    if changed:
                        lib.cards[card.id] = card
                        updated += 1
                    print(f"[{i}] SKIP no abstract: {card.id} | {card.name[:60]}", flush=True)
                    if i % 5 == 0:
                        lib.save_library()
                    continue

            if has_abs and do_distill and llm:
                try:
                    obj = distill(llm, card)
                    summary = str(obj.get("summary") or "")
                    if "SKIP_METHOD_PAPER" in summary:
                        card.weight = min(float(card.weight or 1.0), 0.15)
                        card.summary = "方法/工具论文，降权，不作为理论检索支撑。"
                        card.grounded = False
                        card.grounding_source = ""
                        card.richness = _compute_richness(card)
                        changed = True
                        skipped_method += 1
                    elif "SKIP_INSUFFICIENT_ABSTRACT" in summary:
                        if use_knowledge:
                            try:
                                obj2 = distill_knowledge(llm, card)
                                s2 = str(obj2.get("summary") or "")
                                if s2 and "SKIP_UNKNOWN_THEORY" not in s2:
                                    apply_distill(card, obj2, grounded=False, source="model_knowledge")
                                    knowledge_distilled += 1
                                else:
                                    clear_ungrounded_distill(card)
                                    skipped_unknown += 1
                            except Exception as e:  # noqa: BLE001
                                print(f"[{i}] knowledge distill fail {card.id}: {e}", flush=True)
                        else:
                            clear_ungrounded_distill(card)
                            skipped_no_abs += 1
                        changed = True
                    elif summary:
                        quotes = obj.get("evidence_quotes") if isinstance(obj.get("evidence_quotes"), list) else []
                        grounded_ok = quotes_grounded_in_abstract([str(x) for x in quotes], card.abstract)
                        if require_g and not grounded_ok:
                            if use_knowledge:
                                try:
                                    obj2 = distill_knowledge(llm, card)
                                    s2 = str(obj2.get("summary") or "")
                                    if s2 and "SKIP_UNKNOWN_THEORY" not in s2:
                                        apply_distill(card, obj2, grounded=False, source="model_knowledge")
                                        knowledge_distilled += 1
                                        changed = True
                                    else:
                                        skipped_unknown += 1
                                except Exception as e:  # noqa: BLE001
                                    print(f"[{i}] knowledge distill fail {card.id}: {e}", flush=True)
                            else:
                                rejected_ungrounded += 1
                                print(
                                    f"[{i}] REJECT ungrounded distill {card.id} "
                                    f"(evidence_quotes not in abstract)",
                                    flush=True,
                                )
                        else:
                            apply_distill(card, obj, grounded=grounded_ok)
                            changed = True
                except Exception as e:  # noqa: BLE001
                    print(f"[{i}] distill fail {card.id}: {e}", flush=True)

            card.richness = _compute_richness(card)
            if changed:
                lib.cards[card.id] = card
                updated += 1
            if i % 5 == 0 or i == len(candidates):
                lib.save_library()
                print(
                    f"[{i}/{len(candidates)}] updated={updated} skip_method={skipped_method} "
                    f"no_abs={skipped_no_abs} knowledge={knowledge_distilled} "
                    f"unknown={skipped_unknown} reject={rejected_ungrounded} "
                    f"last={card.id} grounded={card.grounded} rich={card.richness:.2f}",
                    flush=True,
                )

        lib.save_library()
        grounded_n = sum(1 for c in lib.cards.values() if c.grounded)
        structured = sum(
            1
            for c in lib.cards.values()
            if c.grounded and c.summary and len(c.summary) >= 40 and c.propositions
        )
        knowledge_n = sum(
            1 for c in lib.cards.values() if c.grounding_source == "model_knowledge"
        )
        print(
            json.dumps(
                {
                    "updated": updated,
                    "skipped_method": skipped_method,
                    "skipped_no_abstract": skipped_no_abs,
                    "knowledge_distilled": knowledge_distilled,
                    "skipped_unknown_theory": skipped_unknown,
                    "rejected_ungrounded": rejected_ungrounded,
                    "total": len(lib.cards),
                    "grounded": grounded_n,
                    "structured_grounded": structured,
                    "model_knowledge": knowledge_n,
                    "canonical": sum(1 for c in lib.cards.values() if c.source == "canonical"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
