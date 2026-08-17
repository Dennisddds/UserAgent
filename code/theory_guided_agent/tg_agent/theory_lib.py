from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from .models import MatchedTheory, TheoryCard

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore


def _tokenize(text: str) -> set[str]:
    text = (text or "").lower()
    hans = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    words = re.findall(r"[a-z0-9_]{3,}", text)
    out = set(words)
    for h in hans:
        out.add(h)
        if len(h) >= 4:
            for i in range(len(h) - 1):
                out.add(h[i : i + 2])
    return out


COORD_KEYWORDS: dict[str, list[str]] = {
    "risk_perception": ["risk", "dread", "fear appeal", "危险", "风险", "恐惧"],
    "trust": ["trust", "credibility", "epistemic", "信任", "辟谣", "credibility"],
    "identity_threat": ["identity threat", "national identity", "身份", "认同", "威胁"],
    "fairness": ["fair", "justice", "equity", "公平", "正义", "双标"],
    "technology_threat": ["technology", "automation", "algorithm", "ai anxiety", "技术", "算法"],
    "uncertainty_reduction": ["uncertainty", "visual evidence", "不确定", "可视化"],
    "motivated_reasoning": [
        "motivated reasoning",
        "persuasion",
        "elaboration",
        "backfire",
        "说服",
    ],
    "social_identity": ["social identity", "in-group", "out-group", "群体", "认同"],
    "framing": ["framing", "frame", "reframing", "框架"],
    "agenda_setting": ["agenda setting", "agenda-setting", "议程"],
    "spiral_of_silence": ["spiral of silence", "沉默螺旋"],
    "inoculation": ["inoculation", "prebunk", "prebunking", "接种"],
    "misinformation": [
        "misinformation",
        "disinformation",
        "fake news",
        "rumor",
        "debunk",
        "假新闻",
        "谣言",
    ],
    "moral_foundations": ["moral foundation", "haidt", "道德基础"],
    "affective_polarization": ["affective polarization", "partisan animosity", "极化"],
    "selective_exposure": ["selective exposure", "confirmation bias", "选择性接触"],
    "narrative_persuasion": ["narrative", "transportation", "exemplification", "叙事"],
    "cognitive_dissonance": ["cognitive dissonance", "festinger", "认知失调"],
    "prospect_theory": ["prospect theory", "loss aversion", "前景理论"],
    "cultural_cognition": ["cultural cognition", "worldview", "文化认知"],
    "source_credibility": ["source credibility", "hovland", "来源可信"],
    "collective_action": ["collective action", "bandwagon", "social proof", "集体行动"],
    "third_person_effect": ["third-person", "third person effect", "第三人效果"],
    "hostile_media": ["hostile media", "敌意媒体"],
    "opinion_leadership": ["opinion leader", "two-step flow", "意见领袖"],
    "dual_process": ["dual process", "system 1", "affect heuristic", "双过程"],
    "face_culture": ["face", "guanxi", "high-context", "mianzi", "面子", "关系"],
    "public_opinion_china": ["weibo", "china", "chinese", "censorship", "微博", "中国"],
    "organizational_behavior": [
        "organizational identification",
        "employee voice",
        "psychological safety",
        "workplace",
        "组织认同",
    ],
    "impression_management": [
        "impression management",
        "self-presentation",
        "goffman",
        "自我呈现",
        "印象管理",
    ],
    "developmental_media": [
        "adolescent",
        "peer influence",
        "emerging adulthood",
        "identity development",
        "青少年",
    ],
    "parasocial": ["parasocial", "influencer", "celebrity endorsement", "准社会"],
    "uses_gratifications": ["uses and gratifications", "media dependency", "使用与满足"],
    "social_capital": ["social capital", "bonding", "bridging", "社会资本"],
    "macro_social_theory": [
        "risk society",
        "foucault",
        "giddens",
        "bauman",
        "liquid modernity",
        "风险社会",
    ],
    "public_sphere": ["public sphere", "habermas", "deliberation", "公共领域"],
    "network_society": ["network society", "castells", "networked individualism", "网络社会"],
    "habitus_capital": ["bourdieu", "habitus", "cultural capital", "惯习", "文化资本"],
    "cultivation": ["cultivation", "gerbner", "mean world", "涵化"],
    "priming": ["priming", "iyengar", "启动"],
    "diffusion_innovation": ["diffusion of innovation", "rogers", "创新扩散"],
    "cmc_theory": [
        "hyperpersonal",
        "social information processing",
        "side model",
        "media richness",
        "social presence",
        "warranting",
        "cmc",
    ],
    "echo_chamber": ["echo chamber", "filter bubble", "回音室", "过滤气泡"],
    "algorithmic_curation": ["algorithmic", "recommender", "personalization", "算法推荐"],
    "online_disinhibition": ["disinhibition", "cyberbullying", "flaming", "去抑制"],
    "privacy_calculus": ["privacy calculus", "contextual integrity", "privacy paradox", "隐私"],
    "crisis_communication": ["crisis communication", "image repair", "coombs", "危机传播"],
    "health_communication": ["health belief", "fear appeal", "health misinformation", "健康传播"],
    "social_comparison": ["social comparison", "festinger", "upward comparison", "社会比较"],
    "media_dependency": ["media dependency", "media system dependency", "媒介依赖"],
    "digital_divide": ["digital divide", "knowledge gap", "数字鸿沟"],
    "system_justification": ["system justification", "jost", "系统合理化"],
    "terror_management": ["terror management", "mortality salience", "恐惧管理"],
    "cancel_culture": ["cancel culture", "public shaming", "moral outrage", "网络舆论审判"],
}


class TheoryLibrary:
    def __init__(
        self,
        seed_path: str | Path,
        library_dir: str | Path,
        *,
        canonical_path: str | Path | None = None,
    ) -> None:
        self.seed_path = Path(seed_path)
        self.library_dir = Path(library_dir)
        self.library_dir.mkdir(parents=True, exist_ok=True)
        self.canonical_path = (
            Path(canonical_path)
            if canonical_path
            else self.seed_path.parent / "canonical_theories.json"
        )
        self.cards: dict[str, TheoryCard] = {}
        self.coordinates: list[str] = []
        self.completed_queries: set[str] = set()
        self._load_seed()
        self._load_canonical()
        self._load_library()
        self._load_progress()
        self._refresh_richness()

    def _load_seed(self) -> None:
        data = json.loads(self.seed_path.read_text(encoding="utf-8"))
        self.coordinates = list(data.get("coordinates") or [])
        for raw in data.get("theories") or []:
            card = TheoryCard.from_dict(raw)
            card.source = card.source or "seed"
            card.grounded = True
            card.grounding_source = "seed"
            self.cards[card.id] = card

    def _load_canonical(self) -> None:
        if not self.canonical_path.exists():
            return
        data = json.loads(self.canonical_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("theories") or []
        for raw in data:
            card = TheoryCard.from_dict(raw)
            card.source = "canonical"
            card.grounded = True
            card.grounding_source = "canonical"
            card.richness = max(card.richness, _compute_richness(card))
            self.cards[card.id] = card

    def _load_library(self) -> None:
        path = self.library_dir / "cards.jsonl"
        if not path.exists():
            return
        bad = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                card = TheoryCard.from_dict(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
                continue
            self.cards[card.id] = card
        if bad:
            print(f"[theory_lib] skipped {bad} corrupt jsonl lines", flush=True)

    def _progress_path(self) -> Path:
        return self.library_dir / "crawl_progress.json"

    def _load_progress(self) -> None:
        p = self._progress_path()
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            self.completed_queries = set(data.get("completed_queries") or [])

    def _save_progress(self) -> None:
        self._progress_path().write_text(
            json.dumps(
                {
                    "completed_queries": sorted(self.completed_queries),
                    "num_cards": len(self.cards),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def save_library(self) -> None:
        path = self.library_dir / "cards.jsonl"
        tmp = path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for card in self.cards.values():
                if card.source in {"seed", "canonical"}:
                    continue
                f.write(json.dumps(card.to_dict(), ensure_ascii=False) + "\n")
        tmp.replace(path)
        by_coord: dict[str, int] = {}
        by_source: dict[str, int] = {}
        for c in self.cards.values():
            by_coord[c.coordinate or "unknown"] = by_coord.get(c.coordinate or "unknown", 0) + 1
            by_source[c.source or "unknown"] = by_source.get(c.source or "unknown", 0) + 1
        meta = {
            "num_cards": len(self.cards),
            "num_crawled": sum(1 for c in self.cards.values() if c.source != "seed"),
            "coordinates": self.coordinates,
            "by_coordinate": dict(sorted(by_coord.items(), key=lambda x: (-x[1], x[0]))),
            "by_source": by_source,
            "completed_queries": len(self.completed_queries),
        }
        (self.library_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._save_progress()

    def merge_coordinates(self, coords: list[str]) -> None:
        seen = set(self.coordinates)
        for c in coords:
            if c and c not in seen:
                self.coordinates.append(c)
                seen.add(c)

    def _refresh_richness(self) -> None:
        for card in self.cards.values():
            card.richness = _compute_richness(card)

    def match(
        self,
        stimulus: str,
        *,
        top_k: int = 3,
        user_weights: dict[str, float] | None = None,
        env_weights: dict[str, float] | None = None,
        prefer_rich: bool = True,
        prefer_grounded: bool = True,
        min_richness: float = 0.0,
    ) -> list[MatchedTheory]:
        """Retrieve theory cards; blend stimulus overlap with user-environment priors."""
        q = _tokenize(stimulus)
        user_weights = user_weights or {}
        env_weights = env_weights or {}
        scored: list[MatchedTheory] = []
        for card in self.cards.values():
            if prefer_rich and card.is_thin() and card.source not in {"canonical", "seed"}:
                richness_mul = 0.25
            else:
                richness_mul = 0.55 + 0.45 * float(card.richness or _compute_richness(card))
            if min_richness and float(card.richness or 0) < min_richness and card.source not in {
                "canonical",
                "seed",
            }:
                continue
            if prefer_grounded and card.source not in {"canonical", "seed"}:
                if not getattr(card, "grounded", False) and card.is_thin():
                    continue

            body_toks = _tokenize(card.retrieval_text())
            trig = set(t.lower() for t in card.trigger) | _tokenize(card.coordinate)
            inter_trig = len(q & trig)
            inter_body = len(q & body_toks)
            denom = math_sqrt(len(q) * max(1, len(trig | body_toks)))
            base = (1.4 * inter_trig + inter_body) / max(1.0, denom)
            # environment prior: allow retrieval even when stimulus lexical overlap is weak
            env_prior = 0.0
            if card.coordinate in env_weights:
                env_prior = 0.10 * float(env_weights[card.coordinate])
            source_boost = 1.35 if card.source == "canonical" else 1.0
            if card.source == "seed" and not card.is_thin():
                source_boost = 1.2
            if getattr(card, "grounded", False) and card.grounding_source == "abstract":
                source_boost *= 1.15
            elif card.source not in {"canonical", "seed"} and not getattr(card, "grounded", False):
                source_boost *= 0.7
            w = user_weights.get(card.id, user_weights.get(card.coordinate, 1.0))
            env_mul = float(env_weights.get(card.coordinate, 1.0)) if env_weights else 1.0
            # soft env: if coord in env profile, multiply; else slight downweight when env known
            if env_weights:
                env_mul = float(env_weights.get(card.coordinate, 0.72))
            score = (base + env_prior) * float(card.weight) * float(w) * richness_mul * source_boost * env_mul
            if score <= 0:
                continue
            hits = sorted(q & set(t.lower() for t in card.trigger))
            why = (
                f"triggers={hits[:6]} coord={card.coordinate} "
                f"rich={card.richness:.2f} src={card.source} "
                f"grounded={getattr(card, 'grounded', False)} "
                f"env={env_mul:.2f} w={w:.2f}"
            )
            scored.append(MatchedTheory(card=card, score=score, why=why))
        scored.sort(key=lambda m: m.score, reverse=True)
        picked: list[MatchedTheory] = []
        seen_coord: set[str] = set()
        for m in scored:
            if m.card.coordinate in seen_coord:
                if not (m.card.richness >= 0.7 and len(picked) < top_k):
                    continue
            picked.append(m)
            seen_coord.add(m.card.coordinate)
            if len(picked) >= top_k:
                break
        return picked

    def crawl_openalex(
        self,
        queries: list[str],
        *,
        per_query: int = 8,
        mailto: str = "",
        pages: int = 1,
        default_coordinate: str | None = None,
        query_coords: dict[str, str] | None = None,
    ) -> int:
        """Crawl OpenAlex works. per_query is per page (max 200); pages paginate via cursor."""
        added = 0
        per_page = max(1, min(int(per_query), 200))
        pages = max(1, int(pages))
        query_coords = query_coords or {}
        pending = [q for q in queries if q not in self.completed_queries]
        print(
            f"[openalex] resume: {len(self.completed_queries)} done, "
            f"{len(pending)} pending / {len(queries)} total; cards={len(self.cards)}",
            flush=True,
        )
        for qi, q in enumerate(pending):
            cursor = "*"
            got = 0
            ok = True
            try:
                for _page in range(pages):
                    url = (
                        "https://api.openalex.org/works?"
                        + urllib.parse.urlencode(
                            {
                                "search": q,
                                "per_page": per_page,
                                "sort": "cited_by_count:desc",
                                "cursor": cursor,
                            }
                        )
                    )
                    payload = _http_get_json(url, mailto=mailto)
                    results = payload.get("results") or []
                    if not results:
                        break
                    for work in results:
                        card = _work_to_card(
                            work,
                            query=q,
                            coordinates=self.coordinates,
                            default_coordinate=query_coords.get(q) or default_coordinate,
                        )
                        if card is None or card.id in self.cards:
                            continue
                        self.cards[card.id] = card
                        added += 1
                        got += 1
                    cursor = (payload.get("meta") or {}).get("next_cursor")
                    if not cursor:
                        break
                    time.sleep(1.2)
            except Exception as e:  # noqa: BLE001 — continue other queries
                ok = False
                print(f"[openalex] FAILED query={q[:60]!r}: {e}", flush=True)
            if ok:
                self.completed_queries.add(q)
            self.save_library()
            print(
                f"[openalex] ({qi+1}/{len(pending)}) query={q[:60]!r} "
                f"added≈{got} total={len(self.cards)} ok={ok}",
                flush=True,
            )
            time.sleep(2.0)
        self.save_library()
        return added

    def crawl_crossref(
        self,
        queries: list[str],
        *,
        per_query: int = 40,
        pages: int = 2,
        mailto: str = "",
        query_coords: dict[str, str] | None = None,
        default_coordinate: str | None = None,
    ) -> int:
        """Crossref works API — usually more tolerant than OpenAlex for bulk harvest."""
        added = 0
        per_page = max(1, min(int(per_query), 100))
        pages = max(1, int(pages))
        query_coords = query_coords or {}
        mail = mailto or "research@useragent.local"
        existing_dois = {c.doi.lower() for c in self.cards.values() if c.doi}
        pending = [q for q in queries if q not in self.completed_queries]
        print(
            f"[crossref] resume: {len(self.completed_queries)} done, "
            f"{len(pending)} pending / {len(queries)} total; cards={len(self.cards)}",
            flush=True,
        )
        for qi, q in enumerate(pending):
            got = 0
            ok = True
            try:
                for page in range(pages):
                    offset = page * per_page
                    url = (
                        "https://api.crossref.org/works?"
                        + urllib.parse.urlencode(
                            {
                                "query": q,
                                "rows": per_page,
                                "offset": offset,
                                "sort": "is-referenced-by-count",
                                "order": "desc",
                                "mailto": mail,
                            }
                        )
                    )
                    payload = _http_get_json(url, mailto=mail, label="crossref")
                    items = ((payload.get("message") or {}).get("items")) or []
                    if not items:
                        break
                    for item in items:
                        card = _crossref_to_card(
                            item,
                            query=q,
                            coordinates=self.coordinates,
                            default_coordinate=query_coords.get(q) or default_coordinate,
                        )
                        if card is None or card.id in self.cards:
                            continue
                        if card.doi and card.doi.lower() in existing_dois:
                            continue
                        self.cards[card.id] = card
                        if card.doi:
                            existing_dois.add(card.doi.lower())
                        added += 1
                        got += 1
                    time.sleep(1.0)
            except Exception as e:  # noqa: BLE001
                ok = False
                print(f"[crossref] FAILED query={q[:60]!r}: {e}", flush=True)
            if ok:
                self.completed_queries.add(q)
            self.save_library()
            print(
                f"[crossref] ({qi+1}/{len(pending)}) query={q[:60]!r} "
                f"added≈{got} total={len(self.cards)} ok={ok}",
                flush=True,
            )
            time.sleep(1.5)
        self.save_library()
        return added

    def crawl_semanticscholar(
        self,
        queries: list[str],
        *,
        per_query: int = 40,
        mailto: str = "",
        query_coords: dict[str, str] | None = None,
        default_coordinate: str | None = None,
    ) -> int:
        added = 0
        limit = max(1, min(int(per_query), 100))
        query_coords = query_coords or {}
        pending = [q for q in queries if q not in self.completed_queries]
        existing_dois = {c.doi.lower() for c in self.cards.values() if c.doi}
        print(
            f"[s2] resume: {len(self.completed_queries)} done, "
            f"{len(pending)} pending; cards={len(self.cards)}",
            flush=True,
        )
        for qi, q in enumerate(pending):
            got = 0
            ok = True
            try:
                url = (
                    "https://api.semanticscholar.org/graph/v1/paper/search?"
                    + urllib.parse.urlencode(
                        {
                            "query": q,
                            "limit": limit,
                            "fields": "paperId,title,abstract,year,authors,citationCount,externalIds,venue",
                        }
                    )
                )
                payload = _http_get_json(url, mailto=mailto, label="s2")
                for item in payload.get("data") or []:
                    card = _s2_to_card(
                        item,
                        query=q,
                        coordinates=self.coordinates,
                        default_coordinate=query_coords.get(q) or default_coordinate,
                    )
                    if card is None or card.id in self.cards:
                        continue
                    if card.doi and card.doi.lower() in existing_dois:
                        continue
                    self.cards[card.id] = card
                    if card.doi:
                        existing_dois.add(card.doi.lower())
                    added += 1
                    got += 1
            except Exception as e:  # noqa: BLE001
                ok = False
                print(f"[s2] FAILED query={q[:60]!r}: {e}", flush=True)
            if ok:
                self.completed_queries.add(q)
            self.save_library()
            print(
                f"[s2] ({qi+1}/{len(pending)}) query={q[:60]!r} "
                f"added≈{got} total={len(self.cards)} ok={ok}",
                flush=True,
            )
            time.sleep(3.5)  # S2 unauthenticated is strict
        self.save_library()
        return added

    def crawl_from_domains(
        self,
        domains_path: str | Path,
        *,
        per_query: int = 50,
        pages: int = 2,
        mailto: str = "",
        source: str = "openalex",
    ) -> int:
        path = Path(domains_path)
        if yaml is None:
            raise RuntimeError("PyYAML required to load theory_domains.yaml")
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        coords = list(data.get("coordinates") or [])
        self.merge_coordinates(coords)
        entries = data.get("queries") or []
        queries: list[str] = []
        query_coords: dict[str, str] = {}
        for e in entries:
            if isinstance(e, str):
                queries.append(e)
            else:
                q = (e.get("q") or "").strip()
                if not q:
                    continue
                queries.append(q)
                if e.get("coordinate"):
                    query_coords[q] = str(e["coordinate"])
        source = (source or "openalex").lower()
        if source in {"crossref", "cr"}:
            return self.crawl_crossref(
                queries,
                per_query=per_query,
                pages=pages,
                mailto=mailto,
                query_coords=query_coords,
            )
        if source in {"s2", "semanticscholar", "semantic"}:
            return self.crawl_semanticscholar(
                queries,
                per_query=per_query,
                mailto=mailto,
                query_coords=query_coords,
            )
        return self.crawl_openalex(
            queries,
            per_query=per_query,
            pages=pages,
            mailto=mailto,
            query_coords=query_coords,
        )

    def crawl_serper(self, queries: list[str], *, per_query: int = 5) -> int:
        key = os.environ.get("SERPER_API_KEY", "")
        if not key or key.startswith("your_"):
            return 0
        added = 0
        for q in queries:
            data = json.dumps(
                {"q": q + " site:arxiv.org OR site:acm.org OR scholarly", "num": per_query}
            ).encode()
            req = urllib.request.Request(
                "https://google.serper.dev/search",
                data=data,
                headers={"X-API-KEY": key, "Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            for item in body.get("organic") or []:
                title = item.get("title") or ""
                snippet = item.get("snippet") or ""
                link = item.get("link") or ""
                tid = "serper_" + re.sub(r"[^a-zA-Z0-9]+", "_", link)[-48:]
                if tid in self.cards:
                    continue
                coord = _infer_coordinate(q, title, snippet, self.coordinates)
                card = TheoryCard(
                    id=tid,
                    name=title[:200],
                    authors="",
                    year="",
                    coordinate=coord,
                    trigger=_extract_triggers(title, snippet, []),
                    mechanism=snippet[:400],
                    prediction=f"Web-scholar hit for query '{q}' may inform {coord}.",
                    boundary="Non-peer-reviewed snippet; verify before strong claims.",
                    source="serper",
                    url=link,
                    abstract=snippet,
                    weight=0.6,
                    query=q,
                )
                self.cards[tid] = card
                added += 1
            time.sleep(0.2)
        self.save_library()
        return added


def math_sqrt(x: float) -> float:
    return x**0.5


def _compute_richness(card: TheoryCard) -> float:
    """0–1 score of how usable the card is as retrieval support.

    Structured theory fields (summary/propositions/constructs) weigh more than
    raw abstracts alone — the library is for retrieval matching, not citation dump.
    """
    score = 0.0
    summary = (card.summary or "").strip()
    mech = (card.mechanism or "").strip()
    abstract = (card.abstract or "").strip()
    placeholder_mech = mech.lower().startswith(
        ("empirical/theoretical", "crossref scholarly", "semantic scholar", "scholarly work retrieved")
    )
    # Chinese summaries are denser; 40 chars ≈ useful digest
    if len(summary) >= 100:
        score += 0.35
    elif len(summary) >= 40:
        score += 0.25
    if mech and len(mech) >= 30 and not placeholder_mech:
        score += 0.15
    if abstract and len(abstract) >= 80:
        # abstract alone is weak support; only partial credit
        score += 0.1 if summary else 0.15
    if card.propositions:
        score += 0.12
    if card.constructs:
        score += 0.1
    if card.conditions:
        score += 0.06
    if card.outcomes:
        score += 0.06
    if card.prediction and len(card.prediction) >= 20:
        score += 0.05
    if card.boundary and len(card.boundary) >= 20:
        score += 0.05
    # demote pure citation stubs even if abstract is long
    if not summary and not card.propositions and not card.constructs:
        score *= 0.55
    return round(min(1.0, score), 3)


def _http_get_json(
    url: str, *, mailto: str = "", retries: int = 8, label: str = "openalex"
) -> dict[str, Any]:
    # OpenAlex/Crossref polite pool: include mailto in User-Agent
    mail = mailto or "research@useragent.local"
    headers = {
        "User-Agent": f"theory-guided-agent/0.2 (mailto:{mail}; research corpus builder)",
        "Accept": "application/json",
    }
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=90) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code == 429:
                wait = 20.0
                ra = e.headers.get("Retry-After") if e.headers else None
                if ra:
                    try:
                        wait = float(ra)
                    except ValueError:
                        wait = 20.0
                wait = max(5.0, min(wait, 90.0))
                wait = max(wait, min(90.0, 8.0 * (1.6**attempt)))
                print(f"[{label}] 429 rate-limit; sleep {wait:.0f}s (attempt {attempt+1})", flush=True)
                time.sleep(wait)
                continue
            if e.code in (500, 502, 503, 504):
                time.sleep(min(60.0, 2.0 * (2**attempt)))
                continue
            raise
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last_err = e
            time.sleep(min(30.0, 1.2 * (2**attempt)))
    raise RuntimeError(f"{label} request failed after retries: {url}") from last_err


def _work_to_card(
    work: dict[str, Any],
    *,
    query: str,
    coordinates: list[str],
    default_coordinate: str | None = None,
) -> TheoryCard | None:
    wid = work.get("id") or ""
    if not wid:
        return None
    tid = "oa_" + wid.rsplit("/", 1)[-1]
    title = work.get("display_name") or ""
    abstract = _invert_abstract(work.get("abstract_inverted_index"))
    concepts = [c.get("display_name", "") for c in (work.get("concepts") or [])[:12] if c]
    coord = default_coordinate or _infer_coordinate(query, title, abstract, coordinates)
    authors = ", ".join(
        (a.get("author") or {}).get("display_name", "")
        for a in (work.get("authorships") or [])[:6]
    )
    year = work.get("publication_year") or ""
    triggers = _extract_triggers(title, abstract, concepts)
    cited = int(work.get("cited_by_count") or 0)
    doi = (work.get("doi") or "").replace("https://doi.org/", "")
    primary = work.get("primary_location") or {}
    source = primary.get("source") or {}
    venue = source.get("display_name") or ""
    weight = 0.55 + min(0.45, (cited**0.5) / 200.0)
    mech = abstract[:500] if abstract else f"Scholarly work retrieved for query: {query}"
    return TheoryCard(
        id=tid,
        name=title[:240],
        authors=authors,
        year=year,
        coordinate=coord,
        trigger=triggers,
        mechanism=mech,
        prediction=(
            f"When stimuli match [{', '.join(triggers[:5])}], "
            f"expect effects discussed in: {title[:120]}"
        ),
        boundary="Group-level finding; calibrate with individual GenMinds history.",
        source="openalex",
        url=wid,
        abstract=abstract[:1200],
        weight=round(weight, 3),
        doi=doi,
        cited_by_count=cited,
        concepts=concepts,
        venue=venue[:200],
        query=query,
    )


def _crossref_to_card(
    item: dict[str, Any],
    *,
    query: str,
    coordinates: list[str],
    default_coordinate: str | None = None,
) -> TheoryCard | None:
    doi = (item.get("DOI") or "").strip()
    title_list = item.get("title") or []
    title = title_list[0] if title_list else ""
    if not title and not doi:
        return None
    tid = "cr_" + (re.sub(r"[^a-zA-Z0-9]+", "_", doi)[:60] if doi else re.sub(r"[^a-zA-Z0-9]+", "_", title)[:48])
    abstract = re.sub(r"<[^>]+>", " ", item.get("abstract") or "")
    abstract = re.sub(r"\s+", " ", abstract).strip()
    authors = ", ".join(
        f"{(a.get('given') or '')} {(a.get('family') or '')}".strip()
        for a in (item.get("author") or [])[:6]
    )
    year = ""
    for key in ("published-print", "published-online", "created"):
        parts = ((item.get(key) or {}).get("date-parts") or [[]])[0]
        if parts:
            year = parts[0]
            break
    cited = int(item.get("is-referenced-by-count") or 0)
    venue_list = item.get("container-title") or []
    venue = venue_list[0] if venue_list else ""
    subjects = item.get("subject") or []
    coord = default_coordinate or _infer_coordinate(query, title, abstract, coordinates)
    triggers = _extract_triggers(title, abstract, list(subjects)[:8])
    weight = 0.55 + min(0.45, (cited**0.5) / 200.0)
    mech = abstract[:500] if abstract else f"Crossref scholarly record for: {query}"
    return TheoryCard(
        id=tid,
        name=title[:240],
        authors=authors,
        year=year,
        coordinate=coord,
        trigger=triggers,
        mechanism=mech,
        prediction=(
            f"When stimuli match [{', '.join(triggers[:5])}], "
            f"expect effects discussed in: {title[:120]}"
        ),
        boundary="Group-level finding; calibrate with individual GenMinds history.",
        source="crossref",
        url=f"https://doi.org/{doi}" if doi else "",
        abstract=abstract[:1200],
        weight=round(weight, 3),
        doi=doi,
        cited_by_count=cited,
        concepts=list(subjects)[:12],
        venue=str(venue)[:200],
        query=query,
    )


def _s2_to_card(
    item: dict[str, Any],
    *,
    query: str,
    coordinates: list[str],
    default_coordinate: str | None = None,
) -> TheoryCard | None:
    pid = item.get("paperId") or ""
    title = item.get("title") or ""
    if not pid and not title:
        return None
    tid = "s2_" + (pid or re.sub(r"[^a-zA-Z0-9]+", "_", title)[:48])
    abstract = item.get("abstract") or ""
    authors = ", ".join((a.get("name") or "") for a in (item.get("authors") or [])[:6])
    year = item.get("year") or ""
    cited = int(item.get("citationCount") or 0)
    ext = item.get("externalIds") or {}
    doi = ext.get("DOI") or ""
    venue = item.get("venue") or ""
    coord = default_coordinate or _infer_coordinate(query, title, abstract, coordinates)
    triggers = _extract_triggers(title, abstract, [])
    weight = 0.55 + min(0.45, (cited**0.5) / 200.0)
    mech = abstract[:500] if abstract else f"Semantic Scholar record for: {query}"
    return TheoryCard(
        id=tid,
        name=title[:240],
        authors=authors,
        year=year,
        coordinate=coord,
        trigger=triggers,
        mechanism=mech,
        prediction=(
            f"When stimuli match [{', '.join(triggers[:5])}], "
            f"expect effects discussed in: {title[:120]}"
        ),
        boundary="Group-level finding; calibrate with individual GenMinds history.",
        source="semanticscholar",
        url=f"https://www.semanticscholar.org/paper/{pid}" if pid else "",
        abstract=abstract[:1200],
        weight=round(weight, 3),
        doi=doi,
        cited_by_count=cited,
        concepts=[],
        venue=str(venue)[:200],
        query=query,
    )


def _invert_abstract(inv: dict[str, list[int]] | None) -> str:
    if not inv:
        return ""
    pairs: list[tuple[int, str]] = []
    for word, idxs in inv.items():
        for i in idxs:
            pairs.append((i, word))
    pairs.sort()
    return " ".join(w for _, w in pairs)


def _infer_coordinate(query: str, title: str, abstract: str, coordinates: list[str]) -> str:
    blob = f"{query} {title} {abstract}".lower()
    coords = coordinates or list(COORD_KEYWORDS.keys())
    best, best_s = (coords[0] if coords else "motivated_reasoning"), -1
    for coord in coords:
        keys = COORD_KEYWORDS.get(coord, [coord.replace("_", " ")])
        s = sum(1 for k in keys if k.lower() in blob)
        if s > best_s:
            best, best_s = coord, s
    return best


def _extract_triggers(title: str, abstract: str, concepts: list[str]) -> list[str]:
    toks = list(_tokenize(f"{title} {abstract}"))
    out = [c for c in concepts if c][:6]
    for t in toks:
        if len(t) >= 4 and t not in out:
            out.append(t)
        if len(out) >= 14:
            break
    return out[:14]
