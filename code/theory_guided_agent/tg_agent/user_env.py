from __future__ import annotations

"""Build user communication / psychological / social environment profiles
for Theory-Guided retrieval (environment-keyed theory match).
"""

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

# Map lexical cues → theory coordinates (weights accumulated then normalized)
_COMM_CUES: list[tuple[str, list[str], float]] = [
    ("sarcasm|irony|讽刺|阴阳|笑死", ["online_disinhibition", "impression_management"], 0.8),
    ("slang|黑话|梗|互联网", ["uses_gratifications", "cmc_theory"], 0.5),
    ("short|简短|短评", ["uses_gratifications"], 0.3),
    ("self-presentation|自我呈现|人设", ["impression_management"], 0.7),
    ("influencer|网红|博主", ["parasocial", "opinion_leadership"], 0.5),
    ("匿名|喷|杠", ["online_disinhibition", "cancel_culture"], 0.6),
]

_PSYCH_CUES: list[tuple[str, list[str], float]] = [
    ("nationalist|爱国|辱华|西方|精日|我们.*他们", ["identity_threat", "social_identity", "public_opinion_china"], 0.9),
    ("rational|理性|逻辑|双标|公平", ["motivated_reasoning", "fairness", "cultural_cognition"], 0.7),
    ("cynical|愤世|犬儒", ["motivated_reasoning", "trust"], 0.5),
    ("threat|威胁|安全|崩溃|危机", ["risk_perception", "technology_threat"], 0.6),
    ("trust|辟谣|谣言|假", ["trust", "misinformation", "inoculation"], 0.7),
    ("persuasion|说服|洗脑|叙事", ["framing", "narrative_persuasion", "motivated_reasoning"], 0.6),
    ("identity|认同|群体", ["social_identity", "identity_threat"], 0.7),
    ("moral|道德|正义|Outrage", ["moral_foundations", "cancel_culture"], 0.6),
    ("compare|攀比|焦虑|FoMO|孤独", ["social_comparison", "developmental_media"], 0.5),
    ("privacy|隐私|监控", ["privacy_calculus"], 0.5),
    ("filter|茧房|回音|推荐算法", ["echo_chamber", "algorithmic_curation", "selective_exposure"], 0.7),
]

_SOCIAL_CUES: list[tuple[str, list[str], float]] = [
    ("china|中国|微博|weibo|审查", ["public_opinion_china", "face_culture"], 0.8),
    ("japan|日本|韩国|美国|西方", ["identity_threat", "public_opinion_china", "affective_polarization"], 0.7),
    ("face|面子|关系|人情", ["face_culture", "impression_management"], 0.6),
    ("protest|动员|爱国行动", ["collective_action"], 0.5),
    ("class|阶层|资本|精英", ["habitus_capital", "fairness", "macro_social_theory"], 0.5),
    ("org|单位|公司|职场", ["organizational_behavior"], 0.4),
    ("public.?sphere|舆论场|公共讨论", ["public_sphere", "spiral_of_silence"], 0.5),
    ("network|社群|圈子", ["network_society", "social_capital"], 0.4),
    ("crisis|舆情|公关", ["crisis_communication"], 0.5),
]


def _blob(*parts: Any) -> str:
    chunks: list[str] = []
    for p in parts:
        if p is None:
            continue
        if isinstance(p, str):
            chunks.append(p)
        elif isinstance(p, list):
            chunks.extend(str(x) for x in p)
        elif isinstance(p, dict):
            chunks.append(json.dumps(p, ensure_ascii=False))
        else:
            chunks.append(str(p))
    return "\n".join(chunks).lower()


def _score_cues(text: str, cues: list[tuple[str, list[str], float]]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for pat, coords, w in cues:
        if re.search(pat, text, re.I):
            for c in coords:
                scores[c] = scores.get(c, 0.0) + w
    return scores


def _normalize(scores: dict[str, float], floor: float = 0.15, cap: float = 1.35) -> dict[str, float]:
    if not scores:
        return {}
    m = max(scores.values()) or 1.0
    out = {k: max(floor, min(cap, 0.35 + 0.9 * (v / m))) for k, v in scores.items()}
    return dict(sorted(out.items(), key=lambda x: -x[1]))


def _topic_counts_from_csv(csv_path: Path, max_rows: int = 200) -> Counter:
    import csv
    import io

    topics: Counter = Counter()
    if not csv_path.exists():
        return topics
    raw = csv_path.read_bytes()
    text = None
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = raw.decode("gb18030", errors="ignore")
    reader = csv.DictReader(io.StringIO(text))
    for i, row in enumerate(reader):
        if i >= max_rows:
            break
        if row.get("是否目标用户作者") == "否":
            continue
        text_body = row.get("正文") or row.get("text") or ""
        tags = row.get("话题") or ""
        for t in re.findall(r"[\u4e00-\u9fffA-Za-z0-9_]{2,}", tags):
            topics[t] += 1
        for t in re.findall(r"#([^#]+)#", text_body):
            topics[t.strip()] += 2
    return topics


def build_user_environment(
    user_id: str,
    *,
    persona: dict[str, Any],
    memory_static: dict[str, Any],
    csv_path: Path | None = None,
    analysis_extra: str = "",
) -> dict[str, Any]:
    """Extract communication / psych / social env + coordinate weights."""
    comm_list = list(memory_static.get("communication") or persona.get("communication") or [])
    values = list(memory_static.get("persona_values") or persona.get("values") or [])
    interests = list(memory_static.get("persona_interests") or persona.get("interests") or [])
    beliefs = list(memory_static.get("beliefs") or [])
    motifs = list(memory_static.get("cognitive_motifs") or [])
    demo = persona.get("demographics") or {}
    analysis = str(persona.get("analysis") or "") + "\n" + analysis_extra

    topic_counts = _topic_counts_from_csv(csv_path) if csv_path else Counter()
    top_topics = [t for t, _ in topic_counts.most_common(20)]

    comm_text = _blob(comm_list, persona.get("statistics"), analysis)
    psych_text = _blob(analysis, values, beliefs, motifs, interests)
    social_text = _blob(analysis, demo, interests, beliefs, top_topics, "微博 weibo china")

    comm_scores = _score_cues(comm_text, _COMM_CUES)
    psych_scores = _score_cues(psych_text, _PSYCH_CUES)
    social_scores = _score_cues(social_text, _SOCIAL_CUES)

    # topic-driven boosts for China / identity / media
    topic_blob = " ".join(top_topics).lower()
    for pat, coords, w in _PSYCH_CUES + _SOCIAL_CUES:
        if re.search(pat, topic_blob, re.I):
            for c in coords:
                social_scores[c] = social_scores.get(c, 0.0) + 0.35 * w

    # always soft-prior Weibo Chinese public opinion for this corpus
    social_scores["public_opinion_china"] = social_scores.get("public_opinion_china", 0.0) + 0.5
    social_scores["uses_gratifications"] = social_scores.get("uses_gratifications", 0.0) + 0.3

    comm_w = _normalize(comm_scores)
    psych_w = _normalize(psych_scores)
    social_w = _normalize(social_scores)

    merged: dict[str, float] = {}
    for d, mul in ((comm_w, 0.9), (psych_w, 1.0), (social_w, 1.05)):
        for k, v in d.items():
            merged[k] = max(merged.get(k, 0.0), v * mul)
    # keep top coordinates only (sparse env prior)
    merged = dict(sorted(merged.items(), key=lambda x: -x[1])[:18])

    evidence = []
    if analysis:
        evidence.append({"source": "persona.analysis", "text": analysis[:400]})
    for label, items in (
        ("communication", comm_list[:6]),
        ("values", values[:6]),
        ("beliefs", beliefs[:6]),
        ("topics", top_topics[:10]),
    ):
        if items:
            evidence.append({"source": label, "text": " | ".join(str(x)[:80] for x in items)})

    return {
        "user_id": user_id,
        "communication": {
            "style": comm_list[:10],
            "statistics": persona.get("statistics") or [],
            "coordinate_weights": comm_w,
        },
        "psychological": {
            "analysis_excerpt": analysis[:500],
            "values": values[:12],
            "beliefs": beliefs[:12],
            "motifs": motifs[:8],
            "coordinate_weights": psych_w,
        },
        "social": {
            "demographics": demo,
            "interests": interests[:12],
            "top_topics": top_topics[:15],
            "coordinate_weights": social_w,
        },
        "coordinate_weights": merged,
        "evidence": evidence,
        "method": "heuristic_cues+persona+genminds+csv_topics",
    }


def env_path_for(state_dir: Path, user_id: str) -> Path:
    return Path(state_dir) / f"{user_id}_env.json"


def load_env(state_dir: Path, user_id: str) -> dict[str, Any] | None:
    p = env_path_for(state_dir, user_id)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def save_env(state_dir: Path, profile: dict[str, Any]) -> Path:
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    p = env_path_for(state_dir, str(profile["user_id"]))
    p.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    return p
