from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TheoryCard:
    """Retrieval unit: must carry enough theory text to match & explain."""

    id: str
    name: str
    authors: str = ""
    year: int | str = ""
    coordinate: str = ""
    trigger: list[str] = field(default_factory=list)
    # --- retrieval-critical theory body ---
    mechanism: str = ""  # how the process works
    prediction: str = ""  # if stimulus matches, what outcome to expect
    boundary: str = ""  # when NOT to apply
    summary: str = ""  # 150–400 word theory digest for retrieval
    constructs: list[str] = field(default_factory=list)  # key variables
    propositions: list[str] = field(default_factory=list)  # testable claims
    conditions: list[str] = field(default_factory=list)  # situational triggers
    outcomes: list[str] = field(default_factory=list)  # typical effects
    # --- provenance ---
    source: str = "seed"
    url: str = ""
    abstract: str = ""
    weight: float = 1.0
    doi: str = ""
    cited_by_count: int = 0
    concepts: list[str] = field(default_factory=list)
    venue: str = ""
    query: str = ""
    richness: float = 0.0  # 0–1, filled by enrichment
    # --- grounding / provenance for distilled fields ---
    grounded: bool = False
    grounding_source: str = ""  # abstract | canonical | seed | ""
    evidence_quotes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TheoryCard":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    def retrieval_text(self) -> str:
        parts = [
            self.name,
            self.coordinate,
            self.summary,
            self.mechanism,
            self.prediction,
            self.boundary,
            self.abstract,
            " ".join(self.constructs),
            " ".join(self.propositions),
            " ".join(self.conditions),
            " ".join(self.outcomes),
            " ".join(self.trigger),
            " ".join(self.concepts),
        ]
        return " ".join(p for p in parts if p)

    def is_thin(self) -> bool:
        body = f"{self.summary} {self.mechanism} {self.abstract}".strip()
        placeholder = body.lower().startswith("empirical/theoretical") or body.lower().startswith(
            "crossref scholarly"
        )
        return placeholder or len(body) < 120


@dataclass
class MatchedTheory:
    card: TheoryCard
    score: float
    why: str


@dataclass
class RetrievedEvent:
    map_id: str
    text: str
    score: float
    event_title: str = ""
    user_opinion: str = ""
    topics: list[str] = field(default_factory=list)


@dataclass
class AgentOutput:
    user_id: str
    stimulus: str
    predicted_opinion: str
    stance: str
    activated_coordinates: list[str]
    matched_theories: list[dict[str, Any]]
    evidence_events: list[dict[str, Any]]
    verbalization: str
    c_trace: dict[str, Any]
    u_snapshot: dict[str, Any]
    v_snapshot: dict[str, Any]
    caveats: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
