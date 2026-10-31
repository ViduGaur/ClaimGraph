"""Core domain model shared by every pipeline stage.

These Pydantic models are the single source of truth for what a paper, a claim,
a relationship and a conflict look like. The API returns them directly and the
frontend mirrors them in ``frontend/src/types.ts``.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


# --------------------------------------------------------------------------- papers
class Paper(BaseModel):
    paper_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    url: str | None = None
    pdf_url: str | None = None
    abstract: str | None = None
    is_open_access: bool = False
    oa_status: str | None = None
    citation_count: int | None = None
    sources: list[str] = Field(default_factory=list)  # which APIs returned it
    source_ids: dict[str, str] = Field(default_factory=dict)
    relevance: float = 0.0
    text_source: Literal["full_text", "abstract", "none"] = "none"
    acquisition_note: str | None = None

    @field_validator("doi")
    @classmethod
    def _normalize_doi(cls, v: str | None) -> str | None:
        if not v:
            return None
        v = v.strip()
        for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
            if v.lower().startswith(prefix):
                v = v[len(prefix):]
        return v.lower() or None

    @property
    def citation(self) -> str:
        first = self.authors[0].split()[-1] if self.authors else "Unknown"
        etal = " et al." if len(self.authors) > 1 else ""
        return f"{first}{etal} ({self.year or 'n.d.'})"


class Passage(BaseModel):
    """A contiguous span of source text, the unit of evidence."""

    passage_id: str
    paper_id: str
    section: str
    text: str
    char_start: int | None = None
    relevance: float = 0.0


# --------------------------------------------------------------------------- claims
class ClaimType(str, Enum):
    FINDING = "finding"  # empirical result reported by the paper
    COMPARISON = "comparison"  # X better/worse/similar than Y
    METHOD = "method"  # contribution: proposes / introduces
    LIMITATION = "limitation"  # weakness, failure mode, caveat
    FUTURE_WORK = "future_work"
    BACKGROUND = "background"  # belief attributed to prior work / the field


class Polarity(str, Enum):
    POSITIVE = "positive"  # improves, outperforms, is effective
    NEGATIVE = "negative"  # worse, fails, not necessary, underperforms
    NEUTRAL = "neutral"  # comparable, matches
    NONE = "none"


class ClaimContext(BaseModel):
    """Experimental/methodological context a claim was made under.

    Empty lists mean *not stated in the evidence*, never *absent*.
    """

    datasets: list[str] = Field(default_factory=list)
    models: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    population: list[str] = Field(default_factory=list)
    sample_size: str | None = None
    conditions: list[str] = Field(default_factory=list)  # e.g. pre-training, from scratch
    training: list[str] = Field(default_factory=list)  # augmentation, distillation, optimizer
    baselines: list[str] = Field(default_factory=list)
    task: list[str] = Field(default_factory=list)
    data_scale: str | None = None  # small / medium / large / web-scale
    time_period: str | None = None
    methodology: str | None = None


class Claim(BaseModel):
    claim_id: str
    paper_id: str
    claim_text: str
    normalized_text: str
    claim_type: ClaimType
    polarity: Polarity = Polarity.NONE
    subject: str | None = None  # primary entity the claim is about
    evidence: list[Passage]
    section: str
    context: ClaimContext = Field(default_factory=ClaimContext)
    inherited_context: list[str] = Field(default_factory=list)  # fields filled from paper-level text, not the claim's passage
    reported_result: str | None = None  # numeric/qualitative result span
    confidence: float = Field(ge=0.0, le=1.0)
    # heuristic = cue rules; llm_selected = an LLM chose among verbatim cue candidates; llm = LLM quote, grounding-checked
    extraction_method: Literal["heuristic", "llm_selected", "llm"] = "heuristic"
    grounding_score: float = Field(1.0, ge=0.0, le=1.0)  # fuzzy match claim ↔ evidence
    source_citation: str
    relevance: float = 0.0
    uncertainty_notes: list[str] = Field(default_factory=list)

    @field_validator("claim_text")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if len(v.strip()) < 12:
            raise ValueError("claim_text is too short to be a claim")
        return v.strip()

    @field_validator("evidence")
    @classmethod
    def _must_be_grounded(cls, v: list[Passage]) -> list[Passage]:
        if not v:
            raise ValueError("every claim must carry at least one evidence passage")
        return v


# --------------------------------------------------------------------------- relations
class RelationType(str, Enum):
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    QUALIFIES = "QUALIFIES"
    EXTENDS = "EXTENDS"


class FactorComparison(BaseModel):
    factor: str  # dataset, model, metric, ...
    a: list[str] = Field(default_factory=list)
    b: list[str] = Field(default_factory=list)
    status: Literal["same", "overlap", "different", "unknown"]
    inherited: bool = False  # at least one side comes from paper-level context (weaker evidence)
    # set by the LLM pair analyser: the verbatim source spans that ground each side's value
    a_quote: str | None = None
    b_quote: str | None = None
    note: str | None = None  # e.g. why a stated difference is (not) material to the findings


class RelationSignals(BaseModel):
    similarity: float
    nli_entailment: float
    nli_contradiction: float
    nli_neutral: float
    polarity_conflict: bool
    context_divergence: float  # 0 = same conditions, 1 = entirely different
    conflict_signal: bool = False  # the relation came from a face-value disagreement (so it gets a conflict verdict)


class Relationship(BaseModel):
    relation_id: str
    source_claim_id: str  # claim B (the one that relates)
    target_claim_id: str  # claim A (the one being related to)
    relation: RelationType
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    signals: RelationSignals
    factors: list[FactorComparison] = Field(default_factory=list)
    decided_by: Literal["rules", "llm", "rules+llm"] = "rules"  # rules+llm = LLM-analysed stance/context, rule decision
    uncertain: bool = False
    llm_note: str | None = None  # the LLM analyser's one-sentence rationale (never used as evidence)


# --------------------------------------------------------------------------- conflicts
class ConflictVerdict(str, Enum):
    CONTEXT_DEPENDENT = "context_dependent"  # likely not a real contradiction
    GENUINE = "genuine_disagreement"  # similar conditions, opposing findings
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class ConflictAnalysis(BaseModel):
    conflict_id: str
    relation_id: str
    claim_a_id: str
    claim_b_id: str
    verdict: ConflictVerdict
    headline: str
    explanation: str
    differing_factors: list[FactorComparison]
    shared_factors: list[FactorComparison]
    unknown_factors: list[str]
    evidence_passage_ids: list[str]
    confidence: float
    llm_explanation: str | None = None  # grounded LLM wording of the verdict (never the source of the verdict)
    apparent_conflict: str | None = None  # how the two claims oppose at face value
    explanation_source: Literal["template", "llm"] = "template"
    explanation_note: str | None = None  # why an LLM explanation was rejected, if it was


# --------------------------------------------------------------------------- landscape
class Theme(BaseModel):
    theme_id: str
    label: str
    keywords: list[str]
    claim_ids: list[str]
    paper_ids: list[str]
    support_edges: int = 0
    conflict_edges: int = 0
    status: Literal["consensus", "disputed", "emerging", "single_source"]
    summary: str


class ResearchGap(BaseModel):
    gap_id: str
    title: str  # always phrased as a *potential* gap
    signal: Literal[
        "repeated_limitation",
        "unresolved_conflict",
        "single_source",
        "understudied_condition",
        "future_work",
        "missing_comparison",
    ]
    rationale: str
    evidence_claim_ids: list[str]
    strength: Literal["weak", "moderate", "strong"]


class Landscape(BaseModel):
    summary: str
    consensus: list[Theme]
    disagreements: list[Theme]
    emerging: list[Theme]
    unresolved_questions: list[str]
    gaps: list[ResearchGap]


# --------------------------------------------------------------------------- graph
class GraphNode(BaseModel):
    id: str
    kind: Literal["paper", "claim"]
    label: str
    data: dict = Field(default_factory=dict)


class GraphEdge(BaseModel):
    id: str
    source: str
    target: str
    kind: Literal["MAKES", "SUPPORTS", "CONTRADICTS", "QUALIFIES", "EXTENDS"]
    data: dict = Field(default_factory=dict)


class ClaimGraphModel(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    stats: dict = Field(default_factory=dict)
