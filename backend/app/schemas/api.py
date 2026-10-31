"""Request/response schemas for the HTTP API."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.schemas.domain import (
    Claim,
    ClaimGraphModel,
    ConflictAnalysis,
    Landscape,
    Paper,
    Relationship,
    ResearchGap,
)


class AnalyzeRequest(BaseModel):
    question: str = Field(min_length=8, max_length=500)
    max_papers: int | None = Field(None, ge=2, le=20)
    use_llm: bool | None = Field(None, description="override CLAIMGRAPH_LLM_PROVIDER usage for this run")
    demo: bool | None = Field(None, description="force the curated demo corpus for this run")

    @field_validator("question")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = " ".join(v.split())
        if len(v) < 8:
            raise ValueError("question is too short")
        return v


class AnalysisStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


STAGES: list[tuple[str, str]] = [
    ("search", "Searching literature"),
    ("select", "Selecting papers"),
    ("acquire", "Acquiring papers"),
    ("extract", "Extracting claims"),
    ("compare", "Comparing claims"),
    ("graph", "Building claim graph"),
    ("landscape", "Generating research landscape"),
]


class StageProgress(BaseModel):
    key: str
    label: str
    status: Literal["pending", "running", "done", "failed", "skipped"] = "pending"
    detail: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


class PipelineWarning(BaseModel):
    stage: str
    message: str


class QueryPlan(BaseModel):
    question: str
    search_queries: list[str]
    key_terms: list[str]
    entities: list[str]
    method: Literal["heuristic", "llm"] = "heuristic"


class RunInfo(BaseModel):
    demo_mode: bool
    llm_provider: str
    llm_model: str | None
    llm_calls: int
    embedding_backend: str
    nli_backend: str
    sources: list[str]
    duration_s: float | None = None


class AnalysisResult(BaseModel):
    query_plan: QueryPlan
    papers: list[Paper]
    claims: list[Claim]
    relationships: list[Relationship]
    conflicts: list[ConflictAnalysis]
    graph: ClaimGraphModel
    landscape: Landscape
    run: RunInfo


class AnalysisCounts(BaseModel):
    papers: int = 0
    claims: int = 0
    relationships: int = 0
    conflicts: int = 0
    gaps: int = 0


class AnalysisSummary(BaseModel):
    id: str
    question: str
    status: AnalysisStatus
    created_at: datetime
    updated_at: datetime
    stages: list[StageProgress]
    warnings: list[PipelineWarning] = Field(default_factory=list)
    error: str | None = None
    counts: AnalysisCounts = Field(default_factory=AnalysisCounts)
    demo_mode: bool = False


class AnalysisDetail(AnalysisSummary):
    result: AnalysisResult | None = None


class GapsResponse(BaseModel):
    gaps: list[ResearchGap]
    landscape: Landscape


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    demo_mode: bool
    llm_provider: str
    llm_available: bool
    sources: list[str]
    keys_configured: dict[str, bool]
