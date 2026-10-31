"""HTTP API (v1). Route handlers only translate HTTP ↔ service calls."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel

from app import __version__
from app.core.config import Settings, get_settings
from app.providers.llm import create_provider
from app.schemas.api import (
    AnalysisDetail,
    AnalysisResult,
    AnalysisSummary,
    AnalyzeRequest,
    GapsResponse,
    HealthResponse,
)
from app.schemas.domain import Claim, ClaimGraphModel, ConflictAnalysis, Paper, Relationship
from app.services.analysis_service import AnalysisNotFound, AnalysisService

router = APIRouter(prefix="/api/v1")


def get_service(request: Request) -> AnalysisService:
    return request.app.state.analysis_service


def _completed(service: AnalysisService, analysis_id: str) -> AnalysisResult:
    try:
        result = service.get_result(analysis_id)
    except AnalysisNotFound:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"analysis {analysis_id} not found") from None
    if result is None:
        summary = service.get_summary(analysis_id)
        raise HTTPException(status.HTTP_409_CONFLICT, f"analysis is {summary.status.value}; results are not available yet")
    return result


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    llm_ok = False
    try:
        provider = create_provider(settings)
        llm_ok = bool(provider and provider.is_available())
    except Exception:  # noqa: BLE001
        llm_ok = False
    return HealthResponse(
        status="ok", version=__version__, demo_mode=settings.demo_mode, llm_provider=settings.llm_provider_name,
        llm_available=llm_ok, sources=settings.source_list,
        keys_configured={
            "openalex_api_key": bool(settings.openalex_api_key),
            "semantic_scholar_api_key": bool(settings.semantic_scholar_api_key),
            "contact_email": bool(settings.contact_email),
            "llm_api_key": bool(settings.llm_api_key),
        },
    )


@router.post("/research/analyze", response_model=AnalysisSummary, status_code=status.HTTP_202_ACCEPTED, tags=["research"])
def analyze(req: AnalyzeRequest, service: AnalysisService = Depends(get_service)) -> AnalysisSummary:
    return service.submit(req)


@router.get("/research", response_model=list[AnalysisSummary], tags=["research"])
def list_analyses(limit: int = Query(20, ge=1, le=100), service: AnalysisService = Depends(get_service)) -> list[AnalysisSummary]:
    return service.list_recent(limit)


@router.get("/research/{analysis_id}", response_model=AnalysisDetail, tags=["research"])
def get_analysis(analysis_id: str, include_result: bool = True, service: AnalysisService = Depends(get_service)) -> AnalysisDetail:
    try:
        if include_result:
            return service.get_detail(analysis_id)
        return AnalysisDetail(**service.get_summary(analysis_id).model_dump())
    except AnalysisNotFound:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"analysis {analysis_id} not found") from None


@router.get("/research/{analysis_id}/papers", response_model=list[Paper], tags=["research"])
def papers(analysis_id: str, service: AnalysisService = Depends(get_service)) -> list[Paper]:
    return _completed(service, analysis_id).papers


@router.get("/research/{analysis_id}/claims", response_model=list[Claim], tags=["research"])
def claims(analysis_id: str, paper_id: str | None = None, claim_type: str | None = None,
           service: AnalysisService = Depends(get_service)) -> list[Claim]:
    out = _completed(service, analysis_id).claims
    if paper_id:
        out = [c for c in out if c.paper_id == paper_id]
    if claim_type:
        out = [c for c in out if c.claim_type.value == claim_type]
    return out


class ClaimExplanation(BaseModel):
    """Everything needed to answer "Why did ClaimGraph say this?" for one claim."""

    claim: Claim
    paper: Paper
    relationships: list[Relationship]
    related_claims: list[Claim]


@router.get("/research/{analysis_id}/claims/{claim_id}", response_model=ClaimExplanation, tags=["research"])
def claim_detail(analysis_id: str, claim_id: str, service: AnalysisService = Depends(get_service)) -> ClaimExplanation:
    result = _completed(service, analysis_id)
    by_id = {c.claim_id: c for c in result.claims}
    claim = by_id.get(claim_id)
    if claim is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"claim {claim_id} not found")
    rels = [r for r in result.relationships if claim_id in (r.source_claim_id, r.target_claim_id)]
    related_ids = {r.source_claim_id for r in rels} | {r.target_claim_id for r in rels}
    paper = next(p for p in result.papers if p.paper_id == claim.paper_id)
    return ClaimExplanation(claim=claim, paper=paper, relationships=rels,
                            related_claims=[by_id[i] for i in related_ids if i != claim_id])


@router.get("/research/{analysis_id}/relationships", response_model=list[Relationship], tags=["research"])
def relationships(analysis_id: str, service: AnalysisService = Depends(get_service)) -> list[Relationship]:
    return _completed(service, analysis_id).relationships


@router.get("/research/{analysis_id}/graph", response_model=ClaimGraphModel, tags=["research"])
def graph(analysis_id: str, service: AnalysisService = Depends(get_service)) -> ClaimGraphModel:
    return _completed(service, analysis_id).graph


@router.get("/research/{analysis_id}/conflicts", response_model=list[ConflictAnalysis], tags=["research"])
def conflicts(analysis_id: str, service: AnalysisService = Depends(get_service)) -> list[ConflictAnalysis]:
    return _completed(service, analysis_id).conflicts


@router.get("/research/{analysis_id}/gaps", response_model=GapsResponse, tags=["research"])
def gaps(analysis_id: str, service: AnalysisService = Depends(get_service)) -> GapsResponse:
    result = _completed(service, analysis_id)
    return GapsResponse(gaps=result.landscape.gaps, landscape=result.landscape)
