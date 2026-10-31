"""Analysis service: runs pipelines in a background worker and persists progress.

A single worker thread processes analyses one at a time — the local models are
CPU-bound, so running several at once would only slow each of them down.
"""
from __future__ import annotations

import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from sqlalchemy import select

from app.core.config import Settings
from app.db.session import AnalysisRow, get_sessionmaker
from app.pipelines.analysis import AnalysisPipeline, PipelineError
from app.schemas.api import (
    STAGES,
    AnalysisCounts,
    AnalysisDetail,
    AnalysisResult,
    AnalysisStatus,
    AnalysisSummary,
    AnalyzeRequest,
    PipelineWarning,
    StageProgress,
)

log = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AnalysisNotFound(KeyError):
    pass


class AnalysisService:
    def __init__(self, settings: Settings, pipeline_factory=None):
        self.settings = settings
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="claimgraph-worker")
        self._lock = threading.Lock()
        self._pipeline_factory = pipeline_factory or (lambda progress, warn: AnalysisPipeline(settings, progress, warn))

    # ------------------------------------------------------------------ lifecycle
    def recover_interrupted(self) -> int:
        """Mark runs left 'running'/'queued' by a previous process as failed."""
        with get_sessionmaker()() as db:
            rows = db.scalars(select(AnalysisRow).where(AnalysisRow.status.in_(["running", "queued"]))).all()
            for row in rows:
                row.status = AnalysisStatus.FAILED.value
                row.error = "The server restarted before this analysis finished. Please run it again."
            db.commit()
            return len(rows)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------------ commands
    def submit(self, req: AnalyzeRequest, run_async: bool = True) -> AnalysisSummary:
        demo = bool(req.demo) if req.demo is not None else self.settings.demo_mode
        row = AnalysisRow(
            id=uuid.uuid4().hex[:16],
            question=req.question,
            status=AnalysisStatus.QUEUED.value,
            demo_mode=demo,
            request=req.model_dump(),
            stages=[StageProgress(key=k, label=label).model_dump(mode="json") for k, label in STAGES],
            warnings=[],
            counts={},
        )
        with get_sessionmaker()() as db:
            db.add(row)
            db.commit()
        if run_async:
            self._pool.submit(self._run, row.id, req)
        else:
            self._run(row.id, req)
        return self.get_summary(row.id)

    # ------------------------------------------------------------------ worker
    def _update(self, analysis_id: str, **fields) -> None:
        with self._lock, get_sessionmaker()() as db:
            row = db.get(AnalysisRow, analysis_id)
            if row is None:
                return
            for k, v in fields.items():
                setattr(row, k, v)
            db.commit()

    def _mutate_stage(self, analysis_id: str, key: str, status: str, detail: str | None) -> None:
        with self._lock, get_sessionmaker()() as db:
            row = db.get(AnalysisRow, analysis_id)
            if row is None:
                return
            stages = [dict(s) for s in row.stages]
            for s in stages:
                if s["key"] == key:
                    if s["status"] == "pending" or status != "running":
                        s["started_at"] = s.get("started_at") or _now()
                    s["status"] = status
                    if detail is not None:
                        s["detail"] = detail
                    if status in ("done", "failed", "skipped"):
                        s["finished_at"] = _now()
            row.stages = stages
            db.commit()

    def _add_warning(self, analysis_id: str, stage: str, message: str) -> None:
        with self._lock, get_sessionmaker()() as db:
            row = db.get(AnalysisRow, analysis_id)
            if row is None:
                return
            warnings = list(row.warnings or [])
            if not any(w["message"] == message for w in warnings):
                warnings.append({"stage": stage, "message": message})
            row.warnings = warnings[:50]
            db.commit()

    def _run(self, analysis_id: str, req: AnalyzeRequest) -> None:
        self._update(analysis_id, status=AnalysisStatus.RUNNING.value)
        pipeline = self._pipeline_factory(
            lambda key, status, detail: self._mutate_stage(analysis_id, key, status, detail),
            lambda stage, message: self._add_warning(analysis_id, stage, message),
        )
        try:
            result = pipeline.run(req)
        except PipelineError as exc:
            self._fail(analysis_id, str(exc))
            return
        except Exception as exc:  # noqa: BLE001 — surface, never crash the worker
            log.exception("analysis %s crashed", analysis_id)
            self._fail(analysis_id, f"Internal error: {exc.__class__.__name__}: {exc}")
            return
        counts = AnalysisCounts(papers=len(result.papers), claims=len(result.claims), relationships=len(result.relationships),
                                conflicts=len(result.conflicts), gaps=len(result.landscape.gaps))
        self._update(analysis_id, status=AnalysisStatus.COMPLETED.value, result=result.model_dump(mode="json"),
                     counts=counts.model_dump())

    def _fail(self, analysis_id: str, message: str) -> None:
        with self._lock, get_sessionmaker()() as db:
            row = db.get(AnalysisRow, analysis_id)
            if row is None:
                return
            stages = [dict(s) for s in row.stages]
            for s in stages:
                if s["status"] == "running":
                    s["status"] = "failed"
                    s["finished_at"] = _now()
                elif s["status"] == "pending":
                    s["status"] = "skipped"
            row.stages = stages
            row.status = AnalysisStatus.FAILED.value
            row.error = message
            db.commit()

    # ------------------------------------------------------------------ queries
    def _row(self, analysis_id: str) -> AnalysisRow:
        with get_sessionmaker()() as db:
            row = db.get(AnalysisRow, analysis_id)
            if row is None:
                raise AnalysisNotFound(analysis_id)
            return row

    @staticmethod
    def _summary_from(row: AnalysisRow) -> dict:
        return dict(
            id=row.id, question=row.question, status=row.status, created_at=row.created_at, updated_at=row.updated_at,
            stages=[StageProgress(**s) for s in row.stages], warnings=[PipelineWarning(**w) for w in row.warnings or []],
            error=row.error, counts=AnalysisCounts(**(row.counts or {})), demo_mode=row.demo_mode,
        )

    def get_summary(self, analysis_id: str) -> AnalysisSummary:
        return AnalysisSummary(**self._summary_from(self._row(analysis_id)))

    def get_detail(self, analysis_id: str) -> AnalysisDetail:
        row = self._row(analysis_id)
        result = AnalysisResult.model_validate(row.result) if row.result else None
        return AnalysisDetail(**self._summary_from(row), result=result)

    def get_result(self, analysis_id: str) -> AnalysisResult | None:
        row = self._row(analysis_id)
        return AnalysisResult.model_validate(row.result) if row.result else None

    def list_recent(self, limit: int = 20) -> list[AnalysisSummary]:
        with get_sessionmaker()() as db:
            rows = db.scalars(select(AnalysisRow).order_by(AnalysisRow.created_at.desc()).limit(limit)).all()
            return [AnalysisSummary(**self._summary_from(r)) for r in rows]
