"""ClaimGraph FastAPI application."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.routes import router
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.db.session import init_db
from app.services.analysis_service import AnalysisService

log = logging.getLogger("claimgraph")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(settings.log_level)
    init_db(settings.database_url)
    service = AnalysisService(settings)
    stale = service.recover_interrupted()
    if stale:
        log.info("marked %d interrupted analyses as failed", stale)
    app.state.analysis_service = service
    log.info("ClaimGraph %s ready (demo_mode=%s, llm=%s, sources=%s)", __version__, settings.demo_mode,
             settings.llm_provider_name, settings.source_list)
    yield
    service.shutdown()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="ClaimGraph API",
        version=__version__,
        description="Cross-paper scientific claim analysis: discovery, claim extraction, relationship classification, "
                    "contextual conflict analysis and research landscape.",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origin_list, allow_methods=["*"], allow_headers=["*"])
    app.include_router(router)
    return app


app = create_app()
