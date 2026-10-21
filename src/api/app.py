"""FastAPI application factory for the Research Intelligence Engine."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import system, research, papers
from src.api.dependencies import initialize_service, shutdown_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("research_engine_api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifecycle."""
    logger.info("Starting Research Intelligence Engine API...")
    # Initialize the core engine service once at startup
    initialize_service()
    yield
    logger.info("Shutting down Research Intelligence Engine API...")
    shutdown_service()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(
        title="Research Intelligence Engine API",
        description="Production REST API for evidence-grounded research synthesis with page citations.",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Allow local Next.js frontend development origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:8501",
            "http://127.0.0.1:8501",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register API routers
    app.include_router(system.router)
    app.include_router(research.router)
    app.include_router(papers.router)

    @app.get("/api/health")
    def health_alias():
        """Alias for /api/system/health."""
        return {"status": "ok", "service": "Research Intelligence Engine API", "version": "1.0.0"}

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error(f"Unhandled error on {request.method} {request.url.path}: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "detail": "An internal server error occurred. Credentials and internal traces have been protected.",
                "path": str(request.url.path),
            },
        )

    return app


app = create_app()
