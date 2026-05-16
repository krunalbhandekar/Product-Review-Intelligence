from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.requests import Request

from app.api.routes import health, weekly_pulse
from app.config import get_settings
from app.core.exceptions import AppError
from app.core.logging import configure_logging, get_logger


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    log = get_logger("app")
    settings = get_settings()
    log.info(
        "app.startup",
        environment=settings.app.ENVIRONMENT,
        api_key_configured=bool(settings.api_key),
        google_doc_id_configured=bool(settings.google_doc_id),
        mcp_server_url=settings.mcp_server_url or None,
        groq_model=settings.groq.MODEL,
        groq_fallback_models=settings.groq.FALLBACK_MODELS,
        groq_token_threshold=settings.groq.TOKEN_THRESHOLD,
        groq_max_429_streak=settings.groq.MAX_429_STREAK,
        groq_request_concurrency=settings.groq.REQUEST_CONCURRENCY,
        summarization_concurrency=settings.pipeline.SUMMARIZATION_CONCURRENCY,
        max_reviews_per_run=settings.pipeline.MAX_REVIEWS_PER_RUN,
        lookback_weeks=settings.pipeline.LOOKBACK_WEEKS,
        chunk_size=settings.pipeline.CHUNK_SIZE,
        chunk_target_tokens=settings.pipeline.CHUNK_TARGET_TOKENS,
        max_review_body_chars=settings.pipeline.MAX_REVIEW_BODY_CHARS,
    )
    # Dedicated, easily-grepped line for the pulse window. Operators
    # tend to scan for this when verifying which window a run used.
    log.info(f"weekly_pulse.lookback_weeks={settings.pipeline.LOOKBACK_WEEKS}")

    if not settings.api_key:
        log.error("app.startup.api_key_missing")
    elif (
        settings.app.ENVIRONMENT != "development"
        and settings.api_key == "dev-key"
    ):
        log.warning("app.startup.api_key_is_default_dev_key")
    if not settings.google_doc_id:
        # Non-fatal: health checks still work. Delivery stage will raise
        # a clear ValidationError when actually invoked.
        log.warning("app.startup.google_doc_id_missing")
    yield
    log.info("app.shutdown")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app.APP_NAME, version="0.1.0", lifespan=lifespan)

    @app.exception_handler(AppError)
    async def on_app_error(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": type(exc).__name__, "message": exc.message},
        )

    app.include_router(health.router)
    app.include_router(weekly_pulse.router)
    return app


app = create_app()
