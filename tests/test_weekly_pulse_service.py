"""Tests for ``WeeklyPulseService.run`` end-to-end orchestration.

The pipeline is wired together with injected fakes:

* ``ingest_play_fn`` / ``ingest_app_fn`` return canned review lists,
  bypassing the public Play Store + App Store sources.
* A fake ``SummarizationService`` returns a fixed ``SummaryReport`` so
  Groq is never called.
* The MCP server is driven via ``httpx.MockTransport`` so we can
  exercise success, partial, and doc-failure paths deterministically.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import cast

import httpx
import pytest

from app.config import Settings
from app.domain.enums import Platform
from app.domain.models import Review
from app.services.mcp import MCPClient, PulseDeliveryService
from app.services.summarization.schemas import (
    FinalTheme,
    SummaryReport,
    SummaryStats,
)
from app.services.summarization.service import SummarizationService
from app.services.weekly_pulse import (
    WeeklyPulseRequest,
    WeeklyPulseService,
)

BASE_URL = "https://mcp.test"


def _settings(**overrides: object) -> Settings:
    """Build a test Settings instance, routing MCP/pipeline kwargs into
    the nested ``PipelineConfig``."""
    from app.config import PipelineConfig

    pipeline_overrides: dict[str, object] = {
        "MCP_TIMEOUT": 5.0,
        "MCP_MAX_RETRIES": 1,
        "MCP_RETRY_BACKOFF_SECONDS": 0.0,
        "LOOKBACK_WEEKS": 4,
    }
    settings_overrides: dict[str, object] = {
        "mcp_server_url": BASE_URL,
        "email_to": "leads@example.com",
        "playstore_app_id": "com.example.app",
        "appstore_app_id": "12345",
        "google_doc_id": "test-doc-id",
    }
    nested_keys = {
        "mcp_timeout", "mcp_max_retries", "mcp_retry_backoff_seconds",
        "lookback_weeks",
    }
    for k, v in overrides.items():
        if k in nested_keys:
            pipeline_overrides[k.upper()] = v
        else:
            settings_overrides[k] = v
    settings_overrides["pipeline"] = PipelineConfig(**pipeline_overrides)  # type: ignore[arg-type]
    return Settings(**settings_overrides)  # type: ignore[arg-type]


def _review(platform: Platform, idx: int) -> Review:
    return Review(
        source=platform,
        app_id="app",
        review_id=f"{platform.value}-{idx}",
        rating=4,
        title=None,
        body=f"Solid update. Login still flaky sometimes. ({idx})",
        author=None,
        posted_at=datetime(2026, 5, 14, tzinfo=UTC),
        lang="en",
        country="us",
    )


def _report(total_reviews: int) -> SummaryReport:
    return SummaryReport(
        executive_summary="Login intermittency dominates this window.",
        themes=[
            FinalTheme(
                label="Login flakiness",
                description="Repeated sign-in failures since the last update.",
                sentiment="negative",
                prevalence="high",
                supporting_quotes=["Login fails 50% of the time."],
                action_hint="Audit auth retry logic.",
            )
        ],
        action_ideas=["Ship a login hotfix."],
        stats=SummaryStats(
            total_reviews=total_reviews,
            chunks_processed=2,
            chunks_failed=0,
            themes_premerged=2,
            themes_final=1,
        ),
    )


class _FakeSummarizer:
    """Stand-in for SummarizationService that ignores Groq entirely."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    async def summarize(
        self,
        reviews,  # type: ignore[no-untyped-def]
        *,
        client=None,  # type: ignore[no-untyped-def]
    ) -> SummaryReport:
        review_list = list(reviews)
        self.calls.append(len(review_list))
        return _report(len(review_list))


def _delivery_with_handler(
    settings: Settings,
    handler,  # type: ignore[no-untyped-def]
) -> PulseDeliveryService:
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(base_url=BASE_URL, transport=transport)
    client = MCPClient(settings=settings, client=http)
    return PulseDeliveryService(client=client, settings=settings)


def _typed_summarizer(fake: _FakeSummarizer) -> SummarizationService:
    # Tests use the same call surface (``summarize(reviews)``) as the real
    # service. Cast to satisfy the orchestrator's typed slot without
    # importing structural protocols.
    return cast(SummarizationService, fake)


@pytest.mark.asyncio
async def test_run_success_end_to_end() -> None:
    settings = _settings()
    fake_summ = _FakeSummarizer()
    seen_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_paths.append(request.url.path)
        if request.url.path == "/append_to_doc":
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "document_url": "https://docs.google.com/d/ok",
                },
            )
        if request.url.path == "/create_email_draft":
            return httpx.Response(
                200, json={"success": True, "draft_id": "draft_1"}
            )
        return httpx.Response(404)

    async def ingest_play(app_id, since, until):  # type: ignore[no-untyped-def]
        assert app_id == "com.example.app"
        assert since < until
        return [_review(Platform.ANDROID, i) for i in range(3)]

    async def ingest_app(app_id, since, until):  # type: ignore[no-untyped-def]
        assert app_id == "12345"
        return [_review(Platform.IOS, i) for i in range(2)]

    svc = WeeklyPulseService(
        settings=settings,
        ingest_play_fn=ingest_play,
        ingest_app_fn=ingest_app,
        summarizer=_typed_summarizer(fake_summ),
        delivery=_delivery_with_handler(settings, handler),
    )

    result = await svc.run(WeeklyPulseRequest())

    assert result.status == "succeeded"
    assert result.reviews_ingested == 5
    assert result.chunks_summarized == 4  # 2 reports × 2 chunks each
    assert result.email_sent is True
    assert fake_summ.calls == [3, 2]
    assert seen_paths == ["/append_to_doc", "/create_email_draft"]

    delivery = result.details["delivery"]
    assert delivery["status"] == "succeeded"  # type: ignore[index]
    assert delivery["document_url"].endswith("/ok")  # type: ignore[index]
    assert delivery["draft_id"] == "draft_1"  # type: ignore[index]


@pytest.mark.asyncio
async def test_run_partial_when_draft_fails() -> None:
    settings = _settings()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/append_to_doc":
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "document_url": "https://docs.google.com/d/partial",
                },
            )
        return httpx.Response(503, json={"error": "gmail down"})

    async def ingest_play(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        return [_review(Platform.ANDROID, 0)]

    async def ingest_app(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        return []

    svc = WeeklyPulseService(
        settings=settings,
        ingest_play_fn=ingest_play,
        ingest_app_fn=ingest_app,
        summarizer=_typed_summarizer(_FakeSummarizer()),
        delivery=_delivery_with_handler(settings, handler),
    )

    result = await svc.run(WeeklyPulseRequest())

    assert result.status == "partial"
    assert result.email_sent is False
    delivery = result.details["delivery"]
    assert delivery["status"] == "partial"  # type: ignore[index]
    assert delivery["draft_id"] is None  # type: ignore[index]
    assert delivery["failure_stage"] == "create_email_draft"  # type: ignore[index]
    assert delivery["document_url"].endswith("/partial")  # type: ignore[index]


@pytest.mark.asyncio
async def test_run_failed_when_doc_append_fails() -> None:
    settings = _settings()

    def handler(request: httpx.Request) -> httpx.Response:
        # Every doc append returns 502; retries exhaust → MCPClientError.
        return httpx.Response(502, json={"error": "bad gateway"})

    async def ingest_play(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        return [_review(Platform.ANDROID, 0)]

    async def ingest_app(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        return []

    svc = WeeklyPulseService(
        settings=settings,
        ingest_play_fn=ingest_play,
        ingest_app_fn=ingest_app,
        summarizer=_typed_summarizer(_FakeSummarizer()),
        delivery=_delivery_with_handler(settings, handler),
    )

    result = await svc.run(WeeklyPulseRequest())

    assert result.status == "failed"
    assert result.email_sent is False
    assert result.details["stage"] == "mcp_append_to_doc"
    assert result.details["status_code"] == 502


@pytest.mark.asyncio
async def test_run_no_data_short_circuits_delivery() -> None:
    settings = _settings()
    summarizer = _FakeSummarizer()

    def handler(request: httpx.Request) -> httpx.Response:
        # Should never be hit — the orchestrator must skip delivery when no
        # reviews land in the window.
        raise AssertionError(f"unexpected MCP call to {request.url.path}")

    async def ingest_zero(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        return []

    svc = WeeklyPulseService(
        settings=settings,
        ingest_play_fn=ingest_zero,
        ingest_app_fn=ingest_zero,
        summarizer=_typed_summarizer(summarizer),
        delivery=_delivery_with_handler(settings, handler),
    )

    result = await svc.run(WeeklyPulseRequest())

    assert result.status == "no_data"
    assert result.reviews_ingested == 0
    assert result.chunks_summarized == 0
    assert result.email_sent is False
    assert summarizer.calls == []  # no summarisation attempted


@pytest.mark.asyncio
async def test_run_dry_run_skips_pipeline() -> None:
    settings = _settings()

    async def boom(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("dry_run must not ingest")

    svc = WeeklyPulseService(
        settings=settings,
        ingest_play_fn=boom,
        ingest_app_fn=boom,
        summarizer=_typed_summarizer(_FakeSummarizer()),
        delivery=_delivery_with_handler(
            settings, lambda _r: httpx.Response(500)
        ),
    )

    result = await svc.run(WeeklyPulseRequest(dry_run=True))

    assert result.status == "succeeded"
    assert result.dry_run is True
    assert result.reviews_ingested == 0
    assert result.email_sent is False
    assert "would_email_to" in result.details
