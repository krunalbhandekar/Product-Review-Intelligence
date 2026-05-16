"""Tests for the MCP client + delivery layer.

We use ``httpx.MockTransport`` to drive the client without hitting the
real MCP server. That keeps the tests deterministic and gives us full
control over response shape, latency, and error sequencing.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from app.config import PipelineConfig, Settings
from app.core.exceptions import ValidationError
from app.services.mcp import MCPClient, MCPClientError, PulseDeliveryService
from app.services.pulse import WeeklyPulseGenerator
from app.services.summarization.schemas import (
    FinalTheme,
    SummaryReport,
    SummaryStats,
)

BASE_URL = "https://mcp.test"


_PIPELINE_OVERRIDE_KEYS = {"mcp_max_retries", "mcp_timeout", "mcp_retry_backoff_seconds"}


def _settings(**overrides: object) -> Settings:
    """Build a test Settings instance with friendly kwargs.

    Accepts flat ``mcp_max_retries=N`` style kwargs (legacy ergonomics)
    and routes them into the new nested ``PipelineConfig`` so the call
    sites in this file stay readable.
    """
    pipeline_overrides: dict[str, object] = {
        "MCP_TIMEOUT": 5.0,
        "MCP_MAX_RETRIES": 2,
        "MCP_RETRY_BACKOFF_SECONDS": 0.0,  # no real sleep in tests
    }
    settings_overrides: dict[str, object] = {
        "mcp_base_url": BASE_URL,
        "email_to": "team@example.com",
        "google_doc_id": "test-doc-id",
    }
    for k, v in overrides.items():
        if k in _PIPELINE_OVERRIDE_KEYS:
            pipeline_overrides[k.upper()] = v
        else:
            settings_overrides[k] = v
    settings_overrides["pipeline"] = PipelineConfig(**pipeline_overrides)  # type: ignore[arg-type]
    return Settings(**settings_overrides)  # type: ignore[arg-type]


def _make_client(
    settings: Settings,
    handler: Callable[[httpx.Request], httpx.Response],
) -> MCPClient:
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(base_url=BASE_URL, transport=transport)
    return MCPClient(settings=settings, client=http)


@pytest.mark.asyncio
async def test_append_to_doc_success() -> None:
    settings = _settings()
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = request.content
        return httpx.Response(
            200,
            json={
                "success": True,
                "document_url": "https://docs.google.com/abc",
            },
        )

    client = _make_client(settings, handler)
    async with client:
        resp = await client.append_to_doc(doc_id="DOC123", content="# body")

    assert resp.success is True
    assert resp.document_url == "https://docs.google.com/abc"
    assert captured["path"] == "/append_to_doc"
    assert b"DOC123" in captured["body"]  # type: ignore[operator]


@pytest.mark.asyncio
async def test_create_email_draft_success() -> None:
    settings = _settings()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/create_email_draft"
        return httpx.Response(
            200, json={"success": True, "draft_id": "draft_42"}
        )

    client = _make_client(settings, handler)
    async with client:
        resp = await client.create_email_draft(
            to="user@example.com", subject="Hi", body="See doc."
        )

    assert resp.draft_id == "draft_42"


@pytest.mark.asyncio
async def test_retries_on_5xx_then_succeeds() -> None:
    settings = _settings(mcp_max_retries=2)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, json={"error": "unavailable"})
        return httpx.Response(
            200,
            json={"success": True, "document_url": "https://docs.google.com/xyz"},
        )

    client = _make_client(settings, handler)
    async with client:
        resp = await client.append_to_doc(doc_id="t", content="c")

    assert calls["n"] == 3
    assert resp.document_url.endswith("xyz")


@pytest.mark.asyncio
async def test_retries_exhausted_raises() -> None:
    settings = _settings(mcp_max_retries=1)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(502, json={"error": "bad gateway"})

    client = _make_client(settings, handler)
    with pytest.raises(MCPClientError) as exc_info:
        async with client:
            await client.append_to_doc(doc_id="t", content="c")

    assert calls["n"] == 2  # 1 initial + 1 retry
    assert exc_info.value.status_code == 502


@pytest.mark.asyncio
async def test_4xx_is_not_retried() -> None:
    settings = _settings(mcp_max_retries=3)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, json={"error": "bad request"})

    client = _make_client(settings, handler)
    with pytest.raises(MCPClientError) as exc_info:
        async with client:
            await client.append_to_doc(doc_id="t", content="c")

    assert calls["n"] == 1
    assert exc_info.value.status_code == 400
    assert exc_info.value.payload == {"error": "bad request"}


@pytest.mark.asyncio
async def test_transport_error_is_retried() -> None:
    settings = _settings(mcp_max_retries=1)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("boom", request=request)
        return httpx.Response(
            200,
            json={"success": True, "document_url": "https://docs.google.com/ok"},
        )

    client = _make_client(settings, handler)
    async with client:
        resp = await client.append_to_doc(doc_id="t", content="c")

    assert calls["n"] == 2
    assert resp.success is True


@pytest.mark.asyncio
async def test_missing_base_url_raises() -> None:
    settings = Settings(mcp_base_url="")
    with pytest.raises(ValueError):
        MCPClient(settings=settings)


def _tiny_report() -> SummaryReport:
    return SummaryReport(
        executive_summary="Users love the new dark mode but hit login bugs.",
        themes=[
            FinalTheme(
                label="Login Failures",
                description="Sign-in regression after v4.2.",
                sentiment="negative",
                prevalence="high",
                supporting_quotes=["Can't log in."],
                action_hint="Hotfix v4.2 auth.",
            )
        ],
        action_ideas=["Ship a hotfix."],
        stats=SummaryStats(
            total_reviews=100,
            chunks_processed=2,
            chunks_failed=0,
            themes_premerged=2,
            themes_final=1,
        ),
    )


@pytest.mark.asyncio
async def test_delivery_service_appends_then_drafts() -> None:
    settings = _settings()
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path == "/append_to_doc":
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "document_url": "https://docs.google.com/d/xyz",
                },
            )
        if request.url.path == "/create_email_draft":
            body = request.content.decode()
            # The doc URL must be woven into the email body.
            assert "https://docs.google.com/d/xyz" in body
            return httpx.Response(
                200, json={"success": True, "draft_id": "draft_99"}
            )
        return httpx.Response(404)

    client = _make_client(settings, handler)
    artifacts = WeeklyPulseGenerator().generate([_tiny_report()])

    async with client:
        result = await PulseDeliveryService(
            client=client, settings=settings
        ).deliver(artifacts)

    assert seen == ["/append_to_doc", "/create_email_draft"]
    assert result.status == "succeeded"
    assert result.document_url.endswith("xyz")
    assert result.draft_id == "draft_99"
    assert result.email_to == "team@example.com"
    assert settings.pipeline.EMAIL_SUBJECT_PREFIX in result.email_subject
    assert result.failure_stage is None
    assert result.error is None


@pytest.mark.asyncio
async def test_delivery_partial_when_draft_fails() -> None:
    """Doc append succeeds, draft step fails after retries → status='partial'."""
    settings = _settings(mcp_max_retries=1)
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path == "/append_to_doc":
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "document_url": "https://docs.google.com/d/partial",
                },
            )
        if request.url.path == "/create_email_draft":
            return httpx.Response(503, json={"error": "gmail down"})
        return httpx.Response(404)

    client = _make_client(settings, handler)
    artifacts = WeeklyPulseGenerator().generate([_tiny_report()])

    async with client:
        result = await PulseDeliveryService(
            client=client, settings=settings
        ).deliver(artifacts)

    assert result.status == "partial"
    assert result.document_url.endswith("partial")
    assert result.draft_id is None
    assert result.failure_stage == "create_email_draft"
    assert result.error is not None
    # Verify the draft endpoint was retried per mcp_max_retries before giving up.
    assert seen.count("/create_email_draft") == 2


@pytest.mark.asyncio
async def test_delivery_raises_when_doc_fails() -> None:
    """Doc-append failure after retries surfaces as MCPClientError (no partial)."""
    settings = _settings(mcp_max_retries=1)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/append_to_doc":
            return httpx.Response(502, json={"error": "bad gateway"})
        return httpx.Response(500)

    client = _make_client(settings, handler)
    artifacts = WeeklyPulseGenerator().generate([_tiny_report()])

    with pytest.raises(MCPClientError):
        async with client:
            await PulseDeliveryService(
                client=client, settings=settings
            ).deliver(artifacts)


@pytest.mark.asyncio
async def test_delivery_requires_recipient() -> None:
    settings = _settings(email_to="")
    client = _make_client(settings, lambda _r: httpx.Response(200, json={}))
    artifacts = WeeklyPulseGenerator().generate([_tiny_report()])

    with pytest.raises(ValidationError):
        await PulseDeliveryService(client=client, settings=settings).deliver(
            artifacts
        )
