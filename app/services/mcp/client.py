"""Async HTTP client for the external MCP server.

Design notes
------------
* The MCP server is a thin HTTP wrapper around Google Docs + Gmail —
  we only talk to it via JSON POSTs. No SDK required.
* Retries cover *both* transport failures (connection/read/timeout) and
  5xx responses. 4xx is treated as a non-retryable client error.
* Timeouts and retry budget come from :class:`Settings`, so the same
  client can be tuned per environment without code changes.
* The client is reusable as an async context manager so the underlying
  ``httpx.AsyncClient`` (and connection pool) can be reused across calls.
"""

from __future__ import annotations

import asyncio
from types import TracebackType
from typing import Any

import httpx
from pydantic import ValidationError as PydanticValidationError

from app.config import Settings, get_settings
from app.core.exceptions import UpstreamError
from app.core.logging import get_logger
from app.services.mcp.schemas import (
    AppendDocRequest,
    AppendDocResponse,
    CreateEmailDraftRequest,
    CreateEmailDraftResponse,
)

log = get_logger("service.mcp.client")


class MCPClientError(UpstreamError):
    """Raised when the MCP server call cannot be completed.

    Wraps both retry-exhausted transport errors and non-retryable
    4xx/5xx responses. Carries ``status_code`` (HTTP status if known)
    and ``payload`` (parsed body if available) for the caller.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        payload: Any = None,
    ) -> None:
        super().__init__(message, status_code=status_code or 502)
        self.payload = payload


# Statuses we'll retry. Everything else in 4xx is a permanent client
# error (bad request, auth, etc.) and retrying just wastes time.
_RETRYABLE_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})


class MCPClient:
    """Async HTTP client for the Google Docs + Gmail MCP server.

    Use as an async context manager to amortise the underlying connection
    pool across calls::

        async with MCPClient() as mcp:
            doc = await mcp.append_to_doc(title=..., content=...)
            draft = await mcp.create_email_draft(to=..., subject=..., body=...)
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        backoff_seconds: float | None = None,
        settings: Settings | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        cfg = settings or get_settings()
        pipeline = cfg.pipeline

        resolved_base = (base_url or cfg.mcp_base_url).rstrip("/")
        if not resolved_base:
            raise ValueError(
                "MCP base URL not configured. Set MCP_SERVER_URL."
            )

        self._base_url = resolved_base
        self._timeout = timeout if timeout is not None else pipeline.MCP_TIMEOUT
        self._max_retries = (
            max_retries if max_retries is not None else pipeline.MCP_MAX_RETRIES
        )
        self._backoff = (
            backoff_seconds
            if backoff_seconds is not None
            else pipeline.MCP_RETRY_BACKOFF_SECONDS
        )
        # Allow injecting a client for tests (e.g. with MockTransport).
        # When injected, we don't own its lifecycle.
        self._injected_client = client
        self._client: httpx.AsyncClient | None = client

    # -- lifecycle ---------------------------------------------------

    async def __aenter__(self) -> MCPClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
                headers=self._default_headers(),
            )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._injected_client is None and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def aclose(self) -> None:
        await self.__aexit__(None, None, None)

    def _default_headers(self) -> dict[str, str]:
        return {"Accept": "application/json", "Content-Type": "application/json"}

    # -- public API --------------------------------------------------

    async def append_to_doc(
        self, *, doc_id: str, content: str, idempotency_key: str | None = None
    ) -> AppendDocResponse:
        """Append ``content`` (markdown) to the Google Doc identified by ``doc_id``.

        ``idempotency_key`` is forwarded as the ``Idempotency-Key`` header
        so a server-side retry of the same logical operation is a no-op.

        Fails fast with :class:`MCPClientError` if ``doc_id`` or ``content``
        is empty — these are configuration bugs, not upstream issues, and
        sending them produces an opaque 422 from the MCP server.
        """
        if not doc_id or not doc_id.strip():
            raise MCPClientError(
                "append_to_doc requires a non-empty doc_id (set GOOGLE_DOC_ID)",
                status_code=400,
            )
        if not content or not content.strip():
            raise MCPClientError(
                "append_to_doc requires non-empty content",
                status_code=400,
            )
        payload = AppendDocRequest(doc_id=doc_id, content=content).model_dump()
        data = await self._post(
            "/append_to_doc", payload, idempotency_key=idempotency_key
        )
        return _parse_response(AppendDocResponse, data, endpoint="/append_to_doc")

    async def create_email_draft(
        self, *, to: str, subject: str, body: str, idempotency_key: str | None = None
    ) -> CreateEmailDraftResponse:
        """Create a Gmail draft addressed to ``to`` with ``subject``/``body``."""
        payload = CreateEmailDraftRequest(
            to=to, subject=subject, body=body
        ).model_dump()
        data = await self._post(
            "/create_email_draft", payload, idempotency_key=idempotency_key
        )
        return _parse_response(
            CreateEmailDraftResponse, data, endpoint="/create_email_draft"
        )

    # -- internals ---------------------------------------------------

    async def _post(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """POST JSON to ``path`` with retries + structured logging.

        Retries up to ``max_retries`` *additional* attempts (so a value of
        3 means up to 4 total tries) on transport errors and retryable
        5xx/429 responses, with exponential backoff.
        """
        if self._client is None:
            # Allow ad-hoc use without ``async with`` — create + close per call.
            async with self:
                return await self._post(
                    path, payload, idempotency_key=idempotency_key
                )

        attempts = self._max_retries + 1
        last_exc: Exception | None = None

        # Include the same Idempotency-Key on every retry so the server can
        # collapse duplicate logical writes (e.g. retried doc append).
        per_request_headers: dict[str, str] | None = None
        if idempotency_key:
            per_request_headers = {"Idempotency-Key": idempotency_key}

        # Log payload *keys* (not values) so request shape is visible in
        # logs without leaking content. The full URL is the base + path so
        # operators can spot misrouting at a glance.
        endpoint_url = f"{self._base_url}{path}"
        bound = log.bind(
            mcp_path=path,
            mcp_url=endpoint_url,
            payload_keys=sorted(payload.keys()),
        )

        for attempt in range(1, attempts + 1):
            bound.debug("mcp.request", attempt=attempt, max_attempts=attempts)
            try:
                response = await self._client.post(
                    path, json=payload, headers=per_request_headers
                )
            except httpx.TimeoutException as exc:
                last_exc = exc
                bound.warning(
                    "mcp.request.timeout",
                    attempt=attempt,
                    error=str(exc),
                )
                await self._sleep_backoff(attempt)
                continue
            except httpx.TransportError as exc:
                last_exc = exc
                bound.warning(
                    "mcp.request.transport_error",
                    attempt=attempt,
                    error=str(exc),
                )
                await self._sleep_backoff(attempt)
                continue

            if response.status_code in _RETRYABLE_STATUSES:
                bound.warning(
                    "mcp.response.retryable_error",
                    attempt=attempt,
                    status_code=response.status_code,
                )
                last_exc = MCPClientError(
                    f"MCP server returned retryable status {response.status_code}",
                    status_code=response.status_code,
                    payload=_safe_json(response),
                )
                await self._sleep_backoff(attempt)
                continue

            if response.is_error:
                # Non-retryable 4xx — fail fast. Include the server's
                # response body in the error message so the pipeline-level
                # ``details.error`` surfaces the actual validation reason
                # (e.g. FastAPI's ``{"detail": [...]}`` for a 422), not
                # just the opaque status code.
                payload_body = _safe_json(response)
                bound.error(
                    "mcp.response.client_error",
                    status_code=response.status_code,
                    body=payload_body,
                )
                body_repr = _short_body_repr(payload_body)
                message = (
                    f"MCP server rejected request with status "
                    f"{response.status_code}"
                )
                if body_repr:
                    message = f"{message}: {body_repr}"
                raise MCPClientError(
                    message,
                    status_code=response.status_code,
                    payload=payload_body,
                )

            try:
                data: Any = response.json()
            except ValueError as exc:
                bound.error("mcp.response.invalid_json", error=str(exc))
                raise MCPClientError(
                    "MCP server returned a non-JSON response",
                    status_code=response.status_code,
                ) from exc

            if not isinstance(data, dict):
                bound.error("mcp.response.unexpected_shape", body_type=type(data).__name__)
                raise MCPClientError(
                    "MCP server returned a non-object JSON body",
                    status_code=response.status_code,
                    payload=data,
                )

            bound.info(
                "mcp.response.ok",
                attempt=attempt,
                status_code=response.status_code,
            )
            return data

        # Retries exhausted.
        bound.error(
            "mcp.request.retries_exhausted",
            attempts=attempts,
            last_error=str(last_exc) if last_exc else None,
        )
        if isinstance(last_exc, MCPClientError):
            raise last_exc
        raise MCPClientError(
            f"MCP request to {path} failed after {attempts} attempts: {last_exc}",
            status_code=502,
        ) from last_exc

    async def _sleep_backoff(self, attempt: int) -> None:
        # Exponential backoff: backoff * 2^(attempt-1). attempt starts at 1.
        delay = self._backoff * (2 ** (attempt - 1))
        await asyncio.sleep(delay)


def _parse_response(
    model_cls: type, data: dict[str, Any], *, endpoint: str
) -> Any:
    """Validate a successful MCP response body, logging the raw keys.

    The MCP server's response shape varies (e.g. ``status`` vs
    ``success``); the response models normalise this transparently.
    Logging the raw key set on every successful call gives us a
    breadcrumb when the contract drifts again — without ever logging
    user-content values like draft bodies or doc text.
    """
    log.info(
        "mcp.response.raw",
        endpoint=endpoint,
        raw_keys=sorted(data.keys()) if isinstance(data, dict) else None,
    )
    try:
        return model_cls.model_validate(data)
    except PydanticValidationError as exc:
        log.error(
            "mcp.response.schema_mismatch",
            endpoint=endpoint,
            raw_keys=sorted(data.keys()) if isinstance(data, dict) else None,
            errors=exc.errors(),
        )
        raise MCPClientError(
            f"MCP {endpoint} response did not match expected schema: "
            f"{exc.errors()}",
            status_code=502,
            payload=data,
        ) from exc


def _safe_json(response: httpx.Response) -> Any:
    """Best-effort JSON decode of an error response body, for logging."""
    try:
        return response.json()
    except ValueError:
        return response.text[:500] if response.text else None


def _short_body_repr(body: Any) -> str:
    """Compact stringification of an error body for inclusion in messages."""
    if body is None:
        return ""
    text = body if isinstance(body, str) else str(body)
    text = text.strip()
    return text[:300] + ("…" if len(text) > 300 else "")


__all__ = ["MCPClient", "MCPClientError"]
