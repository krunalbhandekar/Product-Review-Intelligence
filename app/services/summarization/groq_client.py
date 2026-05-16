"""Async Groq chat-completions client with retry and JSON-mode helpers.

Uses Groq's OpenAI-compatible HTTP endpoint via ``httpx`` (already a project
dependency — no extra SDK). Two design choices worth calling out:

  * **JSON mode by default.** Every call asks Groq for
    ``response_format={"type": "json_object"}`` so we can validate the
    response against a Pydantic model. The caller still does the parsing
    so each pipeline stage can attach its own schema and recover from
    drift independently.
  * **Retry on transient classes only.** 429, 5xx, network/timeout errors
    are retried with capped exponential backoff. 4xx other than 429
    indicates a programmer error (bad prompt, missing key) and is raised
    immediately — silent retries hide bugs.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
from dataclasses import dataclass
from typing import Any

import httpx

from app.core.exceptions import UpstreamError
from app.core.logging import get_logger

log = get_logger("summarization.groq")

_GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
_RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


def _parse_retry_after(value: str | None) -> float | None:
    """Best-effort parse of ``Retry-After`` (seconds form only)."""
    if not value:
        return None
    try:
        seconds = float(value.strip())
    except (TypeError, ValueError):
        return None
    return seconds if seconds >= 0 else None


@dataclass(frozen=True)
class GroqConfig:
    """Connection + retry knobs for the Groq client."""

    api_key: str
    model: str
    temperature: float = 0.2
    timeout: float = 30.0
    max_retries: int = 3
    base_backoff: float = 0.5
    max_backoff: float = 8.0
    base_url: str = _GROQ_CHAT_URL
    # Ordered fallback chain. Used when the primary model returns 429:
    # the next model is tried for the same prompt instead of retrying
    # the throttled one. Empty tuple disables fallback.
    fallback_models: tuple[str, ...] = ()
    # Consecutive-429 threshold for the in-process circuit breaker.
    # When tripped, the active model is pinned one step down the chain
    # for the remainder of the client's lifetime.
    max_consecutive_429: int = 3
    # Cap on in-flight requests this client will make at once. ``1``
    # serialises calls — the safest default on TPM-limited Groq tiers.
    request_concurrency: int = 1


class _RateLimited(Exception):
    """Internal signal that a single per-model attempt hit 429.

    Surfaces to :meth:`GroqClient.complete_json` which advances the
    fallback chain rather than retrying the same throttled model.
    """


# ``json_object`` mode usually returns clean JSON, but in the wild we
# still see:
#   * a ```json …``` markdown fence wrapping the object,
#   * a prose preamble ("Here is the JSON: { … }"),
#   * trailing commas inside arrays/objects.
# These don't represent a bug we can fix server-side, so we salvage on
# the client. Anything we cannot salvage raises ``json.JSONDecodeError``
# and the caller logs the raw content via ``groq.raw_response``.
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def _salvage_json_loads(raw: str) -> tuple[dict[str, Any], str]:
    """Parse ``raw`` as JSON, attempting progressively looser strategies.

    Returns ``(parsed, mode)`` where ``mode`` is one of
    ``"direct" | "fence" | "object_slice" | "repaired"`` so the caller
    can log which path was taken. Raises :class:`json.JSONDecodeError`
    if nothing works.
    """
    text = raw.strip()
    try:
        return json.loads(text), "direct"
    except json.JSONDecodeError:
        pass

    fence_match = _FENCE_RE.search(text)
    if fence_match:
        fenced = fence_match.group(1).strip()
        try:
            return json.loads(fenced), "fence"
        except json.JSONDecodeError:
            text = fenced  # try further salvage on the unfenced body

    # Slice from the first ``{`` to the matching ``}``. Cheap heuristic
    # that handles a preamble like "Sure, here's the JSON: {...}".
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        sliced = text[start : end + 1]
        try:
            return json.loads(sliced), "object_slice"
        except json.JSONDecodeError:
            text = sliced

    repaired = _TRAILING_COMMA_RE.sub(r"\1", text)
    return json.loads(repaired), "repaired"


class GroqClient:
    """Thin async client around Groq's chat-completions endpoint.

    Re-uses a single ``httpx.AsyncClient`` across calls so connection
    pooling is preserved. Use as an async context manager::

        async with GroqClient(config) as client:
            data = await client.complete_json(system=..., user=...)
    """

    def __init__(
        self,
        config: GroqConfig,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if not config.api_key:
            raise ValueError("GroqConfig.api_key is required")
        self._config = config
        self._http = http_client
        self._owns_http = http_client is None
        # Mutable run-state. ``_active_model`` is the model used as the
        # *first* attempt on each call; the fallback chain is appended
        # after it. The circuit breaker pins this one step down the
        # chain when too many consecutive 429s land.
        self._models: tuple[str, ...] = (config.model, *config.fallback_models)
        self._active_index: int = 0
        self._consecutive_429: int = 0
        # ``request_concurrency=1`` serialises Groq calls — the safest
        # default for TPM-limited tiers. ``asyncio.Semaphore`` is created
        # lazily on first acquire so the client is safe to construct
        # outside an event loop.
        self._semaphore: asyncio.Semaphore | None = None

    async def __aenter__(self) -> GroqClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=self._config.timeout)
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._owns_http and self._http is not None:
            await self._http.aclose()
            self._http = None

    @property
    def active_model(self) -> str:
        """Currently preferred model. Shifts down the chain when the
        circuit breaker trips.

        Falls back to ``self._config.model`` if the fallback chain has
        not been initialised — e.g. for test doubles that bypass
        ``__init__``.
        """
        models = getattr(self, "_models", None)
        idx = getattr(self, "_active_index", 0)
        if models:
            return models[idx]
        config = getattr(self, "_config", None)
        return getattr(config, "model", "unknown") if config else "unknown"

    async def complete_json(
        self,
        *,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stage: str | None = None,
    ) -> dict[str, Any]:
        """Send a chat-completion in JSON mode with model-fallback on 429.

        Behaviour:
        * Acquires the in-process semaphore so concurrent callers can't
          overwhelm Groq's TPM ceiling.
        * Tries the active model first; on HTTP 429 walks down the
          fallback chain without retrying the throttled model.
        * Trips an in-process circuit breaker after
          ``max_consecutive_429`` 429s in a row, pinning the active
          model one step down the chain for the rest of this client's
          lifetime.
        * Transient failures (timeout, transport error, 5xx) are still
          retried with capped exponential backoff *within* a single
          per-model attempt.

        Raises :class:`UpstreamError` if every model in the chain is
        rate-limited or fails fatally.
        """
        if self._http is None:
            raise RuntimeError("GroqClient must be used as an async context manager")

        semaphore = self._get_semaphore()
        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
        }
        effective_temperature = (
            self._config.temperature if temperature is None else temperature
        )
        # Char/4 heuristic, same as the chunker. Logged per-attempt so
        # we can attribute TPM/TPD pressure to specific prompts/stages.
        estimated_prompt_tokens = (len(system) + len(user)) // 4

        async with semaphore:
            chain = self._models[self._active_index :]
            last_429: Exception | None = None
            for offset, model in enumerate(chain):
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": effective_temperature,
                    "response_format": {"type": "json_object"},
                }
                if max_tokens is not None:
                    payload["max_tokens"] = max_tokens

                log.info(
                    "groq.request",
                    stage=stage,
                    model=model,
                    is_fallback=offset > 0,
                    estimated_prompt_tokens=estimated_prompt_tokens,
                    max_tokens=max_tokens,
                )

                try:
                    content = await self._post_with_retry(
                        payload, headers, stage=stage, model=model
                    )
                except _RateLimited as exc:
                    last_429 = exc
                    self._on_429(failed_model=model, stage=stage)
                    log.warning(
                        "groq.fallback.advance",
                        stage=stage,
                        failed_model=model,
                        next_model=(
                            chain[offset + 1] if offset + 1 < len(chain) else None
                        ),
                        consecutive_429=self._consecutive_429,
                    )
                    continue

                # Success on this model — reset the 429 streak so a
                # later transient 429 doesn't immediately trip the
                # circuit breaker.
                self._consecutive_429 = 0
                # Full raw response is logged at debug-equivalent level
                # so we can always reconstruct what the model said even
                # if downstream sanitation/validation rejects it. We
                # truncate the inline preview to keep log lines scannable
                # and emit the full payload on parse failure below.
                log.info(
                    "groq.raw_response",
                    stage=stage,
                    model=model,
                    length=len(content),
                    preview=content[:500],
                )
                try:
                    parsed, mode = _salvage_json_loads(content)
                except json.JSONDecodeError as exc:
                    log.error(
                        "groq.parse_failed",
                        stage=stage,
                        model=model,
                        error=str(exc),
                        raw_repr=repr(content)[:2000],
                    )
                    raise UpstreamError(
                        f"Groq returned non-JSON content from {model}: "
                        f"{exc.msg} (pos {exc.pos})"
                    ) from exc
                log.info(
                    "groq.parsed",
                    stage=stage,
                    model=model,
                    parse_mode=mode,
                    top_level_keys=(
                        sorted(parsed.keys())
                        if isinstance(parsed, dict)
                        else None
                    ),
                )
                if not isinstance(parsed, dict):
                    log.error(
                        "groq.parse_unexpected_shape",
                        stage=stage,
                        model=model,
                        body_type=type(parsed).__name__,
                        raw_repr=repr(content)[:2000],
                    )
                    raise UpstreamError(
                        f"Groq {model} returned non-object JSON "
                        f"({type(parsed).__name__})"
                    )
                return parsed

            # Whole chain exhausted on 429.
            raise UpstreamError(
                f"All Groq models rate-limited (tried {list(chain)}): {last_429}"
            )

    async def _post_with_retry(
        self,
        payload: dict[str, Any],
        headers: dict[str, str],
        *,
        stage: str | None = None,
        model: str | None = None,
    ) -> str:
        """Single per-model attempt, with retry only for transient errors.

        429 raises :class:`_RateLimited` immediately so the caller can
        try the next model in the fallback chain — retrying the same
        throttled model wastes the rate-limit window. 5xx, timeouts,
        and transport errors are retried with capped exponential backoff
        because they are usually flaps, not rate-limits.
        """
        assert self._http is not None  # for type-checker; guarded in caller
        used_model = model or payload.get("model")
        last_error: Exception | None = None

        for attempt in range(self._config.max_retries + 1):
            try:
                response = await self._http.post(
                    self._config.base_url,
                    json=payload,
                    headers=headers,
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                log.warning(
                    "groq.network_error",
                    attempt=attempt,
                    stage=stage,
                    model=used_model,
                    error=type(exc).__name__,
                    detail=str(exc),
                )
            else:
                if response.status_code == 200:
                    data = response.json()
                    usage = data.get("usage") if isinstance(data, dict) else None
                    if isinstance(usage, dict):
                        log.info(
                            "groq.usage",
                            stage=stage,
                            model=used_model,
                            prompt_tokens=usage.get("prompt_tokens"),
                            completion_tokens=usage.get("completion_tokens"),
                            total_tokens=usage.get("total_tokens"),
                        )
                    try:
                        return data["choices"][0]["message"]["content"]
                    except (KeyError, IndexError, TypeError) as exc:
                        raise UpstreamError(
                            f"Groq response missing content: {data!r}"
                        ) from exc

                # Surface 429 immediately — caller picks next model.
                if response.status_code == 429:
                    retry_after = _parse_retry_after(
                        response.headers.get("retry-after")
                    )
                    log.warning(
                        "groq.rate_limited",
                        stage=stage,
                        model=used_model,
                        attempt=attempt,
                        retry_after=retry_after,
                        body=response.text[:200],
                    )
                    raise _RateLimited(
                        f"Groq 429 on {used_model}: {response.text[:200]}"
                    )

                if response.status_code not in _RETRYABLE_STATUS:
                    raise UpstreamError(
                        f"Groq error {response.status_code} on {used_model}: "
                        f"{response.text[:500]}"
                    )

                last_error = UpstreamError(
                    f"Groq retryable {response.status_code} on {used_model}: "
                    f"{response.text[:200]}"
                )
                log.warning(
                    "groq.retryable_status",
                    attempt=attempt,
                    status=response.status_code,
                    stage=stage,
                    model=used_model,
                )

            if attempt >= self._config.max_retries:
                break

            backoff = min(
                self._config.base_backoff * (2**attempt),
                self._config.max_backoff,
            )
            jitter = random.uniform(0, backoff * 0.25)
            await asyncio.sleep(backoff + jitter)

        raise UpstreamError(
            f"Groq exhausted retries on {used_model} "
            f"({self._config.max_retries}): {last_error}"
        )

    def _get_semaphore(self) -> asyncio.Semaphore:
        """Lazily build the semaphore on first acquire — keeps the
        client safe to construct outside an event loop (tests, DI)."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(
                max(1, self._config.request_concurrency)
            )
        return self._semaphore

    def _on_429(self, *, failed_model: str, stage: str | None) -> None:
        """Bookkeeping after a 429 from a per-model attempt.

        Bumps the consecutive-429 counter and, when the breaker
        threshold is crossed, pins the active model one step down the
        chain so subsequent calls skip the throttled model entirely.
        """
        self._consecutive_429 += 1
        if (
            self._consecutive_429 >= self._config.max_consecutive_429
            and self._active_index < len(self._models) - 1
        ):
            old = self._models[self._active_index]
            self._active_index += 1
            new = self._models[self._active_index]
            log.warning(
                "groq.circuit_breaker.tripped",
                stage=stage,
                from_model=old,
                to_model=new,
                consecutive_429=self._consecutive_429,
            )
            # Reset the streak so we don't immediately re-trip and
            # skip past every fallback after a single bad window.
            self._consecutive_429 = 0
