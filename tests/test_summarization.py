"""Tests for the chunk-based summarisation pipeline.

The Groq client is faked end-to-end — no network calls. The fake records
every prompt it sees and returns a script of canned JSON dicts, so we
can assert both the *contract* with Groq (system/user prompts, JSON
schema discipline) and the *behaviour* of the pipeline (map fan-out,
reduce merge, per-chunk failure isolation).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from app.domain.enums import Platform
from app.domain.models import Review
from app.services.summarization import (
    ChunkSummary,
    ChunkTheme,
    ReviewChunker,
    SummarizationService,
    ThemeAggregator,
    estimate_tokens,
    premerge_themes,
)
from app.services.summarization.groq_client import GroqClient


def _review(rid: str, body: str, *, rating: int = 4, title: str | None = None) -> Review:
    return Review(
        source=Platform.IOS,
        app_id="310633997",
        review_id=rid,
        rating=rating,
        title=title,
        body=body,
        author=None,
        posted_at=datetime(2026, 5, 10, tzinfo=UTC),
        lang="en",
        country="us",
    )


# --- chunker --------------------------------------------------------------


def test_estimate_tokens_chars_over_four() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 100) == 25


def test_chunker_packs_until_token_budget() -> None:
    body = "x" * 400  # ~100 tokens once rendered
    reviews = [_review(f"r{i}", body) for i in range(10)]
    chunker = ReviewChunker(target_tokens=250, max_reviews=50)
    chunks = list(chunker.chunk(reviews))

    assert len(chunks) >= 4  # ~2-3 reviews per chunk at this budget
    assert all(c.token_estimate <= 250 or c.review_count == 1 for c in chunks)
    # Every review survives, in order.
    flat = [r.review_id for c in chunks for r in c.reviews]
    assert flat == [r.review_id for r in reviews]


def test_chunker_respects_max_reviews_cap() -> None:
    reviews = [_review(f"r{i}", "ok") for i in range(25)]
    chunker = ReviewChunker(target_tokens=10_000, max_reviews=10)
    chunks = list(chunker.chunk(reviews))
    assert [c.review_count for c in chunks] == [10, 10, 5]


def test_chunker_isolates_oversized_review() -> None:
    big = _review("big", "y" * 20_000)
    small = _review("small", "short")
    chunker = ReviewChunker(target_tokens=200, max_reviews=50)
    chunks = list(chunker.chunk([big, small]))
    # The oversized review goes in its own chunk; the small one tags along
    # in the next chunk rather than being squeezed in next to the giant.
    assert chunks[0].reviews == (big,)
    assert chunks[0].token_estimate > 200
    assert chunks[1].reviews == (small,)


def test_chunker_rejects_bad_config() -> None:
    with pytest.raises(ValueError):
        ReviewChunker(target_tokens=0)
    with pytest.raises(ValueError):
        ReviewChunker(max_reviews=0)


def test_chunker_renders_review_with_id_rating_country() -> None:
    chunker = ReviewChunker(target_tokens=10_000, max_reviews=50)
    [chunk] = list(chunker.chunk([_review("abc", "hello world", rating=5)]))
    assert "[abc|5*|us]" in chunk.rendered
    assert "hello world" in chunk.rendered


# --- aggregator pre-merge -------------------------------------------------


def test_premerge_themes_combines_identical_labels() -> None:
    chunks = [
        ChunkSummary(
            chunk_id=0,
            review_count=10,
            themes=[
                ChunkTheme(
                    label="Login Failures",
                    description="Users cannot sign in.",
                    sentiment="negative",
                    evidence_count=3,
                    sample_quotes=["Can't log in"],
                ),
            ],
        ),
        ChunkSummary(
            chunk_id=1,
            review_count=10,
            themes=[
                ChunkTheme(
                    label="login failures!",  # normalises identically
                    description="Same issue different chunk.",
                    sentiment="negative",
                    evidence_count=2,
                    sample_quotes=["Login is broken", "Can't log in"],  # dup quote
                ),
            ],
        ),
    ]
    merged = premerge_themes(chunks)
    assert len(merged) == 1
    assert merged[0].evidence_count == 5
    # Quote union, de-duplicated, capped at 3.
    assert merged[0].sample_quotes == ["Can't log in", "Login is broken"]


def test_premerge_themes_marks_mixed_when_sentiments_differ() -> None:
    chunks = [
        ChunkSummary(
            chunk_id=0,
            review_count=1,
            themes=[ChunkTheme(label="UI", description="ok", sentiment="positive")],
        ),
        ChunkSummary(
            chunk_id=1,
            review_count=1,
            themes=[ChunkTheme(label="ui", description="bad", sentiment="negative")],
        ),
    ]
    merged = premerge_themes(chunks)
    assert merged[0].sentiment == "mixed"


# --- service end-to-end with a fake client --------------------------------


class FakeGroqClient(GroqClient):
    """In-process stand-in for ``GroqClient``.

    Bypasses ``__init__`` (no API key needed) and overrides ``complete_json``
    to consume a scripted queue of responses. Records every prompt so tests
    can assert on the contract Groq sees.
    """

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        # Intentionally skip super().__init__ — we never touch httpx.
        self._responses: list[dict[str, Any]] = list(responses)
        self.calls: list[dict[str, str]] = []

    async def __aenter__(self) -> FakeGroqClient:  # type: ignore[override]
        return self

    async def __aexit__(self, *exc: object) -> None:  # type: ignore[override]
        return None

    async def complete_json(  # type: ignore[override]
        self,
        *,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stage: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append({"system": system, "user": user})
        if not self._responses:
            raise AssertionError("FakeGroqClient ran out of scripted responses")
        return self._responses.pop(0)


def _chunk_response(label: str, *, count: int, quote: str) -> dict[str, Any]:
    return {
        "summary": f"Users mention {label}.",
        "themes": [
            {
                "label": label,
                "description": f"{label} issue.",
                "sentiment": "negative",
                "evidence_count": count,
                "sample_quotes": [quote],
            }
        ],
    }


def _reduce_response(labels: list[str]) -> dict[str, Any]:
    return {
        "executive_summary": "Overall users complain about " + ", ".join(labels) + ".",
        "themes": [
            {
                "label": lab,
                "description": f"Consolidated {lab}.",
                "sentiment": "negative",
                "prevalence": "high",
                "supporting_quotes": [f"quote-{lab}"],
                "action_hint": f"Fix {lab}.",
            }
            for lab in labels
        ],
    }


async def test_service_summarize_happy_path() -> None:
    # Two chunks (2 reviews each, force-split by tiny token budget).
    reviews = [
        _review("r1", "Login is broken " * 20),
        _review("r2", "Cannot sign in " * 20),
        _review("r3", "Crashes on launch " * 20),
        _review("r4", "App won't open " * 20),
    ]
    chunker = ReviewChunker(target_tokens=10_000, max_reviews=2)
    service = SummarizationService(
        chunker=chunker,
        aggregator=ThemeAggregator(max_themes=5),
    )

    fake = FakeGroqClient(
        responses=[
            _chunk_response("Login Failures", count=2, quote="Can't log in"),
            _chunk_response("Crash on Launch", count=2, quote="App crashes"),
            _reduce_response(["Login Failures", "Crash on Launch"]),
        ]
    )

    report = await service.summarize(reviews, client=fake)

    # 2 map calls + 1 reduce call.
    assert len(fake.calls) == 3
    assert report.stats.total_reviews == 4
    assert report.stats.chunks_processed == 2
    assert report.stats.chunks_failed == 0
    assert report.stats.themes_final == 2
    assert [t.label for t in report.themes] == ["Login Failures", "Crash on Launch"]
    assert "Login Failures" in report.executive_summary


async def test_service_caps_final_themes() -> None:
    reviews = [_review(f"r{i}", "feedback " * 20) for i in range(2)]
    chunker = ReviewChunker(target_tokens=10_000, max_reviews=50)
    service = SummarizationService(
        chunker=chunker,
        aggregator=ThemeAggregator(max_themes=3),
    )

    # Reduce step returns 6 themes — service must clip to max_themes=3.
    fake = FakeGroqClient(
        responses=[
            _chunk_response("A", count=1, quote="qa"),
            _reduce_response(["A", "B", "C", "D", "E", "F"]),
        ]
    )
    report = await service.summarize(reviews, client=fake)
    assert len(report.themes) == 3
    assert [t.label for t in report.themes] == ["A", "B", "C"]


async def test_service_isolates_chunk_failures() -> None:
    reviews = [
        _review("r1", "good thing " * 20),
        _review("r2", "bad thing " * 20),
    ]
    chunker = ReviewChunker(target_tokens=80, max_reviews=1)

    class FlakyClient(FakeGroqClient):
        async def complete_json(  # type: ignore[override]
            self,
            *,
            system: str,
            user: str,
            temperature: float | None = None,
            max_tokens: int | None = None,
            stage: str | None = None,
        ) -> dict[str, Any]:
            self.calls.append({"system": system, "user": user})
            # First map call blows up; subsequent calls return scripted data.
            if len(self.calls) == 1:
                raise RuntimeError("simulated upstream blip")
            if not self._responses:
                raise AssertionError("out of scripted responses")
            return self._responses.pop(0)

    fake = FlakyClient(
        responses=[
            _chunk_response("UI Bugs", count=1, quote="bug"),
            _reduce_response(["UI Bugs"]),
        ]
    )

    service = SummarizationService(chunker=chunker)
    report = await service.summarize(reviews, client=fake)

    assert report.stats.chunks_processed == 1
    assert report.stats.chunks_failed == 1
    assert [t.label for t in report.themes] == ["UI Bugs"]


async def test_service_empty_reviews_short_circuits() -> None:
    service = SummarizationService()
    fake = FakeGroqClient(responses=[])  # must not be touched
    report = await service.summarize([], client=fake)
    assert fake.calls == []
    assert report.stats.total_reviews == 0
    assert report.themes == []


async def test_service_sends_json_only_system_prompt() -> None:
    """Guards the prompt contract: system prompt must instruct JSON-only output."""
    reviews = [_review("r1", "ok " * 20)]
    fake = FakeGroqClient(
        responses=[
            _chunk_response("Latency", count=1, quote="slow"),
            _reduce_response(["Latency"]),
        ]
    )
    await SummarizationService().summarize(reviews, client=fake)

    for call in fake.calls:
        assert "JSON" in call["system"]
        assert "no prose" in call["system"].lower()
    # Reduce prompt must echo the max_themes cap.
    reduce_call = fake.calls[-1]
    # Default settings.max_themes is 5.
    assert "AT MOST 5" in reduce_call["system"]


# --- groq client retry surface --------------------------------------------


async def test_groq_client_requires_api_key() -> None:
    from app.services.summarization.groq_client import GroqConfig

    with pytest.raises(ValueError):
        # constructing the client does the validation
        GroqClient(GroqConfig(api_key="", model="m"))


async def test_groq_client_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Networking is faked via a stub AsyncClient so we exercise the retry loop."""
    from app.services.summarization.groq_client import GroqClient, GroqConfig

    calls = {"n": 0}

    class _Response:
        def __init__(self, status_code: int, payload: dict[str, Any] | str) -> None:
            self.status_code = status_code
            self._payload = payload
            self.text = payload if isinstance(payload, str) else json.dumps(payload)

        def json(self) -> dict[str, Any]:
            assert isinstance(self._payload, dict)
            return self._payload

    class _StubHttp:
        async def post(self, *_a: object, **_kw: object) -> _Response:
            calls["n"] += 1
            if calls["n"] < 3:
                return _Response(503, "service unavailable")
            return _Response(
                200,
                {"choices": [{"message": {"content": json.dumps({"ok": True})}}]},
            )

        async def aclose(self) -> None:
            return None

    # No real sleeping.
    async def _no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("app.services.summarization.groq_client.asyncio.sleep", _no_sleep)

    client = GroqClient(
        GroqConfig(api_key="k", model="m", max_retries=3, base_backoff=0),
        http_client=_StubHttp(),  # type: ignore[arg-type]
    )
    async with client:
        data = await client.complete_json(system="s", user="u")
    assert data == {"ok": True}
    assert calls["n"] == 3


async def test_groq_client_non_retryable_raises_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.core.exceptions import UpstreamError
    from app.services.summarization.groq_client import GroqClient, GroqConfig

    calls = {"n": 0}

    class _Response:
        status_code = 401
        text = "unauthorised"

    class _StubHttp:
        async def post(self, *_a: object, **_kw: object) -> _Response:
            calls["n"] += 1
            return _Response()

        async def aclose(self) -> None:
            return None

    async def _no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("app.services.summarization.groq_client.asyncio.sleep", _no_sleep)

    client = GroqClient(
        GroqConfig(api_key="k", model="m", max_retries=3),
        http_client=_StubHttp(),  # type: ignore[arg-type]
    )
    with pytest.raises(UpstreamError):
        async with client:
            await client.complete_json(system="s", user="u")
    assert calls["n"] == 1  # not retried
