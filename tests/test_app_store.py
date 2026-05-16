from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest

from app.core.exceptions import UpstreamError
from app.domain.enums import Platform
from app.services.ingest.app_store import (
    AppStoreSource,
    _entries,
    _entry_to_review,
    _parse_datetime,
)
from app.services.ingest.time_window import lookback_window


def _entry(
    *,
    review_id: str = "1234",
    rating: str = "5",
    title: str = "Great",
    body: str = "Body text",
    author: str | None = "alice",
    version: str | None = "1.2.3",
    updated: str = "2026-05-01T12:34:56-07:00",
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "id": {"label": review_id},
        "im:rating": {"label": rating},
        "title": {"label": title},
        "content": {"label": body, "attributes": {"type": "text"}},
        "updated": {"label": updated},
    }
    if author is not None:
        entry["author"] = {"name": {"label": author}}
    if version is not None:
        entry["im:version"] = {"label": version}
    return entry


def test_parse_datetime_normalizes_to_utc() -> None:
    dt = _parse_datetime("2026-05-01T12:00:00-07:00")
    assert dt is not None
    assert dt.tzinfo is UTC
    assert dt.hour == 19


def test_parse_datetime_handles_naive_as_utc() -> None:
    dt = _parse_datetime("2026-05-01T12:00:00")
    assert dt is not None
    assert dt.tzinfo is UTC
    assert dt.hour == 12


def test_parse_datetime_returns_none_on_garbage() -> None:
    assert _parse_datetime(None) is None
    assert _parse_datetime("") is None
    assert _parse_datetime("not-a-date") is None


def test_entries_handles_list_dict_and_missing() -> None:
    assert _entries({"feed": {"entry": [{"a": 1}, "skip", {"b": 2}]}}) == [
        {"a": 1},
        {"b": 2},
    ]
    assert _entries({"feed": {"entry": {"a": 1}}}) == [{"a": 1}]
    assert _entries({"feed": {}}) == []
    assert _entries({}) == []
    assert _entries(None) == []


def test_entry_to_review_maps_required_fields() -> None:
    r = _entry_to_review(_entry(), app_id="310633997", country="us", lang="en")
    assert r is not None
    assert r.source == Platform.IOS
    assert r.app_id == "310633997"
    assert r.review_id == "1234"
    assert r.rating == 5
    assert r.title == "Great"
    assert r.body == "Body text"
    assert r.author == "alice"
    assert r.app_version == "1.2.3"
    assert r.country == "us"
    assert r.lang == "en"
    assert r.posted_at.tzinfo is UTC


def test_entry_to_review_drops_invalid_rating() -> None:
    assert _entry_to_review(_entry(rating="0"), app_id="x", country="us", lang="en") is None
    assert _entry_to_review(_entry(rating="6"), app_id="x", country="us", lang="en") is None
    assert _entry_to_review(_entry(rating="foo"), app_id="x", country="us", lang="en") is None


def test_entry_to_review_handles_missing_optional_fields() -> None:
    r = _entry_to_review(
        _entry(author=None, version=None, title="", body=""),
        app_id="x",
        country="us",
        lang="en",
    )
    assert r is not None
    assert r.author is None
    assert r.app_version is None
    # empty title label collapses to None; body coerces to empty string
    assert r.title is None
    assert r.body == ""


def test_lookback_window_clamps_to_band() -> None:
    end = datetime(2026, 5, 14, tzinfo=UTC)
    since, until = lookback_window(2, until=end)
    assert until == end
    assert until - since == timedelta(weeks=8)

    since, until = lookback_window(20, until=end)
    assert until - since == timedelta(weeks=12)

    since, until = lookback_window(10, until=end)
    assert until - since == timedelta(weeks=10)


def _feed(entries: list[dict[str, Any]]) -> dict[str, Any]:
    return {"feed": {"entry": entries}}


def _make_transport(pages: list[Any]) -> tuple[httpx.MockTransport, list[str]]:
    """Mock transport that returns one payload per page in order.

    Items in ``pages`` may be:
      - a ``dict`` -> 200 with that JSON body
      - an ``int``  -> that HTTP status with empty body
    """
    calls: list[str] = []
    iterator = iter(pages)

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        item = next(iterator)
        if isinstance(item, int):
            return httpx.Response(item)
        return httpx.Response(200, json=item)

    return httpx.MockTransport(handler), calls


async def _collect(source: AppStoreSource, **kwargs: Any) -> list[Any]:
    return [r async for r in source.stream(**kwargs)]


async def test_stream_yields_reviews_within_window() -> None:
    page1 = _feed(
        [
            _entry(review_id="r1", updated="2026-05-10T00:00:00+00:00"),
            _entry(review_id="r2", updated="2026-05-05T00:00:00+00:00"),
        ]
    )
    page2 = _feed([_entry(review_id="r3", updated="2026-04-30T00:00:00+00:00")])
    transport, calls = _make_transport([page1, page2, _feed([])])

    async with httpx.AsyncClient(transport=transport) as http:
        src = AppStoreSource(client=http, max_pages=10)
        until = datetime(2026, 5, 14, tzinfo=UTC)
        since = until - timedelta(weeks=8)
        out = await _collect(src, app_id="310633997", since=since, until=until)

    assert [r.review_id for r in out] == ["r1", "r2", "r3"]
    assert all(r.source == Platform.IOS for r in out)
    assert "page=1" in calls[0]
    assert "page=2" in calls[1]


async def test_stream_stops_when_window_exhausted() -> None:
    page1 = _feed(
        [
            _entry(review_id="r1", updated="2026-05-10T00:00:00+00:00"),
            _entry(review_id="too-old", updated="2026-01-01T00:00:00+00:00"),
        ]
    )
    transport, calls = _make_transport([page1])

    async with httpx.AsyncClient(transport=transport) as http:
        src = AppStoreSource(client=http)
        until = datetime(2026, 5, 14, tzinfo=UTC)
        since = until - timedelta(weeks=8)
        out = await _collect(src, app_id="x", since=since, until=until)

    assert [r.review_id for r in out] == ["r1"]
    assert len(calls) == 1  # second page never requested


async def test_stream_respects_max_reviews() -> None:
    page1 = _feed(
        [
            _entry(review_id="r1", updated="2026-05-10T00:00:00+00:00"),
            _entry(review_id="r2", updated="2026-05-09T00:00:00+00:00"),
            _entry(review_id="r3", updated="2026-05-08T00:00:00+00:00"),
        ]
    )
    transport, _calls = _make_transport([page1])

    async with httpx.AsyncClient(transport=transport) as http:
        src = AppStoreSource(client=http, max_reviews=2)
        until = datetime(2026, 5, 14, tzinfo=UTC)
        since = until - timedelta(weeks=8)
        out = await _collect(src, app_id="x", since=since, until=until)

    assert [r.review_id for r in out] == ["r1", "r2"]


async def test_stream_retries_on_transient_status() -> None:
    page1 = _feed([_entry(review_id="r1", updated="2026-05-10T00:00:00+00:00")])
    transport, calls = _make_transport([503, 429, page1, _feed([])])

    async with httpx.AsyncClient(transport=transport) as http:
        src = AppStoreSource(
            client=http,
            max_retries=3,
            retry_backoff=0.0,
            max_pages=1,
        )
        until = datetime(2026, 5, 14, tzinfo=UTC)
        since = until - timedelta(weeks=8)
        out = await _collect(src, app_id="x", since=since, until=until)

    assert [r.review_id for r in out] == ["r1"]
    assert len(calls) == 3  # 2 retries + success


async def test_stream_raises_after_exhausting_retries() -> None:
    transport, _calls = _make_transport([503, 503, 503])

    async with httpx.AsyncClient(transport=transport) as http:
        src = AppStoreSource(client=http, max_retries=3, retry_backoff=0.0)
        until = datetime(2026, 5, 14, tzinfo=UTC)
        since = until - timedelta(weeks=8)
        with pytest.raises(UpstreamError):
            await _collect(src, app_id="x", since=since, until=until)


async def test_stream_raises_on_fatal_status() -> None:
    transport, _calls = _make_transport([404])

    async with httpx.AsyncClient(transport=transport) as http:
        src = AppStoreSource(client=http, max_retries=3, retry_backoff=0.0)
        until = datetime(2026, 5, 14, tzinfo=UTC)
        since = until - timedelta(weeks=8)
        with pytest.raises(UpstreamError):
            await _collect(src, app_id="bogus", since=since, until=until)


async def test_stream_rejects_inverted_window() -> None:
    transport, _calls = _make_transport([])
    async with httpx.AsyncClient(transport=transport) as http:
        src = AppStoreSource(client=http)
        with pytest.raises(ValueError):
            await _collect(
                src,
                app_id="x",
                since=datetime(2026, 5, 14, tzinfo=UTC),
                until=datetime(2026, 5, 1, tzinfo=UTC),
            )
