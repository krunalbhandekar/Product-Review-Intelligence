"""Apple App Store public review ingestion.

Uses Apple's public customer-reviews RSS feed
(``itunes.apple.com/<country>/rss/customerreviews/...``). The feed exposes
public review data only — no login or private API surface is touched.

Reviews are sorted NEWEST first; the stream stops as soon as a review older
than ``since`` is observed. Apple caps the feed at 10 pages of ~50 entries
per country, so the importer naturally bounds itself.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from types import TracebackType
from typing import Any
from urllib.parse import urlsplit

import httpx

from app.core.exceptions import UpstreamError
from app.core.logging import get_logger
from app.domain.enums import Platform
from app.domain.models import Review

log = get_logger("ingest.app_store")

_FEED_URL = (
    "https://itunes.apple.com/{country}/rss/customerreviews"
    "/page={page}/id={app_id}/sortby=mostrecent/json"
)
_APPLE_MAX_PAGES = 10
_DEFAULT_TIMEOUT = httpx.Timeout(15.0, connect=5.0)
_RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


def _label(node: Any) -> str | None:
    if isinstance(node, dict):
        value = node.get("label")
        if value is None:
            return None
        text = str(value).strip()
        return text or None
    return None


def _parse_datetime(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return dt.astimezone(UTC) if dt.tzinfo else dt.replace(tzinfo=UTC)


def _entries(payload: Any) -> list[dict[str, Any]]:
    feed = payload.get("feed") if isinstance(payload, dict) else None
    entry = feed.get("entry") if isinstance(feed, dict) else None
    if entry is None:
        return []
    if isinstance(entry, list):
        return [e for e in entry if isinstance(e, dict)]
    if isinstance(entry, dict):
        return [entry]
    return []


def _entry_to_review(
    entry: dict[str, Any],
    *,
    app_id: str,
    country: str,
    lang: str,
) -> Review | None:
    """Map a single RSS ``entry`` dict to a domain ``Review``.

    Returns ``None`` for entries that are not reviews (e.g. the app-metadata
    entry Apple sometimes includes) or that fail validation.
    """
    rating_raw = _label(entry.get("im:rating"))
    review_id = _label(entry.get("id"))
    posted = _parse_datetime(_label(entry.get("updated")))
    if rating_raw is None or review_id is None or posted is None:
        return None
    try:
        rating = int(rating_raw)
    except ValueError:
        return None
    if not 1 <= rating <= 5:
        return None

    author_node = entry.get("author")
    author = _label(author_node.get("name")) if isinstance(author_node, dict) else None

    return Review(
        source=Platform.IOS,
        app_id=app_id,
        review_id=review_id,
        rating=rating,
        title=_label(entry.get("title")),
        body=_label(entry.get("content")) or "",
        author=author,
        posted_at=posted,
        lang=lang,
        country=country,
        app_version=_label(entry.get("im:version")),
    )


class AppStoreSource:
    """Streams public App Store reviews from the iTunes RSS feed."""

    def __init__(
        self,
        *,
        lang: str = "en",
        country: str = "us",
        max_reviews: int | None = None,
        max_pages: int = _APPLE_MAX_PAGES,
        max_retries: int = 3,
        retry_backoff: float = 1.5,
        timeout: httpx.Timeout | float = _DEFAULT_TIMEOUT,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")
        self.lang = lang
        self.country = country.lower()
        self.max_reviews = max_reviews
        self.max_pages = max(1, min(max_pages, _APPLE_MAX_PAGES))
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self._timeout = timeout
        self._client = client
        self._owns_client = client is None

    async def __aenter__(self) -> AppStoreSource:
        await self._get_client()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def _fetch_page(self, *, app_id: str, page: int) -> dict[str, Any]:
        # Defensive validation: reject empty / URL-breaking values up front
        # so we fail fast on a config bug instead of producing a malformed
        # request that looks like an upstream outage.
        clean_id = (app_id or "").strip()
        if not clean_id or any(c in clean_id for c in " /?#"):
            raise UpstreamError(
                f"App Store app_id is empty or malformed: {app_id!r}"
            )
        if not self.country or any(c in self.country for c in " /?#"):
            raise UpstreamError(
                f"App Store country is empty or malformed: {self.country!r}"
            )

        url = _FEED_URL.format(country=self.country, page=page, app_id=clean_id)
        parsed = urlsplit(url)
        host = parsed.hostname or ""
        if parsed.scheme not in {"http", "https"} or not host:
            raise UpstreamError(f"App Store RSS URL malformed: {url!r}")

        client = await self._get_client()
        last_error: str = ""

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = await client.get(url, headers={"Accept": "application/json"})
            except httpx.ConnectError as exc:
                # DNS/TCP failures are common transient issues — log host
                # explicitly so operators can distinguish DNS resolution
                # problems from URL construction bugs.
                last_error = f"ConnectError: {exc}"
                log.warning(
                    "app_store.page.connect_error",
                    url=url,
                    host=host,
                    page=page,
                    attempt=attempt,
                    error=last_error,
                )
            except httpx.TimeoutException as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                log.warning(
                    "app_store.page.timeout",
                    url=url,
                    host=host,
                    page=page,
                    attempt=attempt,
                    error=last_error,
                )
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                log.warning(
                    "app_store.page.transport_error",
                    url=url,
                    host=host,
                    page=page,
                    attempt=attempt,
                    error=last_error,
                )
            else:
                status = resp.status_code
                if status == 200:
                    try:
                        data = resp.json()
                    except ValueError as exc:
                        last_error = f"invalid JSON: {exc}"
                        log.warning(
                            "app_store.page.bad_json",
                            page=page,
                            attempt=attempt,
                        )
                    else:
                        if not isinstance(data, dict):
                            last_error = "unexpected payload shape"
                            log.warning(
                                "app_store.page.bad_shape",
                                page=page,
                                attempt=attempt,
                            )
                        else:
                            return data
                elif status in _RETRYABLE_STATUS:
                    last_error = f"HTTP {status}"
                    log.warning(
                        "app_store.page.retryable_status",
                        url=url,
                        host=host,
                        page=page,
                        attempt=attempt,
                        status=status,
                    )
                else:
                    log.error(
                        "app_store.page.fatal_status",
                        url=url,
                        host=host,
                        page=page,
                        status=status,
                    )
                    raise UpstreamError(
                        f"App Store RSS returned HTTP {status} for page {page}"
                    )

            if attempt < self.max_retries:
                await asyncio.sleep(self.retry_backoff * (2 ** (attempt - 1)))

        raise UpstreamError(
            f"App Store RSS failed after {self.max_retries} attempts "
            f"(page={page}): {last_error}"
        )

    async def stream(
        self,
        *,
        app_id: str,
        since: datetime,
        until: datetime,
    ) -> AsyncIterator[Review]:
        """Yield reviews for ``app_id`` posted within ``[since, until]`` (UTC)."""
        if since > until:
            raise ValueError("since must be <= until")

        yielded = 0
        for page in range(1, self.max_pages + 1):
            payload = await self._fetch_page(app_id=app_id, page=page)
            entries = _entries(payload)
            if not entries:
                log.info(
                    "app_store.stream.end",
                    reason="empty_page",
                    page=page,
                    yielded=yielded,
                )
                return

            window_exhausted = False
            page_yielded = 0
            for entry in entries:
                review = _entry_to_review(
                    entry,
                    app_id=app_id,
                    country=self.country,
                    lang=self.lang,
                )
                if review is None:
                    continue
                if review.posted_at > until:
                    continue
                if review.posted_at < since:
                    window_exhausted = True
                    break

                yield review
                yielded += 1
                page_yielded += 1

                if self.max_reviews is not None and yielded >= self.max_reviews:
                    log.info(
                        "app_store.stream.end",
                        reason="max_reviews",
                        page=page,
                        yielded=yielded,
                    )
                    return

            if window_exhausted:
                log.info(
                    "app_store.stream.end",
                    reason="window_exhausted",
                    page=page,
                    yielded=yielded,
                )
                return

            if page_yielded == 0:
                log.info(
                    "app_store.stream.end",
                    reason="no_eligible_entries",
                    page=page,
                    yielded=yielded,
                )
                return

        log.info(
            "app_store.stream.end",
            reason="max_pages",
            page=self.max_pages,
            yielded=yielded,
        )
