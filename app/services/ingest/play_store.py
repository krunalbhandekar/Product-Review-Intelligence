import asyncio
import random
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

from google_play_scraper import Sort, reviews

from app.core.exceptions import UpstreamError
from app.core.logging import get_logger
from app.domain.enums import Platform
from app.domain.models import Review

log = get_logger("ingest.play_store")

_DEFAULT_PAGE_TIMEOUT = 20.0
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_BASE_BACKOFF = 1.0
_DEFAULT_MAX_BACKOFF = 8.0


def _ensure_utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt


def _to_review(raw: dict[str, Any], *, app_id: str, country: str, lang: str) -> Review:
    return Review(
        source=Platform.ANDROID,
        app_id=app_id,
        review_id=str(raw["reviewId"]),
        rating=int(raw["score"]),
        title=None,
        body=str(raw.get("content") or ""),
        author=raw.get("userName"),
        posted_at=_ensure_utc(raw["at"]),
        lang=lang,
        country=country,
    )


class PlayStoreSource:
    """Streams public Play Store reviews via google-play-scraper.

    Reviews are sorted NEWEST first; the stream stops as soon as a review
    older than ``since`` is observed, so we never buffer the full corpus.
    """

    def __init__(
        self,
        *,
        lang: str = "en",
        country: str = "us",
        page_size: int = 200,
        max_reviews: int | None = None,
        page_timeout: float = _DEFAULT_PAGE_TIMEOUT,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        base_backoff: float = _DEFAULT_BASE_BACKOFF,
        max_backoff: float = _DEFAULT_MAX_BACKOFF,
    ) -> None:
        self.lang = lang
        self.country = country
        self.page_size = page_size
        self.max_reviews = max_reviews
        self.page_timeout = page_timeout
        self.max_retries = max(1, max_retries)
        self.base_backoff = base_backoff
        self.max_backoff = max_backoff

    async def _fetch_page(
        self, *, app_id: str, token: Any, page: int
    ) -> tuple[list[dict[str, Any]], Any]:
        """Fetch a single Play Store page with timeout + retry/backoff.

        ``google_play_scraper.reviews`` is a blocking call that hits a public
        Google endpoint with no rate-limit visibility — wrap it in a thread
        plus ``asyncio.wait_for`` so a single slow page can't stall the run,
        and retry on timeout / transient errors with exponential backoff.
        """
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(
                        reviews,
                        app_id,
                        lang=self.lang,
                        country=self.country,
                        sort=Sort.NEWEST,
                        count=self.page_size,
                        continuation_token=token,
                    ),
                    timeout=self.page_timeout,
                )
            except TimeoutError as exc:
                last_error = exc
                log.warning(
                    "play_store.page.timeout",
                    page=page,
                    attempt=attempt,
                    timeout=self.page_timeout,
                )
            except Exception as exc:  # pragma: no cover - defensive
                # google_play_scraper raises bare Exception subclasses on
                # transport / parsing issues; retry rather than failing the
                # whole pipeline on a transient blip.
                last_error = exc
                log.warning(
                    "play_store.page.error",
                    page=page,
                    attempt=attempt,
                    error=type(exc).__name__,
                    detail=str(exc)[:200],
                )

            if attempt >= self.max_retries:
                break
            backoff = min(self.base_backoff * (2 ** (attempt - 1)), self.max_backoff)
            await asyncio.sleep(backoff + random.uniform(0, backoff * 0.25))

        raise UpstreamError(
            f"Play Store fetch failed after {self.max_retries} attempts "
            f"(page={page}): {last_error}"
        )

    async def stream(
        self,
        *,
        app_id: str,
        since: datetime,
        until: datetime,
    ) -> AsyncIterator[Review]:
        token: Any = None
        yielded = 0
        pages = 0
        while True:
            result, token = await self._fetch_page(
                app_id=app_id, token=token, page=pages + 1
            )
            pages += 1
            if not result:
                log.info("play_store.stream.end", reason="empty_page", pages=pages)
                return

            for raw in result:
                posted = _ensure_utc(raw["at"])
                if posted > until:
                    continue
                if posted < since:
                    log.info(
                        "play_store.stream.end",
                        reason="window_exhausted",
                        pages=pages,
                        yielded=yielded,
                    )
                    return
                yield _to_review(raw, app_id=app_id, country=self.country, lang=self.lang)
                yielded += 1
                if self.max_reviews is not None and yielded >= self.max_reviews:
                    log.info(
                        "play_store.stream.end",
                        reason="max_reviews",
                        pages=pages,
                        yielded=yielded,
                    )
                    return

            if token is None or getattr(token, "token", None) is None:
                log.info(
                    "play_store.stream.end",
                    reason="no_continuation",
                    pages=pages,
                    yielded=yielded,
                )
                return
