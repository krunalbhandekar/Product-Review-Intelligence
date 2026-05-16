"""Example: stream the last 8 weeks of App Store reviews for one app.

Run with::

    python -m examples.app_store_demo 310633997

(310633997 is WhatsApp Messenger on the US App Store.)
"""

from __future__ import annotations

import argparse
import asyncio

from app.core.logging import configure_logging, get_logger
from app.services.ingest.app_store import AppStoreSource
from app.services.ingest.time_window import lookback_window


async def run(app_id: str, country: str, weeks: int, limit: int) -> None:
    log = get_logger("examples.app_store")
    since, until = lookback_window(weeks)

    async with AppStoreSource(country=country, max_reviews=limit) as source:
        count = 0
        async for review in source.stream(app_id=app_id, since=since, until=until):
            count += 1
            log.info(
                "review",
                review_id=review.review_id,
                rating=review.rating,
                posted_at=review.posted_at.isoformat(),
                app_version=review.app_version,
                title=review.title,
                body_preview=review.body[:80],
            )
        log.info("done", total=count, since=since.isoformat(), until=until.isoformat())


def main() -> None:
    parser = argparse.ArgumentParser(description="App Store review importer demo")
    parser.add_argument("app_id", help="Numeric App Store app id, e.g. 310633997")
    parser.add_argument("--country", default="us", help="Two-letter country code (default: us)")
    parser.add_argument("--weeks", type=int, default=8, help="Lookback window in weeks (8-12)")
    parser.add_argument("--limit", type=int, default=50, help="Maximum reviews to fetch")
    args = parser.parse_args()

    configure_logging()
    asyncio.run(run(args.app_id, args.country, args.weeks, args.limit))


if __name__ == "__main__":
    main()
