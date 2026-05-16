"""Example: drive the chunk-based summarisation pipeline against Groq.

Run with::

    GROQ_API_KEY=... python -m examples.summarization_demo

The script feeds a handful of hand-written reviews through the preprocessing
service and then ``SummarizationService``. It exists so you can:

  * confirm a fresh ``GROQ_API_KEY`` works end-to-end,
  * eyeball whether the prompts produce sensible themes on a known input,
  * verify the JSON schema round-trips cleanly into ``SummaryReport``.

If ``GROQ_API_KEY`` is empty the script exits without making any network
calls, so it's safe to run in CI as a smoke test.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime

from app.config import get_settings
from app.domain.enums import Platform
from app.domain.models import Review
from app.services.preprocessing import PreprocessingService
from app.services.summarization import SummarizationService


def _sample_reviews() -> list[Review]:
    base = {
        "source": Platform.ANDROID,
        "app_id": "com.example.app",
        "rating": 2,
        "posted_at": datetime(2026, 5, 10, tzinfo=UTC),
        "lang": "en",
        "country": "us",
    }
    bodies = [
        ("r1", "App crashes every time I open it after the latest update.", 1),
        ("r2", "Login keeps failing — says invalid password but it's correct.", 1),
        ("r3", "Cannot sign in at all. Frustrating!", 1),
        ("r4", "Crashes on launch on my Pixel 7. Force-close every time.", 1),
        ("r5", "Love the new dark mode, looks great!", 5),
        ("r6", "UI is beautiful but the app is very slow to load.", 3),
        ("r7", "Notifications never arrive on time. Often hours late.", 2),
        ("r8", "Push notifications are delayed and sometimes missing.", 2),
        ("r9", "Battery drain is insane since the v4 update.", 1),
        ("r10", "Drains my battery in a few hours when running in background.", 1),
        ("r11", "Great app overall, but please fix the crashes.", 3),
        ("r12", "Dark theme is gorgeous and customisable. Five stars.", 5),
    ]
    return [
        Review(review_id=rid, title=None, body=body, author=None,
               **{**base, "rating": rating})
        for rid, body, rating in bodies
    ]


async def main() -> None:
    settings = get_settings()
    if not settings.groq_api_key:
        print("GROQ_API_KEY is empty — skipping live demo. "
              "Set it in .env to run end-to-end.")
        return

    raw = _sample_reviews()
    cleaned = list(PreprocessingService(dedupe="both").preprocess(raw))
    print(f"Preprocessed: {len(cleaned)} reviews (from {len(raw)} raw)")

    service = SummarizationService(settings=settings)
    report = await service.summarize(cleaned)

    print("\n=== Executive Summary ===")
    print(report.executive_summary or "(empty)")
    print("\n=== Final Themes ===")
    for i, theme in enumerate(report.themes, 1):
        print(f"\n{i}. {theme.label}  [{theme.sentiment} · {theme.prevalence}]")
        print(f"   {theme.description}")
        if theme.supporting_quotes:
            print("   Quotes:")
            for q in theme.supporting_quotes:
                print(f"     - {q}")
        if theme.action_hint:
            print(f"   Action: {theme.action_hint}")

    print("\n=== Stats ===")
    print(json.dumps(report.stats.model_dump(), indent=2))


if __name__ == "__main__":
    # SAFETY: ensure we never hit Groq from a CI runner that forgot to set the key.
    if not os.getenv("GROQ_API_KEY"):
        print("Hint: export GROQ_API_KEY before running this demo.")
    asyncio.run(main())
