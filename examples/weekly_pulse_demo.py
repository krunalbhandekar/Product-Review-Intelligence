"""Example: produce a weekly pulse from two aggregated summaries.

Run with::

    python -m examples.weekly_pulse_demo

This demo *does not* call Groq. It feeds two hand-built
:class:`SummaryReport` objects (one for iOS, one for Android) into
:class:`WeeklyPulseGenerator` and prints both the structured JSON and
the rendered markdown view.

The intent is to make the pulse contract eyeballable in isolation
from the LLM pipeline — useful for prompt iteration, formatting
tweaks, and stakeholder previews.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from app.services.pulse import WeeklyPulseGenerator
from app.services.summarization.schemas import (
    FinalTheme,
    SummaryReport,
    SummaryStats,
)


def _ios_report() -> SummaryReport:
    return SummaryReport(
        executive_summary=(
            "iOS users this week are dominated by sign-in pain after the "
            "v4.2 release. Crashes on launch are also up sharply on "
            "older devices. Positive notes around the redesigned dark "
            "mode remain, but are drowned out by reliability complaints."
        ),
        themes=[
            FinalTheme(
                label="Login Failures",
                description=(
                    "Widespread sign-in failures since v4.2 — users report "
                    "valid credentials being rejected and no recovery path."
                ),
                sentiment="negative",
                prevalence="high",
                supporting_quotes=[
                    "Can't log in — says invalid password but it's correct.",
                    "Locked out of my account for three days now.",
                ],
                action_hint="Audit auth service error rates since v4.2 rollout.",
            ),
            FinalTheme(
                label="Crash on Launch",
                description="App force-closes immediately on iPhone 11 and older.",
                sentiment="negative",
                prevalence="high",
                supporting_quotes=[
                    "App won't open at all on my iPhone 11.",
                ],
                action_hint="Roll back the v4.2 launch path on iOS <15.",
            ),
            FinalTheme(
                label="Dark Mode",
                description="Redesigned dark mode is well received.",
                sentiment="positive",
                prevalence="medium",
                supporting_quotes=["Love the new dark theme, looks gorgeous."],
                action_hint=None,
            ),
        ],
        action_ideas=[
            "Hot-fix the v4.2 sign-in regression.",
            "Add a status-page banner for impacted users.",
        ],
        stats=SummaryStats(
            total_reviews=812,
            chunks_processed=18,
            chunks_failed=0,
            themes_premerged=9,
            themes_final=3,
        ),
    )


def _android_report() -> SummaryReport:
    return SummaryReport(
        executive_summary=(
            "Android complaints concentrate on battery drain and delayed "
            "notifications. Sign-in issues echo the iOS pattern, suggesting "
            "a backend cause rather than a platform-specific bug."
        ),
        themes=[
            FinalTheme(
                label="Battery Drain",
                description="Background battery usage spikes since v4.2.",
                sentiment="negative",
                prevalence="high",
                supporting_quotes=[
                    "Battery drains in 3 hours since the update.",
                    "Phone is hot all day with the app installed.",
                ],
                action_hint="Profile background sync introduced in v4.2.",
            ),
            FinalTheme(
                label="Login Failures",
                description="Same sign-in regression as iOS, slightly lower volume.",
                sentiment="negative",
                prevalence="medium",
                supporting_quotes=["Login keeps failing on my Pixel 7."],
                action_hint="Audit auth service error rates since v4.2 rollout.",
            ),
            FinalTheme(
                label="Delayed Notifications",
                description="Push notifications arrive hours late or not at all.",
                sentiment="negative",
                prevalence="medium",
                supporting_quotes=[
                    "Notifications show up four hours after the event.",
                ],
                action_hint="Investigate FCM token refresh path.",
            ),
        ],
        action_ideas=[
            "Add background-work telemetry to catch battery regressions earlier.",
        ],
        stats=SummaryStats(
            total_reviews=1_104,
            chunks_processed=24,
            chunks_failed=1,
            themes_premerged=11,
            themes_final=3,
        ),
    )


def main() -> None:
    window_end = datetime(2026, 5, 15, tzinfo=UTC)
    window_start = window_end - timedelta(days=7)

    artifacts = WeeklyPulseGenerator().generate(
        [_ios_report(), _android_report()],
        window_start=window_start,
        window_end=window_end,
        generated_at=window_end,
    )

    print("=== Structured JSON ===")
    print(json.dumps(artifacts.to_json(), indent=2, default=str))
    print()
    print("=== Markdown ===")
    print(artifacts.markdown)
    print(f"(word count: {artifacts.pulse.meta.word_count})")


if __name__ == "__main__":
    main()
