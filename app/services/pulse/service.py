"""Top-level entrypoint for weekly pulse generation.

``WeeklyPulseGenerator.generate`` is the one-call surface a caller needs:
hand it one or more :class:`SummaryReport`s and an optional reporting
window, get back a :class:`WeeklyPulse` (structured JSON) plus a rendered
markdown view.

The generator composes :class:`PulseAggregator`, :class:`PulseRanker`,
and :class:`PulseFormatter`. Each piece is independently testable and
swappable; the service layer just orchestrates them and enforces the
output shape (250-word cap, top-3 limits, headline derivation).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

from app.core.logging import get_logger
from app.services.pulse.aggregation import (
    AggregatedPulseInput,
    PulseAggregator,
    ThemeBucket,
)
from app.services.pulse.formatting import (
    PulseFormatter,
    clip_to_word_budget,
    count_words,
)
from app.services.pulse.ranking import PulseRanker, RankedPulse
from app.services.pulse.schemas import (
    PulseTone,
    WeeklyPulse,
    WeeklyPulseMeta,
)
from app.services.pulse.text_utils import clamp_text
from app.services.summarization.schemas import SummaryReport

log = get_logger("service.pulse")

DEFAULT_WORD_BUDGET = 250
# Reserve ~80 words across the headline + themes + quotes + actions
# scaffolding so the executive paragraph itself doesn't blow the budget.
_EXEC_SUMMARY_WORD_BUDGET = 170
# Mirrors WeeklyPulse.headline schema cap; see ranking.py for rationale.
_PULSE_HEADLINE_MAX = 160


@dataclass(frozen=True)
class WeeklyPulseArtifacts:
    """Bundled output: structured pulse + rendered markdown."""

    pulse: WeeklyPulse
    markdown: str

    def to_json(self) -> dict:
        """Convenience: JSON-serialisable dict view of the structured pulse."""
        return self.pulse.model_dump(mode="json")


class WeeklyPulseGenerator:
    """Compose aggregation + ranking + formatting into a single call."""

    def __init__(
        self,
        *,
        aggregator: PulseAggregator | None = None,
        ranker: PulseRanker | None = None,
        formatter: PulseFormatter | None = None,
        word_budget: int = DEFAULT_WORD_BUDGET,
    ) -> None:
        if word_budget <= 0:
            raise ValueError("word_budget must be positive")
        self._aggregator = aggregator or PulseAggregator()
        self._ranker = ranker or PulseRanker()
        self._formatter = formatter or PulseFormatter()
        self._word_budget = word_budget

    def generate(
        self,
        reports: Iterable[SummaryReport],
        *,
        window_start: datetime | None = None,
        window_end: datetime | None = None,
        title: str | None = None,
        generated_at: datetime | None = None,
    ) -> WeeklyPulseArtifacts:
        aggregated = self._aggregator.aggregate(reports)
        ranked = self._ranker.rank(aggregated)

        executive_summary = self._build_executive_summary(aggregated, ranked)
        headline = title or self._build_headline(window_start, window_end)
        tone = self._overall_tone(ranked.ranked_buckets)

        pulse = WeeklyPulse(
            headline=clamp_text(
                headline,
                max_length=_PULSE_HEADLINE_MAX,
                field="WeeklyPulse.headline",
                ellipsis=False,
            ),
            executive_summary=executive_summary,
            overall_tone=tone,
            themes=ranked.themes,
            quotes=ranked.quotes,
            actions=ranked.actions,
            meta=WeeklyPulseMeta(
                generated_at=generated_at or datetime.now(tz=UTC),
                window_start=window_start,
                window_end=window_end,
                reports_count=aggregated.reports_count,
                total_reviews=aggregated.total_reviews,
                word_count=count_words(executive_summary),
            ),
        )

        markdown = self._formatter.render_markdown(pulse)
        log.info(
            "pulse.generated",
            themes=len(pulse.themes),
            quotes=len(pulse.quotes),
            actions=len(pulse.actions),
            words=pulse.meta.word_count,
            reports=aggregated.reports_count,
        )
        return WeeklyPulseArtifacts(pulse=pulse, markdown=markdown)

    def _build_executive_summary(
        self,
        aggregated: AggregatedPulseInput,
        ranked: RankedPulse,
    ) -> str:
        """Prefer an upstream executive summary; otherwise synthesise one.

        Either path is clipped to the per-summary word budget so the
        whole pulse stays under the 250-word target with room for the
        themes/quotes/actions scaffolding.
        """
        if aggregated.executive_summaries:
            # Use the longest upstream summary as the spine; it tends to be
            # the most informative. Anything trimmed off is fine — themes
            # and quotes carry the specifics.
            spine = max(aggregated.executive_summaries, key=len)
            return clip_to_word_budget(spine, budget=_EXEC_SUMMARY_WORD_BUDGET)

        if not ranked.themes:
            return "No themes surfaced from the reporting window."

        labels = ", ".join(t.label for t in ranked.themes)
        synthesised = (
            f"Across {aggregated.reports_count} report(s) covering "
            f"{aggregated.total_reviews:,} reviews, the dominant themes were "
            f"{labels}. See below for representative quotes and recommended "
            "next steps."
        )
        return clip_to_word_budget(synthesised, budget=_EXEC_SUMMARY_WORD_BUDGET)

    def _build_headline(
        self,
        window_start: datetime | None,
        window_end: datetime | None,
    ) -> str:
        if window_end:
            return f"Weekly Pulse — {window_end.date().isoformat()}"
        if window_start:
            return f"Weekly Pulse — week of {window_start.date().isoformat()}"
        return "Weekly Pulse"

    def _overall_tone(self, buckets: list[ThemeBucket]) -> PulseTone:
        """Derive overall tone from the top themes' sentiment mix.

        Plurality wins; ties resolve to ``mixed`` so leadership doesn't
        see false confidence about a fundamentally split signal.
        """
        if not buckets:
            return "neutral"
        top = buckets[: self._ranker._max_themes]  # noqa: SLF001 — same package
        counts: dict[str, int] = {}
        for b in top:
            counts[b.sentiment] = counts.get(b.sentiment, 0) + 1
        ordered = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        if len(ordered) >= 2 and ordered[0][1] == ordered[1][1]:
            return "mixed"
        return ordered[0][0]  # type: ignore[return-value]


__all__ = ["WeeklyPulseArtifacts", "WeeklyPulseGenerator"]
