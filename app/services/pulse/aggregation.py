"""Aggregation of multiple :class:`SummaryReport`s into theme buckets.

A weekly pulse may combine summaries from several sources (iOS + Android,
multiple app IDs, last week + carryover) — each is already an *aggregated
summary* in its own right. This module fuses them by normalised label,
the same trick the summarisation aggregator uses for its pre-merge step.

Aggregation here is intentionally cheap and deterministic: no LLM calls,
no fuzzy matching beyond a casefold + alnum normalisation. The
:mod:`ranking` module decides which buckets make the final cut.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass, field

from app.services.summarization.schemas import (
    FinalTheme,
    Prevalence,
    Sentiment,
    SummaryReport,
)

_LABEL_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")

_PREVALENCE_RANK: dict[Prevalence, int] = {"low": 1, "medium": 2, "high": 3}
_RANK_PREVALENCE: dict[int, Prevalence] = {1: "low", 2: "medium", 3: "high"}


def _normalize_label(label: str) -> str:
    """``'Login failures!' -> 'login failures'`` — collapses to a join key."""
    return _LABEL_NORMALIZE_RE.sub(" ", label.lower()).strip()


@dataclass
class ThemeBucket:
    """One merged theme across all input reports.

    A bucket records every observation of the underlying theme so the
    ranker can score by both prevalence (qualitative LLM judgement) and
    evidence_count (quantitative chunk-level signal).
    """

    label: str
    description: str
    sentiment: Sentiment = "neutral"
    prevalence: Prevalence = "medium"
    quotes: list[str] = field(default_factory=list)
    action_hints: list[str] = field(default_factory=list)
    evidence_count: int = 0
    occurrences: int = 0

    def absorb(self, theme: FinalTheme) -> None:
        """Fold another :class:`FinalTheme` into this bucket."""
        self.occurrences += 1
        self.evidence_count += max(1, len(theme.supporting_quotes))
        if _PREVALENCE_RANK[theme.prevalence] > _PREVALENCE_RANK[self.prevalence]:
            self.prevalence = theme.prevalence
        if self.sentiment != theme.sentiment:
            self.sentiment = "mixed"
        for q in theme.supporting_quotes:
            cleaned = q.strip()
            if cleaned and cleaned not in self.quotes:
                self.quotes.append(cleaned)
        if theme.action_hint:
            hint = theme.action_hint.strip()
            if hint and hint not in self.action_hints:
                self.action_hints.append(hint)


@dataclass
class AggregatedPulseInput:
    """Output of :class:`PulseAggregator` — the ranker's input."""

    buckets: list[ThemeBucket]
    executive_summaries: list[str]
    global_actions: list[str]
    total_reviews: int
    reports_count: int


class PulseAggregator:
    """Merge a sequence of :class:`SummaryReport`s into theme buckets.

    Stateless: instances are cheap to construct and safe to reuse.
    """

    def aggregate(
        self, reports: Iterable[SummaryReport]
    ) -> AggregatedPulseInput:
        buckets: OrderedDict[str, ThemeBucket] = OrderedDict()
        execs: list[str] = []
        actions: list[str] = []
        total_reviews = 0
        reports_count = 0

        for report in reports:
            reports_count += 1
            total_reviews += report.stats.total_reviews
            if report.executive_summary.strip():
                execs.append(report.executive_summary.strip())
            for action in report.action_ideas:
                cleaned = action.strip()
                if cleaned and cleaned not in actions:
                    actions.append(cleaned)
            for theme in report.themes:
                key = _normalize_label(theme.label)
                if not key:
                    continue
                bucket = buckets.get(key)
                if bucket is None:
                    bucket = ThemeBucket(
                        label=theme.label,
                        description=theme.description,
                        sentiment=theme.sentiment,
                        prevalence=theme.prevalence,
                        quotes=[
                            q.strip() for q in theme.supporting_quotes if q.strip()
                        ],
                        action_hints=(
                            [theme.action_hint.strip()]
                            if theme.action_hint and theme.action_hint.strip()
                            else []
                        ),
                        evidence_count=max(1, len(theme.supporting_quotes)),
                        occurrences=1,
                    )
                    buckets[key] = bucket
                else:
                    bucket.absorb(theme)

        return AggregatedPulseInput(
            buckets=list(buckets.values()),
            executive_summaries=execs,
            global_actions=actions,
            total_reviews=total_reviews,
            reports_count=reports_count,
        )


__all__ = [
    "AggregatedPulseInput",
    "PulseAggregator",
    "ThemeBucket",
]
