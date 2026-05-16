"""Ranking of themes, quotes, and action ideas for the weekly pulse.

The summarisation aggregator already produces a *ranked* theme list per
report; this module re-ranks across reports and applies pulse-specific
tie-breakers so leadership sees the three signals that matter most:

* themes ranked by prevalence + evidence + cross-report agreement,
* one quote per top theme (so the quote set spans the headline issues),
* actions skewed toward the top themes, padded with global ideas.

All scoring is pure Python, deterministic, and side-effect free.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.services.pulse.aggregation import AggregatedPulseInput, ThemeBucket
from app.services.pulse.schemas import PulseAction, PulseQuote, PulseTheme
from app.services.pulse.text_utils import clamp_text

# Caps mirror the schema field constraints in ``pulse/schemas.py``. They
# are duplicated here so the normalisation layer can shape input before
# pydantic validates it; if a schema cap changes, update both. The
# schema remains the authoritative contract — clamp_text is the safe
# normal path, pydantic is the safety net.
_THEME_LABEL_MAX = 80
_THEME_HEADLINE_MAX = 240
_QUOTE_TEXT_MAX = 300
_QUOTE_LABEL_MAX = 80
_ACTION_TEXT_MAX = 240
_ACTION_LABEL_MAX = 80

# Prevalence carries the most weight: it's the LLM's qualitative judgement
# over the whole input. Evidence count breaks ties; cross-report occurrences
# break further ties (a theme that shows up in iOS *and* Android matters
# more than one that only shows up in one platform).
_PREVALENCE_WEIGHT: dict[str, int] = {"low": 1, "medium": 3, "high": 6}
_NEGATIVE_BONUS = 1  # pain points get a small bump — they need leadership air-time


def _score(bucket: ThemeBucket) -> tuple[int, int, int, int]:
    """Sort key: higher tuples rank first.

    Returns ``(prevalence_score, evidence, occurrences, neg_bonus)`` —
    each component is monotonically meaningful so ties cascade cleanly.
    """
    prev = _PREVALENCE_WEIGHT.get(bucket.prevalence, 2)
    neg = _NEGATIVE_BONUS if bucket.sentiment == "negative" else 0
    return (prev + neg, bucket.evidence_count, bucket.occurrences, neg)


@dataclass
class RankedPulse:
    """Output of :class:`PulseRanker`."""

    themes: list[PulseTheme]
    quotes: list[PulseQuote]
    actions: list[PulseAction]
    ranked_buckets: list[ThemeBucket]


class PulseRanker:
    """Rank theme buckets and derive quote/action shortlists."""

    def __init__(
        self,
        *,
        max_themes: int = 3,
        max_quotes: int = 3,
        max_actions: int = 3,
    ) -> None:
        if max_themes <= 0 or max_quotes <= 0 or max_actions <= 0:
            raise ValueError("max_* knobs must be positive")
        self._max_themes = max_themes
        self._max_quotes = max_quotes
        self._max_actions = max_actions

    def rank(self, aggregated: AggregatedPulseInput) -> RankedPulse:
        ranked = sorted(aggregated.buckets, key=_score, reverse=True)
        top = ranked[: self._max_themes]

        themes = [
            PulseTheme(
                rank=i + 1,
                label=clamp_text(
                    bucket.label,
                    max_length=_THEME_LABEL_MAX,
                    field="PulseTheme.label",
                    ellipsis=False,
                ),
                headline=clamp_text(
                    bucket.description,
                    max_length=_THEME_HEADLINE_MAX,
                    field="PulseTheme.headline",
                ),
                sentiment=bucket.sentiment,
                prevalence=bucket.prevalence,
                evidence_count=bucket.evidence_count,
            )
            for i, bucket in enumerate(top)
        ]

        quotes = self._pick_quotes(top)
        actions = self._pick_actions(top, aggregated.global_actions)

        return RankedPulse(
            themes=themes,
            quotes=quotes,
            actions=actions,
            ranked_buckets=ranked,
        )

    def _pick_quotes(self, top: list[ThemeBucket]) -> list[PulseQuote]:
        """One quote per top theme, then fall back to remaining quotes.

        Picking one-per-theme keeps the quote set diverse: leadership
        sees three distinct user voices, not three variations of the
        same complaint.
        """
        chosen: list[PulseQuote] = []
        seen: set[str] = set()

        def _quote(text: str, label: str) -> PulseQuote:
            return PulseQuote(
                text=clamp_text(
                    text,
                    max_length=_QUOTE_TEXT_MAX,
                    field="PulseQuote.text",
                ),
                theme_label=clamp_text(
                    label,
                    max_length=_QUOTE_LABEL_MAX,
                    field="PulseQuote.theme_label",
                    ellipsis=False,
                ),
            )

        for bucket in top:
            for quote in bucket.quotes:
                if quote not in seen:
                    chosen.append(_quote(quote, bucket.label))
                    seen.add(quote)
                    break
            if len(chosen) >= self._max_quotes:
                return chosen[: self._max_quotes]

        # Top-up: fill remaining slots from any remaining quote in rank order.
        for bucket in top:
            for quote in bucket.quotes:
                if len(chosen) >= self._max_quotes:
                    return chosen
                if quote not in seen:
                    chosen.append(_quote(quote, bucket.label))
                    seen.add(quote)

        return chosen[: self._max_quotes]

    def _pick_actions(
        self, top: list[ThemeBucket], global_actions: list[str]
    ) -> list[PulseAction]:
        """Prefer theme-tied actions over global ones — they're more concrete."""
        chosen: list[PulseAction] = []
        seen: set[str] = set()

        def _action(text: str, label: str | None) -> PulseAction:
            return PulseAction(
                text=clamp_text(
                    text,
                    max_length=_ACTION_TEXT_MAX,
                    field="PulseAction.text",
                ),
                theme_label=(
                    clamp_text(
                        label,
                        max_length=_ACTION_LABEL_MAX,
                        field="PulseAction.theme_label",
                        ellipsis=False,
                    )
                    if label is not None
                    else None
                ),
            )

        for bucket in top:
            for hint in bucket.action_hints:
                if hint not in seen:
                    chosen.append(_action(hint, bucket.label))
                    seen.add(hint)
                    break
            if len(chosen) >= self._max_actions:
                return chosen[: self._max_actions]

        for action in global_actions:
            if len(chosen) >= self._max_actions:
                break
            if action not in seen:
                chosen.append(_action(action, None))
                seen.add(action)

        return chosen[: self._max_actions]


__all__ = ["PulseRanker", "RankedPulse"]
