"""LLM-free summarisation fallback.

If every Groq call in the map stage fails (rate-limited across the
entire fallback chain, model outage, etc.), the pipeline must still
hand the pulse generator *something* — leadership reading "No themes
surfaced" the day Groq has an incident is worse than reading a
degraded but real signal pulled from raw review text.

This module builds a :class:`SummaryReport` from the raw review list
using cheap, deterministic heuristics:

* keyword frequency over a stop-word-filtered token stream,
* rating-distribution-derived sentiment,
* a synthetic executive summary string that explicitly flags the
  degraded mode so consumers know the LLM didn't run.

It is intentionally simple: it is the floor of useful output, not a
replacement for the LLM pipeline.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence

from app.domain.models import Review
from app.services.summarization.schemas import (
    FinalTheme,
    SummaryReport,
    SummaryStats,
)

# Minimal English stop-list. Deliberately short — we want common
# product-review verbs/adjectives ("crash", "slow", "great") to show
# up; only noise words are filtered.
_STOPWORDS: frozenset[str] = frozenset(
    """
    a about all also am an and any are as at be been being but by can could
    did do does doing don for from get got had has have having he her here
    him his how i if in into is it its just like me more my no not of on
    one only or other our out so some than that the their them then there
    these they this those to too us was we were what when where which while
    who why will with would you your
    app phone use using used really very much get make made one two too can
    """.split()
)
_TOKEN_RE = re.compile(r"[a-z][a-z']{2,}")
# Words that strongly indicate a complaint when they show up frequently.
# Used to flag the heuristic "pain point" theme.
_NEGATIVE_HINT_WORDS: frozenset[str] = frozenset(
    {
        "crash", "crashes", "crashing", "bug", "buggy", "broken", "issue",
        "issues", "slow", "stuck", "fails", "failed", "error", "problem",
        "problems", "worst", "terrible", "useless", "delay", "delays",
        "delayed", "wait", "waiting", "loading", "freeze", "freezes",
        "login", "logout", "otp", "verification", "support", "fraud",
        "money", "withdraw", "withdrawal", "lost", "missing",
    }
)


def _tokenise(text: str) -> list[str]:
    return [
        t
        for t in _TOKEN_RE.findall(text.lower())
        if t not in _STOPWORDS and len(t) >= 4
    ]


def _ratings_sentiment(reviews: Sequence[Review]) -> tuple[str, dict[int, int]]:
    """Average rating → coarse sentiment, plus the full histogram."""
    histogram: dict[int, int] = {i: 0 for i in range(1, 6)}
    for r in reviews:
        histogram[r.rating] = histogram.get(r.rating, 0) + 1
    n = sum(histogram.values())
    if n == 0:
        return "neutral", histogram
    avg = sum(rating * count for rating, count in histogram.items()) / n
    if avg >= 4.0:
        return "positive", histogram
    if avg <= 2.5:
        return "negative", histogram
    return "mixed", histogram


def _quote_for(reviews: Sequence[Review], keyword: str) -> str | None:
    """Return the shortest review body containing ``keyword`` — short
    bodies make for the cleanest leadership-facing quotes."""
    matches = [
        r.body.strip()
        for r in reviews
        if r.body and keyword in r.body.lower()
    ]
    if not matches:
        return None
    matches.sort(key=len)
    # Keep quotes short so they fit downstream caps cleanly.
    chosen = matches[0]
    return chosen if len(chosen) <= 240 else chosen[:237] + "…"


def build_heuristic_report(
    reviews: Sequence[Review],
    *,
    max_themes: int = 3,
    failure_reasons: Sequence[str] = (),
) -> SummaryReport:
    """Construct a degraded but informative ``SummaryReport`` from raw reviews.

    Always returns a valid report. ``failure_reasons`` is folded into
    the executive summary so the reader (and future operators reading
    the doc) can see why the LLM pipeline didn't run.
    """
    if not reviews:
        return SummaryReport(
            executive_summary=(
                "Degraded mode: no reviews were available and the LLM "
                "summarisation stage did not run."
            ),
            themes=[],
            stats=SummaryStats(total_reviews=0, chunks_failed=len(failure_reasons)),
        )

    overall_sentiment, histogram = _ratings_sentiment(reviews)
    counter: Counter[str] = Counter()
    for r in reviews:
        text_parts = [r.body or ""]
        if r.title:
            text_parts.append(r.title)
        counter.update(_tokenise(" ".join(text_parts)))

    top_keywords = [kw for kw, _ in counter.most_common(max_themes * 3)][
        : max_themes * 3
    ]

    themes: list[FinalTheme] = []
    seen_labels: set[str] = set()
    for kw in top_keywords:
        if len(themes) >= max_themes:
            break
        label = kw.title()
        if label.lower() in seen_labels:
            continue
        seen_labels.add(label.lower())
        count = counter[kw]
        is_negative = kw in _NEGATIVE_HINT_WORDS
        theme_sentiment = "negative" if is_negative else overall_sentiment
        prevalence = "high" if count >= max(5, len(reviews) // 5) else "medium"
        quote = _quote_for(reviews, kw)
        themes.append(
            FinalTheme(
                label=label,
                description=(
                    f"Mentioned in {count} review(s) within the window. "
                    f"Heuristic theme — generated without LLM "
                    f"summarisation; treat as directional."
                ),
                sentiment=theme_sentiment,  # type: ignore[arg-type]
                prevalence=prevalence,  # type: ignore[arg-type]
                supporting_quotes=[quote] if quote else [],
                action_hint=None,
            )
        )

    n = len(reviews)
    avg = sum(rating * count for rating, count in histogram.items()) / max(1, n)
    reasons_blurb = ""
    if failure_reasons:
        # Show only a handful so the doc body stays readable.
        sample = "; ".join(sorted({r for r in failure_reasons})[:3])
        reasons_blurb = f" LLM failure reasons: {sample}."

    exec_summary = (
        f"Degraded heuristic summary: {n} review(s) analysed without LLM "
        f"summarisation. Average rating {avg:.2f}/5; rating distribution "
        f"{histogram}. Themes below are derived from keyword frequency and "
        f"should be treated as directional, not authoritative."
        f"{reasons_blurb}"
    )

    return SummaryReport(
        executive_summary=exec_summary,
        themes=themes,
        action_ideas=[],
        stats=SummaryStats(
            total_reviews=n,
            chunks_processed=0,
            chunks_failed=len(failure_reasons),
            themes_premerged=len(top_keywords),
            themes_final=len(themes),
        ),
    )


__all__ = ["build_heuristic_report"]
