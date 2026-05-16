"""Theme aggregation: cheap pre-merge + LLM reduce.

The aggregation step turns N chunk summaries into <= ``max_themes`` final
themes. It runs in two passes:

  1. **Pre-merge**: themes whose labels normalise to the same key are
     fused locally — ``evidence_count`` summed, ``sample_quotes`` unioned
     (deduplicated, capped). This is pure Python, has no API cost, and
     materially shrinks the payload sent to the reduce LLM call. Worst
     case it's a no-op; best case (lots of chunks repeating the same
     issue) it cuts the reduce input by 5-10x.

  2. **LLM reduce**: one Groq call ranks, merges semantically-similar
     themes (e.g. "Crash on Launch" vs "App Won't Open"), and produces
     the final ranked output. This is where the real consolidation
     happens — the pre-merge only catches exact-label duplicates.
"""

from __future__ import annotations

import json
import re
from collections import OrderedDict

from pydantic import ValidationError

from app.core.logging import get_logger
from app.services.summarization.groq_client import GroqClient
from app.services.summarization.prompts import (
    REDUCE_SYSTEM_PROMPT_TEMPLATE,
    REDUCE_USER_PROMPT_TEMPLATE,
)
from app.services.summarization.sanitization import (
    sanitize_action_ideas,
    sanitize_executive_summary,
    sanitize_llm_output,
)
from app.services.summarization.schemas import (
    ChunkSummary,
    ChunkTheme,
    FinalTheme,
    SummaryReport,
    SummaryStats,
)

log = get_logger("summarization.aggregator")

_LABEL_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")
_REDUCE_MAX_TOKENS = 1400


def _normalize_label(label: str) -> str:
    """Lowercase + strip non-alnum so 'Login failures!' == 'login_failures'."""
    return _LABEL_NORMALIZE_RE.sub(" ", label.lower()).strip()


def premerge_themes(chunks: list[ChunkSummary]) -> list[ChunkTheme]:
    """Merge themes across chunks by normalised label.

    Quotes are unioned (preserving order, dropping exact duplicates) and
    capped at 3 to keep the reduce-call payload bounded. Description is
    kept from the first occurrence — the LLM rewrites it in the reduce
    step anyway, so concatenating descriptions here would only waste
    tokens.
    """
    merged: OrderedDict[str, ChunkTheme] = OrderedDict()

    for chunk in chunks:
        for theme in chunk.themes:
            key = _normalize_label(theme.label)
            if not key:
                continue
            existing = merged.get(key)
            if existing is None:
                merged[key] = ChunkTheme(
                    label=theme.label,
                    description=theme.description,
                    sentiment=theme.sentiment,
                    evidence_count=theme.evidence_count,
                    sample_quotes=list(theme.sample_quotes)[:3],
                )
                continue

            quotes = list(existing.sample_quotes)
            for quote in theme.sample_quotes:
                if quote not in quotes:
                    quotes.append(quote)
                if len(quotes) >= 3:
                    break

            merged[key] = existing.model_copy(
                update={
                    "evidence_count": existing.evidence_count + theme.evidence_count,
                    "sample_quotes": quotes,
                    "sentiment": (
                        existing.sentiment
                        if existing.sentiment == theme.sentiment
                        else "mixed"
                    ),
                }
            )

    return list(merged.values())


class ThemeAggregator:
    """Drives the reduce step against a Groq client."""

    def __init__(self, *, max_themes: int = 5) -> None:
        if max_themes <= 0:
            raise ValueError("max_themes must be positive")
        self._max_themes = max_themes

    async def aggregate(
        self,
        chunks: list[ChunkSummary],
        client: GroqClient,
        *,
        total_reviews: int,
        chunks_failed: int = 0,
    ) -> SummaryReport:
        """Pre-merge then LLM-reduce ``chunks`` into a final report."""
        premerged = premerge_themes(chunks)

        stats = SummaryStats(
            total_reviews=total_reviews,
            chunks_processed=len(chunks),
            chunks_failed=chunks_failed,
            themes_premerged=len(premerged),
        )

        if not premerged:
            log.info("aggregator.no_themes")
            return SummaryReport(executive_summary="", themes=[], stats=stats)

        themes_payload = [t.model_dump() for t in premerged]
        user_prompt = REDUCE_USER_PROMPT_TEMPLATE.format(
            themes_json=json.dumps(themes_payload, ensure_ascii=False, indent=2)
        )
        system_prompt = REDUCE_SYSTEM_PROMPT_TEMPLATE.replace(
            "__MAX_THEMES__", str(self._max_themes)
        )

        # Guard the LLM call itself — if Groq fails outright we still
        # want a partial report (pre-merged themes as the spine) rather
        # than dragging the pipeline down with us.
        try:
            raw = await client.complete_json(
                system=system_prompt,
                user=user_prompt,
                max_tokens=_REDUCE_MAX_TOKENS,
                stage="reduce",
            )
        except Exception as exc:
            log.warning(
                "aggregator.reduce_call_failed",
                error=type(exc).__name__,
                detail=str(exc)[:200],
                premerged=len(premerged),
            )
            return _partial_report_from_premerged(premerged, stats, self._max_themes)

        # Sanitise before Pydantic. ``sanitize_llm_output`` clips
        # oversized arrays (the bug that started this fix — 6 quotes
        # vs. the 3-cap), coerces wrong types, strips nulls, and drops
        # malformed entries entirely. Pydantic stays the safety net.
        raw_themes = raw.get("themes") if isinstance(raw, dict) else None
        before_count = len(raw_themes) if isinstance(raw_themes, list) else 0
        sanitised = sanitize_llm_output(raw, kind="final")

        final_themes: list[FinalTheme] = []
        skipped = 0
        for entry in sanitised[: self._max_themes]:
            try:
                final_themes.append(FinalTheme.model_validate(entry))
            except ValidationError as exc:
                skipped += 1
                log.warning(
                    "aggregator.theme_skipped",
                    label=entry.get("label"),
                    error=str(exc)[:200],
                )

        action_ideas = sanitize_action_ideas(raw.get("action_ideas"))
        executive_summary = sanitize_executive_summary(raw.get("executive_summary"))

        stats = stats.model_copy(update={"themes_final": len(final_themes)})

        # If validation removed *every* candidate, fall back to the
        # pre-merged themes instead of returning an empty list — the
        # whole point of this hardening is that we never ship a
        # themeless report when we have signal in hand.
        if not final_themes and sanitised:
            log.warning(
                "aggregator.all_themes_invalid.fallback_premerged",
                themes_before_validation=before_count,
                themes_after_sanitization=len(sanitised),
                skipped_invalid_themes=skipped,
            )
            return _partial_report_from_premerged(
                premerged,
                stats,
                self._max_themes,
                executive_summary=executive_summary,
                action_ideas=action_ideas,
            )

        report = SummaryReport(
            executive_summary=executive_summary,
            themes=final_themes,
            action_ideas=action_ideas,
            stats=stats,
        )
        log.info(
            "aggregator.done",
            premerged=len(premerged),
            themes_before_validation=before_count,
            themes_after_sanitization=len(sanitised),
            skipped_invalid_themes=skipped,
            final=len(final_themes),
        )
        return report


def _partial_report_from_premerged(
    premerged: list[ChunkTheme],
    stats: SummaryStats,
    max_themes: int,
    *,
    executive_summary: str = "",
    action_ideas: list[str] | None = None,
) -> SummaryReport:
    """Last-resort report built from pre-merged chunk themes.

    Used when either the reduce LLM call fails outright or every theme
    in its response is malformed. Pre-merged themes are pure-Python
    artefacts, so this path can never raise — guaranteeing the caller
    always gets back something rather than an exception.
    """
    fallback_themes: list[FinalTheme] = []
    skipped = 0
    for ct in premerged[:max_themes]:
        try:
            fallback_themes.append(
                FinalTheme(
                    label=ct.label,
                    description=ct.description,
                    sentiment=ct.sentiment,
                    prevalence="medium",
                    supporting_quotes=list(ct.sample_quotes)[:3],
                    action_hint=None,
                )
            )
        except ValidationError as exc:
            # Should be impossible (ChunkTheme passed its own validation
            # to get here), but keep the safety net anyway so this path
            # can never raise.
            skipped += 1
            log.warning(
                "aggregator.premerged_fallback_invalid",
                label=ct.label,
                error=str(exc)[:200],
            )
    stats = stats.model_copy(update={"themes_final": len(fallback_themes)})
    log.warning(
        "aggregator.partial_report",
        themes=len(fallback_themes),
        skipped=skipped,
    )
    return SummaryReport(
        executive_summary=executive_summary,
        themes=fallback_themes,
        action_ideas=list(action_ideas or []),
        stats=stats,
    )
