"""End-to-end chunk-based summarisation service.

Glues together the chunker, the per-chunk map step, and the reduce step.

Concurrency model
-----------------
Chunks are summarised in parallel under a bounded ``asyncio.Semaphore``
so we never exceed ``SUMMARIZATION_CONCURRENCY`` in-flight Groq calls. A single
``GroqClient`` (and therefore a single ``httpx.AsyncClient``) is shared
across all chunks, so connection pooling is preserved end-to-end.

Failure model
-------------
A single chunk failure does NOT poison the whole run. The map step
catches per-chunk exceptions, logs them, and continues with the chunks
that did succeed. The reduce step still runs as long as at least one
chunk produced themes. The number of failed chunks is reported via
``SummaryReport.stats.chunks_failed`` so callers can decide whether to
trust the output.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterable, Sequence
from typing import Any

from pydantic import ValidationError

from app.config import Settings, get_settings
from app.core.logging import get_logger
from app.domain.models import Review
from app.services.summarization.aggregator import ThemeAggregator
from app.services.summarization.chunker import (
    ReviewChunk,
    ReviewChunker,
    split_if_oversized,
)
from app.services.summarization.groq_client import GroqClient, GroqConfig
from app.services.summarization.heuristic import build_heuristic_report
from app.services.summarization.prompts import (
    CHUNK_SYSTEM_PROMPT,
    CHUNK_USER_PROMPT_TEMPLATE,
)
from app.services.summarization.sanitization import sanitize_llm_output
from app.services.summarization.schemas import (
    ChunkSummary,
    ChunkTheme,
    SummaryReport,
    SummaryStats,
)

log = get_logger("summarization.service")

# Output-size caps for the two Groq stages. Chunk summaries are tight by
# design; the reduce step needs more room for the merged narrative.
_CHUNK_MAX_TOKENS = 900
_REDUCE_MAX_TOKENS = 1400


class SummarizationService:
    """Map-reduce summarisation over a stream of (preprocessed) reviews."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        chunker: ReviewChunker | None = None,
        aggregator: ThemeAggregator | None = None,
        groq_config: GroqConfig | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        pipeline = self._settings.pipeline
        groq = self._settings.groq
        self._chunker = chunker or ReviewChunker(
            target_tokens=pipeline.CHUNK_TARGET_TOKENS,
            max_reviews=pipeline.CHUNK_SIZE,
            max_body_chars=pipeline.MAX_REVIEW_BODY_CHARS,
        )
        self._aggregator = aggregator or ThemeAggregator(
            max_themes=self._settings.summary.MAX_THEMES,
        )
        self._groq_config = groq_config or GroqConfig(
            api_key=self._settings.groq_api_key,
            model=groq.MODEL,
            temperature=groq.TEMPERATURE,
            timeout=groq.TIMEOUT,
            max_retries=groq.MAX_RETRIES,
            fallback_models=groq.FALLBACK_MODELS,
            max_consecutive_429=groq.MAX_429_STREAK,
            request_concurrency=groq.REQUEST_CONCURRENCY,
        )
        # Sequential by default so we don't blow Groq's TPM ceiling on
        # free/dev tiers. Production can raise SUMMARIZATION_CONCURRENCY
        # alongside a higher Groq tier.
        self._max_concurrency = max(1, pipeline.SUMMARIZATION_CONCURRENCY)
        self._token_threshold = max(1, groq.TOKEN_THRESHOLD)
        self._max_body_chars = pipeline.MAX_REVIEW_BODY_CHARS

    async def summarize(
        self,
        reviews: Iterable[Review],
        *,
        client: GroqClient | None = None,
    ) -> SummaryReport:
        """Run the full pipeline. ``client`` is injectable for tests.

        Guarantees a usable :class:`SummaryReport` even under adverse
        conditions: if every chunk fails (rate-limit storm, model
        outage), it falls back to :func:`build_heuristic_report` so
        leadership sees directional themes instead of "no themes".
        """
        review_list: list[Review] = list(reviews)
        chunks = list(self._chunker.chunk(review_list))
        # Adaptive splitting: re-pack any chunk that exceeds the token
        # threshold so a single batch of long reviews can't push the
        # prompt past the model context limit or the TPM ceiling.
        expanded: list[ReviewChunk] = []
        for c in chunks:
            expanded.extend(
                split_if_oversized(
                    c,
                    token_threshold=self._token_threshold,
                    max_body_chars=self._max_body_chars,
                )
            )
        log.info(
            "summarization.start",
            reviews=len(review_list),
            chunks_initial=len(chunks),
            chunks_after_split=len(expanded),
            concurrency=self._max_concurrency,
            token_threshold=self._token_threshold,
        )

        if not expanded:
            return SummaryReport(
                executive_summary="",
                themes=[],
                stats=SummaryStats(total_reviews=0),
            )

        async def _run(c: GroqClient) -> SummaryReport:
            chunk_summaries, failed, reasons = await self._map(expanded, c)
            if chunk_summaries:
                report = await self._aggregator.aggregate(
                    chunk_summaries,
                    c,
                    total_reviews=len(review_list),
                    chunks_failed=failed,
                )
                if not report.themes:
                    log.warning(
                        "summarization.aggregator_empty.fallback_heuristic",
                        chunks_ok=len(chunk_summaries),
                        chunks_failed=failed,
                    )
                    return build_heuristic_report(
                        review_list,
                        max_themes=self._settings.summary.MAX_THEMES,
                        failure_reasons=reasons,
                    )
                return report
            log.warning(
                "summarization.all_chunks_failed.fallback_heuristic",
                chunks_failed=failed,
                reasons=list(reasons)[:5],
            )
            return build_heuristic_report(
                review_list,
                max_themes=self._settings.summary.MAX_THEMES,
                failure_reasons=reasons,
            )

        if client is not None:
            return await _run(client)

        async with GroqClient(self._groq_config) as owned:
            return await _run(owned)

    async def _map(
        self,
        chunks: Sequence[ReviewChunk],
        client: GroqClient,
    ) -> tuple[list[ChunkSummary], int, list[str]]:
        """Map step. Sequential by default; failures don't poison the run.

        Returns ``(ok_summaries, failed_count, failure_reasons)``.
        ``failure_reasons`` carries short labels for each failure so the
        heuristic fallback (and operators) can see *why* a chunk
        dropped.
        """
        sem = asyncio.Semaphore(self._max_concurrency)
        failures: list[str] = []
        failures_lock = asyncio.Lock()

        async def _one(chunk: ReviewChunk) -> ChunkSummary | None:
            async with sem:
                started = time.monotonic()
                result = await self._summarize_chunk(chunk, client)
                elapsed_ms = int((time.monotonic() - started) * 1000)
                log.info(
                    "summarization.chunk.done",
                    chunk_id=chunk.chunk_id,
                    review_count=chunk.review_count,
                    token_estimate=chunk.token_estimate,
                    active_model=client.active_model,
                    elapsed_ms=elapsed_ms,
                    ok=result is not None,
                )
                return result

        async def _gather_one(chunk: ReviewChunk) -> ChunkSummary | None:
            try:
                return await _one(chunk)
            except Exception as exc:
                reason = f"{type(exc).__name__}: {str(exc)[:120]}"
                async with failures_lock:
                    failures.append(reason)
                log.warning(
                    "summarization.chunk_failed",
                    chunk_id=chunk.chunk_id,
                    error=type(exc).__name__,
                    detail=str(exc)[:200],
                )
                return None

        results = await asyncio.gather(*(_gather_one(c) for c in chunks))
        ok = [r for r in results if r is not None]
        failed = len(results) - len(ok)
        log.info(
            "summarization.map.done",
            ok=len(ok),
            failed=failed,
            success_rate=round(len(ok) / max(1, len(results)), 3),
            active_model=client.active_model,
        )
        return ok, failed, failures

    async def _summarize_chunk(
        self,
        chunk: ReviewChunk,
        client: GroqClient,
    ) -> ChunkSummary | None:
        user_prompt = CHUNK_USER_PROMPT_TEMPLATE.format(rendered=chunk.rendered)
        try:
            raw = await client.complete_json(
                system=CHUNK_SYSTEM_PROMPT,
                user=user_prompt,
                max_tokens=_CHUNK_MAX_TOKENS,
                stage="chunk",
            )
        except Exception as exc:
            # Re-raise so ``_map._gather_one`` records the reason. We
            # used to swallow this into ``return None`` which lost the
            # cause attribution the heuristic fallback now needs.
            log.warning(
                "summarization.chunk.groq_failed",
                chunk_id=chunk.chunk_id,
                error=type(exc).__name__,
                detail=str(exc)[:200],
            )
            raise

        raw_themes = raw.get("themes") if isinstance(raw, dict) else None
        before_count = len(raw_themes) if isinstance(raw_themes, list) else 0
        sanitised_theme_dicts = sanitize_llm_output(raw, kind="chunk")
        themes: list[ChunkTheme] = []
        skipped = 0
        for t in sanitised_theme_dicts:
            try:
                themes.append(ChunkTheme.model_validate(t))
            except ValidationError as exc:
                # Per-entry rather than per-list — one malformed theme
                # shouldn't blank the rest of a successful chunk.
                skipped += 1
                log.warning(
                    "summarization.chunk_invalid_theme",
                    chunk_id=chunk.chunk_id,
                    label=t.get("label"),
                    error=str(exc)[:200],
                )

        log.info(
            "summarization.chunk.themes_validated",
            chunk_id=chunk.chunk_id,
            themes_before_validation=before_count,
            themes_after_sanitization=len(sanitised_theme_dicts),
            themes_accepted=len(themes),
            skipped_invalid_themes=skipped,
        )

        summary_text = _clip(str(raw.get("summary") or "").strip(), 1200)
        return ChunkSummary(
            chunk_id=chunk.chunk_id,
            review_count=chunk.review_count,
            summary=summary_text,
            themes=themes,
        )


def _clip(text: str, max_len: int) -> str:
    """Hard char-cap for sanitisation. We don't add an ellipsis here —
    the Pulse layer's ``clamp_text`` is the user-facing presenter."""
    if max_len <= 0 or len(text) <= max_len:
        return text
    return text[:max_len]


