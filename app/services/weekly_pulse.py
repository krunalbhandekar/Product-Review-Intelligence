"""End-to-end weekly pulse orchestration.

Wires the existing review pipeline into a single async entrypoint that
the API route and the scheduled job both call:

    ingest (Play Store + App Store)
      → preprocess (PII scrub, dedup)
      → summarize (Groq map/reduce)
      → generate pulse (aggregation + formatting)
      → deliver via MCP (Google Docs + Gmail draft)

Status semantics
----------------
* ``succeeded`` — pipeline ran and both MCP steps (doc + email) worked.
* ``partial``   — pipeline ran, the Google Doc was published, but the
                  Gmail draft step failed after retries. ``details.delivery``
                  carries the doc URL so the operator can send manually.
* ``no_data``   — no reviews fell in the window; delivery was skipped.
* ``failed``    — ingest/summarisation raised, OR the doc append step
                  failed after retries. ``details.error`` carries the cause.

The service is dependency-injection-friendly: ingest, summarisation,
generation, and delivery can each be replaced for tests (see
:class:`tests.test_weekly_pulse_service`).
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from app.config import Settings, get_settings
from app.core.exceptions import UpstreamError
from app.core.logging import get_logger
from app.domain.enums import Platform
from app.domain.models import Review
from app.services.ingest.app_store import AppStoreSource
from app.services.ingest.play_store import PlayStoreSource
from app.services.mcp.client import MCPClientError
from app.services.mcp.delivery import PulseDeliveryService
from app.services.preprocessing.service import PreprocessingService
from app.services.pulse.service import WeeklyPulseGenerator
from app.services.summarization.groq_client import GroqClient, GroqConfig
from app.services.summarization.schemas import SummaryReport
from app.services.summarization.service import SummarizationService

log = get_logger("service.weekly_pulse")

IngestFn = Callable[[str, datetime, datetime], Awaitable[list[Review]]]


@dataclass(frozen=True)
class WeeklyPulseRequest:
    """Input parameters for a weekly pulse run.

    Note: the lookback window is intentionally not a request input — it
    is loaded from ``Settings.lookback_weeks`` at service-init time so
    that operators (not API callers) control window size.
    """

    playstore_app_id: str | None = None
    appstore_app_id: str | None = None
    email_to: str | None = None
    dry_run: bool = False


@dataclass
class WeeklyPulseResult:
    """Outcome of a weekly pulse run."""

    run_id: str
    status: str
    dry_run: bool
    started_at: datetime
    finished_at: datetime | None = None
    window_start: datetime | None = None
    window_end: datetime | None = None
    reviews_ingested: int = 0
    chunks_summarized: int = 0
    email_sent: bool = False
    details: dict[str, Any] = field(default_factory=dict)


class WeeklyPulseService:
    """Orchestrates the weekly review-pulse pipeline.

    Stages: ingest reviews → preprocess → chunk + summarize → generate
    pulse → deliver via MCP. ``dry_run`` short-circuits all side effects
    (no network calls, no LLM, no MCP) while still exercising config,
    window calculation, and logging.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        ingest_play_fn: IngestFn | None = None,
        ingest_app_fn: IngestFn | None = None,
        summarizer: SummarizationService | None = None,
        generator: WeeklyPulseGenerator | None = None,
        delivery: PulseDeliveryService | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._ingest_play = ingest_play_fn or self._default_ingest_play
        self._ingest_app = ingest_app_fn or self._default_ingest_app
        self._summarizer = summarizer
        self._generator = generator or WeeklyPulseGenerator(
            word_budget=self.settings.summary.SUMMARY_MAX_WORDS
        )
        self._delivery = delivery

    # -- public --------------------------------------------------------

    async def run(self, request: WeeklyPulseRequest) -> WeeklyPulseResult:
        run_id = uuid.uuid4().hex
        started_at = datetime.now(tz=UTC)

        window_start, window_end = self._window(self.settings.pipeline.LOOKBACK_WEEKS)
        playstore_app_id = request.playstore_app_id or self.settings.playstore_app_id
        appstore_app_id = request.appstore_app_id or self.settings.appstore_app_id
        email_to = request.email_to or self.settings.email_to

        bound = log.bind(run_id=run_id, dry_run=request.dry_run)
        bound.info(
            "weekly_pulse.start",
            playstore_app_id=playstore_app_id or None,
            appstore_app_id=appstore_app_id or None,
            window_start=window_start.isoformat(),
            window_end=window_end.isoformat(),
            email_to=email_to or None,
        )

        result = WeeklyPulseResult(
            run_id=run_id,
            status="succeeded",
            dry_run=request.dry_run,
            started_at=started_at,
            window_start=window_start,
            window_end=window_end,
        )

        if request.dry_run:
            bound.info("weekly_pulse.dry_run.skip_pipeline")
            result.details = {
                "message": "dry_run: pipeline stages skipped",
                "would_ingest_from": {
                    "playstore_app_id": playstore_app_id or None,
                    "appstore_app_id": appstore_app_id or None,
                },
                "would_email_to": email_to or None,
            }
            result.finished_at = datetime.now(tz=UTC)
            bound.info("weekly_pulse.finish", status=result.status)
            return result

        try:
            reviews_by_platform = await self._ingest_all(
                playstore_app_id=playstore_app_id,
                appstore_app_id=appstore_app_id,
                window_start=window_start,
                window_end=window_end,
                bound=bound,
            )
            result.reviews_ingested = sum(
                len(rs) for rs in reviews_by_platform.values()
            )

            if result.reviews_ingested == 0:
                result.status = "no_data"
                result.details = {
                    "message": "No reviews found in window",
                    "playstore_app_id": playstore_app_id or None,
                    "appstore_app_id": appstore_app_id or None,
                }
                result.finished_at = datetime.now(tz=UTC)
                bound.info("weekly_pulse.finish", status=result.status)
                return result

            reports = await self._summarize_all(reviews_by_platform, bound=bound)
            result.chunks_summarized = sum(r.stats.chunks_processed for r in reports)

            artifacts = self._generator.generate(
                reports,
                window_start=window_start,
                window_end=window_end,
            )

            delivery = self._delivery or PulseDeliveryService(settings=self.settings)
            # Derive a stable idempotency key from run_id + window end date.
            # The run_id makes each run unique; the window_end_date keeps
            # the key human-debuggable in MCP server logs.
            idem_key = f"{run_id}-{window_end.date().isoformat()}"
            delivery_result = await delivery.deliver(
                artifacts, to=email_to, idempotency_key=idem_key
            )

            result.status = delivery_result.status
            result.email_sent = delivery_result.draft_id is not None
            result.details = {
                "delivery": {
                    "status": delivery_result.status,
                    "document_url": delivery_result.document_url,
                    "draft_id": delivery_result.draft_id,
                    "doc_title": delivery_result.doc_title,
                    "email_to": delivery_result.email_to,
                    "email_subject": delivery_result.email_subject,
                    "failure_stage": delivery_result.failure_stage,
                    "error": delivery_result.error,
                },
                "reports": [
                    {
                        "total_reviews": r.stats.total_reviews,
                        "chunks_processed": r.stats.chunks_processed,
                        "chunks_failed": r.stats.chunks_failed,
                    }
                    for r in reports
                ],
            }

        except MCPClientError as exc:
            # Only reachable when `append_to_doc` fails — draft failures are
            # handled inside PulseDeliveryService and surface as `partial`.
            bound.error(
                "weekly_pulse.delivery.doc_failed",
                status_code=exc.status_code,
                error=str(exc),
            )
            result.status = "failed"
            result.details = {
                "stage": "mcp_append_to_doc",
                "status_code": exc.status_code,
                "error": str(exc),
            }
        except UpstreamError as exc:
            bound.error(
                "weekly_pulse.upstream_failed",
                status_code=exc.status_code,
                error=str(exc),
            )
            result.status = "failed"
            result.details = {
                "stage": "pipeline",
                "status_code": exc.status_code,
                "error": str(exc),
            }

        result.finished_at = datetime.now(tz=UTC)
        bound.info("weekly_pulse.finish", status=result.status)
        return result

    # -- pipeline stages -----------------------------------------------

    async def _ingest_all(
        self,
        *,
        playstore_app_id: str | None,
        appstore_app_id: str | None,
        window_start: datetime,
        window_end: datetime,
        bound: Any,
    ) -> dict[Platform, list[Review]]:
        # Run Play Store + App Store ingestion concurrently — they hit
        # independent upstreams, so there's no reason to serialise them.
        async def _play() -> tuple[Platform, list[Review]] | None:
            if not playstore_app_id:
                return None
            reviews = await self._ingest_play(
                playstore_app_id, window_start, window_end
            )
            bound.info(
                "weekly_pulse.ingest.play_store",
                app_id=playstore_app_id,
                count=len(reviews),
            )
            return (Platform.ANDROID, reviews) if reviews else None

        async def _app() -> tuple[Platform, list[Review]] | None:
            if not appstore_app_id:
                return None
            reviews = await self._ingest_app(
                appstore_app_id, window_start, window_end
            )
            bound.info(
                "weekly_pulse.ingest.app_store",
                app_id=appstore_app_id,
                count=len(reviews),
            )
            return (Platform.IOS, reviews) if reviews else None

        # return_exceptions=True so one platform's upstream failure cannot
        # cancel the other. We surface the partial outcome via logs; the
        # caller continues with whatever data did arrive. Only if BOTH
        # platforms fail do we re-raise so the run is marked failed.
        raw = await asyncio.gather(_play(), _app(), return_exceptions=True)
        platforms = ("play_store", "app_store")
        out: dict[Platform, list[Review]] = {}
        errors: list[tuple[str, BaseException]] = []
        for name, item in zip(platforms, raw, strict=True):
            if isinstance(item, BaseException):
                bound.warning(
                    "weekly_pulse.ingest.platform_failed",
                    platform=name,
                    error=f"{type(item).__name__}: {item}",
                )
                errors.append((name, item))
            elif item is not None:
                platform, reviews = item
                out[platform] = reviews

        if errors and not out:
            # All platforms failed — propagate the first error so the
            # pipeline reports a clear failure rather than `no_data`.
            raise errors[0][1]
        return out

    async def _summarize_all(
        self,
        reviews_by_platform: dict[Platform, list[Review]],
        *,
        bound: Any,
    ) -> list[SummaryReport]:
        # Platforms are independent — summarise them concurrently. The Groq
        # client already bounds in-flight calls via the request semaphore
        # so the shared client absorbs the extra parallelism safely.
        async def _one(
            summ: SummarizationService,
            platform: Platform,
            reviews: list[Review],
            client: GroqClient | None,
        ) -> SummaryReport:
            report = await summ.summarize(reviews, client=client) if client is not None else await summ.summarize(reviews)
            bound.info(
                "weekly_pulse.summarize.platform",
                platform=platform.value,
                chunks=report.stats.chunks_processed,
                themes=len(report.themes),
            )
            return report

        if self._summarizer is not None:
            return await asyncio.gather(
                *(
                    _one(self._summarizer, platform, reviews, None)
                    for platform, reviews in reviews_by_platform.items()
                )
            )

        summarizer = SummarizationService(self.settings)
        groq_cfg = self.settings.groq
        cfg = GroqConfig(
            api_key=self.settings.groq_api_key,
            model=groq_cfg.MODEL,
            temperature=groq_cfg.TEMPERATURE,
            timeout=groq_cfg.TIMEOUT,
            max_retries=groq_cfg.MAX_RETRIES,
            fallback_models=groq_cfg.FALLBACK_MODELS,
            max_consecutive_429=groq_cfg.MAX_429_STREAK,
            request_concurrency=groq_cfg.REQUEST_CONCURRENCY,
        )
        async with GroqClient(cfg) as groq:
            return list(
                await asyncio.gather(
                    *(
                        _one(summarizer, platform, reviews, groq)
                        for platform, reviews in reviews_by_platform.items()
                    )
                )
            )

    # -- default ingest implementations --------------------------------

    async def _default_ingest_play(
        self, app_id: str, since: datetime, until: datetime
    ) -> list[Review]:
        source = PlayStoreSource(max_reviews=self.settings.pipeline.MAX_REVIEWS_PER_RUN)
        preprocess = PreprocessingService()
        return [
            r
            async for r in preprocess.preprocess_stream(
                source.stream(app_id=app_id, since=since, until=until)
            )
        ]

    async def _default_ingest_app(
        self, app_id: str, since: datetime, until: datetime
    ) -> list[Review]:
        preprocess = PreprocessingService()
        async with AppStoreSource(
            max_reviews=self.settings.pipeline.MAX_REVIEWS_PER_RUN
        ) as source:
            return [
                r
                async for r in preprocess.preprocess_stream(
                    source.stream(app_id=app_id, since=since, until=until)
                )
            ]

    # -- helpers -------------------------------------------------------

    def _window(self, lookback_weeks: int) -> tuple[datetime, datetime]:
        now = datetime.now(tz=UTC)
        return now - timedelta(weeks=lookback_weeks), now


__all__ = [
    "IngestFn",
    "WeeklyPulseRequest",
    "WeeklyPulseResult",
    "WeeklyPulseService",
]
