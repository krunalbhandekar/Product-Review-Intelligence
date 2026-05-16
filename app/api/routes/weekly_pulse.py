from fastapi import APIRouter, Depends, Response, status
from pydantic import BaseModel, ConfigDict

from app.api.deps import require_api_key
from app.services.weekly_pulse import (
    WeeklyPulseRequest,
    WeeklyPulseService,
)

router = APIRouter(tags=["pipeline"])


# Map orchestration status → HTTP status code.
#   succeeded → 200 (everything shipped)
#   partial   → 207 Multi-Status (doc published, draft failed)
#   no_data   → 204 No Content (window empty; body suppressed)
#   failed    → 500 (pipeline raised after retries)
_STATUS_HTTP = {
    "succeeded": status.HTTP_200_OK,
    "partial": status.HTTP_207_MULTI_STATUS,
    "no_data": status.HTTP_204_NO_CONTENT,
    "failed": status.HTTP_500_INTERNAL_SERVER_ERROR,
}


class RunWeeklyPulseBody(BaseModel):
    """Request body for ``POST /run-weekly-pulse``.

    The lookback window is intentionally not accepted from the client —
    it is a server-side operations knob loaded from ``LOOKBACK_WEEKS``.
    ``extra="forbid"`` makes a stale client that still sends
    ``lookback_weeks`` (or any other unknown field) fail with a clear
    422 instead of being silently ignored.
    """

    model_config = ConfigDict(extra="forbid")

    playstore_app_id: str | None = None
    appstore_app_id: str | None = None
    email_to: str | None = None
    dry_run: bool = False


class RunWeeklyPulseResponse(BaseModel):
    run_id: str
    status: str
    dry_run: bool
    started_at: str
    finished_at: str | None
    window_start: str | None
    window_end: str | None
    reviews_ingested: int
    chunks_summarized: int
    email_sent: bool
    details: dict[str, object]


@router.post(
    "/run-weekly-pulse",
    response_model=RunWeeklyPulseResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(require_api_key)],
    responses={
        204: {"description": "No reviews found in window"},
        207: {"description": "Pipeline ran; delivery partially succeeded"},
        500: {"description": "Pipeline failed"},
    },
)
async def run_weekly_pulse(
    body: RunWeeklyPulseBody, response: Response
) -> RunWeeklyPulseResponse | Response:
    service = WeeklyPulseService()
    result = await service.run(
        WeeklyPulseRequest(
            playstore_app_id=body.playstore_app_id,
            appstore_app_id=body.appstore_app_id,
            email_to=body.email_to,
            dry_run=body.dry_run,
        )
    )
    http_status = _STATUS_HTTP.get(result.status, status.HTTP_200_OK)

    # 204 must not carry a body. Return a bare Response so FastAPI doesn't
    # try to serialise the result model.
    if http_status == status.HTTP_204_NO_CONTENT:
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    response.status_code = http_status
    return RunWeeklyPulseResponse(
        run_id=result.run_id,
        status=result.status,
        dry_run=result.dry_run,
        started_at=result.started_at.isoformat(),
        finished_at=result.finished_at.isoformat() if result.finished_at else None,
        window_start=result.window_start.isoformat() if result.window_start else None,
        window_end=result.window_end.isoformat() if result.window_end else None,
        reviews_ingested=result.reviews_ingested,
        chunks_summarized=result.chunks_summarized,
        email_sent=result.email_sent,
        details=result.details,
    )
